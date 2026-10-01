# Offline setup: the periodic hexahedral mesh and its materials, the 8-node element matrices, the influence
# functions E and P of the actual composite and P0 of the homogeneous reference, and their cache.

import hashlib
import sys
import time
from typing import NamedTuple
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import tqdm

import config
from lattice_fft import get_P0_transformed

## ------- Mesh, Geometry and Materials ------- ##

class Mesh(NamedTuple):
    element_nodes: np.ndarray
    element_partition_ids: np.ndarray
    element_material_ids: np.ndarray

class PartitionMaterials(NamedTuple):
    L: np.ndarray
    yield_stress: np.ndarray
    hardening_modulus: np.ndarray

def get_grid_id(i, j, k, count_per_side):
    return i + count_per_side * j + count_per_side**2 * k

def is_inside_inclusion(centre):
    offset_from_inclusion_axis = centre[:2] - config.domain_side_length / 2
    if config.inclusion_shape == "circle":
        return np.sum(offset_from_inclusion_axis**2) < config.inclusion_radius**2
    return np.all(np.abs(offset_from_inclusion_axis) < config.inclusion_side_length / 2)

def get_partition_material_ids():
    partition_material_ids = np.zeros(config.partition_count, dtype=int)

    for j in range(config.partition_number_per_side):
        for i in range(config.partition_number_per_side):
            partition_id = get_grid_id(i, j, 0, config.partition_number_per_side)
            partition_centre = (np.array([i, j]) + 0.5) * config.partition_side_length
            partition_material_ids[partition_id] = int(is_inside_inclusion(partition_centre))

    return partition_material_ids

def get_periodic_mesh(partition_material_ids):
    element_nodes = np.zeros((config.element_count, 8), dtype=int)
    element_partition_ids = np.zeros(config.element_count, dtype=int)

    for k in range(config.element_number_along_z):
        for j in range(config.element_number_per_side):
            for i in range(config.element_number_per_side):
                element_id = get_grid_id(i, j, k, config.element_number_per_side)

                for offset_k in range(2):
                    for offset_j in range(2):
                        for offset_i in range(2):
                            corner = get_grid_id(offset_i, offset_j, offset_k, 2)
                            element_nodes[element_id, corner] = get_grid_id(
                                (i + offset_i) % config.element_number_per_side,
                                (j + offset_j) % config.element_number_per_side,
                                (k + offset_k) % config.element_number_along_z,
                                config.element_number_per_side)

                element_partition_ids[element_id] = get_grid_id(i // config.elements_per_partition_side,
                                                                j // config.elements_per_partition_side,
                                                                0,
                                                                config.partition_number_per_side)

    # Elements take their partition's material, so every partition is a single material and the inclusion is
    # resolved at partition, not element, resolution.
    element_material_ids = partition_material_ids[element_partition_ids]
    return Mesh(element_nodes, element_partition_ids, element_material_ids)

def get_value_per_material(matrix_value, inclusion_value, material_ids):
    value_by_material_id = np.array([matrix_value, inclusion_value])
    return value_by_material_id[material_ids]

def get_L(elastic_modulus, poisson_ratio):
    lame_lambda   = elastic_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    shear_modulus = elastic_modulus / (2 * (1 + poisson_ratio))
    normal_block = lame_lambda * np.ones((3, 3)) + 2 * shear_modulus * np.eye(3)
    shear_block  = shear_modulus * np.eye(3)
    L = np.zeros((6, 6))
    L[:3, :3] = normal_block
    L[3:, 3:] = shear_block
    return L

def get_partition_materials(L_matrix, L_inclusion, partition_material_ids):
    return PartitionMaterials(
        L=get_value_per_material(L_matrix, L_inclusion, partition_material_ids),
        yield_stress=get_value_per_material(config.matrix_yield_stress, config.inclusion_yield_stress,
                                            partition_material_ids),
        hardening_modulus=get_value_per_material(config.matrix_hardening_modulus, config.inclusion_hardening_modulus,
                                                 partition_material_ids))

## ------- Element Matrices and Assembly ------- ##

corner_natural_coordinates = np.array([[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
                                       [-1, -1,  1], [1, -1,  1], [-1, 1,  1], [1, 1,  1]])

def get_shape_function_derivatives(xi, eta, zeta):
    corner_xi, corner_eta, corner_zeta = corner_natural_coordinates.T
    xi_bracket   = 1 + corner_xi   * xi
    eta_bracket  = 1 + corner_eta  * eta
    zeta_bracket = 1 + corner_zeta * zeta
    derivative_wrt_xi   = corner_xi   * eta_bracket * zeta_bracket / 8
    derivative_wrt_eta  = corner_eta  * xi_bracket  * zeta_bracket / 8
    derivative_wrt_zeta = corner_zeta * xi_bracket  * eta_bracket  / 8
    return derivative_wrt_xi, derivative_wrt_eta, derivative_wrt_zeta

def get_B():
    gauss_point_coordinates = corner_natural_coordinates / np.sqrt(3)
    B = np.zeros((8, 6, 24))

    for gauss_index, (xi, eta, zeta) in enumerate(gauss_point_coordinates):
        derivatives_wrt_xi, derivatives_wrt_eta, derivatives_wrt_zeta = get_shape_function_derivatives(xi, eta, zeta)
        derivatives_wrt_x = derivatives_wrt_xi   * 2 / config.element_side_length
        derivatives_wrt_y = derivatives_wrt_eta  * 2 / config.element_side_length
        derivatives_wrt_z = derivatives_wrt_zeta * 2 / config.element_length_along_z

        for corner in range(8):
            column = 3 * corner
            B[gauss_index, 0, column]     = derivatives_wrt_x[corner]
            B[gauss_index, 1, column + 1] = derivatives_wrt_y[corner]
            B[gauss_index, 2, column + 2] = derivatives_wrt_z[corner]
            B[gauss_index, 3, column]     = derivatives_wrt_y[corner]
            B[gauss_index, 3, column + 1] = derivatives_wrt_x[corner]
            B[gauss_index, 4, column + 1] = derivatives_wrt_z[corner]
            B[gauss_index, 4, column + 2] = derivatives_wrt_y[corner]
            B[gauss_index, 5, column]     = derivatives_wrt_z[corner]
            B[gauss_index, 5, column + 2] = derivatives_wrt_x[corner]

    return B

def get_element_dofs(element_nodes):
    node_dofs = 3 * element_nodes[:, :, None] + np.arange(3)
    return node_dofs.reshape(config.element_count, 24)

def get_partition_components(element_partition_ids):
    # The six entries of each element's partition in a partition-stacked vector such as ε, μ or b.
    return 6 * element_partition_ids[:, None] + np.arange(6)

def assemble_element_blocks(element_blocks, element_rows, element_columns, shape):
    # Adds each element's block into a sparse global matrix at the given global rows and columns. Blocks that share
    # an entry are summed, as in finite-element assembly.
    rows = np.broadcast_to(element_rows[:, :, None], element_blocks.shape)
    columns = np.broadcast_to(element_columns[:, None, :], element_blocks.shape)
    return scipy.sparse.coo_matrix((element_blocks.ravel(), (rows.ravel(), columns.ravel())), shape=shape).tocsr()

def get_K_element(B, L):
    K_element = np.zeros((24, 24))
    for gauss_index in range(8):
        K_element += config.gauss_point_volume_weight * B[gauss_index].T @ L @ B[gauss_index]
    return K_element

def get_K(B, L_per_element, element_nodes):
    element_dofs = get_element_dofs(element_nodes)
    K_elements = np.array([get_K_element(B, L) for L in L_per_element])
    return assemble_element_blocks(K_elements, element_dofs, element_dofs, (config.dof_count, config.dof_count))

def get_integrated_B(B):
    return config.gauss_point_volume_weight * np.sum(B, axis=0)

def get_F_elements(B, L_per_element):
    # Each element's nodal load per unit eigenstrain, ∫ Bᵀ L dV (24 × 6).
    return get_integrated_B(B).T @ L_per_element

def get_free_dofs():
    # Pinning one node's three displacements removes the rigid translation of the periodic cell.
    pinned_dofs = np.arange(3)
    return np.setdiff1d(np.arange(config.dof_count), pinned_dofs)

## ------- Influence Functions (Offline) ------- ##

def get_F_macrostrain(B, L_per_element, element_nodes):
    # Nodal loads per unit macrostrain. Moving the applied strain to the right-hand side gives -∫ Bᵀ L ε̄ dV.
    macrostrain_columns = np.tile(np.arange(6), (config.element_count, 1))
    return -assemble_element_blocks(get_F_elements(B, L_per_element), get_element_dofs(element_nodes),
                                    macrostrain_columns, (config.dof_count, 6))

def get_F_eigenstrain(B, L_per_element, element_nodes, element_partition_ids):
    # F_μ in ER-24: nodal loads per unit partition eigenstrain, one block of six columns per partition.
    return assemble_element_blocks(get_F_elements(B, L_per_element), get_element_dofs(element_nodes),
                                   get_partition_components(element_partition_ids),
                                   (config.dof_count, 6 * config.partition_count))

def get_partition_averaging_operator(B, element_nodes, element_partition_ids):
    # A_ε in ER-24: maps nodal displacements to partition-average strain.
    element_blocks = np.broadcast_to(get_integrated_B(B) / config.partition_volume, (config.element_count, 6, 24))
    return assemble_element_blocks(element_blocks, get_partition_components(element_partition_ids),
                                   get_element_dofs(element_nodes), (6 * config.partition_count, config.dof_count))

def solve_influence_function(K, F, A):
    # The partition-average strain response A K⁻¹ F to every load column of F.
    free_dofs = get_free_dofs()
    K_free = K[free_dofs][:, free_dofs].tocsc()
    print(f"  {f'factor K ({K_free.shape[0]} DOFs)':<26}: ", end="", flush=True)
    start_time = time.perf_counter()
    K_free_factorization = scipy.sparse.linalg.splu(K_free, permc_spec='MMD_AT_PLUS_A')
    print(f"{time.perf_counter() - start_time:.1f} s")

    F_free = F.tocsr()[free_dofs].tocsc()
    columns_per_solve = 100
    average_strain = np.zeros((6 * config.partition_count, F.shape[1]))
    column_batches = tqdm.tqdm(range(0, F.shape[1], columns_per_solve),
                               desc=f"  {f'solve {F.shape[1]} load columns':<26}",
                               bar_format="{desc}: {percentage:3.0f}% |{bar:25}| {n_fmt}/{total_fmt} batches "
                                          "[{elapsed} elapsed, {remaining} left]", file=sys.stdout)
    for first_column in column_batches:
        columns = slice(first_column, first_column + columns_per_solve)
        batch_F_free = F_free[:, columns].toarray()
        batch_displacements = np.zeros((config.dof_count, batch_F_free.shape[1]))
        batch_displacements[free_dofs] = K_free_factorization.solve(batch_F_free)
        average_strain[:, columns] = A @ batch_displacements
    return average_strain

def get_influence_functions(B, L_per_element, mesh, A):
    K = get_K(B, L_per_element, mesh.element_nodes)
    F_macrostrain_and_eigenstrain = scipy.sparse.hstack([
        get_F_macrostrain(B, L_per_element, mesh.element_nodes),
        get_F_eigenstrain(B, L_per_element, mesh.element_nodes, mesh.element_partition_ids)])
    average_strain = solve_influence_function(K, F_macrostrain_and_eigenstrain, A)
    # ER-1: E is the applied macrostrain plus the fluctuation it causes; P is A_ε K⁻¹ F_μ (ER-24).
    E = np.tile(np.eye(6), (config.partition_count, 1)) + average_strain[:, :6]
    P = average_strain[:, 6:]
    return E, P

def get_homogenized_L(E, L_per_partition):
    E_blocks = E.reshape(config.partition_count, 6, 6)
    return np.mean(L_per_partition @ E_blocks, axis=0)

def get_reference_L(E, L_per_partition, L_matrix):
    # The homogeneous reference stiffness C0 chosen by config.reference_stiffness. Every partition has the same volume,
    # so the averages are plain means over the partitions.
    if config.reference_stiffness == "homogenized":
        return get_homogenized_L(E, L_per_partition)
    if config.reference_stiffness == "voigt":
        return np.mean(L_per_partition, axis=0)
    return L_matrix

def get_P0_offset_blocks(B, reference_L_per_element, mesh, A):
    # The reference response to eigenstrain in partition 0 alone. Reference interactions depend only on the offset
    # between partitions (ER-6e), so block B of this response is P0[B - 0] and one solve gives the whole kernel.
    K = get_K(B, reference_L_per_element, mesh.element_nodes)
    F_eigenstrain = get_F_eigenstrain(B, reference_L_per_element, mesh.element_nodes, mesh.element_partition_ids)
    F_eigenstrain_in_first_partition = F_eigenstrain[:, :6]
    return solve_influence_function(K, F_eigenstrain_in_first_partition, A)

## ------- Cache ------- ##

# Increase whenever a code change alters the offline operators, so caches made by the old code are recomputed.
cache_version = 1

def get_cache_path(kind, parameters):
    # One file per parameter set, so switching between configurations or entry points reuses each one's operators
    # instead of overwriting a single file. The parameters are also stored in the file and checked on load.
    digest = hashlib.sha256()
    for name in sorted(parameters):
        value = np.asarray(parameters[name])
        digest.update(f"{name}:{value.dtype.str}:{value.shape}:".encode())
        digest.update(value.tobytes())
    return config.cache_folder / f"{kind}_{digest.hexdigest()[:12]}.npz"

def load_cache(cache_path, parameters):
    if not cache_path.exists():
        print(f"{cache_path.name}: no cache found, computing.")
        return None
    with np.load(cache_path) as cached:
        for name, value in parameters.items():
            if name not in cached.files or not np.array_equal(cached[name], value):
                print(f"{cache_path.name}: {name} missing or changed, recomputing.")
                return None
        print(f"{cache_path.name}: loaded from cache.")
        return {name: cached[name] for name in cached.files}

def save_cache(cache_path, parameters, **arrays):
    config.cache_folder.mkdir(exist_ok=True)
    np.savez(cache_path, **arrays, **parameters)

def get_E_and_P(mesh, B, L_per_element, L_matrix, L_inclusion, A):
    E_P_parameters = {"cache_version": cache_version,
                      "domain_side_length": config.domain_side_length,
                      "inclusion_shape": config.inclusion_shape,
                      "element_number_per_side": config.element_number_per_side,
                      "element_number_along_z": config.element_number_along_z,
                      "partition_number_per_side": config.partition_number_per_side,
                      "L_matrix": L_matrix,
                      "L_inclusion": L_inclusion}
    if config.inclusion_shape == "square":
        E_P_parameters["inclusion_side_length"] = config.inclusion_side_length
    else:
        E_P_parameters["inclusion_radius"] = config.inclusion_radius
    E_P_cache_path = get_cache_path("E_P", E_P_parameters)
    cached_E_P = load_cache(E_P_cache_path, E_P_parameters)
    if cached_E_P is None:
        E, P = get_influence_functions(B, L_per_element, mesh, A)
        save_cache(E_P_cache_path, E_P_parameters, E=E, P=P)
    else:
        E, P = cached_E_P["E"], cached_E_P["P"]
    # Column-major, so the columns of a run of consecutive partitions are contiguous and P products can read just the
    # yielding partitions' columns without copying them (tfa_solvers.apply_P_to_partitions).
    return E, np.asfortranarray(P)

def get_reference_kernel(mesh, B, reference_L, A):
    # P0 for the given reference stiffness, cached under that stiffness, so each reference choice has its own file.
    P0_parameters = {"cache_version": cache_version,
                     "domain_side_length": config.domain_side_length,
                     "element_number_per_side": config.element_number_per_side,
                     "element_number_along_z": config.element_number_along_z,
                     "partition_number_per_side": config.partition_number_per_side,
                     "reference_L": reference_L}
    P0_cache_path = get_cache_path("P0_offset_blocks", P0_parameters)
    cached_P0 = load_cache(P0_cache_path, P0_parameters)
    if cached_P0 is None:
        start_time = time.perf_counter()
        reference_L_per_element = np.tile(reference_L, (config.element_count, 1, 1))
        P0_offset_blocks = get_P0_offset_blocks(B, reference_L_per_element, mesh, A)
        print(f"P0 offline solve (for the FFT reference and the LS model): {time.perf_counter() - start_time:.2f} "
              "seconds.")
        save_cache(P0_cache_path, P0_parameters, P0_offset_blocks=P0_offset_blocks)
    else:
        P0_offset_blocks = cached_P0["P0_offset_blocks"]
    return get_P0_transformed(P0_offset_blocks)

## ------- Problem Setup ------- ##

class Problem(NamedTuple):
    partition_material_ids: np.ndarray
    partition_materials: PartitionMaterials
    mesh: Mesh
    B: np.ndarray
    E: np.ndarray
    P: np.ndarray
    P0_transformed: np.ndarray
    # The homogeneous reference stiffness C0 chosen by config.reference_stiffness, from which P0 is built, and its
    # inverse. Shared by the LS model and the TFA FFT reference.
    reference_L: np.ndarray
    reference_compliance: np.ndarray

def get_problem():
    # Everything an entry point needs, built from config, with the offline operators taken from the cache when it
    # matches.
    partition_material_ids = get_partition_material_ids()
    L_matrix = get_L(config.elastic_modulus_matrix, config.poisson_ratio_matrix)
    L_inclusion = get_L(config.elastic_modulus_inclusion, config.poisson_ratio_inclusion)
    if config.matched_stiffness_control:
        L_inclusion = L_matrix
        print("matched-stiffness control: the inclusion elastic stiffness is overridden to the matrix value, which "
              "makes the TFA and LS models the same equation. Yield properties still differ.")
    partition_materials = get_partition_materials(L_matrix, L_inclusion, partition_material_ids)
    mesh = get_periodic_mesh(partition_material_ids)
    B = get_B()
    L_per_element = get_value_per_material(L_matrix, L_inclusion, mesh.element_material_ids)
    A = get_partition_averaging_operator(B, mesh.element_nodes, mesh.element_partition_ids)
    E, P = get_E_and_P(mesh, B, L_per_element, L_matrix, L_inclusion, A)
    reference_L = get_reference_L(E, partition_materials.L, L_matrix)
    print(f"reference stiffness C0: {config.reference_stiffness}, C0_1111 = {reference_L[0, 0] / 1e6:.2f} MPa")
    P0_transformed = get_reference_kernel(mesh, B, reference_L, A)
    return Problem(partition_material_ids, partition_materials, mesh, B, E, P, P0_transformed, reference_L,
                   np.linalg.inv(reference_L))
