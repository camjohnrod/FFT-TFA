# Offline setup: the periodic hexahedral mesh and its materials, the 8-node element matrices, the influence
# functions E and P of the actual composite and P0 of the homogeneous reference, and their cache. Everything is built
# for a Discretization (a mesh and its partition lattice), so the same code serves the main, fine and coarse ones.

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

## ------- Discretizations ------- ##

class Discretization(NamedTuple):
    # A mesh of element_number_per_side² × element_number_along_z hexahedra over the periodic cell, and its lattice of
    # partition_number_per_side² partitions, each a column of elements through the thickness.
    element_number_per_side: int
    element_number_along_z: int
    partition_number_per_side: int

    @property
    def elements_per_partition_side(self):
        return self.element_number_per_side // self.partition_number_per_side

    @property
    def element_count(self):
        return self.element_number_per_side**2 * self.element_number_along_z

    @property
    def partition_count(self):
        return self.partition_number_per_side**2

    @property
    def dof_count(self):
        return 3 * self.element_count

    @property
    def element_side_length(self):
        return config.domain_side_length / self.element_number_per_side

    @property
    def element_length_along_z(self):
        return config.domain_side_length / self.element_number_along_z

    @property
    def element_volume(self):
        return self.element_side_length**2 * self.element_length_along_z

    @property
    def gauss_point_volume_weight(self):
        return self.element_volume / 8

    @property
    def partition_volume(self):
        return self.elements_per_partition_side**2 * self.element_number_along_z * self.element_volume

def get_main_discretization():
    # The one config.py describes: TFA and partitioned LS live on its partitions.
    return Discretization(config.element_number_per_side, config.element_number_along_z,
                          config.partition_number_per_side)

def get_fine_discretization():
    # The main mesh with one partition per element column: the lattice of the fine, non-partitioned LS solver.
    return Discretization(config.element_number_per_side, config.element_number_along_z,
                          config.element_number_per_side)

def get_coarse_discretization():
    # A coarse mesh with one element per main partition: the lattice of the coarse, non-partitioned LS scheme.
    return Discretization(config.partition_number_per_side, config.element_number_along_z,
                          config.partition_number_per_side)

## ------- Mesh, Geometry and Materials ------- ##

class Mesh(NamedTuple):
    discretization: Discretization
    element_nodes: np.ndarray
    element_partition_ids: np.ndarray
    element_material_ids: np.ndarray
    # The strain-displacement matrix at each of the 8 Gauss points, shared by every element (8 × 6 × 24).
    B: np.ndarray

class PartitionMaterials(NamedTuple):
    # Everything a lattice's partitions are made of, one row per partition: the phase (0 matrix, 1 inclusion), the
    # elastic stiffness and the yield properties.
    material_ids: np.ndarray
    L: np.ndarray
    yield_stress: np.ndarray
    hardening_modulus: np.ndarray

def get_grid_id(i, j, k, count_per_side):
    return i + count_per_side * j + count_per_side**2 * k

def is_inside_inclusion(centres):
    # Whether each point of the cross-section, one (x, y) row each, lies inside the inclusion.
    offsets_from_inclusion_axis = centres - config.domain_side_length / 2
    if config.inclusion_shape == "circle":
        return np.sum(offsets_from_inclusion_axis**2, axis=1) < config.inclusion_radius**2
    if config.inclusion_shape == "laminate":
        return np.abs(offsets_from_inclusion_axis[:, 0]) < config.inclusion_side_length / 2
    return np.all(np.abs(offsets_from_inclusion_axis) < config.inclusion_side_length / 2, axis=1)

def get_element_column_material_ids(discretization):
    # The geometry, resolved on the elements: 1 for an element column whose centre lies inside the inclusion, else 0,
    # in get_grid_id order (i fastest). Every element of a column shares its material, since the geometry does not vary
    # through the thickness.
    centre_coordinates = (np.arange(discretization.element_number_per_side) + 0.5) * discretization.element_side_length
    x, y = np.meshgrid(centre_coordinates, centre_coordinates)
    return is_inside_inclusion(np.column_stack([x.ravel(), y.ravel()])).astype(int)

def get_partition_material_ids(mesh):
    # Each partition's material, read off its elements. TFA and partitioned LS give every partition one material law,
    # so a partition holding both phases stops the run here, before any offline solve.
    partition_count = mesh.discretization.partition_count
    if mesh.element_material_ids.min() == mesh.element_material_ids.max():
        raise ValueError("Every element is the same phase, so the cell is homogeneous: the inclusion is too small or "
                         "too large for this element grid.")
    inclusion_elements = np.bincount(mesh.element_partition_ids, weights=mesh.element_material_ids,
                                     minlength=partition_count)
    elements = np.bincount(mesh.element_partition_ids, minlength=partition_count)
    mixed_partitions = (inclusion_elements > 0) & (inclusion_elements < elements)
    if mixed_partitions.any():
        raise ValueError(f"{np.count_nonzero(mixed_partitions)} of {partition_count} partitions contain both phases, "
                         "but every partition must be a single phase: it has one material law in TFA and in "
                         "partitioned LS. Align the inclusion with the partition grid, or use one partition per "
                         "element (partition_number_per_side = element_number_per_side).")
    return (inclusion_elements > 0).astype(int)

def get_periodic_mesh(discretization):
    element_number_per_side = discretization.element_number_per_side
    element_nodes = np.zeros((discretization.element_count, 8), dtype=int)
    element_partition_ids = np.zeros(discretization.element_count, dtype=int)

    for k in range(discretization.element_number_along_z):
        for j in range(element_number_per_side):
            for i in range(element_number_per_side):
                element_id = get_grid_id(i, j, k, element_number_per_side)

                for offset_k in range(2):
                    for offset_j in range(2):
                        for offset_i in range(2):
                            corner = get_grid_id(offset_i, offset_j, offset_k, 2)
                            element_nodes[element_id, corner] = get_grid_id(
                                (i + offset_i) % element_number_per_side,
                                (j + offset_j) % element_number_per_side,
                                (k + offset_k) % discretization.element_number_along_z,
                                element_number_per_side)

                element_partition_ids[element_id] = get_grid_id(i // discretization.elements_per_partition_side,
                                                                j // discretization.elements_per_partition_side,
                                                                0,
                                                                discretization.partition_number_per_side)

    element_material_ids = np.tile(get_element_column_material_ids(discretization),
                                   discretization.element_number_along_z)
    return Mesh(discretization, element_nodes, element_partition_ids, element_material_ids, get_B(discretization))

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
        material_ids=partition_material_ids,
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

def get_B(discretization):
    gauss_point_coordinates = corner_natural_coordinates / np.sqrt(3)
    B = np.zeros((8, 6, 24))

    for gauss_index, (xi, eta, zeta) in enumerate(gauss_point_coordinates):
        derivatives_wrt_xi, derivatives_wrt_eta, derivatives_wrt_zeta = get_shape_function_derivatives(xi, eta, zeta)
        derivatives_wrt_x = derivatives_wrt_xi   * 2 / discretization.element_side_length
        derivatives_wrt_y = derivatives_wrt_eta  * 2 / discretization.element_side_length
        derivatives_wrt_z = derivatives_wrt_zeta * 2 / discretization.element_length_along_z

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

def get_element_dofs(mesh):
    node_dofs = 3 * mesh.element_nodes[:, :, None] + np.arange(3)
    return node_dofs.reshape(-1, 24)

def get_partition_components(element_partition_ids):
    # The six entries of each element's partition in a partition-stacked vector such as ε, μ or b.
    return 6 * element_partition_ids[:, None] + np.arange(6)

def assemble_element_blocks(element_blocks, element_rows, element_columns, shape):
    # Adds each element's block into a sparse global matrix at the given global rows and columns. Blocks that share
    # an entry are summed, as in finite-element assembly.
    rows = np.broadcast_to(element_rows[:, :, None], element_blocks.shape)
    columns = np.broadcast_to(element_columns[:, None, :], element_blocks.shape)
    return scipy.sparse.coo_matrix((element_blocks.ravel(), (rows.ravel(), columns.ravel())), shape=shape).tocsr()

def get_K_element(mesh, L):
    K_element = np.zeros((24, 24))
    for gauss_index in range(8):
        K_element += (mesh.discretization.gauss_point_volume_weight * mesh.B[gauss_index].T @ L
                      @ mesh.B[gauss_index])
    return K_element

def get_K(mesh, L_per_element):
    element_dofs = get_element_dofs(mesh)
    K_elements = np.array([get_K_element(mesh, L) for L in L_per_element])
    dof_count = mesh.discretization.dof_count
    return assemble_element_blocks(K_elements, element_dofs, element_dofs, (dof_count, dof_count))

def get_integrated_B(mesh):
    return mesh.discretization.gauss_point_volume_weight * np.sum(mesh.B, axis=0)

def get_F_elements(mesh, L_per_element):
    # Each element's nodal load per unit eigenstrain, ∫ Bᵀ L dV (24 × 6).
    return get_integrated_B(mesh).T @ L_per_element

def get_free_dofs(mesh):
    # Pinning one node's three displacements removes the rigid translation of the periodic cell.
    pinned_dofs = np.arange(3)
    return np.setdiff1d(np.arange(mesh.discretization.dof_count), pinned_dofs)

## ------- Influence Functions (Offline) ------- ##

def get_F_macrostrain(mesh, L_per_element):
    # Nodal loads per unit macrostrain. Moving the applied strain to the right-hand side gives -∫ Bᵀ L ε̄ dV.
    macrostrain_columns = np.tile(np.arange(6), (mesh.discretization.element_count, 1))
    return -assemble_element_blocks(get_F_elements(mesh, L_per_element), get_element_dofs(mesh), macrostrain_columns,
                                    (mesh.discretization.dof_count, 6))

def get_F_eigenstrain(mesh, L_per_element):
    # F_μ in ER-24: nodal loads per unit partition eigenstrain, one block of six columns per partition.
    return assemble_element_blocks(get_F_elements(mesh, L_per_element), get_element_dofs(mesh),
                                   get_partition_components(mesh.element_partition_ids),
                                   (mesh.discretization.dof_count, 6 * mesh.discretization.partition_count))

def get_partition_averaging_operator(mesh):
    # A_ε in ER-24: maps nodal displacements to partition-average strain.
    discretization = mesh.discretization
    element_blocks = np.broadcast_to(get_integrated_B(mesh) / discretization.partition_volume,
                                     (discretization.element_count, 6, 24))
    return assemble_element_blocks(element_blocks, get_partition_components(mesh.element_partition_ids),
                                   get_element_dofs(mesh), (6 * discretization.partition_count,
                                                            discretization.dof_count))

def solve_influence_function(mesh, K, F):
    # The partition-average strain response A_ε K⁻¹ F to every load column of F.
    free_dofs = get_free_dofs(mesh)
    K_free = K[free_dofs][:, free_dofs].tocsc()
    print(f"  {f'factor K ({K_free.shape[0]} DOFs)':<26}: ", end="", flush=True)
    start_time = time.perf_counter()
    K_free_factorization = scipy.sparse.linalg.splu(K_free, permc_spec='MMD_AT_PLUS_A')
    print(f"{time.perf_counter() - start_time:.1f} s")

    A = get_partition_averaging_operator(mesh)
    F_free = F.tocsr()[free_dofs].tocsc()
    columns_per_solve = 100
    average_strain = np.zeros((A.shape[0], F.shape[1]))
    column_batches = tqdm.tqdm(range(0, F.shape[1], columns_per_solve),
                               desc=f"  {f'solve {F.shape[1]} load columns':<26}",
                               bar_format="{desc}: {percentage:3.0f}% |{bar:25}| {n_fmt}/{total_fmt} batches "
                                          "[{elapsed} elapsed, {remaining} left]", file=sys.stdout)
    for first_column in column_batches:
        columns = slice(first_column, first_column + columns_per_solve)
        batch_F_free = F_free[:, columns].toarray()
        batch_displacements = np.zeros((mesh.discretization.dof_count, batch_F_free.shape[1]))
        batch_displacements[free_dofs] = K_free_factorization.solve(batch_F_free)
        average_strain[:, columns] = A @ batch_displacements
    return average_strain

def get_influence_functions(mesh, L_per_element):
    K = get_K(mesh, L_per_element)
    F_macrostrain_and_eigenstrain = scipy.sparse.hstack([get_F_macrostrain(mesh, L_per_element),
                                                         get_F_eigenstrain(mesh, L_per_element)])
    average_strain = solve_influence_function(mesh, K, F_macrostrain_and_eigenstrain)
    # ER-1: E is the applied macrostrain plus the fluctuation it causes; P is A_ε K⁻¹ F_μ (ER-24).
    E = np.tile(np.eye(6), (mesh.discretization.partition_count, 1)) + average_strain[:, :6]
    P = average_strain[:, 6:]
    return E, P

def get_homogenized_L(E, L_per_partition):
    E_blocks = E.reshape(-1, 6, 6)
    return np.mean(L_per_partition @ E_blocks, axis=0)

def get_reference_L(E, L_per_partition, L_matrix):
    # The homogeneous reference stiffness C0 chosen by config.reference_stiffness. Every partition has the same volume,
    # so the averages are plain means over the partitions.
    if config.reference_stiffness == "homogenized":
        return get_homogenized_L(E, L_per_partition)
    if config.reference_stiffness == "voigt":
        return np.mean(L_per_partition, axis=0)
    return L_matrix

def get_P0_offset_blocks(mesh, reference_L):
    # The reference response to eigenstrain in partition 0 alone. Reference interactions depend only on the offset
    # between partitions (ER-6e), so block B of this response is P0[B - 0] and one solve gives the whole kernel.
    reference_L_per_element = np.tile(reference_L, (mesh.discretization.element_count, 1, 1))
    K = get_K(mesh, reference_L_per_element)
    F_eigenstrain_in_first_partition = get_F_eigenstrain(mesh, reference_L_per_element)[:, :6]
    return solve_influence_function(mesh, K, F_eigenstrain_in_first_partition)

## ------- Cache ------- ##

# Increase whenever a code change alters the offline operators, so caches made by the old code are recomputed.
# 2: the geometry is resolved on the elements, not the partitions.
cache_version = 2

def get_cache_path(kind, parameters):
    # One file per parameter set, so switching between configurations or entry points reuses each one's operators
    # instead of overwriting a single file. The name is a hash of every parameter, so a file that exists matches them.
    digest = hashlib.sha256()
    for name in sorted(parameters):
        value = np.asarray(parameters[name])
        digest.update(f"{name}:{value.dtype.str}:{value.shape}:".encode())
        digest.update(value.tobytes())
    return config.cache_folder / f"{kind}_{digest.hexdigest()[:12]}.npz"

def get_cached(kind, parameters, compute):
    # The arrays compute() returns, as a dict by name, from the cache file for these parameters when it exists,
    # otherwise computed and saved there.
    cache_path = get_cache_path(kind, parameters)
    if cache_path.exists():
        print(f"{cache_path.name}: loaded from cache.")
        with np.load(cache_path) as cached:
            return {name: cached[name] for name in cached.files}
    print(f"{cache_path.name}: no cache found, computing.")
    arrays = compute()
    config.cache_folder.mkdir(exist_ok=True)
    np.savez(cache_path, **arrays)
    return arrays

def get_E_and_P(mesh, L_matrix, L_inclusion):
    discretization = mesh.discretization
    E_P_parameters = {"cache_version": cache_version,
                      "domain_side_length": config.domain_side_length,
                      "inclusion_shape": config.inclusion_shape,
                      "element_number_per_side": discretization.element_number_per_side,
                      "element_number_along_z": discretization.element_number_along_z,
                      "partition_number_per_side": discretization.partition_number_per_side,
                      "L_matrix": L_matrix,
                      "L_inclusion": L_inclusion}
    if config.inclusion_shape in ("square", "laminate"):
        E_P_parameters["inclusion_side_length"] = config.inclusion_side_length
    else:
        E_P_parameters["inclusion_radius"] = config.inclusion_radius

    def compute():
        E, P = get_influence_functions(mesh, get_value_per_material(L_matrix, L_inclusion, mesh.element_material_ids))
        return {"E": E, "P": P}

    E_P = get_cached("E_P", E_P_parameters, compute)
    # Column-major, so the columns of a run of consecutive partitions are contiguous and P products can read just the
    # yielding partitions' columns without copying them (tfa_solvers.apply_P_to_partitions).
    return E_P["E"], np.asfortranarray(E_P["P"])

def get_reference_kernel(mesh, reference_L):
    # P0 on the mesh's partition lattice, for the given reference stiffness, cached under both, so each discretization
    # and each reference choice has its own file. P0 does not depend on the geometry, since the reference is
    # homogeneous.
    discretization = mesh.discretization
    P0_parameters = {"cache_version": cache_version,
                     "domain_side_length": config.domain_side_length,
                     "element_number_per_side": discretization.element_number_per_side,
                     "element_number_along_z": discretization.element_number_along_z,
                     "partition_number_per_side": discretization.partition_number_per_side,
                     "reference_L": reference_L}

    def compute():
        start_time = time.perf_counter()
        P0_offset_blocks = get_P0_offset_blocks(mesh, reference_L)
        print(f"P0 offline solve ({discretization.element_number_per_side} elements, "
              f"{discretization.partition_number_per_side} partitions per side): "
              f"{time.perf_counter() - start_time:.2f} seconds.")
        return {"P0_offset_blocks": P0_offset_blocks}

    P0_offset_blocks = get_cached("P0_offset_blocks", P0_parameters, compute)["P0_offset_blocks"]
    return get_P0_transformed(P0_offset_blocks, discretization.partition_number_per_side)

## ------- Problem Setup ------- ##

class Lattice(NamedTuple):
    # A partition lattice with what an LS solver on it needs: its partitions' materials and its P0 kernel.
    partition_materials: PartitionMaterials
    P0_transformed: np.ndarray

def get_lattice(discretization, L_matrix, L_inclusion, reference_L):
    # The lattice of a discretization, with the geometry resolved on its own elements and P0 built with the given C0.
    mesh = get_periodic_mesh(discretization)
    partition_materials = get_partition_materials(L_matrix, L_inclusion, get_partition_material_ids(mesh))
    return Lattice(partition_materials, get_reference_kernel(mesh, reference_L))

class Problem(NamedTuple):
    # The main lattice (TFA and partitioned LS): its partitions' materials, its mesh, on which E, P and P0 are built,
    # and those operators.
    partition_materials: PartitionMaterials
    mesh: Mesh
    E: np.ndarray
    P: np.ndarray
    P0_transformed: np.ndarray
    # The homogeneous reference stiffness C0 chosen by config.reference_stiffness, from which every P0 is built, and
    # its inverse. Shared by every LS solver and the TFA FFT reference.
    reference_L: np.ndarray
    reference_compliance: np.ndarray
    # The lattices of the non-partitioned LS solvers (offline.get_fine_discretization, get_coarse_discretization).
    fine_lattice: Lattice
    coarse_lattice: Lattice

def get_problem():
    # Everything an entry point needs, built from config, with the offline operators taken from the cache when it
    # matches.
    mesh = get_periodic_mesh(get_main_discretization())
    partition_material_ids = get_partition_material_ids(mesh)
    L_matrix = get_L(config.elastic_modulus_matrix, config.poisson_ratio_matrix)
    L_inclusion = get_L(config.elastic_modulus_inclusion, config.poisson_ratio_inclusion)
    if config.matched_stiffness_control:
        L_inclusion = L_matrix
        print("matched-stiffness control: the inclusion elastic stiffness is overridden to the matrix value, which "
              "makes the TFA and LS models the same equation. Yield properties still differ.")
    partition_materials = get_partition_materials(L_matrix, L_inclusion, partition_material_ids)
    E, P = get_E_and_P(mesh, L_matrix, L_inclusion)
    reference_L = get_reference_L(E, partition_materials.L, L_matrix)
    print(f"reference stiffness C0: {config.reference_stiffness}, C0_1111 = {reference_L[0, 0] / 1e6:.2f} MPa")
    P0_transformed = get_reference_kernel(mesh, reference_L)
    fine_lattice = get_lattice(get_fine_discretization(), L_matrix, L_inclusion, reference_L)
    coarse_lattice = get_lattice(get_coarse_discretization(), L_matrix, L_inclusion, reference_L)
    return Problem(partition_materials, mesh, E, P, P0_transformed, reference_L, np.linalg.inv(reference_L),
                   fine_lattice, coarse_lattice)
