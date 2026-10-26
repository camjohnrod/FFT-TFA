# The offline step: the periodic hexahedral mesh and its materials, the 8-node element matrices, the FE influence
# functions E and P of the composite (TFA), the Moulinec–Suquet kernel P0 of the homogeneous reference (CP and
# Spectral LS), and the cache of E and P. Everything is built for a Discretization (a mesh and its partition lattice),
# so the same code serves the partitions of CP and TFA and the element lattice of Spectral LS. get_problem() at the
# end builds everything a run needs and times what each method's offline step costs.

import hashlib
import sys
import time
from typing import NamedTuple
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import tqdm

import config

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

def get_partition_discretization():
    # The one config.py describes: CP and TFA live on its partitions.
    return Discretization(config.element_number_per_side, config.element_number_along_z,
                          config.partition_number_per_side)

def get_spectral_discretization():
    # The same mesh with one partition per element column: the lattice of Spectral LS.
    return Discretization(config.element_number_per_side, config.element_number_along_z,
                          config.element_number_per_side)

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
    # Each partition's material, read off its elements. Every method gives each partition one material law, so a
    # partition holding both phases stops the run here, before any offline solve.
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
                         "but every partition must be a single phase: it has one material law. Align the inclusion "
                         "with the partition grid, or use one partition per element (partition_number_per_side = "
                         "element_number_per_side).")
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

## ------- Influence Functions E and P (TFA) ------- ##

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

def get_influence_function_solver(mesh, K):
    # Factorizes K once and returns solve(F), the partition-average strain response A_ε K⁻¹ F to every load column of
    # F, so that several sets of load columns share one factorization.
    free_dofs = get_free_dofs(mesh)
    K_free = K[free_dofs][:, free_dofs].tocsc()
    print(f"  {f'factor K ({K_free.shape[0]} DOFs)':<26}: ", end="", flush=True)
    start_time = time.perf_counter()
    K_free_factorization = scipy.sparse.linalg.splu(K_free, permc_spec='MMD_AT_PLUS_A')
    print(f"{time.perf_counter() - start_time:.1f} s")
    A = get_partition_averaging_operator(mesh)

    def solve(F):
        F_free = F.tocsr()[free_dofs].tocsc()
        columns_per_solve = config.offline_columns_per_solve
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

    return solve

def get_influence_functions(mesh, L_per_element, include_P):
    # E, and P when include_P (None otherwise), from one factorization of K, each with the seconds it took: E's
    # include assembling and factorizing K, P's only its own solves. E's six load columns are solved on their own, so
    # E is the same to the last bit whether or not P is built.
    start_time = time.perf_counter()
    solve = get_influence_function_solver(mesh, get_K(mesh, L_per_element))
    # ER-1: E is the applied macrostrain plus the fluctuation it causes; P is A_ε K⁻¹ F_μ (ER-24).
    E = np.tile(np.eye(6), (mesh.discretization.partition_count, 1)) + solve(get_F_macrostrain(mesh, L_per_element))
    E_seconds = time.perf_counter() - start_time
    P = solve(get_F_eigenstrain(mesh, L_per_element)) if include_P else None
    P_seconds = time.perf_counter() - start_time - E_seconds
    return E, P, E_seconds, P_seconds

## ------- Reference Stiffness C0 and Kernel P0 (CP and Spectral LS) ------- ##

def get_reference_L(E, L_per_partition, L_matrix):
    # The homogeneous reference stiffness C0 chosen by config.reference_stiffness. Every partition has the same volume,
    # so the averages are plain means over the partitions.
    if config.reference_stiffness == "homogenized":
        return np.mean(L_per_partition @ E.reshape(-1, 6, 6), axis=0)
    if config.reference_stiffness == "voigt":
        return np.mean(L_per_partition, axis=0)
    return L_matrix

# P0 is handled in two forms: in real space, one 6 × 6 block per partition offset P0[a_B - a_A] on a lattice indexed
# [j, i] as get_grid_id orders it, and transformed over the lattice (ER-14a). The kernel is real, so the transform
# keeps only the non-redundant half of the frequencies (rfftn, shape (N, N//2 + 1, 6, 6)).

def get_P0_transformed(P0_offset_blocks, partition_number_per_side):
    P0_offset_lattice = P0_offset_blocks.reshape(partition_number_per_side, partition_number_per_side, 6, 6)
    return np.fft.rfftn(P0_offset_lattice, axes=(0, 1))

def get_P0_offset_blocks_from_transformed(P0_transformed):
    lattice_shape = (P0_transformed.shape[0],) * 2
    return np.fft.irfftn(P0_transformed, s=lattice_shape, axes=(0, 1)).reshape(-1, 6)

def get_direction_strain_matrix(directions):
    # B(n), shape (..., 6, 3): the engineering strain (ER-2) of a displacement u varying along the unit direction n,
    # per unit gradient, ε = B(n) u. The rows are those of get_B with each derivative replaced by its component of n.
    n1, n2, n3 = np.moveaxis(directions, -1, 0)
    zero = np.zeros_like(n1)
    rows = [(n1, zero, zero), (zero, n2, zero), (zero, zero, n3), (n2, n1, zero), (zero, n3, n2), (n3, zero, n1)]
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)

def get_moulinec_suquet_P0_offset_blocks(lattice_side, reference_L):
    # The Moulinec–Suquet kernel on a lattice_side² lattice (LS-21): P̂0(ξ) = Γ̂0(ξ) C0, with the continuum Green
    # operator Γ̂0 = B (Bᵀ C0 B)⁻¹ Bᵀ, B = B(ξ / |ξ|), at every lattice frequency ξ, all in-plane since the fields are
    # uniform through the thickness. P̂0 is zero at ξ = 0 (LS-9). On an even lattice, Γ̂0 is not even in a component
    # equal to lattice_side / 2, so it has no real-valued meaning there; P̂0 is then the identity, Moulinec–Suquet's
    # (1998, section 2.4.2) Γ̂0 = C0⁻¹, which makes the stress at those frequencies zero.
    frequency_x = np.fft.rfftfreq(lattice_side, 1 / lattice_side)
    frequency_y = np.fft.fftfreq(lattice_side, 1 / lattice_side)
    # The half spectrum is indexed [ξ_y, ξ_x], like the lattice and get_P0_transformed.
    wave_vectors = np.stack(np.broadcast_arrays(frequency_x[None, :], frequency_y[:, None], 0.0), axis=-1)
    wave_vectors[0, 0] = [1, 0, 0]  # any direction at ξ = 0, whose block is set to zero below
    B = get_direction_strain_matrix(wave_vectors / np.linalg.norm(wave_vectors, axis=-1, keepdims=True))
    B_transposed = np.swapaxes(B, -1, -2)
    P0_transformed = B @ np.linalg.inv(B_transposed @ reference_L @ B) @ B_transposed @ reference_L
    P0_transformed[0, 0] = 0
    if lattice_side % 2 == 0:
        highest_frequencies = ((np.abs(frequency_y) == lattice_side / 2)[:, None]
                               | (frequency_x == lattice_side / 2)[None, :])
        P0_transformed[highest_frequencies] = np.eye(6)
    return get_P0_offset_blocks_from_transformed(P0_transformed)

def get_partition_averaged_P0_offset_blocks(element_P0_offset_blocks, discretization):
    # The kernel of the discretization's partitions from the kernel of its element columns (one partition per column,
    # same mesh), by linearity. Unit eigenstrain in a partition is unit eigenstrain in each of its k × k columns, and
    # a partition's average strain is the mean over its columns, so P0[D] = (1/k²) Σ_receiver s Σ_source t
    # p0_element[k D + s - t]. Grouped by d = s - t, which k - |d_x| times k - |d_y| pairs (s, t) share:
    #   P0[D] = Σ_d (k - |d_x|)(k - |d_y|) / k² · p0_element[k D + d].
    # Offsets are periodic. With one element per partition (k = 1) the kernel is returned unchanged.
    k = discretization.elements_per_partition_side
    element_side, partition_side = discretization.element_number_per_side, discretization.partition_number_per_side
    element_P0 = element_P0_offset_blocks.reshape(element_side, element_side, 6, 6)
    partition_offsets = k * np.arange(partition_side)
    partition_P0 = np.zeros((partition_side, partition_side, 6, 6))
    for d_j in range(1 - k, k):
        for d_i in range(1 - k, k):
            weight = (k - abs(d_j)) * (k - abs(d_i)) / k**2
            partition_P0 += weight * element_P0[np.ix_((partition_offsets + d_j) % element_side,
                                                       (partition_offsets + d_i) % element_side)]
    return partition_P0.reshape(-1, 6)

def get_reference_kernel(discretization, reference_L):
    # P̂0 on the discretization's partition lattice: the Moulinec–Suquet kernel built on its element grid, then
    # averaged over its partitions, as SCA averages its fine-grid operator over clusters. On a lattice of one element
    # per partition (Spectral LS) the average leaves it unchanged. It takes milliseconds, so it is not cached.
    P0_offset_blocks = get_partition_averaged_P0_offset_blocks(
        get_moulinec_suquet_P0_offset_blocks(discretization.element_number_per_side, reference_L), discretization)
    return get_P0_transformed(P0_offset_blocks, discretization.partition_number_per_side)

## ------- Cache ------- ##

# Increase whenever a code change alters E or P, so caches made by the old code are recomputed.
# 2: the geometry is resolved on the elements, not the partitions.
cache_version = 2

def get_cache_path(kind, parameters):
    # One file per parameter set, so switching between configurations reuses each one's operators instead of
    # overwriting a single file. The name is a hash of every parameter, so a file that exists matches them.
    digest = hashlib.sha256()
    for name in sorted(parameters):
        value = np.asarray(parameters[name])
        digest.update(f"{name}:{value.dtype.str}:{value.shape}:".encode())
        digest.update(value.tobytes())
    return config.cache_folder / f"{kind}_{digest.hexdigest()[:12]}.npz"

def is_cached(cache_path):
    # A cache file counts only if it records how long its array took to build, which the offline times report.
    if not cache_path.exists():
        return False
    with np.load(cache_path) as cached:
        return "build_seconds" in cached.files

def get_cached(names, parameters, compute):
    # The named arrays and the seconds each took to build, as two dicts by name, each from its own cache file for these
    # parameters when they all exist, otherwise computed together by compute(), which returns both dicts, and each
    # saved to its own file. One file per array lets a run load only the arrays it needs (E without P when TFA is off).
    cache_paths = {name: get_cache_path(name, parameters) for name in names}
    if all(is_cached(cache_path) for cache_path in cache_paths.values()):
        arrays, build_seconds = {}, {}
        for name, cache_path in cache_paths.items():
            print(f"{cache_path.name}: loaded from cache.")
            with np.load(cache_path) as cached:
                arrays[name], build_seconds[name] = cached[name], float(cached["build_seconds"])
        return arrays, build_seconds
    print(f"{', '.join(cache_path.name for cache_path in cache_paths.values())}: no cache found, computing.")
    arrays, build_seconds = compute()
    config.cache_folder.mkdir(exist_ok=True)
    for name, cache_path in cache_paths.items():
        np.savez(cache_path, **{name: arrays[name], "build_seconds": build_seconds[name]})
    return arrays, build_seconds

def get_E_and_P(mesh, L_matrix, L_inclusion, include_P):
    # E, and the dense P, (6 × partitions)² doubles, only when include_P (P is None otherwise), and the seconds they
    # took to build, by name ("E", "P"), measured when they were computed, even if they now come from the cache.
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
        E, P, E_seconds, P_seconds = get_influence_functions(
            mesh, get_value_per_material(L_matrix, L_inclusion, mesh.element_material_ids), include_P)
        return {"E": E, "P": P}, {"E": E_seconds, "P": P_seconds}

    E_P, build_seconds = get_cached(("E", "P") if include_P else ("E",), E_P_parameters, compute)
    if not include_P:
        return E_P["E"], None, build_seconds
    # Column-major, so the columns of a run of consecutive partitions are contiguous and P products can read just the
    # yielding partitions' columns without copying them (tfa_model.apply_P_to_partitions).
    return E_P["E"], np.asfortranarray(E_P["P"]), build_seconds

## ------- Problem ------- ##

class Lattice(NamedTuple):
    # A partition lattice with what an LS method on it needs: its partitions' materials and its P0 kernel.
    partition_materials: PartitionMaterials
    P0_transformed: np.ndarray

class Problem(NamedTuple):
    # Everything a run needs. CP and TFA share the partitions; E and P are None when no method needs them.
    partitions: Lattice
    E: np.ndarray
    P: np.ndarray
    # The inverse of the reference stiffness C0, shared by CP and Spectral LS.
    reference_compliance: np.ndarray
    # The element lattice of Spectral LS, None without config.include_spectral_ls.
    spectral_lattice: Lattice
    # The seconds each method's offline step costs, by method name: everything it needs built, as if it ran alone.
    offline_seconds: dict

def get_problem():
    # Built from config, with E and P taken from the cache when it matches. E is needed by TFA and by the
    # "homogenized" reference stiffness, P by TFA only. The offline times count E and P at the time they took to
    # compute, even when they come from the cache, and everything else as built now.
    start_time = time.perf_counter()
    mesh = get_periodic_mesh(get_partition_discretization())
    L_matrix = get_L(config.elastic_modulus_matrix, config.poisson_ratio_matrix)
    L_inclusion = get_L(config.elastic_modulus_inclusion, config.poisson_ratio_inclusion)
    partition_materials = get_partition_materials(L_matrix, L_inclusion, get_partition_material_ids(mesh))
    partition_seconds = time.perf_counter() - start_time

    E, P, build_seconds = None, None, {}
    if config.include_tfa_model or config.reference_stiffness == "homogenized":
        E, P, build_seconds = get_E_and_P(mesh, L_matrix, L_inclusion, include_P=config.include_tfa_model)
    start_time = time.perf_counter()
    reference_L = get_reference_L(E, partition_materials.L, L_matrix)
    reference_seconds = time.perf_counter() - start_time
    if config.reference_stiffness == "homogenized":
        reference_seconds += build_seconds["E"]
    print(f"reference stiffness C0: {config.reference_stiffness}, C0_1111 = {reference_L[0, 0] / 1e6:.2f} MPa")
    start_time = time.perf_counter()
    partitions = Lattice(partition_materials, get_reference_kernel(mesh.discretization, reference_L))
    offline_seconds = {"CP": partition_seconds + reference_seconds + time.perf_counter() - start_time}

    spectral_lattice = None
    if config.include_spectral_ls:
        start_time = time.perf_counter()
        spectral_discretization = get_spectral_discretization()
        spectral_materials = get_partition_materials(
            L_matrix, L_inclusion, get_partition_material_ids(get_periodic_mesh(spectral_discretization)))
        spectral_lattice = Lattice(spectral_materials, get_reference_kernel(spectral_discretization, reference_L))
        # C0 is read off the partitions (Voigt and homogenized), so Spectral LS needs them built too.
        offline_seconds["Spectral LS"] = partition_seconds + reference_seconds + time.perf_counter() - start_time
    if config.include_tfa_model:
        offline_seconds["TFA"] = partition_seconds + build_seconds["E"] + build_seconds["P"]
    return Problem(partitions, E, P, np.linalg.inv(reference_L), spectral_lattice, offline_seconds)
