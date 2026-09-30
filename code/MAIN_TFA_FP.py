import functools
import pathlib
import sys
import time
from typing import NamedTuple
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import tqdm
import plots

## ------- Inputs ------- ##

domain_side_length          = 1e-3
inclusion_shape             = "circle"
inclusion_side_length       = (5/9) * 1e-3
inclusion_radius            = 0.35e-3

element_number_per_side     = 75
element_number_along_z      = 5
partition_number_per_side   = 25

elastic_modulus_inclusion   = 10e9
elastic_modulus_matrix      = 100e6
poisson_ratio_inclusion     = 0.3
poisson_ratio_matrix        = 0.3
inclusion_yield_stress      = np.inf
matrix_yield_stress         = 1.0e6
inclusion_hardening_modulus = 0.0
matrix_hardening_modulus    = 10e6

# max_macro_strain            = np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0])
# Alternative load paths:
# max_macro_strain            = np.array([0.03, 0.018, 0.001, 0.0, 0.0, 0.0])
max_macro_strain            = np.array([0.015, 0.020, 0.0, 0.03, 0.0, 0.0])

strain_increment_count      = 60

fixed_point_tolerance       = 1e-6
fixed_point_max_iterations  = 1000
divergence_residual_limit   = 1.0
residual_strain_scale_floor = 1e-4
relaxation_factor           = 1.0
solver_agreement_tolerance  = 1e-4

timing_repeat_count         = 5
post_process_element_stress = True
von_mises_plot_min_stress   = 1.0e6
von_mises_plot_max_stress   = 3.0e6

## ------- Calculated Values and Checks ------- ##

if min(element_number_per_side, element_number_along_z, partition_number_per_side, strain_increment_count,
       timing_repeat_count) < 1:
    raise ValueError("The element, partition, strain increment and timing repeat counts must all be at least 1.")
if element_number_per_side % partition_number_per_side != 0:
    raise ValueError("The number of elements per side must be divisible by the number of partitions per side.")

elements_per_partition_side = element_number_per_side // partition_number_per_side
element_count = element_number_per_side**2 * element_number_along_z
partition_count = partition_number_per_side**2
dof_count = 3 * element_count

element_side_length = domain_side_length / element_number_per_side
element_length_along_z = domain_side_length / element_number_along_z
element_volume = element_side_length**2 * element_length_along_z
gauss_point_volume_weight = element_volume / 8
partition_side_length = domain_side_length / partition_number_per_side
partition_volume = elements_per_partition_side**2 * element_number_along_z * element_volume

if inclusion_shape == "square":
    matrix_partitions_beside_inclusion = (partition_number_per_side * (domain_side_length - inclusion_side_length)
                                          / (2 * domain_side_length))
    whole_matrix_partitions_beside_inclusion = round(matrix_partitions_beside_inclusion)
    if (not np.isclose(matrix_partitions_beside_inclusion, whole_matrix_partitions_beside_inclusion)
            or not 1 <= whole_matrix_partitions_beside_inclusion < partition_number_per_side / 2):
        raise ValueError("The centred inclusion must have a whole number of matrix partitions (at least 1) on each "
                         "side in x and y, so its edges fall on partition edges. "
                         f"Got {matrix_partitions_beside_inclusion}.")
elif inclusion_shape == "circle":
    if not 0 < inclusion_radius < domain_side_length / 2:
        raise ValueError("The circular inclusion's radius must be positive and below half the domain side, so "
                         f"neighbouring inclusions do not touch. Got {inclusion_radius}.")
    nearest_partition_centre_distance = (0 if partition_number_per_side % 2 == 1
                                         else np.sqrt(2) * partition_side_length / 2)
    if inclusion_radius <= nearest_partition_centre_distance:
        raise ValueError("The circular inclusion contains no partition centre, so no partition would be inclusion. "
                         f"Increase the radius above {nearest_partition_centre_distance} or use an odd partition "
                         "count.")
else:
    raise ValueError(f"inclusion_shape must be \"square\" or \"circle\". Got {inclusion_shape!r}.")

cache_folder = pathlib.Path(__file__).parent / "cache"
output_folder = pathlib.Path(__file__).parent / "output"

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
    offset_from_inclusion_axis = centre[:2] - domain_side_length / 2
    if inclusion_shape == "circle":
        return np.sum(offset_from_inclusion_axis**2) < inclusion_radius**2
    return np.all(np.abs(offset_from_inclusion_axis) < inclusion_side_length / 2)

def get_partition_material_ids():
    partition_material_ids = np.zeros(partition_count, dtype=int)

    for j in range(partition_number_per_side):
        for i in range(partition_number_per_side):
            partition_id = get_grid_id(i, j, 0, partition_number_per_side)
            partition_centre = (np.array([i, j]) + 0.5) * partition_side_length
            partition_material_ids[partition_id] = int(is_inside_inclusion(partition_centre))

    return partition_material_ids

def get_periodic_mesh(partition_material_ids):
    element_nodes = np.zeros((element_count, 8), dtype=int)
    element_partition_ids = np.zeros(element_count, dtype=int)

    for k in range(element_number_along_z):
        for j in range(element_number_per_side):
            for i in range(element_number_per_side):
                element_id = get_grid_id(i, j, k, element_number_per_side)

                for offset_k in range(2):
                    for offset_j in range(2):
                        for offset_i in range(2):
                            corner = get_grid_id(offset_i, offset_j, offset_k, 2)
                            element_nodes[element_id, corner] = get_grid_id((i + offset_i) % element_number_per_side,
                                                                            (j + offset_j) % element_number_per_side,
                                                                            (k + offset_k) % element_number_along_z,
                                                                            element_number_per_side)

                element_partition_ids[element_id] = get_grid_id(i // elements_per_partition_side,
                                                                j // elements_per_partition_side,
                                                                0,
                                                                partition_number_per_side)

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
        yield_stress=get_value_per_material(matrix_yield_stress, inclusion_yield_stress, partition_material_ids),
        hardening_modulus=get_value_per_material(matrix_hardening_modulus, inclusion_hardening_modulus,
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
        derivatives_wrt_x = derivatives_wrt_xi   * 2 / element_side_length
        derivatives_wrt_y = derivatives_wrt_eta  * 2 / element_side_length
        derivatives_wrt_z = derivatives_wrt_zeta * 2 / element_length_along_z

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
    return node_dofs.reshape(element_count, 24)

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
        K_element += gauss_point_volume_weight * B[gauss_index].T @ L @ B[gauss_index]
    return K_element

def get_K(B, L_per_element, element_nodes):
    element_dofs = get_element_dofs(element_nodes)
    K_elements = np.array([get_K_element(B, L) for L in L_per_element])
    return assemble_element_blocks(K_elements, element_dofs, element_dofs, (dof_count, dof_count))

def get_integrated_B(B):
    return gauss_point_volume_weight * np.sum(B, axis=0)

def get_F_elements(B, L_per_element):
    # Each element's nodal load per unit eigenstrain, ∫ Bᵀ L dV (24 × 6).
    return get_integrated_B(B).T @ L_per_element

def get_free_dofs():
    # Pinning one node's three displacements removes the rigid translation of the periodic cell.
    pinned_dofs = np.arange(3)
    return np.setdiff1d(np.arange(dof_count), pinned_dofs)

## ------- Influence Functions (Offline) ------- ##

def get_F_macrostrain(B, L_per_element, element_nodes):
    # Nodal loads per unit macrostrain. Moving the applied strain to the right-hand side gives -∫ Bᵀ L ε̄ dV.
    macrostrain_columns = np.tile(np.arange(6), (element_count, 1))
    return -assemble_element_blocks(get_F_elements(B, L_per_element), get_element_dofs(element_nodes),
                                    macrostrain_columns, (dof_count, 6))

def get_F_eigenstrain(B, L_per_element, element_nodes, element_partition_ids):
    # F_μ in ER-24: nodal loads per unit partition eigenstrain, one block of six columns per partition.
    return assemble_element_blocks(get_F_elements(B, L_per_element), get_element_dofs(element_nodes),
                                   get_partition_components(element_partition_ids), (dof_count, 6 * partition_count))

def get_partition_averaging_operator(B, element_nodes, element_partition_ids):
    # A_ε in ER-24: maps nodal displacements to partition-average strain.
    element_blocks = np.broadcast_to(get_integrated_B(B) / partition_volume, (element_count, 6, 24))
    return assemble_element_blocks(element_blocks, get_partition_components(element_partition_ids),
                                   get_element_dofs(element_nodes), (6 * partition_count, dof_count))

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
    average_strain = np.zeros((6 * partition_count, F.shape[1]))
    column_batches = tqdm.tqdm(range(0, F.shape[1], columns_per_solve),
                               desc=f"  {f'solve {F.shape[1]} load columns':<26}",
                               bar_format="{desc}: {percentage:3.0f}% |{bar:25}| {n_fmt}/{total_fmt} batches "
                                          "[{elapsed} elapsed, {remaining} left]", file=sys.stdout)
    for first_column in column_batches:
        columns = slice(first_column, first_column + columns_per_solve)
        batch_F_free = F_free[:, columns].toarray()
        batch_displacements = np.zeros((dof_count, batch_F_free.shape[1]))
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
    E = np.tile(np.eye(6), (partition_count, 1)) + average_strain[:, :6]
    P = average_strain[:, 6:]
    return E, P

def get_homogenized_L(E, L_per_partition):
    E_blocks = E.reshape(partition_count, 6, 6)
    return np.mean(L_per_partition @ E_blocks, axis=0)

def get_P0_offset_blocks(B, reference_L_per_element, mesh, A):
    # The reference response to eigenstrain in partition 0 alone. Reference interactions depend only on the offset
    # between partitions (ER-6e), so block B of this response is P0[B - 0] and one solve gives the whole kernel.
    K = get_K(B, reference_L_per_element, mesh.element_nodes)
    F_eigenstrain = get_F_eigenstrain(B, reference_L_per_element, mesh.element_nodes, mesh.element_partition_ids)
    F_eigenstrain_in_first_partition = F_eigenstrain[:, :6]
    return solve_influence_function(K, F_eigenstrain_in_first_partition, A)

def get_P0_transformed(P0_offset_blocks):
    # P̂0(ξ) in ER-14a, transformed over the two in-plane lattice directions. The kernel is real, so P̂0(-ξ) is the
    # complex conjugate of P̂0(ξ), and only the non-redundant half of the frequencies is kept (rfftn).
    P0_offset_lattice = P0_offset_blocks.reshape(partition_number_per_side, partition_number_per_side, 6, 6)
    return np.fft.rfftn(P0_offset_lattice, axes=(0, 1))

## ------- Cache ------- ##

# Increase whenever a code change alters the offline operators, so caches made by the old code are recomputed.
cache_version = 1

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
    cache_folder.mkdir(exist_ok=True)
    np.savez(cache_path, **arrays, **parameters)

def get_offline_operators(mesh, B, L_per_element, L_matrix, L_inclusion, L_per_partition):
    A = get_partition_averaging_operator(B, mesh.element_nodes, mesh.element_partition_ids)

    E_P_cache_path = cache_folder / "E_P.npz"
    E_P_parameters = {"cache_version": cache_version,
                      "domain_side_length": domain_side_length,
                      "inclusion_shape": inclusion_shape,
                      "element_number_per_side": element_number_per_side,
                      "element_number_along_z": element_number_along_z,
                      "partition_number_per_side": partition_number_per_side,
                      "L_matrix": L_matrix,
                      "L_inclusion": L_inclusion}
    if inclusion_shape == "square":
        E_P_parameters["inclusion_side_length"] = inclusion_side_length
    else:
        E_P_parameters["inclusion_radius"] = inclusion_radius
    cached_E_P = load_cache(E_P_cache_path, E_P_parameters)
    if cached_E_P is None:
        E, P = get_influence_functions(B, L_per_element, mesh, A)
        save_cache(E_P_cache_path, E_P_parameters, E=E, P=P)
    else:
        E, P = cached_E_P["E"], cached_E_P["P"]

    reference_L = get_homogenized_L(E, L_per_partition)
    reference_L_per_element = np.tile(reference_L, (element_count, 1, 1))

    P0_cache_path = cache_folder / "P0_offset_blocks.npz"
    P0_parameters = {"cache_version": cache_version,
                     "domain_side_length": domain_side_length,
                     "element_number_per_side": element_number_per_side,
                     "element_number_along_z": element_number_along_z,
                     "partition_number_per_side": partition_number_per_side,
                     "reference_L": reference_L}
    cached_P0 = load_cache(P0_cache_path, P0_parameters)
    if cached_P0 is None:
        start_time = time.perf_counter()
        P0_offset_blocks = get_P0_offset_blocks(B, reference_L_per_element, mesh, A)
        print(f"P0 offline solve (extra cost of the FFT solver): {time.perf_counter() - start_time:.2f} seconds.")
        save_cache(P0_cache_path, P0_parameters, P0_offset_blocks=P0_offset_blocks)
    else:
        P0_offset_blocks = cached_P0["P0_offset_blocks"]

    # Column-major, so the columns of a run of consecutive partitions are contiguous and the induced strain can
    # read just the yielding partitions' columns without copying them (get_induced_strain_increment).
    return E, np.asfortranarray(P), get_P0_transformed(P0_offset_blocks)

## ------- Online Timing ------- ##

online_time_groups = ("material_update", "induced_strain", "reference_sensitivity", "reference_inverse",
                      "correction_solve")

class OnlineTimer:
    def __init__(self):
        self.open_group = None
        self.reset()

    def reset(self):
        self.time_per_group = [0.0] * len(online_time_groups)
        self.calls_per_group = [0] * len(online_time_groups)

online_timer = OnlineTimer()

def timed_online(group):
    group_id = online_time_groups.index(group)
    def decorator(function):
        @functools.wraps(function)
        def timed_function(*args, **kwargs):
            if online_timer.open_group is not None:
                raise RuntimeError(f"online time group {group!r} started inside {online_timer.open_group!r}, "
                                   "so its time would be counted twice.")
            online_timer.open_group = group
            start_time = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                online_timer.time_per_group[group_id] += time.perf_counter() - start_time
                online_timer.calls_per_group[group_id] += 1
                online_timer.open_group = None
        return timed_function
    return decorator

def get_online_timer_overhead_per_call(call_count=100_000):
    def untimed_function():
        pass
    timed_function = timed_online(online_time_groups[0])(untimed_function)

    start_time = time.perf_counter()
    for _ in range(call_count):
        timed_function()
    timed_duration = time.perf_counter() - start_time
    start_time = time.perf_counter()
    for _ in range(call_count):
        untimed_function()
    untimed_duration = time.perf_counter() - start_time
    online_timer.reset()
    return max(timed_duration - untimed_duration, 0) / call_count

## ------- Load Step Failures ------- ##

class LoadPathAbandoned(RuntimeError):
    # A load step that cannot be solved. run_strain_path catches only this type and keeps the steps before it, so
    # genuine bugs, such as the online timer's nesting guard or a failed conjugate gradient in post-processing,
    # still stop the run.
    pass

class SolverDidNotConverge(LoadPathAbandoned):
    pass

class MaterialFullySoftened(LoadPathAbandoned):
    pass

## ------- J2 Material ------- ##

class PlasticState(NamedTuple):
    # Partition-stacked plastic history: either the accepted history z_n of ER-3 or a candidate for the current step.
    plastic_strain: np.ndarray
    accumulated_plastic_strain: np.ndarray

class TrialState(NamedTuple):
    stress: np.ndarray
    mean_stress: np.ndarray
    deviatoric_stress: np.ndarray
    equivalent_stress: np.ndarray
    yield_function: np.ndarray
    plastic_multiplier: np.ndarray

volumetric_direction = np.array([1, 1, 1, 0, 0, 0])

deviatoric_projector = np.zeros((6, 6))
deviatoric_projector[:3, :3] = np.eye(3) - np.ones((3, 3)) / 3
deviatoric_projector[3:, 3:] = np.eye(3)

def get_deviatoric_stress(stress):
    return stress @ deviatoric_projector.T

def get_equivalent_stress(deviatoric_stress):
    return np.sqrt(1.5 * np.sum(deviatoric_stress[:, :3]**2, axis=1) + 3 * np.sum(deviatoric_stress[:, 3:]**2, axis=1))

def get_von_mises_stress(stress):
    return get_equivalent_stress(get_deviatoric_stress(stress))

def get_flow_stress(partition_materials, accumulated_plastic_strain):
    return partition_materials.yield_stress + partition_materials.hardening_modulus * accumulated_plastic_strain

def get_trial_state(strain, partition_materials, plastic_history):
    stress = np.einsum('pij,pj->pi', partition_materials.L, strain - plastic_history.plastic_strain)
    mean_stress = np.sum(stress[:, :3], axis=1) / 3
    deviatoric_stress = stress - np.outer(mean_stress, volumetric_direction)
    equivalent_stress = get_equivalent_stress(deviatoric_stress)
    yield_function = equivalent_stress - get_flow_stress(partition_materials,
                                                         plastic_history.accumulated_plastic_strain)
    shear_modulus = partition_materials.L[:, 3, 3]
    plastic_multiplier = np.maximum(yield_function, 0) / (3 * shear_modulus + partition_materials.hardening_modulus)
    return TrialState(stress, mean_stress, deviatoric_stress, equivalent_stress, yield_function, plastic_multiplier)

def get_flow_direction(deviatoric_stress, equivalent_stress):
    flow_direction = np.zeros_like(deviatoric_stress)
    flow_direction[:, :3] = 1.5 * deviatoric_stress[:, :3] / equivalent_stress[:, None]
    flow_direction[:, 3:] = 3 * deviatoric_stress[:, 3:] / equivalent_stress[:, None]
    return flow_direction

@timed_online("material_update")
def get_plastic_eigenstrain(strain, partition_materials, plastic_history):
    # ER-3 by radial return from the accepted history. The eigenstrain μ is the plastic strain.
    trial = get_trial_state(strain, partition_materials, plastic_history)

    plastic_strain = plastic_history.plastic_strain.copy()
    accumulated_plastic_strain = plastic_history.accumulated_plastic_strain.copy()
    stress = trial.stress.copy()

    yielding = trial.yield_function > 0
    yielding_deviatoric_stress = trial.deviatoric_stress[yielding]
    yielding_equivalent_stress = trial.equivalent_stress[yielding]
    shear_modulus = partition_materials.L[yielding, 3, 3]
    plastic_multiplier = trial.plastic_multiplier[yielding]

    flow_direction = get_flow_direction(yielding_deviatoric_stress, yielding_equivalent_stress)
    plastic_strain[yielding] += plastic_multiplier[:, None] * flow_direction
    accumulated_plastic_strain[yielding] += plastic_multiplier

    deviatoric_scaling = 1 - 3 * shear_modulus * plastic_multiplier / yielding_equivalent_stress
    corrected_deviatoric_stress = deviatoric_scaling[:, None] * yielding_deviatoric_stress
    stress[yielding] = np.outer(trial.mean_stress[yielding], volumetric_direction) + corrected_deviatoric_stress

    return PlasticState(plastic_strain, accumulated_plastic_strain), stress

def get_eigenstrain_sensitivity(strain, partition_materials, plastic_history):
    # H_μ = ∂μ/∂ε at the current strain (ER-8), zero in partitions that stay elastic.
    trial = get_trial_state(strain, partition_materials, plastic_history)

    sensitivity = np.zeros((partition_count, 6, 6))

    yielding = trial.yield_function > 0
    yielding_deviatoric_stress = trial.deviatoric_stress[yielding]
    yielding_equivalent_stress = trial.equivalent_stress[yielding]
    shear_modulus = partition_materials.L[yielding, 3, 3]
    hardening_modulus = partition_materials.hardening_modulus[yielding]
    plastic_multiplier = trial.plastic_multiplier[yielding]

    flow_direction = get_flow_direction(yielding_deviatoric_stress, yielding_equivalent_stress)
    normalized_deviatoric_stress = yielding_deviatoric_stress / yielding_equivalent_stress[:, None]
    flow_projector = np.einsum('pi,pj->pij', flow_direction, normalized_deviatoric_stress)

    radial_sensitivity = 3 * shear_modulus / (3 * shear_modulus + hardening_modulus)
    rotational_sensitivity = 3 * shear_modulus * plastic_multiplier / yielding_equivalent_stress

    sensitivity[yielding] = (radial_sensitivity[:, None, None] * flow_projector
                             + rotational_sensitivity[:, None, None] * (deviatoric_projector - flow_projector))
    return sensitivity

def check_flow_stress_is_positive(partition_materials, plastic_state):
    # Linear softening drives the flow stress to zero at finite plastic strain. Past that point the return map
    # flips the sign of the deviatoric stress, which the solvers would otherwise accept as a converged answer.
    flow_stress = get_flow_stress(partition_materials, plastic_state.accumulated_plastic_strain)
    if np.any(flow_stress <= 0):
        raise MaterialFullySoftened(f"{np.count_nonzero(flow_stress <= 0)} partitions softened to zero flow stress "
                                    f"(lowest {flow_stress.min():.3g} Pa)")

## ------- Solvers ------- ##

@timed_online("induced_strain")
def get_induced_strain(P, eigenstrain):
    return (P @ eigenstrain.reshape(-1)).reshape(partition_count, 6)

def get_partition_runs(partitions):
    # (first, stop) of every run of consecutive partitions in the mask, in partition order.
    edges = np.diff(np.concatenate(([0], partitions.astype(int), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))

@timed_online("induced_strain")
def get_induced_strain_increment(P, eigenstrain_increment, changed_partitions):
    # PΔμ from only the columns of P that belong to changed partitions. P is column-major, so each run of
    # consecutive changed partitions is one contiguous block of columns, used without a copy.
    increment = np.zeros(6 * partition_count)
    for first, stop in get_partition_runs(changed_partitions):
        increment += P[:, 6 * first:6 * stop] @ eigenstrain_increment[first:stop].reshape(-1)
    return increment.reshape(partition_count, 6)

def get_induced_strain_reusing_history(P, plastic_state, plastic_history, induced_strain_history):
    # Pμ = Pμₙ + PΔμ, where Δμ is zero outside the partitions yielding in this step, so only their columns of P
    # are needed. The mask is taken from Δμ itself, so every skipped column multiplies an exact zero.
    eigenstrain_increment = plastic_state.plastic_strain - plastic_history.plastic_strain
    changed_partitions = np.any(eigenstrain_increment != 0, axis=1)
    if not np.any(changed_partitions):
        return induced_strain_history
    return induced_strain_history + get_induced_strain_increment(P, eigenstrain_increment, changed_partitions)

# Weights that turn the engineering shear convention of ER-2 into the tensor norm ε:ε, since γ = 2ε for shear.
tensor_norm_weights = np.array([1, 1, 1, 0.5, 0.5, 0.5])

def get_strain_norm(strain):
    # Tensor norm of one strain, or the root-mean-square over partitions of a partition-stacked strain. Every
    # partition has the same volume, so the mean is also the volume average.
    squared_tensor_norms = np.sum(tensor_norm_weights * np.atleast_2d(strain)**2, axis=1)
    return np.sqrt(np.mean(squared_tensor_norms))

def get_relative_residual(residual, macro_strain):
    # Scaled by the imposed strain rather than the current iterate, and never by less than a fixed floor, so the
    # criterion stays meaningful when a load path passes through zero strain (Formulation.md section 12).
    strain_scale = max(get_strain_norm(macro_strain), residual_strain_scale_floor)
    return get_strain_norm(residual) / strain_scale

def has_converged(relative_residual, iteration_count, solver_name, divergence_limit=divergence_residual_limit,
                  max_iterations=fixed_point_max_iterations):
    if not np.isfinite(relative_residual) or relative_residual > divergence_limit:
        raise SolverDidNotConverge(f"{solver_name} diverged after {iteration_count} iterations "
                                   f"(relative residual {relative_residual})")
    if relative_residual < fixed_point_tolerance:
        return True
    if iteration_count >= max_iterations:
        raise SolverDidNotConverge(f"{solver_name} did not converge within {max_iterations} iterations "
                                   f"(relative residual {relative_residual})")
    return False

def reset_elastic_partitions(strain, b, P, induced_strain_history, partition_materials, plastic_history):
    # Updates the material at the given strain, solves the partitions that stay elastic exactly, and returns the
    # residual r = ε - b - Pμ(ε) of ER-4, which is then zero in every elastic partition.
    #
    # An elastic partition's eigenstrain stays at its accepted value, so setting its strain to b + Pμ zeroes its
    # residual without changing μ. Yielding partitions keep their strain, so their μ does not change either. The
    # second material update catches elastic partitions that the reset pushed past yield: only then does μ, and
    # so Pμ, change.
    plastic_state, stress = get_plastic_eigenstrain(strain, partition_materials, plastic_history)
    induced_strain = get_induced_strain_reusing_history(P, plastic_state, plastic_history, induced_strain_history)
    yielding_partitions = plastic_state.accumulated_plastic_strain > plastic_history.accumulated_plastic_strain

    strain = strain.copy()
    strain[~yielding_partitions] = b[~yielding_partitions] + induced_strain[~yielding_partitions]

    plastic_state, stress = get_plastic_eigenstrain(strain, partition_materials, plastic_history)
    yielding_partitions_after_reset = (plastic_state.accumulated_plastic_strain
                                       > plastic_history.accumulated_plastic_strain)
    if not np.array_equal(yielding_partitions_after_reset, yielding_partitions):
        induced_strain = get_induced_strain_reusing_history(P, plastic_state, plastic_history, induced_strain_history)

    residual = strain - b - induced_strain
    return strain, plastic_state, stress, residual

@timed_online("correction_solve")
def get_standard_correction(residual):
    return -residual

def standard_richardson_iteration(E, P, macro_strain, partition_materials, plastic_history):
    b = (E @ macro_strain).reshape(partition_count, 6)
    induced_strain_history = get_induced_strain(P, plastic_history.plastic_strain)
    strain = b + induced_strain_history

    residual_history = []
    while True:
        strain, plastic_state, stress, residual = reset_elastic_partitions(
            strain, b, P, induced_strain_history, partition_materials, plastic_history)
        residual_history.append(get_relative_residual(residual, macro_strain))

        if has_converged(residual_history[-1], len(residual_history), "standard_richardson_iteration"):
            break

        strain = strain + relaxation_factor * get_standard_correction(residual)

    return stress, plastic_state, np.array(residual_history)

@timed_online("reference_sensitivity")
def get_reference_sensitivity(strain, partition_materials, plastic_history):
    # H_μ,0 (ER-8): the partition average of the actual sensitivity, one 6 × 6 block shared by every partition so
    # the reference stays a convolution.
    sensitivity = get_eigenstrain_sensitivity(strain, partition_materials, plastic_history)
    return np.mean(sensitivity, axis=0)

@timed_online("reference_inverse")
def get_reference_fourier_inverse(P0_transformed, reference_sensitivity):
    # Inverts M̂0(ξ) = I - P̂0(ξ) H_μ,0 (ER-15) at every frequency.
    M0_transformed = np.eye(6) - P0_transformed @ reference_sensitivity
    try:
        return np.linalg.inv(M0_transformed)
    except np.linalg.LinAlgError:
        raise SolverDidNotConverge("the reference operator M0 is singular at some frequency, so the FFT "
                                   "correction is undefined") from None

def apply_on_partition_lattice(fourier_blocks, partition_field):
    # Applies a translation-invariant operator, given by its 6 × 6 block at every lattice frequency, to a
    # partition-stacked field: transform the field, multiply frequency by frequency, transform back (ER-14a).
    # The field and the operator are real, so only the non-redundant half of the frequencies is used.
    lattice_shape = (partition_number_per_side, partition_number_per_side)
    field_transformed = np.fft.rfftn(partition_field.reshape(*lattice_shape, 6), axes=(0, 1))
    result_transformed = np.einsum('...ij,...j->...i', fourier_blocks, field_transformed)
    return np.fft.irfftn(result_transformed, s=lattice_shape, axes=(0, 1)).reshape(partition_count, 6)

@timed_online("correction_solve")
def get_fft_correction(reference_fourier_inverse, residual):
    # ER-15 steps 2-4: solve M0 δε = -r one frequency at a time.
    return apply_on_partition_lattice(reference_fourier_inverse, -residual)

def fft_preconditioned_richardson_iteration(E, P, macro_strain, partition_materials, plastic_history,
                                            P0_transformed):
    b = (E @ macro_strain).reshape(partition_count, 6)
    induced_strain_history = get_induced_strain(P, plastic_history.plastic_strain)
    strain = b + induced_strain_history

    yielding_partitions_at_last_rebuild = None
    residual_history = []
    while True:
        strain, plastic_state, stress, residual = reset_elastic_partitions(
            strain, b, P, induced_strain_history, partition_materials, plastic_history)
        residual_history.append(get_relative_residual(residual, macro_strain))

        if has_converged(residual_history[-1], len(residual_history), "fft_preconditioned_richardson_iteration"):
            break

        # The reference is rebuilt only when the set of yielding partitions changes, not every iteration.
        yielding_partitions = plastic_state.accumulated_plastic_strain > plastic_history.accumulated_plastic_strain
        if not np.array_equal(yielding_partitions, yielding_partitions_at_last_rebuild):
            reference_sensitivity = get_reference_sensitivity(strain, partition_materials, plastic_history)
            reference_fourier_inverse = get_reference_fourier_inverse(P0_transformed, reference_sensitivity)
            yielding_partitions_at_last_rebuild = yielding_partitions

        strain = strain + relaxation_factor * get_fft_correction(reference_fourier_inverse, residual)

    return stress, plastic_state, np.array(residual_history)

## ------- Load Path ------- ##

def get_solvers(P0_transformed):
    return {"Standard": standard_richardson_iteration,
            "FFT-preconditioned": functools.partial(fft_preconditioned_richardson_iteration,
                                                    P0_transformed=P0_transformed)}

class LoadPathResult(NamedTuple):
    applied_macro_strain: np.ndarray
    macroscopic_stress: np.ndarray
    iterations_per_step: np.ndarray
    solve_time_per_step: np.ndarray
    group_time_per_step: np.ndarray
    group_calls_per_step: np.ndarray
    final_stress: np.ndarray
    final_plastic_strain: np.ndarray
    # Steps actually solved. Entries past this are untouched zeros, so totals stay honest but anything plotted
    # has to be sliced or the curves fall to zero at the abandoned steps.
    completed_step_count: int
    failure_reason: str

def get_macroscopic_stress(stress):
    # ER-5. Every partition has the same volume, so the volume-weighted sum is a plain mean.
    return np.mean(stress, axis=0)

def run_strain_path(solver_name, solver, E, P, partition_materials):
    plastic_history = PlasticState(np.zeros((partition_count, 6)), np.zeros(partition_count))

    load_fractions = np.linspace(1 / strain_increment_count, 1, strain_increment_count)
    applied_macro_strain = np.outer(load_fractions, max_macro_strain)
    macroscopic_stress = np.zeros((strain_increment_count, 6))
    iterations_per_step = np.zeros(strain_increment_count, dtype=int)
    solve_time_per_step = np.zeros(strain_increment_count)
    group_time_per_step = np.zeros((strain_increment_count, len(online_time_groups)))
    group_calls_per_step = np.zeros((strain_increment_count, len(online_time_groups)), dtype=int)
    completed_step_count = strain_increment_count
    failure_reason = ""

    for step, macro_strain in enumerate(applied_macro_strain):
        online_timer.reset()
        start_time = time.perf_counter()
        try:
            step_stress, step_plastic_state, residual_history = solver(E, P, macro_strain, partition_materials,
                                                                       plastic_history)
            check_flow_stress_is_positive(partition_materials, step_plastic_state)
        except LoadPathAbandoned as failure:
            # Failing on the first step leaves no partial result worth keeping.
            if step == 0:
                raise
            completed_step_count, failure_reason = step, str(failure)
            print(f"  {solver_name}: abandoned the load path at step {step + 1} of {strain_increment_count}. "
                  f"{failure_reason}.")
            break
        solve_time_per_step[step] = time.perf_counter() - start_time
        group_time_per_step[step] = online_timer.time_per_group
        group_calls_per_step[step] = online_timer.calls_per_group
        iterations_per_step[step] = len(residual_history)
        macroscopic_stress[step] = get_macroscopic_stress(step_stress)
        stress, plastic_history = step_stress, step_plastic_state

    return LoadPathResult(applied_macro_strain, macroscopic_stress, iterations_per_step, solve_time_per_step,
                          group_time_per_step, group_calls_per_step, stress, plastic_history.plastic_strain,
                          completed_step_count, failure_reason)

def run_interleaved_repeats(get_solvers_for_one_path, E, P, partition_materials):
    # Every solver runs once per repeat, and the one that runs first rotates between repeats. Taking the fastest
    # repeat per step removes random noise, but not a bias that recurs every repeat, such as the first solver
    # paying the first-touch cost of the dense P. Solvers are built fresh for every load path, so none carries
    # state from one path into the next.
    solver_names = list(get_solvers_for_one_path())
    results = {solver_name: [] for solver_name in solver_names}
    for repeat in range(timing_repeat_count):
        for position in range(len(solver_names)):
            solver_name = solver_names[(repeat + position) % len(solver_names)]
            solver = get_solvers_for_one_path()[solver_name]
            results[solver_name].append(run_strain_path(solver_name, solver, E, P, partition_materials))
    return results

## ------- Post-processing ------- ##

def get_F_state(B, L_per_element, mesh, macro_strain, eigenstrain):
    return (get_F_macrostrain(B, L_per_element, mesh.element_nodes) @ macro_strain
            + get_F_eigenstrain(B, L_per_element, mesh.element_nodes, mesh.element_partition_ids)
            @ eigenstrain.ravel())

def solve_state_displacements(K, F):
    free_dofs = get_free_dofs()
    K_free = K[free_dofs][:, free_dofs]
    jacobi_preconditioner = scipy.sparse.diags(1 / K_free.diagonal())
    conjugate_gradient_tolerance = 1e-10
    free_displacements, info = scipy.sparse.linalg.cg(K_free, F[free_dofs], rtol=conjugate_gradient_tolerance,
                                                      M=jacobi_preconditioner)
    if info != 0:
        raise RuntimeError(f"conjugate gradient did not converge (info {info}).")
    displacements = np.zeros(dof_count)
    displacements[free_dofs] = free_displacements
    return displacements

def get_element_average_fluctuation_strain(displacements, B, element_nodes):
    element_dofs = get_element_dofs(element_nodes)
    return displacements[element_dofs] @ get_integrated_B(B).T / element_volume

def get_element_stress(mesh, B, L_per_element, macro_strain, eigenstrain):
    start_time = time.perf_counter()
    K = get_K(B, L_per_element, mesh.element_nodes)
    F = get_F_state(B, L_per_element, mesh, macro_strain, eigenstrain)
    displacements = solve_state_displacements(K, F)

    element_strain = macro_strain + get_element_average_fluctuation_strain(displacements, B, mesh.element_nodes)
    element_eigenstrain = eigenstrain[mesh.element_partition_ids]
    element_stress = np.einsum('eij,ej->ei', L_per_element, element_strain - element_eigenstrain)
    print(f"element stress post-processing: {time.perf_counter() - start_time:.2f} seconds.")
    return element_stress

## ------- Timing Report ------- ##

def get_completed_step_count(solver_results):
    return min(result.completed_step_count for result in solver_results)

def get_fastest_repeat_per_step(solver_results, field):
    fastest_repeat_per_step = np.argmin([result.solve_time_per_step for result in solver_results], axis=0)
    return np.array([getattr(result, field) for result in solver_results])[fastest_repeat_per_step,
                                                                           np.arange(strain_increment_count)]

def get_solver_label(solver_name, solver_results, step_count):
    iteration_count = get_fastest_repeat_per_step(solver_results, "iterations_per_step")[:step_count].sum()
    solve_time = get_fastest_repeat_per_step(solver_results, "solve_time_per_step")[:step_count].sum()
    label = (f"{solver_name}: {iteration_count} iterations, {solve_time:.2f} s, "
             f"{1000 * solve_time / iteration_count:.2f} ms per iteration")
    last_result = solver_results[-1]
    if last_result.completed_step_count < strain_increment_count:
        label += f"  [stopped at step {last_result.completed_step_count + 1}: {last_result.failure_reason}]"
    return label

def get_time_per_iteration_per_group(solver_results, step_count):
    solve_time = get_fastest_repeat_per_step(solver_results, "solve_time_per_step")[:step_count].sum()
    time_per_group = get_fastest_repeat_per_step(solver_results, "group_time_per_step")[:step_count].sum(axis=0)
    iteration_count = get_fastest_repeat_per_step(solver_results, "iterations_per_step")[:step_count].sum()
    return np.append(time_per_group, solve_time - time_per_group.sum()) / iteration_count

def get_timer_overhead_fraction(solver_results, timer_overhead_per_call, step_count):
    solve_time = get_fastest_repeat_per_step(solver_results, "solve_time_per_step")[:step_count].sum()
    timed_call_count = get_fastest_repeat_per_step(solver_results, "group_calls_per_step")[:step_count].sum()
    return timer_overhead_per_call * timed_call_count / solve_time

## ------- Main ------- ##

def print_comparison(results, comparable_step_count):
    print(f"relaxation_factor = {relaxation_factor}")
    compared_steps = (f"the {comparable_step_count} of {strain_increment_count} steps every solver completed"
                      if comparable_step_count < strain_increment_count else f"all {strain_increment_count} steps")
    print(f"solve times are the minimum per load step over {timing_repeat_count} interleaved runs of each solver, "
          f"summed over {compared_steps}")
    for solver_name, solver_results in results.items():
        print(get_solver_label(solver_name, solver_results, comparable_step_count))

    timer_overhead_per_call = get_online_timer_overhead_per_call()
    overhead_labels = []
    for solver_name, solver_results in results.items():
        overhead_fraction = get_timer_overhead_fraction(solver_results, timer_overhead_per_call, comparable_step_count)
        overhead_labels.append(f"{100 * overhead_fraction:.3f} % ({solver_name})")
    print(f"online timer overhead: {', '.join(overhead_labels)} of online time")

def get_most_complete_solver_name(results):
    # The solver that got furthest along the load path, whose stresses are plotted. The first listed wins a tie.
    return max(results, key=lambda solver_name: get_completed_step_count(results[solver_name]))

def get_solver_steps(solver_results):
    return plots.SolverSteps(iterations_per_step=solver_results[-1].iterations_per_step,
                             solve_time_per_step=get_fastest_repeat_per_step(solver_results, "solve_time_per_step"),
                             completed_step_count=get_completed_step_count(solver_results))

def save_load_path_plots(results, comparable_step_count, plotted_solver_name):
    plots.plot_load_path_summary(
        get_deviatoric_stress(results[plotted_solver_name][-1].macroscopic_stress) / 1e6, plotted_solver_name,
        {solver_name: get_solver_steps(solver_results) for solver_name, solver_results in results.items()},
        plots.get_applied_strain_description(max_macro_strain, strain_increment_count),
        output_folder / "load_path_summary.png")
    plots.plot_time_per_iteration_breakdown(
        {solver_name: 1000 * get_time_per_iteration_per_group(solver_results, comparable_step_count)
         for solver_name, solver_results in results.items()},
        output_folder / "time_per_iteration_breakdown.png")

def save_von_mises_plots(result, mesh, B, L_per_element):
    partition_von_mises_stress_MPa = get_von_mises_stress(result.final_stress) / 1e6
    plots.plot_von_mises_cross_section(
        partition_von_mises_stress_MPa.reshape(partition_number_per_side, partition_number_per_side),
        element_number_per_side, partition_number_per_side, von_mises_plot_min_stress / 1e6,
        von_mises_plot_max_stress / 1e6, output_folder / "von_mises_cross_section.png")

    if post_process_element_stress:
        # The last step the solver actually solved, which is not the last requested one if it stopped early.
        final_macro_strain = result.applied_macro_strain[result.completed_step_count - 1]
        element_stress = get_element_stress(mesh, B, L_per_element, final_macro_strain, result.final_plastic_strain)
        bottom_layer_von_mises_stress_MPa = get_von_mises_stress(element_stress[:element_number_per_side**2]) / 1e6
        plots.plot_von_mises_cross_section(
            bottom_layer_von_mises_stress_MPa.reshape(element_number_per_side, element_number_per_side),
            element_number_per_side, partition_number_per_side, von_mises_plot_min_stress / 1e6,
            von_mises_plot_max_stress / 1e6, output_folder / "von_mises_element_cross_section.png")

def check_solver_agreement(results, comparable_step_count):
    # Both solvers solve the same equations, so their macroscopic stresses must agree to within solver tolerance.
    standard_stress = results["Standard"][-1].macroscopic_stress[:comparable_step_count]
    fft_stress = results["FFT-preconditioned"][-1].macroscopic_stress[:comparable_step_count]
    stress_scale = max(np.abs(standard_stress).max(), np.finfo(float).tiny)
    stress_difference = np.abs(fft_stress - standard_stress).max() / stress_scale

    passed = stress_difference <= solver_agreement_tolerance
    print(f"[{'PASS' if passed else 'FAIL'}] stress agreement between solvers: {stress_difference:.2e} relative "
          f"(tolerance {solver_agreement_tolerance:.0e}), over {comparable_step_count} of {strain_increment_count} "
          "steps.")
    if not passed:
        raise RuntimeError("the standard and FFT-preconditioned solvers disagree on the macroscopic stress.")

def main():
    output_folder.mkdir(exist_ok=True)
    partition_material_ids = get_partition_material_ids()
    print(f"inclusion volume fraction on the partition grid: {partition_material_ids.mean():.4f}")
    plots.plot_cross_section(partition_material_ids.reshape(partition_number_per_side, partition_number_per_side),
                             element_number_per_side, output_folder / "cross_section.png")

    L_matrix = get_L(elastic_modulus_matrix, poisson_ratio_matrix)
    L_inclusion = get_L(elastic_modulus_inclusion, poisson_ratio_inclusion)
    partition_materials = get_partition_materials(L_matrix, L_inclusion, partition_material_ids)
    mesh = get_periodic_mesh(partition_material_ids)
    B = get_B()
    L_per_element = get_value_per_material(L_matrix, L_inclusion, mesh.element_material_ids)
    E, P, P0_transformed = get_offline_operators(mesh, B, L_per_element, L_matrix, L_inclusion, partition_materials.L)

    results = run_interleaved_repeats(functools.partial(get_solvers, P0_transformed), E, P, partition_materials)
    # Timings and stresses are compared only over the steps every solver completed.
    comparable_step_count = min(get_completed_step_count(solver_results) for solver_results in results.values())
    print_comparison(results, comparable_step_count)
    plotted_solver_name = get_most_complete_solver_name(results)
    save_load_path_plots(results, comparable_step_count, plotted_solver_name)
    save_von_mises_plots(results[plotted_solver_name][-1], mesh, B, L_per_element)
    check_solver_agreement(results, comparable_step_count)

if __name__ == "__main__":
    main()
