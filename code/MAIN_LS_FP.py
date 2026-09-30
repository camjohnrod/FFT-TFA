import functools
import pathlib
import sys
import time
from typing import NamedTuple
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import tqdm
import plots_testing as plots

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

max_macro_strain            = np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0])
# Alternative load paths:
# max_macro_strain            = np.array([0.03, 0.018, 0.001, 0.0, 0.0, 0.0])
# max_macro_strain            = np.array([0.015, 0.020, 0.0, 0.03, 0.0, 0.0])

strain_increment_count      = 60

fixed_point_tolerance       = 1e-6
fixed_point_max_iterations  = 1000
divergence_residual_limit   = 1.0
residual_strain_scale_floor = 1e-4
relaxation_factor           = 1.0
solver_agreement_tolerance  = 1e-4
reference_solver_method     = "newton"
reference_relaxation_factor = None
reference_stability_safety  = 0.9
reference_max_iterations    = 20000
reference_newton_max_steps  = 50
reference_divergence_limit  = 1e3
reference_krylov_tolerance  = 1e-3
reference_krylov_restart    = 40

timing_repeat_count         = 5
post_process_element_stress = True
von_mises_plot_min_stress   = 1.0e6
von_mises_plot_max_stress   = 3.0e6

verification_enabled        = True
verification_tolerance      = 1e-9
jacobian_check_tolerance    = 1e-6
verification_max_partitions = 256
matched_stiffness_control   = False
matched_stiffness_tolerance = 1e-5

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

cache_folder = pathlib.Path(__file__).parent / "cache_testing"
output_folder = pathlib.Path(__file__).parent / "output_testing"

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
        # Operator applications made inside another group, which only the Newton solver's Krylov solve does. They
        # are counted and timed separately so the cost breakdown can show all operator work as one segment.
        self.nested_operator_applications = 0
        self.nested_operator_time_per_group = [0.0] * len(online_time_groups)

    def record_nested_operator_application(self, elapsed_time):
        self.nested_operator_applications += 1
        self.nested_operator_time_per_group[online_time_groups.index(self.open_group)] += elapsed_time

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

## ------- Reference Lippmann-Schwinger Solver ------- ##

# The reduced model of notes/Partition_Lippmann_Schwinger_New_9_28_2026: a partitionwise-constant polarization
# interacting through the homogeneous reference medium, so the whole nonlocal operator is the P0 convolution. It is
# a different model from actual E/P, so it matches the two solvers above only under matched phase stiffness.

@timed_online("induced_strain")
def get_reference_induced_strain(P0_transformed, partition_field):
    # P0 applied as a convolution over the partition lattice (LS-17), the reference counterpart of
    # get_induced_strain.
    return apply_on_partition_lattice(P0_transformed, partition_field)

def get_reference_state(strain, b, P0_transformed, reference_compliance, partition_materials, plastic_history):
    # The LS-18 residual r = ε - b - P0 μ*, where μ* = ε - C0⁻¹ σ is the equivalent eigenstrain of LS-6.
    plastic_state, stress = get_plastic_eigenstrain(strain, partition_materials, plastic_history)
    equivalent_eigenstrain = strain - stress @ reference_compliance.T
    residual = strain - b - get_reference_induced_strain(P0_transformed, equivalent_eigenstrain)
    return plastic_state, stress, residual

def get_reference_stiffness_ratio_bounds(partition_materials, reference_L):
    # eig(P0) in [0, 1] makes the Jacobian I - P0(I - C0^-1 L_t) similar to a symmetric matrix with spectrum
    # inside [lambda_min, lambda_max] of C0^-1 L. The elastic L is used because the plastic tangent is softer.
    stiffness_ratio = np.linalg.eigvals(np.linalg.inv(reference_L) @ partition_materials.L).real
    return np.min(stiffness_ratio), np.max(stiffness_ratio)

def get_reference_stability_limit(partition_materials, reference_L):
    return 2 / get_reference_stiffness_ratio_bounds(partition_materials, reference_L)[1]

def get_reference_relaxation_factor(partition_materials, reference_L):
    if reference_relaxation_factor is not None:
        return reference_relaxation_factor
    smallest_ratio, largest_ratio = get_reference_stiffness_ratio_bounds(partition_materials, reference_L)
    return reference_stability_safety * 2 / (smallest_ratio + largest_ratio)

@timed_online("reference_sensitivity")
def get_reference_jacobian_blocks(strain, partition_materials, plastic_history, reference_compliance):
    # The block-diagonal factor I - C0⁻¹ L_t of the LS-20 Jacobian J v = v - P0 [(I - C0⁻¹ L_t) v], with the
    # algorithmic stress tangent L_t = L (I - H_μ).
    sensitivity = get_eigenstrain_sensitivity(strain, partition_materials, plastic_history)
    algorithmic_stiffness = partition_materials.L @ (np.eye(6) - sensitivity)
    return np.eye(6) - reference_compliance @ algorithmic_stiffness

@timed_online("correction_solve")
def get_newton_correction(P0_transformed, jacobian_blocks, residual):
    # Timed as one correction solve covering the whole Krylov solve, so this group means the same thing as for the
    # Richardson solvers: the cost of producing the correction. Its P0 applications are recorded as nested.
    def apply_jacobian(flat_vector):
        vector = flat_vector.reshape(partition_count, 6)
        start_time = time.perf_counter()
        coupled_strain = apply_on_partition_lattice(P0_transformed, np.einsum('pij,pj->pi', jacobian_blocks, vector))
        online_timer.record_nested_operator_application(time.perf_counter() - start_time)
        return (vector - coupled_strain).reshape(-1)

    system_size = 6 * partition_count
    jacobian = scipy.sparse.linalg.LinearOperator((system_size, system_size), matvec=apply_jacobian)
    correction, info = scipy.sparse.linalg.gmres(jacobian, -residual.reshape(-1),
                                                 rtol=reference_krylov_tolerance, restart=reference_krylov_restart)
    if info != 0:
        raise SolverDidNotConverge(f"GMRES did not converge inside the reference Newton step (info {info}). Loosen "
                                   "reference_krylov_tolerance, raise reference_krylov_restart, or use "
                                   "reference_solver_method = \"fixed_point\"")
    return correction.reshape(partition_count, 6)

def reference_iteration(macro_strain, initial_strain, partition_materials, plastic_history, P0_transformed,
                        reference_compliance, relaxation, method):
    using_newton = method == "newton"
    # A Newton step costs a whole Krylov solve, so it gets a far smaller cap than the fixed point: without one, a
    # stagnating solve would grind through hundreds of thousands of convolutions instead of failing.
    max_iterations = reference_newton_max_steps if using_newton else reference_max_iterations
    b = np.tile(macro_strain, (partition_count, 1))
    strain = initial_strain

    residual_history = []
    while True:
        plastic_state, stress, residual = get_reference_state(strain, b, P0_transformed, reference_compliance,
                                                              partition_materials, plastic_history)
        residual_history.append(get_relative_residual(residual, macro_strain))

        if has_converged(residual_history[-1], len(residual_history), f"reference_{method}",
                         reference_divergence_limit, max_iterations):
            break

        if using_newton:
            jacobian_blocks = get_reference_jacobian_blocks(strain, partition_materials, plastic_history,
                                                            reference_compliance)
            strain = strain + get_newton_correction(P0_transformed, jacobian_blocks, residual)
        else:
            # LS-19, damped.
            strain = strain + relaxation * get_standard_correction(residual)

    return stress, plastic_state, np.array(residual_history), strain

class ReferenceSolver:
    # The reference solver as a load-path solver. Each step starts from the previous step's converged strain
    # shifted by the macrostrain increment, so the solver keeps that strain between steps; get_solvers builds a
    # fresh one for every load path.
    def __init__(self, P0_transformed, reference_compliance, relaxation, method):
        self.P0_transformed = P0_transformed
        self.reference_compliance = reference_compliance
        self.relaxation = relaxation
        self.method = method
        self.previous_strain = np.zeros((partition_count, 6))
        self.previous_macro_strain = np.zeros(6)

    def __call__(self, E, P, macro_strain, partition_materials, plastic_history):
        # E and P describe the actual heterogeneous model, which the reference model replaces, so they go unused.
        initial_strain = self.previous_strain + (macro_strain - self.previous_macro_strain)
        stress, plastic_state, residual_history, strain = reference_iteration(
            macro_strain, initial_strain, partition_materials, plastic_history, self.P0_transformed,
            self.reference_compliance, self.relaxation, self.method)
        self.previous_strain, self.previous_macro_strain = strain, macro_strain
        return stress, plastic_state, residual_history

## ------- Load Path ------- ##

actual_model_solver_names = ("Standard", "FFT-preconditioned")
reference_solver_name = f"Reference LS ({reference_solver_method})"

def get_solvers(P0_transformed, reference_compliance, reference_relaxation):
    return {"Standard": standard_richardson_iteration,
            "FFT-preconditioned": functools.partial(fft_preconditioned_richardson_iteration,
                                                    P0_transformed=P0_transformed),
            reference_solver_name: ReferenceSolver(P0_transformed, reference_compliance, reference_relaxation,
                                                   reference_solver_method)}

class LoadPathResult(NamedTuple):
    applied_macro_strain: np.ndarray
    macroscopic_stress: np.ndarray
    iterations_per_step: np.ndarray
    # Applications of the nonlocal operator (dense P or the P0 convolution): the one work measure that means the
    # same for every solver, since a Newton step contains a whole Krylov solve of them.
    operator_applications_per_step: np.ndarray
    # Taken from the material state rather than the iteration count, which only marks elastic steps for the two
    # solvers that update non-yielding partitions in closed form.
    yielding_per_step: np.ndarray
    solve_time_per_step: np.ndarray
    group_time_per_step: np.ndarray
    group_calls_per_step: np.ndarray
    nested_operator_time_per_group_per_step: np.ndarray
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
    operator_applications_per_step = np.zeros(strain_increment_count, dtype=int)
    yielding_per_step = np.zeros(strain_increment_count, dtype=bool)
    solve_time_per_step = np.zeros(strain_increment_count)
    group_time_per_step = np.zeros((strain_increment_count, len(online_time_groups)))
    group_calls_per_step = np.zeros((strain_increment_count, len(online_time_groups)), dtype=int)
    nested_operator_time_per_group_per_step = np.zeros((strain_increment_count, len(online_time_groups)))
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
        operator_applications_per_step[step] = (online_timer.calls_per_group[online_time_groups.index("induced_strain")]
                                                + online_timer.nested_operator_applications)
        nested_operator_time_per_group_per_step[step] = online_timer.nested_operator_time_per_group
        yielding_per_step[step] = np.any(step_plastic_state.accumulated_plastic_strain
                                         > plastic_history.accumulated_plastic_strain)
        macroscopic_stress[step] = get_macroscopic_stress(step_stress)
        stress, plastic_history = step_stress, step_plastic_state

    return LoadPathResult(applied_macro_strain, macroscopic_stress, iterations_per_step,
                          operator_applications_per_step, yielding_per_step, solve_time_per_step,
                          group_time_per_step, group_calls_per_step, nested_operator_time_per_group_per_step,
                          stress, plastic_history.plastic_strain, completed_step_count, failure_reason)

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

## ------- Verification ------- ##

def get_dense_P0(P0_offset_blocks):
    # Expands the offset kernel into the full P0 by P0^BA = p0[a_B - a_A]. get_grid_id puts i fastest, so the
    # C-order reshape indexes the lattice as [j, i] -- the same convention get_P0_transformed relies on.
    # offsets[receiver, source] is the periodic separation along one axis; blocks is then indexed
    # [j_receiver, i_receiver, j_source, i_source] and the transpose interleaves the 6-vector axes to give
    # dense[6 * receiver + component, 6 * source + component].
    offset_lattice = P0_offset_blocks.reshape(partition_number_per_side, partition_number_per_side, 6, 6)
    lattice_indices = np.arange(partition_number_per_side)
    offsets = (lattice_indices[:, None] - lattice_indices[None, :]) % partition_number_per_side
    blocks = offset_lattice[offsets[:, None, :, None], offsets[None, :, None, :]]
    return blocks.transpose(0, 1, 4, 2, 3, 5).reshape(6 * partition_count, 6 * partition_count)

def get_P0_offset_blocks_from_transformed(P0_transformed):
    lattice_shape = (partition_number_per_side, partition_number_per_side)
    return np.fft.irfftn(P0_transformed, s=lattice_shape, axes=(0, 1)).reshape(6 * partition_count, 6)

def get_reference_kernel_checks(P0_transformed, reference_L):
    # P0_transformed holds half the frequencies. That covers them all: the block at -ξ is the complex conjugate of
    # the one at ξ, so it passes each check below exactly when the block at ξ does.
    zero_frequency_error = np.max(np.abs(P0_transformed[0, 0])) / np.max(np.abs(P0_transformed))

    D0_transformed = P0_transformed @ np.linalg.inv(reference_L)
    hermitian_difference = D0_transformed - np.conj(np.swapaxes(D0_transformed, -1, -2))
    hermitian_error = np.max(np.abs(hermitian_difference)) / np.max(np.abs(D0_transformed))

    eigenvalues = np.linalg.eigvals(P0_transformed)
    return [("zero-frequency block vanishes", zero_frequency_error, verification_tolerance),
            ("P0(xi) C0^-1 is Hermitian", hermitian_error, verification_tolerance),
            ("P0(xi) eigenvalues are real", np.max(np.abs(eigenvalues.imag)), verification_tolerance),
            ("P0(xi) eigenvalues lie in [0, 1]",
             max(-np.min(eigenvalues.real), np.max(eigenvalues.real) - 1, 0.0), verification_tolerance)]

def get_dense_reference_checks(problem, dense_P0):
    print(f"  dense reference influence functions ({6 + 6 * partition_count} load columns):")
    reference_L_per_element = np.tile(problem.reference_L, (element_count, 1, 1))
    A = get_partition_averaging_operator(problem.B, problem.mesh.element_nodes, problem.mesh.element_partition_ids)
    E_reference, P_reference = get_influence_functions(problem.B, reference_L_per_element, problem.mesh, A)

    identity_error = np.max(np.abs(E_reference.reshape(partition_count, 6, 6) - np.eye(6)))
    translation_error = np.max(np.abs(P_reference - dense_P0)) / np.max(np.abs(P_reference))

    partition_field = np.random.default_rng(0).standard_normal((partition_count, 6))
    dense_induced_strain = (dense_P0 @ partition_field.reshape(-1)).reshape(partition_count, 6)
    fft_induced_strain = get_reference_induced_strain(problem.P0_transformed, partition_field)
    convolution_error = (np.max(np.abs(fft_induced_strain - dense_induced_strain))
                         / np.max(np.abs(dense_induced_strain)))

    return [("reference E blocks equal the identity", identity_error, verification_tolerance),
            ("dense P0 equals the translated kernel", translation_error, verification_tolerance),
            ("FFT convolution equals the dense product", convolution_error, verification_tolerance)]

def get_jacobian_checks(partition_materials, P0_transformed, reference_compliance):
    # Central-difference check of the LS-20 Jacobian-vector product against the residual it differentiates. This is
    # the only test of the plastic branch of H_mu, so it has to be taken at a partly yielded state, and the active
    # yield set must not move across the perturbation or the one-sided branch derivative is not the true one.
    no_plastic_history = PlasticState(np.zeros((partition_count, 6)), np.zeros(partition_count))

    def get_yielding(trial_strain):
        plastic_state, _ = get_plastic_eigenstrain(trial_strain, partition_materials, no_plastic_history)
        return plastic_state.accumulated_plastic_strain > 0

    # Probe at the smallest of these load fractions that actually yields something. A fixed fraction is fragile:
    # at 0.4 of the default path nothing has yielded yet, which silently reduces this to an elastic-only test.
    load_fraction = next((fraction for fraction in (1.0, 2.0, 4.0, 8.0)
                          if get_yielding(np.tile(fraction * max_macro_strain, (partition_count, 1))).any()), 1.0)
    macro_strain = load_fraction * max_macro_strain
    b = np.tile(macro_strain, (partition_count, 1))
    strain = b.copy()

    def get_residual(trial_strain):
        _, _, residual = get_reference_state(trial_strain, b, P0_transformed, reference_compliance,
                                             partition_materials, no_plastic_history)
        return residual

    direction = np.random.default_rng(1).standard_normal((partition_count, 6))
    direction /= np.linalg.norm(direction)
    step = 1e-6 * np.linalg.norm(strain)

    jacobian_blocks = get_reference_jacobian_blocks(strain, partition_materials, no_plastic_history,
                                                    reference_compliance)
    analytic_product = direction - get_reference_induced_strain(
        P0_transformed, np.einsum('pij,pj->pi', jacobian_blocks, direction))
    finite_difference = (get_residual(strain + step * direction)
                         - get_residual(strain - step * direction)) / (2 * step)

    yielding = get_yielding(strain)
    active_set_moved = not (np.array_equal(get_yielding(strain - step * direction), yielding)
                            and np.array_equal(get_yielding(strain + step * direction), yielding))
    print(f"  Jacobian check state: load fraction {load_fraction:g}, {int(yielding.sum())} of {partition_count} "
          f"partitions yielding")
    return [("LS-20 Jacobian matches a central difference",
             np.max(np.abs(analytic_product - finite_difference)) / np.max(np.abs(analytic_product)),
             jacobian_check_tolerance),
            ("Jacobian check reaches the plastic branch", 0.0 if yielding.any() else 1.0, 0.5),
            ("Jacobian check active yield set is fixed", 1.0 if active_set_moved else 0.0, 0.5)]

def get_model_elastic_stiffness(dense_P0, L_per_partition, reference_compliance):
    # Solves the purely elastic reduced model [I + P0 (C0^-1 L - I)] e = 1 x I_6 directly, so the measured model
    # error carries no iterative solver tolerance, then homogenizes exactly as get_homogenized_L does.
    stiffness_ratio = scipy.linalg.block_diag(*(reference_compliance @ L_per_partition))
    identity = np.eye(6 * partition_count)
    partition_strain = np.linalg.solve(identity + dense_P0 @ (stiffness_ratio - identity),
                                       np.tile(np.eye(6), (partition_count, 1)))
    return np.mean(L_per_partition @ partition_strain.reshape(partition_count, 6, 6), axis=0)

def print_elastic_model_error(dense_P0, problem):
    exact_stiffness = get_homogenized_L(problem.E, problem.partition_materials.L)
    model_stiffness = get_model_elastic_stiffness(dense_P0, problem.partition_materials.L,
                                                  problem.reference_compliance)
    stiffness_error = np.max(np.abs(model_stiffness - exact_stiffness)) / np.max(np.abs(exact_stiffness))
    print("  elastic model error (constant-polarization approximation, no solver tolerance involved):")
    print(f"    homogenized stiffness    : {stiffness_error:.3e} relative, worst component")
    print(f"    C_1111 exact vs model MPa: {exact_stiffness[0, 0] / 1e6:.3f} vs {model_stiffness[0, 0] / 1e6:.3f}")

def run_verification(problem):
    print("verification:")
    checks = get_reference_kernel_checks(problem.P0_transformed, problem.reference_L)
    checks += get_jacobian_checks(problem.partition_materials, problem.P0_transformed, problem.reference_compliance)
    dense_P0 = None
    if partition_count <= verification_max_partitions:
        dense_P0 = get_dense_P0(get_P0_offset_blocks_from_transformed(problem.P0_transformed))
        checks += get_dense_reference_checks(problem, dense_P0)
    else:
        print(f"  dense checks skipped: {partition_count} partitions exceeds verification_max_partitions "
              f"({verification_max_partitions}). Lower partition_number_per_side to run them.")

    for description, error, tolerance in checks:
        print(f"  [{'PASS' if error <= tolerance else 'FAIL'}] {description:<43}: {error:.2e} "
              f"(tolerance {tolerance:.0e})")
    failed_checks = [description for description, error, tolerance in checks if not error <= tolerance]
    if failed_checks:
        raise RuntimeError(f"verification failed: {'; '.join(failed_checks)}.")

    if dense_P0 is not None:
        print_elastic_model_error(dense_P0, problem)

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

def get_total_time_per_group(solver_results, step_count):
    # Totals rather than per-iteration figures, because an iteration is not a comparable unit across the solvers.
    # All operator applications are shown as one segment, wherever they ran: for Newton most of them happen inside
    # the Krylov solve, which would otherwise hide them. The other groups are shown net of the operator time nested
    # inside them.
    solve_time = get_fastest_repeat_per_step(solver_results, "solve_time_per_step")[:step_count].sum()
    time_per_group = get_fastest_repeat_per_step(solver_results, "group_time_per_step")[:step_count].sum(axis=0)
    nested_operator_time_per_group = get_fastest_repeat_per_step(
        solver_results, "nested_operator_time_per_group_per_step")[:step_count].sum(axis=0)
    material_update, induced_strain, sensitivity, reference_inverse, correction_solve = (
        time_per_group - nested_operator_time_per_group)
    operator_time = induced_strain + nested_operator_time_per_group.sum()
    segments = [material_update, operator_time, sensitivity, reference_inverse, correction_solve]
    return np.append(segments, solve_time - sum(segments))

def get_iteration_count(solver_results, step_count):
    return get_fastest_repeat_per_step(solver_results, "iterations_per_step")[:step_count].sum()

def get_operator_application_count(solver_results, step_count):
    return get_fastest_repeat_per_step(solver_results, "operator_applications_per_step")[:step_count].sum()

def get_timer_overhead_fraction(solver_results, timer_overhead_per_call, step_count):
    solve_time = get_fastest_repeat_per_step(solver_results, "solve_time_per_step")[:step_count].sum()
    timed_call_count = get_fastest_repeat_per_step(solver_results, "group_calls_per_step")[:step_count].sum()
    return timer_overhead_per_call * timed_call_count / solve_time

## ------- Problem Setup ------- ##

class Problem(NamedTuple):
    partition_material_ids: np.ndarray
    L_matrix: np.ndarray
    L_inclusion: np.ndarray
    partition_materials: PartitionMaterials
    mesh: Mesh
    B: np.ndarray
    L_per_element: np.ndarray
    E: np.ndarray
    P: np.ndarray
    P0_transformed: np.ndarray
    reference_L: np.ndarray
    reference_compliance: np.ndarray

def get_problem():
    # Everything main and the experiment scripts need, built from the inputs, with the offline operators taken
    # from the cache when it matches.
    partition_material_ids = get_partition_material_ids()
    L_matrix = get_L(elastic_modulus_matrix, poisson_ratio_matrix)
    L_inclusion = get_L(elastic_modulus_inclusion, poisson_ratio_inclusion)
    if matched_stiffness_control:
        L_inclusion = L_matrix
        print("matched-stiffness control: the inclusion elastic stiffness is overridden to the matrix value, which "
              "makes the actual E/P and reference models the same equation. Yield properties still differ.")
    partition_materials = get_partition_materials(L_matrix, L_inclusion, partition_material_ids)
    mesh = get_periodic_mesh(partition_material_ids)
    B = get_B()
    L_per_element = get_value_per_material(L_matrix, L_inclusion, mesh.element_material_ids)
    E, P, P0_transformed = get_offline_operators(mesh, B, L_per_element, L_matrix, L_inclusion, partition_materials.L)
    reference_L = get_homogenized_L(E, partition_materials.L)
    return Problem(partition_material_ids, L_matrix, L_inclusion, partition_materials, mesh, B, L_per_element, E, P,
                   P0_transformed, reference_L, np.linalg.inv(reference_L))

## ------- Main ------- ##

def print_reference_solver_settings(problem, reference_relaxation):
    if reference_solver_method == "newton":
        print(f"reference solver: newton, GMRES rtol {reference_krylov_tolerance:.0e}, "
              f"restart {reference_krylov_restart}")
    else:
        stability_limit = get_reference_stability_limit(problem.partition_materials, problem.reference_L)
        print(f"reference solver: fixed_point, relaxation {reference_relaxation:.4g}, stability limit "
              f"{stability_limit:.4g}")

def get_model_difference_per_step(results, actual_solver_name, step_count):
    # The reference model's macroscopic stress against the actual E/P model's, per step, relative to the largest
    # actual-model stress. This is modelling error, not solver error.
    actual_stress = results[actual_solver_name][-1].macroscopic_stress[:step_count]
    reference_stress = results[reference_solver_name][-1].macroscopic_stress[:step_count]
    stress_scale = max(np.abs(actual_stress).max(), np.finfo(float).tiny)
    return np.abs(reference_stress - actual_stress).max(axis=1) / stress_scale

def print_comparison(results, comparable_step_count, actual_solver_name):
    print(f"relaxation_factor = {relaxation_factor}, reference method = {reference_solver_method}")
    compared_steps = (f"the {comparable_step_count} of {strain_increment_count} steps every solver completed"
                      if comparable_step_count < strain_increment_count else f"all {strain_increment_count} steps")
    print(f"solve times are the minimum per load step over {timing_repeat_count} interleaved runs of each solver, "
          f"summed over {compared_steps}")
    for solver_name, solver_results in results.items():
        print(get_solver_label(solver_name, solver_results, comparable_step_count))

    model_difference = get_model_difference_per_step(results, actual_solver_name, comparable_step_count).max()
    print(f"model discrepancy, different models ({actual_solver_name} actual E/P vs reference): "
          f"{model_difference:.2e} relative. This is a modelling difference, not a solver error.")

    timer_overhead_per_call = get_online_timer_overhead_per_call()
    overhead_labels = []
    for solver_name, solver_results in results.items():
        overhead_fraction = get_timer_overhead_fraction(solver_results, timer_overhead_per_call, comparable_step_count)
        overhead_labels.append(f"{100 * overhead_fraction:.3f} % ({solver_name})")
    print(f"online timer overhead: {', '.join(overhead_labels)} of online time")

def get_most_complete_solver_name(results):
    # The solver that got furthest along the load path, whose stresses are plotted. The first listed wins a tie.
    return max(results, key=lambda solver_name: get_completed_step_count(results[solver_name]))

def get_elastic_steps(result):
    # Steps where no partition yielded. Steps past a stopped solver's last are not elastic, only unsolved.
    elastic_steps = ~result.yielding_per_step
    elastic_steps[result.completed_step_count:] = False
    return elastic_steps

def save_load_path_plots(results, comparable_step_count, actual_solver_name):
    actual_result = results[actual_solver_name][-1]
    solver_series = [plots.SolverSeries(solver_name, color, solver_results[-1].iterations_per_step,
                                        get_fastest_repeat_per_step(solver_results, "solve_time_per_step"),
                                        get_completed_step_count(solver_results))
                     for (solver_name, solver_results), color in zip(results.items(), plots.solver_colors)]
    plots.plot_load_path_summary(
        get_deviatoric_stress(actual_result.macroscopic_stress) / 1e6, actual_result.completed_step_count,
        solver_series, get_elastic_steps(actual_result),
        100 * get_model_difference_per_step(results, actual_solver_name, comparable_step_count),
        f"reference LS model vs actual E/P ({actual_solver_name})",
        plots.get_applied_strain_description(max_macro_strain, strain_increment_count),
        output_folder / "load_path_summary.png")
    plots.plot_cost_breakdown(
        [plots.SolverCost(solver_name, get_total_time_per_group(solver_results, comparable_step_count),
                          get_iteration_count(solver_results, comparable_step_count),
                          get_operator_application_count(solver_results, comparable_step_count))
         for solver_name, solver_results in results.items()],
        output_folder / "cost_breakdown.png")

def save_von_mises_plots(results, actual_solver_name, problem):
    actual_result = results[actual_solver_name][-1]
    partition_von_mises_stress_MPa = get_von_mises_stress(actual_result.final_stress) / 1e6
    plots.plot_von_mises_cross_section(
        partition_von_mises_stress_MPa.reshape(partition_number_per_side, partition_number_per_side),
        element_number_per_side, partition_number_per_side, von_mises_plot_min_stress / 1e6,
        von_mises_plot_max_stress / 1e6, output_folder / "von_mises_cross_section.png")

    # Final stresses are only comparable when both models reached the same final step.
    reference_result = results[reference_solver_name][-1]
    if actual_result.completed_step_count == reference_result.completed_step_count == strain_increment_count:
        reference_von_mises_stress_MPa = get_von_mises_stress(reference_result.final_stress) / 1e6
        plots.plot_von_mises_difference_cross_section(
            (reference_von_mises_stress_MPa - partition_von_mises_stress_MPa).reshape(partition_number_per_side,
                                                                                     partition_number_per_side),
            element_number_per_side, partition_number_per_side, output_folder / "von_mises_model_difference.png")
    else:
        print("von Mises model difference plot skipped: a solver stopped before the last load step.")

    if post_process_element_stress:
        # The last step the solver actually solved, which is not the last requested one if it stopped early.
        final_macro_strain = actual_result.applied_macro_strain[actual_result.completed_step_count - 1]
        element_stress = get_element_stress(problem.mesh, problem.B, problem.L_per_element, final_macro_strain,
                                            actual_result.final_plastic_strain)
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

def check_matched_stiffness(results, comparable_step_count):
    # With matched phase stiffness the reference and actual E/P models are the same equation, so here, and only
    # here, the reference solver must reproduce actual E/P to solver tolerance.
    model_difference = get_model_difference_per_step(results, "Standard", comparable_step_count).max()
    passed = model_difference <= matched_stiffness_tolerance
    print(f"[{'PASS' if passed else 'FAIL'}] matched-stiffness control: reference vs actual E/P "
          f"{model_difference:.2e} relative (tolerance {matched_stiffness_tolerance:.0e}).")
    if not passed:
        raise RuntimeError("matched-stiffness control failed: the reference model must reproduce actual E/P to "
                           "solver tolerance.")

def main():
    output_folder.mkdir(exist_ok=True)
    problem = get_problem()
    print(f"inclusion volume fraction on the partition grid: {problem.partition_material_ids.mean():.4f}")
    plots.plot_cross_section(
        problem.partition_material_ids.reshape(partition_number_per_side, partition_number_per_side),
        element_number_per_side, output_folder / "cross_section.png")
    if verification_enabled:
        run_verification(problem)

    reference_relaxation = get_reference_relaxation_factor(problem.partition_materials, problem.reference_L)
    print_reference_solver_settings(problem, reference_relaxation)
    get_solvers_for_one_path = functools.partial(get_solvers, problem.P0_transformed, problem.reference_compliance,
                                                 reference_relaxation)
    results = run_interleaved_repeats(get_solvers_for_one_path, problem.E, problem.P, problem.partition_materials)
    # Timings and stresses are compared only over the steps every solver completed. Stresses are plotted from the
    # actual-model solver that got furthest, since the reference solver solves a different model.
    comparable_step_count = min(get_completed_step_count(solver_results) for solver_results in results.values())
    actual_solver_name = get_most_complete_solver_name({solver_name: results[solver_name]
                                                        for solver_name in actual_model_solver_names})
    print_comparison(results, comparable_step_count, actual_solver_name)
    save_load_path_plots(results, comparable_step_count, actual_solver_name)
    save_von_mises_plots(results, actual_solver_name, problem)
    check_solver_agreement(results, comparable_step_count)
    if matched_stiffness_control:
        check_matched_stiffness(results, comparable_step_count)

if __name__ == "__main__":
    main()
