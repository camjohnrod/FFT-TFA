import pathlib
import sys
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.colors
import matplotlib.legend
import matplotlib.lines
import time
import tqdm

## ------- Inputs ------- ##

domain_side_length          = 1e-3
inclusion_side_length       = (5/9) * 1e-3
element_number_per_side     = 27
partition_number_per_side   = 9

elastic_modulus_inclusion   = 10e9
elastic_modulus_matrix      = 100e6
poisson_ratio_inclusion     = 0.3
poisson_ratio_matrix        = 0.3
inclusion_yield_stress      = np.inf
matrix_yield_stress         = 1.0e6
inclusion_hardening_modulus = 0.0
matrix_hardening_modulus    = 10e6

max_macro_strain            = np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0])
# max_macro_strain            = np.array([0.03, 0.018, 0.001, 0.0, 0.0, 0.0])
# max_macro_strain            = np.array([0.015, 0.020, 0.0, 0.03, 0.0, 0.0])

strain_increment_count      = 60

timing_repeat_count         = 1

fixed_point_tolerance       = 1e-6
fixed_point_max_iterations  = 1000

relaxation_factor           = 1.0

von_mises_plot_min_stress   = 1.0e6
von_mises_plot_max_stress   = 3.0e6

post_process_element_stress = True

## ------- Calculated Values and Checks ------- ##

elements_per_partition_side = element_number_per_side // partition_number_per_side
if element_number_per_side % partition_number_per_side != 0:
    raise ValueError("The number of elements per side must be divisible by the number of partitions per side.")

element_count = element_number_per_side**3
partition_count = partition_number_per_side**3
dof_count = 3 * element_count

element_side_length = domain_side_length / element_number_per_side
gauss_point_volume_weight = element_side_length ** 3 / 8

matrix_partitions_beside_inclusion = partition_number_per_side * (domain_side_length - inclusion_side_length) / (2 * domain_side_length)
whole_matrix_partitions_beside_inclusion = round(matrix_partitions_beside_inclusion)
if not np.isclose(matrix_partitions_beside_inclusion, whole_matrix_partitions_beside_inclusion) or not 1 <= whole_matrix_partitions_beside_inclusion < partition_number_per_side / 2:
    raise ValueError(f"The centred inclusion must have a whole number of matrix partitions (at least 1) on each side in x and y, so its edges fall on partition edges. Got {matrix_partitions_beside_inclusion}.")

## ------- Mesh and Geometry ------- ##

def get_grid_id(i, j, k, count_per_side):
    return i + count_per_side * j + count_per_side**2 * k

def is_inside_inclusion(centre):
    distance_from_inclusion_axis = np.abs(centre[:2] - domain_side_length / 2)
    return np.all(distance_from_inclusion_axis < inclusion_side_length / 2)

def get_periodic_mesh():
    element_nodes = np.zeros((element_count, 8), dtype=int)
    element_partition_ids = np.zeros(element_count, dtype=int)
    element_material_ids = np.zeros(element_count, dtype=int)

    for k in range(element_number_per_side):
        for j in range(element_number_per_side):
            for i in range(element_number_per_side):
                element_id = get_grid_id(i, j, k, element_number_per_side)

                for offset_k in range(2):
                    for offset_j in range(2):
                        for offset_i in range(2):
                            corner = get_grid_id(offset_i, offset_j, offset_k, 2)
                            element_nodes[element_id, corner] = get_grid_id((i + offset_i) % element_number_per_side,
                                                                            (j + offset_j) % element_number_per_side,
                                                                            (k + offset_k) % element_number_per_side,
                                                                            element_number_per_side)

                element_partition_ids[element_id] = get_grid_id(i // elements_per_partition_side,
                                                                j // elements_per_partition_side,
                                                                k // elements_per_partition_side,
                                                                partition_number_per_side)

                element_centre = (np.array([i, j, k]) + 0.5) * element_side_length
                element_material_ids[element_id] = int(is_inside_inclusion(element_centre))

    return element_nodes, element_partition_ids, element_material_ids

def get_partition_material_ids():
    partition_side_length = domain_side_length / partition_number_per_side
    partition_material_ids = np.zeros(partition_count, dtype=int)

    for k in range(partition_number_per_side):
        for j in range(partition_number_per_side):
            for i in range(partition_number_per_side):
                partition_id = get_grid_id(i, j, k, partition_number_per_side)
                partition_centre = (np.array([i, j, k]) + 0.5) * partition_side_length
                partition_material_ids[partition_id] = int(is_inside_inclusion(partition_centre))

    return partition_material_ids

def get_value_per_material(matrix_value, inclusion_value, material_ids):
    value_by_material_id = np.array([matrix_value, inclusion_value])
    return value_by_material_id[material_ids]

## ------- Element Matrices ------- ##

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
        derivatives_wrt_z = derivatives_wrt_zeta * 2 / element_side_length

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

def get_L(elastic_modulus, poisson_ratio):
    lame_lambda   = elastic_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    shear_modulus = elastic_modulus / (2 * (1 + poisson_ratio))
    normal_block = lame_lambda * np.ones((3, 3)) + 2 * shear_modulus * np.eye(3)
    shear_block  = shear_modulus * np.eye(3)
    L = np.zeros((6, 6))
    L[:3, :3] = normal_block
    L[3:, 3:] = shear_block
    return L

def get_element_dofs(element_nodes):
    node_dofs = 3 * element_nodes[:, :, None] + np.arange(3)
    return node_dofs.reshape(element_count, 24)

def get_K_element(B, L):
    K_element = np.zeros((24, 24))
    for gauss_index in range(8):
        K_element += gauss_point_volume_weight * B[gauss_index].T @ L @ B[gauss_index]
    return K_element

def get_K(B, L_per_element, element_nodes):
    element_dofs = get_element_dofs(element_nodes)
    K_elements = np.zeros((element_count, 24, 24))
    for element_id in range(element_count):
        K_elements[element_id] = get_K_element(B, L_per_element[element_id])
    rows = np.repeat(element_dofs, 24, axis=1)
    columns = np.tile(element_dofs, (1, 24))
    K = scipy.sparse.coo_matrix((K_elements.ravel(), (rows.ravel(), columns.ravel())), shape=(dof_count, dof_count))
    return K.tocsr()

def get_integrated_B(B):
    return gauss_point_volume_weight * np.sum(B, axis=0)

def get_F_element(B, L):
    return get_integrated_B(B).T @ L

## ------- Influence Functions (Offline) ------- ##

def get_F_macrostrain(B, L_per_element, element_nodes):
    element_dofs = get_element_dofs(element_nodes)
    F_macrostrain = np.zeros((dof_count, 6))
    for element_id in range(element_count):
        F_element = get_F_element(B, L_per_element[element_id])
        F_macrostrain[element_dofs[element_id]] -= F_element
    return F_macrostrain

def get_F_eigenstrain(B, L_per_element, element_nodes, element_partition_ids):
    element_dofs = get_element_dofs(element_nodes)
    F_eigenstrain = np.zeros((dof_count, 6 * partition_count))
    for element_id in range(element_count):
        F_element = get_F_element(B, L_per_element[element_id])
        first_column = 6 * element_partition_ids[element_id]
        F_eigenstrain[np.ix_(element_dofs[element_id], np.arange(first_column, first_column + 6))] += F_element
    return F_eigenstrain

def get_F_eigenstrain_in_first_partition(B, L_per_element, element_nodes, element_partition_ids):
    element_dofs = get_element_dofs(element_nodes)
    F_eigenstrain = np.zeros((dof_count, 6))
    for element_id in np.flatnonzero(element_partition_ids == 0):
        F_eigenstrain[element_dofs[element_id]] += get_F_element(B, L_per_element[element_id])
    return F_eigenstrain

def get_partition_average_strain(displacements, B, element_nodes, element_partition_ids):
    element_dofs = get_element_dofs(element_nodes)
    partition_volume = elements_per_partition_side**3 * element_side_length**3
    integrated_B = get_integrated_B(B)
    average_strain = np.zeros((6 * partition_count, displacements.shape[1]))
    for element_id in range(element_count):
        element_displacements = displacements[element_dofs[element_id]]
        first_row = 6 * element_partition_ids[element_id]
        average_strain[first_row:first_row + 6] += integrated_B @ element_displacements
    return average_strain / partition_volume

def solve_influence_function(K, F, B, element_nodes, element_partition_ids):
    pinned_dofs = np.arange(3)
    free_dofs = np.setdiff1d(np.arange(K.shape[0]), pinned_dofs)
    K_free = K[free_dofs][:, free_dofs].tocsc()
    print(f"  {f'factor K ({K_free.shape[0]} DOFs)':<26}: ", end="", flush=True)
    start_time = time.perf_counter()
    K_free_factorization = scipy.sparse.linalg.splu(K_free, permc_spec='MMD_AT_PLUS_A')
    print(f"{time.perf_counter() - start_time:.1f} s")

    columns_per_solve = 100
    average_strain = np.zeros((6 * partition_count, F.shape[1]))
    column_batches = tqdm.tqdm(range(0, F.shape[1], columns_per_solve),
                               desc=f"  {f'solve {F.shape[1]} load columns':<26}",
                               bar_format="{desc}: {percentage:3.0f}% |{bar:25}| {n_fmt}/{total_fmt} batches "
                                          "[{elapsed} elapsed, {remaining} left]", file=sys.stdout)
    for first_column in column_batches:
        columns = slice(first_column, first_column + columns_per_solve)
        batch_displacements = np.zeros((dof_count, F[:, columns].shape[1]))
        batch_displacements[free_dofs] = K_free_factorization.solve(F[free_dofs, columns])
        average_strain[:, columns] = get_partition_average_strain(batch_displacements, B, element_nodes,
                                                                  element_partition_ids)
    return average_strain

def get_influence_functions(B, L_per_element, element_nodes, element_partition_ids):
    K = get_K(B, L_per_element, element_nodes)
    F_macrostrain_and_eigenstrain = np.hstack([get_F_macrostrain(B, L_per_element, element_nodes),
                                               get_F_eigenstrain(B, L_per_element, element_nodes, element_partition_ids)])
    average_strain = solve_influence_function(K, F_macrostrain_and_eigenstrain, B, element_nodes, element_partition_ids)
    E = np.tile(np.eye(6), (partition_count, 1)) + average_strain[:, :6]
    P = average_strain[:, 6:]
    return E, P

def get_homogenized_L(E, L_per_partition):
    E_blocks = E.reshape(partition_count, 6, 6)
    return np.mean(L_per_partition @ E_blocks, axis=0)

def get_P0_offset_blocks(B, reference_L_per_element, element_nodes, element_partition_ids):
    K = get_K(B, reference_L_per_element, element_nodes)
    F_eigenstrain = get_F_eigenstrain_in_first_partition(B, reference_L_per_element, element_nodes, element_partition_ids)
    return solve_influence_function(K, F_eigenstrain, B, element_nodes, element_partition_ids)

def get_P0_transformed(P0_offset_blocks):
    P0_offset_lattice = P0_offset_blocks.reshape(partition_number_per_side, partition_number_per_side,
                                                 partition_number_per_side, 6, 6)
    return np.fft.fftn(P0_offset_lattice, axes=(0, 1, 2))

## ------- Cache ------- ##

cache_folder = pathlib.Path(__file__).parent / "cache"

def load_cache(cache_path, parameters):
    if not cache_path.exists():
        print(f"{cache_path.name}: no cache found, computing.")
        return None
    cached = np.load(cache_path)
    for name, value in parameters.items():
        if name not in cached.files or not np.array_equal(cached[name], value):
            print(f"{cache_path.name}: {name} missing or changed, recomputing.")
            return None
    print(f"{cache_path.name}: loaded from cache.")
    return cached

def save_cache(cache_path, parameters, **arrays):
    cache_folder.mkdir(exist_ok=True)
    np.savez(cache_path, **arrays, **parameters)

## ------- J2 Material ------- ##

volumetric_direction = np.array([1, 1, 1, 0, 0, 0])

deviatoric_projector = np.zeros((6, 6))
deviatoric_projector[:3, :3] = np.eye(3) - np.ones((3, 3)) / 3
deviatoric_projector[3:, 3:] = np.eye(3)

def get_equivalent_stress(deviatoric_stress):
    return np.sqrt(1.5 * np.sum(deviatoric_stress[:, :3]**2, axis=1) + 3 * np.sum(deviatoric_stress[:, 3:]**2, axis=1))

def get_trial_state(strain, L_per_partition, plastic_strain_history, accumulated_plastic_strain_history,
                    yield_stress_per_partition, hardening_modulus_per_partition):
    elastic_trial_strain = strain - plastic_strain_history
    trial_stress = np.einsum('pij,pj->pi', L_per_partition, elastic_trial_strain)
    trial_mean_stress = np.sum(trial_stress[:, :3], axis=1) / 3
    trial_deviatoric_stress = trial_stress - np.outer(trial_mean_stress, volumetric_direction)
    trial_equivalent_stress = get_equivalent_stress(trial_deviatoric_stress)
    trial_yield_function = trial_equivalent_stress - (yield_stress_per_partition
                                                      + hardening_modulus_per_partition * accumulated_plastic_strain_history)
    shear_modulus = L_per_partition[:, 3, 3]
    trial_plastic_multiplier = np.maximum(trial_yield_function, 0) / (3 * shear_modulus + hardening_modulus_per_partition)
    return (trial_stress, trial_mean_stress, trial_deviatoric_stress, trial_equivalent_stress, trial_yield_function,
            trial_plastic_multiplier)

def get_flow_direction(deviatoric_stress, equivalent_stress):
    flow_direction = np.zeros_like(deviatoric_stress)
    flow_direction[:, :3] = 1.5 * deviatoric_stress[:, :3] / equivalent_stress[:, None]
    flow_direction[:, 3:] = 3 * deviatoric_stress[:, 3:] / equivalent_stress[:, None]
    return flow_direction

def get_plastic_eigenstrain(strain, L_per_partition, plastic_strain_history, accumulated_plastic_strain_history,
                            yield_stress_per_partition, hardening_modulus_per_partition):
    (trial_stress, trial_mean_stress, trial_deviatoric_stress, trial_equivalent_stress, trial_yield_function,
     trial_plastic_multiplier) = get_trial_state(strain, L_per_partition, plastic_strain_history,
                                                 accumulated_plastic_strain_history, yield_stress_per_partition,
                                                 hardening_modulus_per_partition)

    plastic_strain_new = plastic_strain_history.copy()
    accumulated_plastic_strain_new = accumulated_plastic_strain_history.copy()
    stress = trial_stress.copy()

    yielding = trial_yield_function > 0
    yielding_trial_deviatoric_stress = trial_deviatoric_stress[yielding]
    yielding_trial_equivalent_stress = trial_equivalent_stress[yielding]
    shear_modulus = L_per_partition[yielding, 3, 3]
    plastic_multiplier = trial_plastic_multiplier[yielding]

    flow_direction = get_flow_direction(yielding_trial_deviatoric_stress, yielding_trial_equivalent_stress)
    plastic_strain_new[yielding] += plastic_multiplier[:, None] * flow_direction
    accumulated_plastic_strain_new[yielding] += plastic_multiplier

    deviatoric_scaling = 1 - 3 * shear_modulus * plastic_multiplier / yielding_trial_equivalent_stress
    corrected_deviatoric_stress = deviatoric_scaling[:, None] * yielding_trial_deviatoric_stress
    stress[yielding] = np.outer(trial_mean_stress[yielding], volumetric_direction) + corrected_deviatoric_stress

    return plastic_strain_new, accumulated_plastic_strain_new, stress

def get_eigenstrain_sensitivity(strain, L_per_partition, plastic_strain_history, accumulated_plastic_strain_history,
                                yield_stress_per_partition, hardening_modulus_per_partition):
    _, _, trial_deviatoric_stress, trial_equivalent_stress, trial_yield_function, trial_plastic_multiplier = get_trial_state(
        strain, L_per_partition, plastic_strain_history, accumulated_plastic_strain_history,
        yield_stress_per_partition, hardening_modulus_per_partition)

    sensitivity = np.zeros((partition_count, 6, 6))

    yielding = trial_yield_function > 0
    yielding_trial_deviatoric_stress = trial_deviatoric_stress[yielding]
    yielding_trial_equivalent_stress = trial_equivalent_stress[yielding]
    shear_modulus = L_per_partition[yielding, 3, 3]
    hardening_modulus = hardening_modulus_per_partition[yielding]
    plastic_multiplier = trial_plastic_multiplier[yielding]

    flow_direction = get_flow_direction(yielding_trial_deviatoric_stress, yielding_trial_equivalent_stress)
    normalized_deviatoric_stress = yielding_trial_deviatoric_stress / yielding_trial_equivalent_stress[:, None]
    flow_projector = np.einsum('pi,pj->pij', flow_direction, normalized_deviatoric_stress)

    radial_sensitivity = 3 * shear_modulus / (3 * shear_modulus + hardening_modulus)
    rotational_sensitivity = 3 * shear_modulus * plastic_multiplier / yielding_trial_equivalent_stress

    sensitivity[yielding] = (radial_sensitivity[:, None, None] * flow_projector
                             + rotational_sensitivity[:, None, None] * (deviatoric_projector - flow_projector))
    return sensitivity

## ------- Solvers ------- ##

def get_induced_strain(P, eigenstrain):
    return (P @ eigenstrain.reshape(-1)).reshape(partition_count, 6)

def get_induced_strain_reusing_history(P, plastic_strain, plastic_strain_history, induced_strain_history):
    if np.array_equal(plastic_strain, plastic_strain_history):
        return induced_strain_history
    return get_induced_strain(P, plastic_strain)

def get_relative_residual(residual, strain):
    strain_norm_floor = 1e-30
    return np.linalg.norm(residual) / max(np.linalg.norm(strain), strain_norm_floor)

def has_converged(relative_residual, iteration_count, solver_name):
    if not np.isfinite(relative_residual) or relative_residual > 1:
        raise RuntimeError(f"{solver_name} diverged after {iteration_count} iterations "
                            f"(relative residual {relative_residual}).")
    if relative_residual < fixed_point_tolerance:
        return True
    if iteration_count >= fixed_point_max_iterations:
        raise RuntimeError(f"{solver_name} did not converge within {fixed_point_max_iterations} iterations "
                            f"(relative residual {relative_residual}).")
    return False

def get_reset_state(strain, b, P, induced_strain_history, L_per_partition, plastic_strain_history,
                    accumulated_plastic_strain_history, yield_stress_per_partition, hardening_modulus_per_partition):
    plastic_strain, accumulated_plastic_strain, stress = get_plastic_eigenstrain(
        strain, L_per_partition, plastic_strain_history, accumulated_plastic_strain_history,
        yield_stress_per_partition, hardening_modulus_per_partition)
    induced_strain = get_induced_strain_reusing_history(P, plastic_strain, plastic_strain_history,
                                                        induced_strain_history)
    yielding_partitions = accumulated_plastic_strain > accumulated_plastic_strain_history
    non_yielding_partitions = ~yielding_partitions

    strain = strain.copy()
    strain[non_yielding_partitions] = b[non_yielding_partitions] + induced_strain[non_yielding_partitions]

    plastic_strain, accumulated_plastic_strain, stress = get_plastic_eigenstrain(
        strain, L_per_partition, plastic_strain_history, accumulated_plastic_strain_history,
        yield_stress_per_partition, hardening_modulus_per_partition)
    if not np.array_equal(accumulated_plastic_strain > accumulated_plastic_strain_history, yielding_partitions):
        induced_strain = get_induced_strain_reusing_history(P, plastic_strain, plastic_strain_history,
                                                            induced_strain_history)

    residual = strain - b - induced_strain
    return strain, plastic_strain, accumulated_plastic_strain, stress, residual

def standard_richardson_iteration(E, P, macro_strain, L_per_partition,
                                   yield_stress_per_partition, hardening_modulus_per_partition,
                                   plastic_strain_history, accumulated_plastic_strain_history):
    b = (E @ macro_strain).reshape(partition_count, 6)
    induced_strain_history = get_induced_strain(P, plastic_strain_history)
    strain = b + induced_strain_history

    residual_history = []
    while True:
        strain, plastic_strain, accumulated_plastic_strain, stress, residual = get_reset_state(
            strain, b, P, induced_strain_history, L_per_partition, plastic_strain_history,
            accumulated_plastic_strain_history, yield_stress_per_partition, hardening_modulus_per_partition)
        residual_history.append(get_relative_residual(residual, strain))

        if has_converged(residual_history[-1], len(residual_history), "standard_richardson_iteration"):
            break

        strain = strain - relaxation_factor * residual

    return stress, plastic_strain, accumulated_plastic_strain, np.array(residual_history)

def get_reference_fourier_inverse(P0_transformed, reference_sensitivity):
    M0_transformed = np.eye(6) - P0_transformed @ reference_sensitivity
    return np.linalg.inv(M0_transformed)

def get_fft_correction(reference_fourier_inverse, residual):
    lattice_shape = (partition_number_per_side, partition_number_per_side, partition_number_per_side, 6)
    residual_transformed = np.fft.fftn(residual.reshape(lattice_shape), axes=(0, 1, 2))
    correction_transformed = np.einsum('...ij,...j->...i', reference_fourier_inverse, -residual_transformed)
    correction = np.fft.ifftn(correction_transformed, axes=(0, 1, 2)).real
    return correction.reshape(partition_count, 6)

def fft_preconditioned_richardson_iteration(E, P, macro_strain, L_per_partition,
                                             yield_stress_per_partition, hardening_modulus_per_partition,
                                             plastic_strain_history, accumulated_plastic_strain_history,
                                             P0_transformed):
    b = (E @ macro_strain).reshape(partition_count, 6)
    induced_strain_history = get_induced_strain(P, plastic_strain_history)
    strain = b + induced_strain_history

    yielding_partitions_at_last_rebuild = None
    residual_history = []
    while True:
        strain, plastic_strain, accumulated_plastic_strain, stress, residual = get_reset_state(
            strain, b, P, induced_strain_history, L_per_partition, plastic_strain_history,
            accumulated_plastic_strain_history, yield_stress_per_partition, hardening_modulus_per_partition)
        residual_history.append(get_relative_residual(residual, strain))

        if has_converged(residual_history[-1], len(residual_history), "fft_preconditioned_richardson_iteration"):
            break

        yielding_partitions = accumulated_plastic_strain > accumulated_plastic_strain_history
        if not np.array_equal(yielding_partitions, yielding_partitions_at_last_rebuild):
            sensitivity = get_eigenstrain_sensitivity(strain, L_per_partition, plastic_strain_history,
                                                      accumulated_plastic_strain_history, yield_stress_per_partition,
                                                      hardening_modulus_per_partition)
            reference_sensitivity = np.mean(sensitivity, axis=0)
            reference_fourier_inverse = get_reference_fourier_inverse(P0_transformed, reference_sensitivity)
            yielding_partitions_at_last_rebuild = yielding_partitions

        strain = strain + relaxation_factor * get_fft_correction(reference_fourier_inverse, residual)

    return stress, plastic_strain, accumulated_plastic_strain, np.array(residual_history)

## ------- Load Path ------- ##

def get_macroscopic_stress(stress):
    return np.mean(stress, axis=0)

def get_deviatoric_stress(stress):
    return stress @ deviatoric_projector.T

def run_strain_path(E, P, L_per_partition, yield_stress_per_partition, hardening_modulus_per_partition,
                    P0_transformed=None):
    plastic_strain_history = np.zeros((partition_count, 6))
    accumulated_plastic_strain_history = np.zeros(partition_count)

    load_fractions = np.linspace(1 / strain_increment_count, 1, strain_increment_count)
    applied_macro_strain = np.outer(load_fractions, max_macro_strain)
    macroscopic_stress = np.zeros((strain_increment_count, 6))
    iterations_per_step = np.zeros(strain_increment_count, dtype=int)
    solve_time_per_step = np.zeros(strain_increment_count)

    for step, macro_strain in enumerate(applied_macro_strain):
        start_time = time.perf_counter()
        if P0_transformed is None:
            stress, plastic_strain_history, accumulated_plastic_strain_history, residual_history = standard_richardson_iteration(
                E, P, macro_strain, L_per_partition, yield_stress_per_partition, hardening_modulus_per_partition,
                plastic_strain_history, accumulated_plastic_strain_history)
        else:
            stress, plastic_strain_history, accumulated_plastic_strain_history, residual_history = fft_preconditioned_richardson_iteration(
                E, P, macro_strain, L_per_partition, yield_stress_per_partition, hardening_modulus_per_partition,
                plastic_strain_history, accumulated_plastic_strain_history, P0_transformed)
        solve_time_per_step[step] = time.perf_counter() - start_time
        iterations_per_step[step] = len(residual_history)
        macroscopic_stress[step] = get_macroscopic_stress(stress)

    return macroscopic_stress, iterations_per_step, solve_time_per_step, stress, plastic_strain_history

## ------- Post-processing ------- ##

def get_F_state(B, L_per_element, element_nodes, element_partition_ids, macro_strain, eigenstrain):
    element_dofs = get_element_dofs(element_nodes)
    F = np.zeros(dof_count)
    for element_id in range(element_count):
        F_element = get_F_element(B, L_per_element[element_id])
        F[element_dofs[element_id]] += F_element @ (eigenstrain[element_partition_ids[element_id]] - macro_strain)
    return F

def solve_state_displacements(K, F):
    pinned_dofs = np.arange(3)
    free_dofs = np.setdiff1d(np.arange(K.shape[0]), pinned_dofs)
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
    element_volume = element_side_length**3
    return displacements[element_dofs] @ get_integrated_B(B).T / element_volume

def get_element_stress(L_matrix, L_inclusion, macro_strain, eigenstrain):
    start_time = time.perf_counter()
    element_nodes, element_partition_ids, element_material_ids = get_periodic_mesh()
    B = get_B()
    L_per_element = get_value_per_material(L_matrix, L_inclusion, element_material_ids)
    K = get_K(B, L_per_element, element_nodes)
    F = get_F_state(B, L_per_element, element_nodes, element_partition_ids, macro_strain, eigenstrain)
    displacements = solve_state_displacements(K, F)

    element_strain = macro_strain + get_element_average_fluctuation_strain(displacements, B, element_nodes)
    element_eigenstrain = eigenstrain[element_partition_ids]
    element_stress = np.einsum('eij,ej->ei', L_per_element, element_strain - element_eigenstrain)
    print(f"element stress post-processing: {time.perf_counter() - start_time:.2f} seconds.")
    return element_stress

## ------- Main ------- ##

def get_partition_material_properties(L_matrix, L_inclusion):
    partition_material_ids = get_partition_material_ids()
    L_per_partition = get_value_per_material(L_matrix, L_inclusion, partition_material_ids)
    yield_stress_per_partition = get_value_per_material(matrix_yield_stress, inclusion_yield_stress,
                                                        partition_material_ids)
    hardening_modulus_per_partition = get_value_per_material(matrix_hardening_modulus, inclusion_hardening_modulus,
                                                             partition_material_ids)
    return L_per_partition, yield_stress_per_partition, hardening_modulus_per_partition

def get_offline_operators(L_matrix, L_inclusion, L_per_partition):
    element_nodes, element_partition_ids, element_material_ids = get_periodic_mesh()
    B = get_B()
    L_per_element = get_value_per_material(L_matrix, L_inclusion, element_material_ids)

    E_P_cache_path = cache_folder / "E_P.npz"
    E_P_parameters = {"domain_side_length": domain_side_length,
                      "fiber_side_length": inclusion_side_length,
                      "element_number_per_side": element_number_per_side,
                      "partition_number_per_side": partition_number_per_side,
                      "L_matrix": L_matrix,
                      "L_inclusion": L_inclusion}
    cached_E_P = load_cache(E_P_cache_path, E_P_parameters)
    if cached_E_P is None:
        E, P = get_influence_functions(B, L_per_element, element_nodes, element_partition_ids)
        save_cache(E_P_cache_path, E_P_parameters, E=E, P=P)
    else:
        E, P = cached_E_P["E"], cached_E_P["P"]

    reference_L = get_homogenized_L(E, L_per_partition)
    reference_L_per_element = np.tile(reference_L, (element_count, 1, 1))

    P0_cache_path = cache_folder / "P0_offset_blocks.npz"
    P0_parameters = {"domain_side_length": domain_side_length,
                     "element_number_per_side": element_number_per_side,
                     "partition_number_per_side": partition_number_per_side,
                     "reference_L": reference_L}
    cached_P0 = load_cache(P0_cache_path, P0_parameters)
    if cached_P0 is None:
        start_time = time.perf_counter()
        P0_offset_blocks = get_P0_offset_blocks(B, reference_L_per_element, element_nodes, element_partition_ids)
        print(f"P0 offline solve (extra cost of the FFT solver): {time.perf_counter() - start_time:.2f} seconds.")
        save_cache(P0_cache_path, P0_parameters, P0_offset_blocks=P0_offset_blocks)
    else:
        P0_offset_blocks = cached_P0["P0_offset_blocks"]

    return E, P, get_P0_transformed(P0_offset_blocks)

def get_solver_label(solver_name, iterations_per_step, solve_time_per_step):
    return (f"{solver_name}: {iterations_per_step.sum()} iterations, {solve_time_per_step.sum():.2f} s, "
            f"{1000 * solve_time_per_step.sum() / iterations_per_step.sum():.2f} ms per iteration")

def shade_elastic_steps(axis, load_steps, iterations_per_step):
    for load_step in load_steps[iterations_per_step == 1]:
        axis.axvspan(load_step - 0.5, load_step + 0.5, color='0.9', linewidth=0, zorder=0)

def label_elastic_steps(axis, load_steps, iterations_per_step):
    if np.any(iterations_per_step == 1):
        first_elastic_step = load_steps[iterations_per_step == 1][0]
        axis.annotate("Elastic\nregime", xy=(first_elastic_step - 0.5, 1), xycoords=axis.get_xaxis_transform(),
                      xytext=(8, -8), textcoords='offset points', ha='left', va='top', color='0.45')

macro_strain_component_labels = [r"$\bar{\varepsilon}_{11}$", r"$\bar{\varepsilon}_{22}$", r"$\bar{\varepsilon}_{33}$",
                                 r"$\bar{\gamma}_{12}$", r"$\bar{\gamma}_{23}$", r"$\bar{\gamma}_{13}$"]

def get_applied_strain_description():
    nonzero_components = [f"{label} = {value:g}" for label, value in zip(macro_strain_component_labels, max_macro_strain)
                          if value != 0]
    return f"Applied Strain: {', '.join(nonzero_components)} ({strain_increment_count} steps)"

def add_solver_legend(axis, line, solver_name, iterations_per_step, solve_time_per_step, location, vertical_anchor):
    no_line = matplotlib.lines.Line2D([], [], linestyle='none')
    totals_label = f"  {iterations_per_step.sum()} iterations,\n  {solve_time_per_step.sum():.2f} s total time"
    solver_legend = matplotlib.legend.Legend(axis, [line, no_line], [f"{solver_name}:", totals_label],
                                             labelspacing=0.4, handlelength=1.5, frameon=False,
                                             loc=location, bbox_to_anchor=(1.01, vertical_anchor))
    solver_name_text, totals_text = solver_legend.get_texts()
    solver_name_text.set_fontweight('bold')
    solver_name_text.set_fontsize(15)
    totals_text.set_fontsize(13)
    axis.add_artist(solver_legend)
    solver_legend.set_clip_on(False)

load_path_summary_style = {'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
                           'legend.fontsize': 15, 'axes.titlesize': 22, 'lines.linewidth': 2.5}
load_path_summary_figure_size = (9.5, 10)

def plot_load_path_summary(macroscopic_stress, iterations_per_step, solve_time_per_step, iterations_per_step_fft,
                           solve_time_per_step_fft):
    load_steps = np.arange(1, strain_increment_count + 1)
    macroscopic_deviatoric_stress_MPa = get_deviatoric_stress(macroscopic_stress) / 1e6
    time_per_iteration_ms = 1000 * solve_time_per_step / iterations_per_step
    time_per_iteration_ms_fft = 1000 * solve_time_per_step_fft / iterations_per_step_fft

    with plt.rc_context(load_path_summary_style):
        figure, (stress_axis, iterations_axis, time_per_iteration_axis) = plt.subplots(
            3, 1, sharex=True, figsize=load_path_summary_figure_size, layout='constrained')
        stress_axis.set_title(get_applied_strain_description(), pad=40)
        # stress_axis.annotate(get_applied_strain_description(), xy=(0.5, 1), xycoords='axes fraction',
        #                      xytext=(0, 12), textcoords='offset points', ha='center', va='bottom', color='0.35')
        for axis in (stress_axis, iterations_axis, time_per_iteration_axis):
            shade_elastic_steps(axis, load_steps, iterations_per_step)
        label_elastic_steps(stress_axis, load_steps, iterations_per_step)

        stress_axis.plot(load_steps, macroscopic_deviatoric_stress_MPa[:, 0], color='red', marker='o', markersize=4,
                         linewidth=2, label=r"$\bar{\mathbf{S}}_{11}$")
        stress_axis.plot(load_steps, macroscopic_deviatoric_stress_MPa[:, 1], color='blue', marker='s', markersize=4,
                         linewidth=2, label=r"$\bar{\mathbf{S}}_{22}$")
        stress_axis.plot(load_steps, macroscopic_deviatoric_stress_MPa[:, 2], color='green', marker='^', markersize=4,
                         linewidth=2, label=r"$\bar{\mathbf{S}}_{33}$")
        stress_axis.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
        stress_axis.set_ylabel("Deviatoric stress (MPa)")

        standard_line, = iterations_axis.step(load_steps, iterations_per_step, where='mid', color='tab:blue')
        fft_line, = iterations_axis.step(load_steps, iterations_per_step_fft, where='mid', color='tab:orange')
        add_solver_legend(iterations_axis, standard_line, "Standard", iterations_per_step, solve_time_per_step,
                          'lower left', 0.52)
        add_solver_legend(iterations_axis, fft_line, "FFT-preconditioned", iterations_per_step_fft,
                          solve_time_per_step_fft, 'upper left', 0.48)
        iterations_axis.set_ylabel("Iterations")
        iterations_axis.set_ylim(bottom=0)
        iterations_axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        time_per_iteration_axis.step(load_steps, time_per_iteration_ms, where='mid', color='tab:blue')
        time_per_iteration_axis.step(load_steps, time_per_iteration_ms_fft, where='mid', color='tab:orange')
        time_per_iteration_axis.set_ylabel("Time per iteration (ms)")
        time_per_iteration_axis.set_ylim(bottom=0)
        time_per_iteration_axis.set_xlabel("Load step", labelpad=12)
        time_per_iteration_axis.set_xlim(0.5, strain_increment_count + 0.5)
        figure.align_ylabels()

        load_path_summary_plot_path = cache_folder / "load_path_summary.png"
        figure.savefig(load_path_summary_plot_path)
    print(f"load path summary plot saved to {load_path_summary_plot_path}")

cross_section_figure_size = (5, 5)

def draw_element_and_partition_edges(axis):
    element_edges = np.linspace(0, domain_side_length, element_number_per_side + 1)
    partition_edges = np.linspace(0, domain_side_length, partition_number_per_side + 1)
    axis.vlines(element_edges, 0, domain_side_length, color='0.75', linewidth=0.5)
    axis.hlines(element_edges, 0, domain_side_length, color='0.75', linewidth=0.5)
    axis.vlines(partition_edges, 0, domain_side_length, color='0.25', linewidth=1.2, clip_on=False)
    axis.hlines(partition_edges, 0, domain_side_length, color='0.25', linewidth=1.2, clip_on=False)

def plot_cross_section():
    partition_material_ids = get_partition_material_ids()
    cross_section_material_ids = partition_material_ids.reshape(partition_number_per_side, partition_number_per_side,
                                                                partition_number_per_side)[0]

    figure, axis = plt.subplots(figsize=cross_section_figure_size, layout='constrained')
    axis.imshow(cross_section_material_ids, cmap=matplotlib.colors.ListedColormap(['0.92', '0.55']), vmin=0, vmax=1,
                origin='lower', extent=(0, domain_side_length, 0, domain_side_length))
    draw_element_and_partition_edges(axis)
    axis.set_axis_off()

    cache_folder.mkdir(exist_ok=True)
    cross_section_plot_path = cache_folder / "cross_section.png"
    figure.savefig(cross_section_plot_path)
    print(f"cross section plot saved to {cross_section_plot_path}")

von_mises_cross_section_figure_size = (6, 5)

def plot_von_mises_cross_section(stress, count_per_side, plot_file_name):
    von_mises_stress_MPa = get_equivalent_stress(get_deviatoric_stress(stress)) / 1e6
    cross_section_von_mises_stress_MPa = von_mises_stress_MPa.reshape(count_per_side, count_per_side, count_per_side)[0]

    figure, axis = plt.subplots(figsize=von_mises_cross_section_figure_size, layout='constrained')
    image = axis.imshow(cross_section_von_mises_stress_MPa, cmap='jet', vmin=von_mises_plot_min_stress / 1e6,
                        vmax=von_mises_plot_max_stress / 1e6, origin='lower',
                        extent=(0, domain_side_length, 0, domain_side_length))
    figure.colorbar(image, ax=axis, extend='both', label="von Mises stress (MPa)")
    draw_element_and_partition_edges(axis)
    axis.set_axis_off()

    von_mises_cross_section_plot_path = cache_folder / plot_file_name
    figure.savefig(von_mises_cross_section_plot_path)
    print(f"von Mises cross section plot saved to {von_mises_cross_section_plot_path}")

def main():
    plot_cross_section()
    L_matrix = get_L(elastic_modulus_matrix, poisson_ratio_matrix)
    L_inclusion = get_L(elastic_modulus_inclusion, poisson_ratio_inclusion)
    L_per_partition, yield_stress_per_partition, hardening_modulus_per_partition = get_partition_material_properties(
        L_matrix, L_inclusion)
    E, P, P0_transformed = get_offline_operators(L_matrix, L_inclusion, L_per_partition)

    solve_time_per_step_per_repeat = np.zeros((timing_repeat_count, strain_increment_count))
    solve_time_per_step_per_repeat_fft = np.zeros((timing_repeat_count, strain_increment_count))
    for repeat in range(timing_repeat_count):
        (macroscopic_stress, iterations_per_step, solve_time_per_step_per_repeat[repeat], final_stress,
         final_plastic_strain) = run_strain_path(
            E, P, L_per_partition, yield_stress_per_partition, hardening_modulus_per_partition)
        macroscopic_stress_fft, iterations_per_step_fft, solve_time_per_step_per_repeat_fft[repeat], _, _ = run_strain_path(
            E, P, L_per_partition, yield_stress_per_partition, hardening_modulus_per_partition, P0_transformed)
    solve_time_per_step = np.min(solve_time_per_step_per_repeat, axis=0)
    solve_time_per_step_fft = np.min(solve_time_per_step_per_repeat_fft, axis=0)

    standard_label = get_solver_label("standard", iterations_per_step, solve_time_per_step)
    fft_label = get_solver_label("FFT-preconditioned", iterations_per_step_fft, solve_time_per_step_fft)
    stress_relative_difference = (np.abs(macroscopic_stress_fft - macroscopic_stress).max()
                                   / np.abs(macroscopic_stress).max())
    print(f"relaxation_factor = {relaxation_factor}")
    print(f"solve times are the minimum per load step over {timing_repeat_count} interleaved runs of each solver")
    print(standard_label)
    print(fft_label)
    print(f"stress agreement between solvers: {stress_relative_difference:.2e} relative.")

    plot_load_path_summary(macroscopic_stress, iterations_per_step, solve_time_per_step, iterations_per_step_fft,
                           solve_time_per_step_fft)
    plot_von_mises_cross_section(final_stress, partition_number_per_side, "von_mises_cross_section.png")

    if post_process_element_stress:
        element_stress = get_element_stress(L_matrix, L_inclusion, max_macro_strain, final_plastic_strain)
        plot_von_mises_cross_section(element_stress, element_number_per_side, "von_mises_element_cross_section.png")

if __name__ == "__main__":
    main()
