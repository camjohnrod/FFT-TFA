import pathlib
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import time

## ------- Inputs ------- ##

domain_side_length          = 1e-3
inclusion_side_length       = (5/9) * 1e-3 #0.7e-3
element_number_per_side     = 27
partition_number_per_side   = 9

elastic_modulus_inclusion   = 10e9
poisson_ratio_inclusion     = 0.3
elastic_modulus_matrix      = 100e6
poisson_ratio_matrix        = poisson_ratio_inclusion
matrix_yield_stress         = 1e6
matrix_hardening_modulus    = 10e6

max_uniaxial_strain         = 0.03
strain_increment_count      = 30

fixed_point_tolerance       = 1e-6
fixed_point_max_iterations  = 500

relaxation_factor           = 0.5

## ------- Calculated Values and Checks ------- ##

elements_per_partition_side = element_number_per_side // partition_number_per_side
if element_number_per_side % partition_number_per_side != 0:
    raise ValueError("The number of elements per side must be divisible by the number of partitions per side.")

element_side_length = domain_side_length / element_number_per_side
gauss_point_volume_weight = element_side_length ** 3 / 8

matrix_partitions_beside_inclusion = partition_number_per_side * (domain_side_length - inclusion_side_length) / (2 * domain_side_length)
whole_matrix_partitions_beside_inclusion = round(matrix_partitions_beside_inclusion)
if not np.isclose(matrix_partitions_beside_inclusion, whole_matrix_partitions_beside_inclusion) or not 1 <= whole_matrix_partitions_beside_inclusion < partition_number_per_side / 2:
    raise ValueError(f"The centred inclusion must have a whole number of matrix partitions (at least 1) on each side, so its edges fall on partition edges. Got {matrix_partitions_beside_inclusion}.")

def get_grid_id(i, j, k, count_per_side):
    return i + count_per_side * j + count_per_side**2 * k

def get_periodic_mesh():
    element_count = element_number_per_side**3
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
                inside_inclusion = np.all(np.abs(element_centre - domain_side_length / 2) < inclusion_side_length / 2)
                element_material_ids[element_id] = int(inside_inclusion)

    return element_nodes, element_partition_ids, element_material_ids

def get_partition_material_ids():
    partition_side_length = domain_side_length / partition_number_per_side
    partition_count = partition_number_per_side**3
    partition_material_ids = np.zeros(partition_count, dtype=int)

    for k in range(partition_number_per_side):
        for j in range(partition_number_per_side):
            for i in range(partition_number_per_side):
                partition_id = get_grid_id(i, j, k, partition_number_per_side)
                partition_centre = (np.array([i, j, k]) + 0.5) * partition_side_length
                inside_inclusion = np.all(np.abs(partition_centre - domain_side_length / 2) < inclusion_side_length / 2)
                partition_material_ids[partition_id] = int(inside_inclusion)

    return partition_material_ids

def solve_influence_function(K, F, B, element_nodes, element_partition_ids):
    displacements = solve_for_displacements(K, F)
    average_strain = get_partition_average_strain(displacements, B, element_nodes, element_partition_ids)
    return average_strain

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

def get_L_per_element(matrix_L, inclusion_L, element_material_ids):
    L_by_material_id = np.array([matrix_L, inclusion_L])
    return L_by_material_id[element_material_ids]

def get_L_per_partition(matrix_L, inclusion_L, partition_material_ids):
    L_by_material_id = np.array([matrix_L, inclusion_L])
    return L_by_material_id[partition_material_ids]

def get_homogenized_L(E, L_per_partition):
    partition_count = L_per_partition.shape[0]
    E_blocks = E.reshape(partition_count, 6, 6)
    return np.mean(L_per_partition @ E_blocks, axis=0)

def get_element_dofs(element_nodes):
    node_dofs = 3 * element_nodes[:, :, None] + np.arange(3)
    return node_dofs.reshape(len(element_nodes), 24)

def get_K_element(B, L):
    K_element = np.zeros((24, 24))
    for gauss_index in range(8):
        K_element += gauss_point_volume_weight * B[gauss_index].T @ L @ B[gauss_index]
    return K_element

def get_K(B, L_per_element, element_nodes):
    element_dofs = get_element_dofs(element_nodes)
    dof_count = 3 * element_number_per_side**3
    K_elements = np.zeros((len(element_nodes), 24, 24))
    for element_id in range(len(element_nodes)):
        K_elements[element_id] = get_K_element(B, L_per_element[element_id])
    rows = np.repeat(element_dofs, 24, axis=1)
    columns = np.tile(element_dofs, (1, 24))
    K = scipy.sparse.coo_matrix((K_elements.ravel(), (rows.ravel(), columns.ravel())), shape=(dof_count, dof_count))
    return K.tocsr()

def get_F_element(B, L):
    F_element = np.zeros((24, 6))
    for gauss_index in range(8):
        F_element += gauss_point_volume_weight * B[gauss_index].T @ L
    return F_element

def get_F_macrostrain(B, L_per_element, element_nodes):
    element_dofs = get_element_dofs(element_nodes)
    dof_count = 3 * element_number_per_side**3
    F_macrostrain = np.zeros((dof_count, 6))
    for element_id in range(len(element_nodes)):
        F_element = get_F_element(B, L_per_element[element_id])
        F_macrostrain[element_dofs[element_id]] -= F_element
    return F_macrostrain

def get_F_eigenstrain(B, L_per_element, element_nodes, element_partition_ids):
    element_dofs = get_element_dofs(element_nodes)
    dof_count = 3 * element_number_per_side**3
    partition_count = partition_number_per_side**3
    F_eigenstrain = np.zeros((dof_count, 6 * partition_count))
    for element_id in range(len(element_nodes)):
        F_element = get_F_element(B, L_per_element[element_id])
        first_column = 6 * element_partition_ids[element_id]
        F_eigenstrain[np.ix_(element_dofs[element_id], np.arange(first_column, first_column + 6))] += F_element
    return F_eigenstrain

def solve_for_displacements(K, loads):
    pinned_dofs = np.arange(3)
    free_dofs = np.setdiff1d(np.arange(K.shape[0]), pinned_dofs)
    K_free = K[free_dofs][:, free_dofs].tocsc()
    K_free_factorization = scipy.sparse.linalg.splu(K_free)

    columns_per_solve = 100
    displacements = np.zeros_like(loads)
    for first_column in range(0, loads.shape[1], columns_per_solve):
        columns = slice(first_column, first_column + columns_per_solve)
        displacements[free_dofs, columns] = K_free_factorization.solve(loads[free_dofs, columns])
    return displacements

def get_partition_average_strain(displacements, B, element_nodes, element_partition_ids):
    element_dofs = get_element_dofs(element_nodes)
    partition_count = partition_number_per_side**3
    partition_volume = elements_per_partition_side**3 * element_side_length**3
    average_strain = np.zeros((6 * partition_count, displacements.shape[1]))
    for element_id in range(len(element_nodes)):
        element_displacements = displacements[element_dofs[element_id]]
        first_row = 6 * element_partition_ids[element_id]
        for gauss_index in range(8):
            average_strain[first_row:first_row + 6] += gauss_point_volume_weight * B[gauss_index] @ element_displacements
    return average_strain / partition_volume

def get_influence_functions(B, L_per_element, element_nodes, element_partition_ids):
    K = get_K(B, L_per_element, element_nodes)
    F_macrostrain = get_F_macrostrain(B, L_per_element, element_nodes)
    F_eigenstrain = get_F_eigenstrain(B, L_per_element, element_nodes, element_partition_ids)
    partition_count = partition_number_per_side**3
    E = np.tile(np.eye(6), (partition_count, 1)) + solve_influence_function(K, F_macrostrain, B, element_nodes, element_partition_ids)
    P = solve_influence_function(K, F_eigenstrain, B, element_nodes, element_partition_ids)
    return E, P

cache_folder = pathlib.Path(__file__).parent / "cache"

def load_cache(cache_path, parameters):
    if not cache_path.exists():
        print(f"{cache_path.name}: no cache found, computing.")
        return None
    cached = np.load(cache_path)
    for name, value in parameters.items():
        if not np.array_equal(cached[name], value):
            print(f"{cache_path.name}: {name} changed, recomputing.")
            return None
    print(f"{cache_path.name}: loaded from cache.")
    return cached

def save_cache(cache_path, parameters, **arrays):
    cache_folder.mkdir(exist_ok=True)
    np.savez(cache_path, **arrays, **parameters)

volumetric_direction = np.array([1, 1, 1, 0, 0, 0])

deviatoric_projector = np.zeros((6, 6))
deviatoric_projector[:3, :3] = np.eye(3) - np.ones((3, 3)) / 3
deviatoric_projector[3:, 3:] = np.eye(3)

def get_trial_state(strain, L, plastic_strain, accumulated_plastic_strain, yield_stress, hardening_modulus):
    elastic_trial_strain = strain - plastic_strain
    trial_stress = L @ elastic_trial_strain
    trial_mean_stress = np.sum(trial_stress[:3]) / 3
    trial_deviatoric_stress = trial_stress - trial_mean_stress * volumetric_direction
    trial_equivalent_stress = np.sqrt(1.5 * np.sum(trial_deviatoric_stress[:3]**2)
                                       + 3 * np.sum(trial_deviatoric_stress[3:]**2))
    trial_yield_function = trial_equivalent_stress - (yield_stress + hardening_modulus * accumulated_plastic_strain)
    return trial_stress, trial_mean_stress, trial_deviatoric_stress, trial_equivalent_stress, trial_yield_function

def get_local_plastic_update(strain, L, plastic_strain, accumulated_plastic_strain, yield_stress, hardening_modulus):
    trial_stress, trial_mean_stress, trial_deviatoric_stress, trial_equivalent_stress, trial_yield_function = get_trial_state(
        strain, L, plastic_strain, accumulated_plastic_strain, yield_stress, hardening_modulus)

    if trial_yield_function <= 0:
        return plastic_strain, accumulated_plastic_strain, trial_stress

    shear_modulus = L[3, 3]
    plastic_multiplier = trial_yield_function / (3 * shear_modulus + hardening_modulus)

    plastic_strain_increment = np.zeros(6)
    plastic_strain_increment[:3] = plastic_multiplier * 1.5 * trial_deviatoric_stress[:3] / trial_equivalent_stress
    plastic_strain_increment[3:] = plastic_multiplier * 3 * trial_deviatoric_stress[3:] / trial_equivalent_stress
    plastic_strain_new = plastic_strain + plastic_strain_increment
    accumulated_plastic_strain_new = accumulated_plastic_strain + plastic_multiplier

    corrected_deviatoric_stress = (1 - 3 * shear_modulus * plastic_multiplier / trial_equivalent_stress) * trial_deviatoric_stress
    stress = trial_mean_stress * volumetric_direction + corrected_deviatoric_stress

    return plastic_strain_new, accumulated_plastic_strain_new, stress

def get_plastic_eigenstrain(strain, L_per_partition, plastic_strain_history, accumulated_plastic_strain_history,
                             yield_stress_per_partition, hardening_modulus_per_partition):
    partition_count = strain.shape[0]
    plastic_strain_new = np.zeros((partition_count, 6))
    accumulated_plastic_strain_new = np.zeros(partition_count)
    stress = np.zeros((partition_count, 6))

    for partition in range(partition_count):
        plastic_strain_new[partition], accumulated_plastic_strain_new[partition], stress[partition] = get_local_plastic_update(
            strain[partition], L_per_partition[partition], plastic_strain_history[partition],
            accumulated_plastic_strain_history[partition], yield_stress_per_partition[partition],
            hardening_modulus_per_partition[partition])

    return plastic_strain_new, accumulated_plastic_strain_new, stress

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

def standard_richardson_iteration(E, P, macro_strain, L_per_partition,
                                   yield_stress_per_partition, hardening_modulus_per_partition,
                                   plastic_strain_history, accumulated_plastic_strain_history,
                                   relaxation_factor):
    partition_count = plastic_strain_history.shape[0]

    b = (E @ macro_strain).reshape(partition_count, 6)
    strain = b + (P @ plastic_strain_history.reshape(-1)).reshape(partition_count, 6)

    residual_history = []
    while True:
        plastic_strain, accumulated_plastic_strain, stress = get_plastic_eigenstrain(
            strain, L_per_partition, plastic_strain_history, accumulated_plastic_strain_history,
            yield_stress_per_partition, hardening_modulus_per_partition)
        residual = strain - b - (P @ plastic_strain.reshape(-1)).reshape(partition_count, 6)
        residual_history.append(get_relative_residual(residual, strain))

        if has_converged(residual_history[-1], len(residual_history), "standard_richardson_iteration"):
            break

        strain = strain - relaxation_factor * residual

    return strain, stress, plastic_strain, accumulated_plastic_strain, np.array(residual_history)

def get_macroscopic_stress(stress):
    return np.mean(stress, axis=0)

def get_local_eigenstrain_sensitivity(strain, L, plastic_strain, accumulated_plastic_strain, yield_stress, hardening_modulus):
    _, _, trial_deviatoric_stress, trial_equivalent_stress, trial_yield_function = get_trial_state(
        strain, L, plastic_strain, accumulated_plastic_strain, yield_stress, hardening_modulus)

    if trial_yield_function <= 0:
        return np.zeros((6, 6))

    shear_modulus = L[3, 3]
    plastic_multiplier = trial_yield_function / (3 * shear_modulus + hardening_modulus)

    flow_direction = np.zeros(6)
    flow_direction[:3] = 1.5 * trial_deviatoric_stress[:3] / trial_equivalent_stress
    flow_direction[3:] = 3 * trial_deviatoric_stress[3:] / trial_equivalent_stress
    normalized_deviatoric_stress = trial_deviatoric_stress / trial_equivalent_stress
    flow_projector = np.outer(flow_direction, normalized_deviatoric_stress)

    radial_sensitivity = 3 * shear_modulus / (3 * shear_modulus + hardening_modulus)
    rotational_sensitivity = 3 * shear_modulus * plastic_multiplier / trial_equivalent_stress

    return radial_sensitivity * flow_projector + rotational_sensitivity * (deviatoric_projector - flow_projector)

def get_eigenstrain_sensitivity(strain, L_per_partition, plastic_strain_history, accumulated_plastic_strain_history,
                                yield_stress_per_partition, hardening_modulus_per_partition):
    partition_count = strain.shape[0]
    sensitivity = np.zeros((partition_count, 6, 6))

    for partition in range(partition_count):
        sensitivity[partition] = get_local_eigenstrain_sensitivity(
            strain[partition], L_per_partition[partition], plastic_strain_history[partition],
            accumulated_plastic_strain_history[partition], yield_stress_per_partition[partition],
            hardening_modulus_per_partition[partition])

    return sensitivity

def get_P0_transformed(P0):
    P0_offset_blocks = P0[:, :6].reshape(partition_number_per_side, partition_number_per_side,
                                         partition_number_per_side, 6, 6)
    return np.fft.fftn(P0_offset_blocks, axes=(0, 1, 2))

def get_reference_fourier_inverse(P0_transformed, reference_sensitivity):
    M0_transformed = np.eye(6) - P0_transformed @ reference_sensitivity
    return np.linalg.inv(M0_transformed)

def fft_preconditioned_richardson_iteration(E, P, macro_strain, L_per_partition,
                                             yield_stress_per_partition, hardening_modulus_per_partition,
                                             plastic_strain_history, accumulated_plastic_strain_history,
                                             P0_transformed, relaxation_factor):
    partition_count = plastic_strain_history.shape[0]
    lattice_shape = (partition_number_per_side, partition_number_per_side, partition_number_per_side, 6)

    b = (E @ macro_strain).reshape(partition_count, 6)
    strain = b + (P @ plastic_strain_history.reshape(-1)).reshape(partition_count, 6)

    sensitivity = get_eigenstrain_sensitivity(strain, L_per_partition, plastic_strain_history,
                                              accumulated_plastic_strain_history, yield_stress_per_partition,
                                              hardening_modulus_per_partition)
    reference_sensitivity = np.mean(sensitivity, axis=0)
    reference_fourier_inverse = get_reference_fourier_inverse(P0_transformed, reference_sensitivity)

    residual_history = []
    while True:
        plastic_strain, accumulated_plastic_strain, stress = get_plastic_eigenstrain(
            strain, L_per_partition, plastic_strain_history, accumulated_plastic_strain_history,
            yield_stress_per_partition, hardening_modulus_per_partition)
        residual = strain - b - (P @ plastic_strain.reshape(-1)).reshape(partition_count, 6)
        residual_history.append(get_relative_residual(residual, strain))

        if has_converged(residual_history[-1], len(residual_history), "fft_preconditioned_richardson_iteration"):
            break

        residual_transformed = np.fft.fftn(residual.reshape(lattice_shape), axes=(0, 1, 2))
        correction_transformed = np.einsum('...ij,...j->...i', reference_fourier_inverse, -residual_transformed)
        correction = np.fft.ifftn(correction_transformed, axes=(0, 1, 2)).real
        strain = strain + relaxation_factor * correction.reshape(partition_count, 6)

    return strain, stress, plastic_strain, accumulated_plastic_strain, np.array(residual_history)

def run_uniaxial_strain_path(E, P, L_per_partition, yield_stress_per_partition, hardening_modulus_per_partition,
                              relaxation_factor, P0_transformed=None):
    partition_count = L_per_partition.shape[0]
    plastic_strain_history = np.zeros((partition_count, 6))
    accumulated_plastic_strain_history = np.zeros(partition_count)

    applied_strain_11 = np.linspace(max_uniaxial_strain / strain_increment_count, max_uniaxial_strain,
                                     strain_increment_count)
    macroscopic_stress = np.zeros((strain_increment_count, 6))
    residual_history_per_step = []
    load_step_boundaries = []
    cumulative_iteration_count = 0

    for step, strain_11 in enumerate(applied_strain_11):
        macro_strain = np.array([strain_11, 0, 0, 0, 0, 0])
        if P0_transformed is None:
            strain, stress, plastic_strain_history, accumulated_plastic_strain_history, residual_history = standard_richardson_iteration(
                E, P, macro_strain, L_per_partition, yield_stress_per_partition, hardening_modulus_per_partition,
                plastic_strain_history, accumulated_plastic_strain_history, relaxation_factor)
        else:
            strain, stress, plastic_strain_history, accumulated_plastic_strain_history, residual_history = fft_preconditioned_richardson_iteration(
                E, P, macro_strain, L_per_partition, yield_stress_per_partition, hardening_modulus_per_partition,
                plastic_strain_history, accumulated_plastic_strain_history, P0_transformed, relaxation_factor)
        macroscopic_stress[step] = get_macroscopic_stress(stress)

        if step > 0:
            load_step_boundaries.append(cumulative_iteration_count)
        residual_history_per_step.append(residual_history)
        cumulative_iteration_count += len(residual_history)

    all_residuals = np.concatenate(residual_history_per_step)
    load_step_boundaries = np.array(load_step_boundaries)

    return applied_strain_11, macroscopic_stress, all_residuals, load_step_boundaries

def main():
    element_nodes, element_partition_ids, element_material_ids = get_periodic_mesh()
    B = get_B()

    L_matrix = get_L(elastic_modulus_matrix, poisson_ratio_matrix)
    L_inclusion = get_L(elastic_modulus_inclusion, poisson_ratio_inclusion)
    L_per_element = get_L_per_element(L_matrix, L_inclusion, element_material_ids)

    E_P_cache_path = cache_folder / "E_P.npz"
    E_P_parameters = {"domain_side_length": domain_side_length,
                      "inclusion_side_length": inclusion_side_length,
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

    partition_material_ids = get_partition_material_ids()
    L_per_partition = get_L_per_partition(L_matrix, L_inclusion, partition_material_ids)

    reference_L = get_homogenized_L(E, L_per_partition)
    reference_L_per_element = get_L_per_element(reference_L, reference_L, element_material_ids)

    P0_cache_path = cache_folder / "P0.npz"
    P0_parameters = {"domain_side_length": domain_side_length,
                     "element_number_per_side": element_number_per_side,
                     "partition_number_per_side": partition_number_per_side,
                     "reference_L": reference_L}
    cached_P0 = load_cache(P0_cache_path, P0_parameters)
    if cached_P0 is None:
        _, P0 = get_influence_functions(B, reference_L_per_element, element_nodes, element_partition_ids)
        save_cache(P0_cache_path, P0_parameters, P0=P0)
    else:
        P0 = cached_P0["P0"]

    yield_stress_by_material_id = np.array([matrix_yield_stress, np.inf])
    yield_stress_per_partition = yield_stress_by_material_id[partition_material_ids]

    hardening_modulus_by_material_id = np.array([matrix_hardening_modulus, 0])
    hardening_modulus_per_partition = hardening_modulus_by_material_id[partition_material_ids]

    P0_transformed = get_P0_transformed(P0)

    t0 = time.time()
    applied_strain_11, macroscopic_stress, all_residuals, load_step_boundaries = run_uniaxial_strain_path(
        E, P, L_per_partition, yield_stress_per_partition, hardening_modulus_per_partition, relaxation_factor)
    t1 = time.time()
    print(f"standard: {len(all_residuals)} iterations, {t1 - t0:.2f} seconds.")

    t0 = time.time()
    _, macroscopic_stress_fft, all_residuals_fft, load_step_boundaries_fft = run_uniaxial_strain_path(
        E, P, L_per_partition, yield_stress_per_partition, hardening_modulus_per_partition, relaxation_factor,
        P0_transformed)
    t1 = time.time()
    stress_relative_difference = (np.abs(macroscopic_stress_fft - macroscopic_stress).max()
                                   / np.abs(macroscopic_stress).max())
    print(f"FFT-preconditioned (relaxation_factor={relaxation_factor}): "
          f"{len(all_residuals_fft)} iterations, {t1 - t0:.2f} seconds, "
          f"stress agreement with standard {stress_relative_difference:.2e} relative.")

    plt.figure(figsize=(6, 4))
    plt.plot(applied_strain_11, macroscopic_stress[:, 0], color='red', marker='o', markersize=4, linestyle='-',
              label=r"$\bar{\sigma}_{11}$")
    plt.plot(applied_strain_11, macroscopic_stress[:, 1], color='blue', marker='o', markersize=4, linestyle='-',
              label=r"$\bar{\sigma}_{22}$")
    plt.plot(applied_strain_11, macroscopic_stress[:, 2], color='green', marker='o', markersize=4, linestyle='-',
              label=r"$\bar{\sigma}_{33}$")
    plt.xlabel(r"applied strain, $\bar{\varepsilon}_{11}$")
    plt.ylabel(r"macroscopic stress, $\bar{\sigma}$")
    plt.legend()
    stress_strain_plot_path = cache_folder / "stress_strain.png"
    plt.savefig(stress_strain_plot_path)
    print(f"stress-strain plot saved to {stress_strain_plot_path}")

    plt.figure(figsize=(10, 3))
    plt.plot(all_residuals, color='blue', label='standard')
    plt.plot(all_residuals_fft, color='orange', label='FFT-preconditioned')
    for boundary in load_step_boundaries:
        plt.axvline(boundary, color='blue', linestyle='--', linewidth=0.5, alpha=0.4)
    for boundary in load_step_boundaries_fft:
        plt.axvline(boundary, color='orange', linestyle='--', linewidth=0.5, alpha=0.4)
    plt.yscale('log')
    plt.xlabel("iteration")
    plt.ylabel("relative residual")
    plt.legend()
    residual_plot_path = cache_folder / "residual_history.png"
    plt.savefig(residual_plot_path)
    print(f"residual history plot saved to {residual_plot_path}")

if __name__ == "__main__":
    main()