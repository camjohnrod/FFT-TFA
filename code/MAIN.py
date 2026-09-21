import numpy as np
import scipy.sparse
import scipy.sparse.linalg


## ------- Inputs ------- ##

domain_side_length         = 1e-3
inclusion_side_length      = (1/3) * 1e-3 #0.7e-3
element_number_per_side    = 9
partition_number_per_side  = 3

elastic_modulus_inclusion  = 10e9
poisson_ratio_inclusion    = 0.3
elastic_modulus_matrix     = 100e6
poisson_ratio_matrix       = poisson_ratio_inclusion
matrix_yield_stress        = 1e6
matrix_hardening_modulus   = 10e6

max_uniaxial_strain        = 0.03
strain_increment_count     = 60

fixed_point_tolerance      = 1e-6

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

def get_plastic_eigenstrain():
    pass

def standard_richardson_iteration():
    pass

def fft_preconditioned_richardson_iteration():
    pass

def main():
    element_nodes, element_partition_ids, element_material_ids = get_periodic_mesh()
    B = get_B()

    L_matrix = get_L(elastic_modulus_matrix, poisson_ratio_matrix)
    L_inclusion = get_L(elastic_modulus_inclusion, poisson_ratio_inclusion)
    L_per_element = get_L_per_element(L_matrix, L_inclusion, element_material_ids)
    
    E, P = get_influence_functions(B, L_per_element, element_nodes, element_partition_ids)
    reference_L_per_element = get_L_per_element(L_matrix, L_matrix, element_material_ids) # using L_matrix for the reference is a placeholder for now
    _, P0 = get_influence_functions(B, reference_L_per_element, element_nodes, element_partition_ids)

if __name__ == "__main__":
    main()