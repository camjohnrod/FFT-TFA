import numpy as np


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

def get_B():
    pass

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
    pass

def get_K(B, L_per_element, element_nodes):
    pass

def get_F_macrostrain(B, L_per_element, element_nodes):
    pass

def get_F_eigenstrain(B, L_per_element, element_nodes, element_partition_ids):
    pass

def solve_for_displacements(K, loads):
    pass

def get_partition_average_strain(displacements, B, element_nodes, element_partition_ids):
    pass

def get_influence_functions(B, L_per_element, element_nodes, element_partition_ids):
    K = get_K(B, L_per_element, element_nodes)
    F_macrostrain = get_F_macrostrain(B, L_per_element, element_nodes)
    F_eigenstrain = get_F_eigenstrain(B, L_per_element, element_nodes, element_partition_ids)
    E = solve_influence_function(K, F_macrostrain, B, element_nodes, element_partition_ids)
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