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

def get_periodic_mesh():
    pass

def get_E(K, F, B, element_nodes, element_partition_ids):
    displacements = solve_for_displacements(K, F)
    average_strain = get_partition_average_strain(displacements, B, element_nodes, element_partition_ids)

def get_P(K, eigenstrain_loads, B, element_nodes, element_partition_ids):
    displacements = solve_for_displacements(K, eigenstrain_loads)
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

def get_F(B, L_per_element, element_nodes):
    pass

def get_eigenstrain_loads(B, L_per_element, element_nodes, element_partition_ids):
    pass

def solve_for_displacements(K, loads):
    pass

def get_partition_average_strain(displacements, B, element_nodes, element_partition_ids):
    pass

def get_influence_functions(B, L_per_element, element_nodes, element_partition_ids):
    K = get_K(B, L_per_element, element_nodes)
    F = get_F(B, L_per_element, element_nodes)
    eigenstrain_loads = get_eigenstrain_loads(B, L_per_element, element_nodes, element_partition_ids)
    E = get_E(K, F, B, element_nodes, element_partition_ids)
    P = get_P(K, eigenstrain_loads, B, element_nodes, element_partition_ids)
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
    reference_L_per_element = get_L_per_element(L_matrix, L_matrix, element_material_ids)
    _, P0 = get_influence_functions(B, reference_L_per_element, element_nodes, element_partition_ids)

if __name__ == "__main__":
    main()