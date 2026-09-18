import numpy as np


## ------- Inputs ------- ##

domain_side_length         = 1e-3
inclusion_side_length      = 0.7e-3
element_number_per_side    = 9
partition_number_per_side  = 3

inclusion_elastic_modulus  = 100e9
inclusion_poisson_ratio    = 0.3
matrix_elastic_modulus     = 10e9
matrix_poisson_ratio       = inclusion_poisson_ratio
matrix_yield_stress        = 1e6
matrix_hardening_modulus   = 10e6

max_uniaxial_strain        = 0.03
strain_increment_count     = 60

fixed_point_tolerance      = 1e-6

## ------- Calculated Values and Checks ------- ##

elements_per_partition_side = element_number_per_side // partition_number_per_side
if element_number_per_side % partition_number_per_side != 0:
    raise ValueError("The number of elements per side must be divisible by the number of partitions per side.")



def get_influence_functions():
    pass

def get_plastic_eigenstrain():
    pass

def standard_richardson_iteration():
    pass

def fft_preconditioned_richardson_iteration():
    pass

def main():
    pass