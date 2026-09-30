# Every input to a run, then the quantities derived from them and the checks that the geometry fits the
# partition grid. This is the only place a run's configuration is changed; every other module reads it as
# config.<name>.

import pathlib
import numpy as np

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
independent_check_round_off = 1e-12

timing_repeat_count         = 5

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

# The inputs that define the physical problem and its load path. They are saved with every run's results, and results
# from two runs are only compared when all of these match.
problem_parameter_names = ("domain_side_length", "inclusion_shape", "inclusion_side_length", "inclusion_radius",
                           "element_number_per_side", "element_number_along_z", "partition_number_per_side",
                           "elastic_modulus_inclusion", "elastic_modulus_matrix", "poisson_ratio_inclusion",
                           "poisson_ratio_matrix", "inclusion_yield_stress", "matrix_yield_stress",
                           "inclusion_hardening_modulus", "matrix_hardening_modulus", "max_macro_strain",
                           "strain_increment_count")

# Shared by every entry point. Cache files are named by a hash of the parameters that produced them, and output files
# are prefixed with the entry point's run name.
cache_folder = pathlib.Path(__file__).parent / "cache"
output_folder = pathlib.Path(__file__).parent / "output"
