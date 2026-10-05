# Every input to a run, then the quantities derived from them and the checks that the geometry fits the
# partition grid. This is the only place a run's configuration is changed; every other module reads it as
# config.<name>.

import pathlib
import numpy as np

## ------- Inputs ------- ##

# Geometry, mesh and partitions
domain_side_length          = 1e-3
inclusion_shape             = "circle"
inclusion_side_length       = (5/9) * 1e-3
inclusion_radius            = 0.35e-3
element_number_per_side     = 64
element_number_along_z      = 3
partition_number_per_side   = 32

# Materials. matched_stiffness_control = True gives the inclusion the matrix's elastic stiffness; the TFA and LS models
# are then the same equation, which is the regression gate for the LS solvers.
elastic_modulus_inclusion   = 10e9
elastic_modulus_matrix      = 100e6
poisson_ratio_inclusion     = 0.3
poisson_ratio_matrix        = 0.19
inclusion_yield_stress      = np.inf
matrix_yield_stress         = 1.0e6
inclusion_hardening_modulus = 0.0
matrix_hardening_modulus    = 10e6
matched_stiffness_control   = False

# Models. The LS model is always solved. include_tfa_model also solves the actual E/P TFA model by the same strategy,
# as the baseline LS is compared against; matched_stiffness_control, the regression gate between the two, needs it.
include_tfa_model           = True

# Reference medium: the homogeneous stiffness C0 from which the P0 kernel is built. It is part of the LS model, so it
# changes the LS answer; for TFA it only preconditions the FFT solvers, changing their convergence, not their answer.
#   "homogenized"  the homogenized stiffness of the actual composite, <L E> over the partitions
#   "voigt"        the Voigt average <L> over the partitions
#   "matrix"       the matrix stiffness
reference_stiffness         = "homogenized"

# Load path: max_macro_strain reached linearly over strain_increment_count steps. Alternatives:
#   np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0])      uniaxial
#   np.array([0.03, 0.018, 0.001, 0.0, 0.0, 0.0])  mixed mode tension
#   np.array([0.015, 0.020, 0.0, 0.03, 0.0, 0.0])  mixed mode with shear
# load_path_shape "monotonic" stops at max_macro_strain; "cyclic" continues 0 -> +max -> -max -> 0 at the same
# increment, 4 * strain_increment_count steps in all, unloading elastically, yielding in reverse and passing through
# zero imposed strain twice.
max_macro_strain            = np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0])
strain_increment_count      = 60
load_path_shape             = "monotonic"

# Convergence, shared by every solver: relative residual (Formulation.md section 12) below convergence_tolerance.
convergence_tolerance       = 1e-6
residual_strain_scale_floor = 1e-4
solver_agreement_tolerance  = 1e-4
independent_check_round_off = 1e-12

# TFA solvers. The relaxation applies to the fixed points; the divergence limit to fixed point and Newton. A solver
# diverges when its relative residual exceeds the limit times the larger of 1 and the step's first relative residual
# (convergence.has_converged), here and for LS.
tfa_relaxation_factor       = 1.0
tfa_max_iterations          = 1000
tfa_divergence_limit        = 1.0

# LS solvers. None picks the fixed point's relaxation from the stiffness-ratio bounds, times the safety factor.
ls_relaxation_factor        = None
ls_stability_safety         = 0.9
ls_max_iterations           = 20000
ls_divergence_limit         = 1e3

# Newton-GMRES, shared by every Newton solver so they are compared at identical settings.
newton_max_steps            = 50
newton_krylov_tolerance     = 1e-3
newton_krylov_restart       = 50

# Checks run before solving, by both entry points (the Jacobian checks by MAIN_Newton.py only).
verification_enabled        = True
verification_tolerance      = 1e-9
jacobian_check_tolerance    = 1e-6
verification_max_partitions = 256
matched_stiffness_tolerance = 1e-5

timing_repeat_count         = 5

# Colour-scale limits (MPa) of the von Mises stress maps, so maps from different runs can share a scale. None takes
# that limit from the plotted maps. The difference map always keeps its own scale, centred on zero.
von_mises_min_MPa           = None
von_mises_max_MPa           = None

## ------- Calculated Values and Checks ------- ##

if min(element_number_per_side, element_number_along_z, partition_number_per_side, strain_increment_count,
       timing_repeat_count) < 1:
    raise ValueError("The element, partition, strain increment and timing repeat counts must all be at least 1.")
if von_mises_min_MPa is not None and von_mises_max_MPa is not None and von_mises_min_MPa >= von_mises_max_MPa:
    raise ValueError("The von Mises colour-scale minimum must be below its maximum.")
if element_number_per_side % partition_number_per_side != 0:
    raise ValueError("The number of elements per side must be divisible by the number of partitions per side.")

# The radial return divides by 3G + H. At or below zero it has no admissible solution: the plastic multiplier is
# infinite or negative, so plastic work runs backwards while the flow stress stays positive, and nothing downstream
# would notice. Checked for every phase that can yield, with the stiffness it actually gets.
inclusion_shear_modulus = (elastic_modulus_matrix / (2 * (1 + poisson_ratio_matrix)) if matched_stiffness_control
                           else elastic_modulus_inclusion / (2 * (1 + poisson_ratio_inclusion)))
for phase, shear_modulus, yield_stress, hardening_modulus in (
        ("matrix", elastic_modulus_matrix / (2 * (1 + poisson_ratio_matrix)), matrix_yield_stress,
         matrix_hardening_modulus),
        ("inclusion", inclusion_shear_modulus, inclusion_yield_stress, inclusion_hardening_modulus)):
    if np.isfinite(yield_stress) and 3 * shear_modulus + hardening_modulus <= 0:
        raise ValueError(f"The {phase} hardening modulus must be above -3G = {-3 * shear_modulus:.4g} Pa, or the "
                         f"return map has no admissible solution. Got {hardening_modulus:.4g} Pa.")

if load_path_shape == "monotonic":
    load_fractions = np.linspace(1 / strain_increment_count, 1, strain_increment_count)
elif load_path_shape == "cyclic":
    load_fractions = np.concatenate([np.arange(1, strain_increment_count + 1),
                                     strain_increment_count - np.arange(1, 2 * strain_increment_count + 1),
                                     np.arange(1 - strain_increment_count, 1)]) / strain_increment_count
else:
    raise ValueError(f"load_path_shape must be \"monotonic\" or \"cyclic\". Got {load_path_shape!r}.")
applied_macro_strain = np.outer(load_fractions, max_macro_strain)
load_step_count = len(applied_macro_strain)

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

if matched_stiffness_control and not include_tfa_model:
    raise ValueError("matched_stiffness_control checks the LS model against TFA, so it needs include_tfa_model = True.")

if reference_stiffness not in ("homogenized", "voigt", "matrix"):
    raise ValueError("reference_stiffness must be \"homogenized\", \"voigt\" or \"matrix\". "
                     f"Got {reference_stiffness!r}.")

# The inputs that define the physical problem and its load path. They are saved with every run's results, and results
# from two runs are only compared when all of these match.
problem_parameter_names = ("domain_side_length", "inclusion_shape", "inclusion_side_length", "inclusion_radius",
                           "element_number_per_side", "element_number_along_z", "partition_number_per_side",
                           "elastic_modulus_inclusion", "elastic_modulus_matrix", "poisson_ratio_inclusion",
                           "poisson_ratio_matrix", "inclusion_yield_stress", "matrix_yield_stress",
                           "inclusion_hardening_modulus", "matrix_hardening_modulus", "matched_stiffness_control",
                           "reference_stiffness", "max_macro_strain", "strain_increment_count", "load_path_shape")

# Shared by every entry point. Cache files are named by a hash of the parameters that produced them, and output files
# are prefixed with the entry point's run name.
cache_folder = pathlib.Path(__file__).parent / "cache"
output_folder = pathlib.Path(__file__).parent / "output"
