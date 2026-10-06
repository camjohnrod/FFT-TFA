# Every input to a run, then the checks on them and the quantities derived from them. This is the only place a run's
# configuration is changed; every other module reads it as config.<name>.

import pathlib
import numpy as np

## ------- Inputs ------- ##

# Geometry, mesh and partitions (lengths in m). The periodic cell is meshed with element_number_per_side² ×
# element_number_along_z hexahedra and divided into partition_number_per_side² partitions, columns of elements through
# the thickness; elements per side must be a multiple of partitions per side. inclusion_shape is "square" or
# "circle", centred and running through the thickness, or "laminate": a centred inclusion layer of thickness
# inclusion_side_length with interfaces normal to x, whose exact solution every solver is checked against
# (laminate.py). The geometry is resolved on the elements, and every partition must be a single phase, or the run
# stops before any solve: a square or laminate must have its edges on partition edges (as 0.5e-3 does on 32
# partitions), and a circle needs one partition per element.
domain_side_length          = 1e-3
inclusion_shape             = "square"
inclusion_side_length       = 0.5e-3
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
# include_non_partitioned_ls adds the same LS scheme on two other lattices, in MAIN_Newton.py only (MAIN_FP.py ignores
# it, since the LS fixed point at fine resolution takes too long):
#   "LS Newton, fine"    one partition per element column: the finest lattice the mesh allows
#   "LS Newton, coarse"  one element per partition: as cheap as partitioned LS, but without the fine mesh's information
include_tfa_model           = True
include_non_partitioned_ls  = True

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

# Verification before solving, by both entry points (the Jacobian checks by MAIN_Newton.py only).
verification_enabled        = True
verification_tolerance      = 1e-9
jacobian_check_tolerance    = 1e-6
verification_max_partitions = 256

# Model checks after solving. LS against TFA under matched_stiffness_control, and every solver against the laminate's
# exact solution, macroscopically and in every partition, each phase on its own stress scale. laminate_tolerance is
# about 25 times the largest difference measured at convergence_tolerance = 1e-6: scale it with convergence_tolerance.
matched_stiffness_tolerance = 1e-5
laminate_tolerance          = 1e-5

# Every solver runs this many full load paths, interleaved, and the reports take the fastest repeat per step.
timing_repeat_count         = 3

# Colour-scale limits (MPa) of the von Mises stress maps, so maps from different runs can share a scale. None takes
# that limit from the plotted maps.
von_mises_min_MPa           = None
von_mises_max_MPa           = None

## ------- Checks ------- ##

if min(element_number_per_side, element_number_along_z, partition_number_per_side, strain_increment_count,
       timing_repeat_count) < 1:
    raise ValueError("The element, partition, strain increment and timing repeat counts must all be at least 1.")
if element_number_per_side % partition_number_per_side != 0:
    raise ValueError("The number of elements per side must be divisible by the number of partitions per side.")

# The geometry is resolved on the elements (offline.get_element_column_material_ids), and offline.py stops the run if
# a partition holds both phases. Here only the inclusion's size is checked.
if inclusion_shape in ("square", "laminate"):
    if not 0 < inclusion_side_length < domain_side_length:
        raise ValueError("The inclusion's side (the layer's thickness for the laminate) must be positive and below the "
                         f"domain side, so matrix surrounds it. Got {inclusion_side_length}.")
elif inclusion_shape == "circle":
    if not 0 < inclusion_radius < domain_side_length / 2:
        raise ValueError("The circular inclusion's radius must be positive and below half the domain side, so "
                         f"neighbouring inclusions do not touch. Got {inclusion_radius}.")
else:
    raise ValueError(f"inclusion_shape must be \"square\", \"circle\" or \"laminate\". Got {inclusion_shape!r}.")

# The radial return divides by 3G + H. At or below zero it has no admissible solution: the plastic multiplier is
# infinite or negative, so plastic work runs backwards while the flow stress stays positive, and nothing downstream
# would notice. Checked for every phase that can yield, with the stiffness it actually gets.
matrix_shear_modulus = elastic_modulus_matrix / (2 * (1 + poisson_ratio_matrix))
inclusion_shear_modulus = (matrix_shear_modulus if matched_stiffness_control
                           else elastic_modulus_inclusion / (2 * (1 + poisson_ratio_inclusion)))
for phase, shear_modulus, yield_stress, hardening_modulus in (
        ("matrix", matrix_shear_modulus, matrix_yield_stress, matrix_hardening_modulus),
        ("inclusion", inclusion_shear_modulus, inclusion_yield_stress, inclusion_hardening_modulus)):
    if np.isfinite(yield_stress) and 3 * shear_modulus + hardening_modulus <= 0:
        raise ValueError(f"The {phase} hardening modulus must be above -3G = {-3 * shear_modulus:.4g} Pa, or the "
                         f"return map has no admissible solution. Got {hardening_modulus:.4g} Pa.")

if load_path_shape not in ("monotonic", "cyclic"):
    raise ValueError(f"load_path_shape must be \"monotonic\" or \"cyclic\". Got {load_path_shape!r}.")
if matched_stiffness_control and not include_tfa_model:
    raise ValueError("matched_stiffness_control checks the LS model against TFA, so it needs include_tfa_model = True.")
if reference_stiffness not in ("homogenized", "voigt", "matrix"):
    raise ValueError("reference_stiffness must be \"homogenized\", \"voigt\" or \"matrix\". "
                     f"Got {reference_stiffness!r}.")
if von_mises_min_MPa is not None and von_mises_max_MPa is not None and von_mises_min_MPa >= von_mises_max_MPa:
    raise ValueError("The von Mises colour-scale minimum must be below its maximum.")

## ------- Derived Values ------- ##

if load_path_shape == "monotonic":
    load_fractions = np.linspace(1 / strain_increment_count, 1, strain_increment_count)
else:
    load_fractions = np.concatenate([np.arange(1, strain_increment_count + 1),
                                     strain_increment_count - np.arange(1, 2 * strain_increment_count + 1),
                                     np.arange(1 - strain_increment_count, 1)]) / strain_increment_count
applied_macro_strain = np.outer(load_fractions, max_macro_strain)
load_step_count = len(applied_macro_strain)

# The main lattice's size, for the checks and reports about it. Every other size belongs to a discretization
# (offline.Discretization), so that one run can hold several.
partition_count = partition_number_per_side**2

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
