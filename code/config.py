# Every setting of the code, and the only file to edit to change a run. Every other module reads these as
# config.<name>. Lengths are in m, stresses and moduli in Pa.
#
#   python code/run.py      solve the problem below with every method switched on, print a timing table, and save it
#                           with the figures in code/output/
#   python code/check.py    check the code on small problems with known answers (a few seconds)

import os
import pathlib

## ------- Threads ------- ##

# The number of CPU threads every method may use, for its dense products (BLAS: TFA's P) and its FFTs (CP and Spectral
# LS) alike, so no method gets more cores than another. It is set here, before numpy loads, which is why run.py and
# check.py import config first; a session that imports numpy before config keeps numpy's default.
thread_count                = 1

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = str(thread_count)

import numpy as np  # noqa: E402, after the thread count is set

## ------- Geometry and Mesh ------- ##

# A periodic cubic cell of side domain_side_length. inclusion_shape is one of
#   "circle"      a centred circular fiber of radius inclusion_radius, running through the thickness
#   "laminate"    a centred fiber layer of thickness layer_thickness, with interfaces normal to x (check.py uses it: its
#                 solution is known exactly)
#   "laminate_z"  the same layer with interfaces normal to z (check.py's test of the 3D kernel)
#   "sphere"      a centred spherical inclusion of radius inclusion_radius
# A shape whose geometry does not vary along z (circle, laminate) is meshed with element_number_per_side² ×
# element_number_along_z hexahedra and divided into partition_number_per_side² partitions, each a column of elements
# through the thickness. Any other shape is meshed with element_number_per_side³ hexahedra (element_number_along_z is
# not used) and divided into partition_number_per_side³ partitions. element_number_per_side must be a multiple of
# partition_number_per_side. The geometry is resolved on the elements. A partition may hold both phases: its part in
# each phase is a cluster, with its own material state.
domain_side_length          = 1e-3
inclusion_shape             = "sphere"
inclusion_radius            = 0.35e-3
layer_thickness             = 0.5e-3
element_number_per_side     = 18
element_number_along_z      = 6
partition_number_per_side   = 9

## ------- Materials ------- ##

# Isotropic elasticity and J2 plasticity with linear isotropic hardening. A yield stress of np.inf keeps a phase
# elastic. A phase that can yield needs a hardening modulus above -3G.
elastic_modulus_inclusion   = 10e9
elastic_modulus_matrix      = 100e6
poisson_ratio_inclusion     = 0.3
poisson_ratio_matrix        = 0.19
inclusion_yield_stress      = np.inf
matrix_yield_stress         = 1.0e6
inclusion_hardening_modulus = 0.0
matrix_hardening_modulus    = 10e6

## ------- Load Path ------- ##

# The macroscopic strain, in engineering notation (ε11, ε22, ε33, γ12, γ23, γ13), reaches max_macro_strain in
# strain_increment_count equal steps. load_path_shape is
#   "monotonic"  0 -> max, strain_increment_count steps
#   "cyclic"     0 -> +max -> -max -> 0, 4 × strain_increment_count steps: unloading, reverse yielding and passing
#                through zero imposed strain
# Examples: np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0]) uniaxial, np.array([0.03, 0.018, 0.001, 0.0, 0.0, 0.0]) mixed
# tension, np.array([0.015, 0.020, 0.0, 0.03, 0.0, 0.0]) mixed with shear.
max_macro_strain            = np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0])
strain_increment_count      = 60
load_path_shape             = "monotonic"

## ------- Methods ------- ##

# CP is always solved: the partition-averaged Lippmann–Schwinger (LS) equation on the partitions, whose partitions
# interact through the homogeneous reference medium C0 by an FFT convolution with the Moulinec–Suquet kernel P0
# (notes/2_ls_model.md). The two other methods can be switched off:
#   include_spectral_ls  the same LS equation with one partition per element column: the Moulinec–Suquet spectral
#                        method on the element grid, the finest lattice the mesh allows
#   include_tfa_model    TFA, the baseline (notes/3_tfa_baseline.md): partitions interact through the exact FE
#                        influence function P of the composite, a dense matrix of (6 × partitions)² doubles, 0.3 GB at
#                        32² partitions and 4.8 GB at 64², built once and then cached
include_spectral_ls         = True
include_tfa_model           = True

# The reference stiffness C0 of CP:
#
#   C0 = reference_stiffness_scale × (the stiffness named by reference_stiffness)
#
# so "matrix" with 0.01 gives one hundredth of the matrix stiffness. reference_stiffness is one of
#   "matrix"       the matrix stiffness
#   "voigt"        the Voigt average <L> over the partitions
#   "homogenized"  the homogenized stiffness of the composite, <L E> over the partitions; it needs the FE influence
#                  function E, so it adds an FE solve even when TFA is off
# One C0 serves the whole model (notes/2_ls_model.md, LS-7 and LS-22): the interaction between partitions (the kernel
# P0), and how a partition holding both phases shares its strain between them, at equal polarization σ − C0 ε (the
# polarization is constant in every partition). C0 must stay below every phase's tangent stiffness all along the load
# path, or that sharing has no solution: a yielding phase's shear tangent drops to about G H / (3G + H), and to zero
# without hardening. A smaller C0 shares the strain closer to equal stress. TFA does not use C0.
reference_stiffness         = "matrix"
reference_stiffness_scale   = 0.05

# The reference stiffness of Spectral LS, its own: its solution does not depend on it (its kernel is a projection), so
# it is only a solver setting, unlike CP's C0, which is part of CP's model. One of
#   "midpoint"  Moulinec–Suquet's rule (1998, section 3.1): the phases' Lamé constants averaged, (L_m + L_f) / 2
#   "matrix"    the matrix stiffness
spectral_ls_reference_stiffness = "midpoint"

## ------- Newton–GMRES Solver ------- ##

# Every method solves each load step by Newton's method, each Newton correction by GMRES, with the same settings.
# A step has converged when its relative residual (the strain residual over the imposed strain, 3_tfa_baseline.md
# section 12) is below convergence_tolerance; the imposed strain is never taken smaller than
# residual_strain_scale_floor, which matters near zero imposed strain. CP's and Spectral LS's residual is a strain of
# their reference medium, which would make the test depend on C0; it is judged as the strain correction an all-matrix
# cell would need to cancel it (notes/2_ls_model.md section 7.5), so the tolerance means the same for any C0.
convergence_tolerance       = 1e-6
residual_strain_scale_floor = 1e-4
newton_max_iterations       = 50
# GMRES stops when its linear residual falls below newton_krylov_tolerance times the Newton residual. It restarts
# every newton_krylov_restart iterations, at most newton_krylov_max_restarts times (None: scipy's default, ten times
# the number of unknowns).
newton_krylov_tolerance     = 1e-3
newton_krylov_restart       = 50
newton_krylov_max_restarts  = None
# The Fourier preconditioner of CP's and Spectral LS's GMRES, the same for both: the best Fourier-diagonal
# approximation of the method's own Newton Jacobian, i.e. the Jacobian of a homogeneous cell with the mean current
# tangent, rebuilt at every Newton correction (notes/2_ls_model.md section 7.4). It changes how many GMRES iterations
# a Newton correction takes, not the correction, so not the answer. It pays off for CP, whose small C0 stretches the
# Jacobian's spectrum; for Spectral LS it changes nothing but costs one more FFT per GMRES iteration, since its
# reference medium already plays that role.
cp_fourier_preconditioner   = True
spectral_ls_fourier_preconditioner = False
# Each Newton correction goes through a backtracking line search: the full step if it lowers the relative residual by
# at least the fraction line_search_sufficient_decrease of the step length, otherwise the step times
# line_search_step_reduction, at most line_search_max_reductions times (the shortest step is taken if none does).
# Where full Newton steps work, the line search changes nothing.
line_search_sufficient_decrease = 1e-4
line_search_step_reduction  = 0.5
line_search_max_reductions  = 10
# A method gives up the rest of the load path when its relative residual exceeds its divergence limit times the
# larger of 1 and the step's first relative residual. ls_divergence_limit is for CP and Spectral LS.
ls_divergence_limit         = 1e3
tfa_divergence_limit        = 1.0

# In a partition holding both phases, the partition strain is shared between the matrix and fiber clusters so that
# both carry the same polarization σ − C0 ε, found by Newton until the polarization difference is below
# mixed_partition_tolerance times the largest cluster stress, within mixed_partition_max_iterations.
mixed_partition_tolerance   = 1e-12
mixed_partition_max_iterations = 50

## ------- Offline Step ------- ##

# The FE influence functions E and P are solved this many load columns at a time (memory against speed). They are
# cached in code/cache/, so a configuration that was already solved loads them instead.
offline_columns_per_solve   = 100

## ------- Figures ------- ##

# Colour-scale limits of the von Mises stress maps (MPa) and the accumulated plastic strain maps (%), so that maps from
# different runs can share a scale. None takes that limit from the plotted maps; for plastic strain, from the Spectral
# LS map alone (every map when Spectral LS is off), so the other methods are read on its scale.
von_mises_min_MPa           = None
von_mises_max_MPa           = 2.0
plastic_strain_min_percent  = None
plastic_strain_max_percent  = None

## ------- Checks (check.py) ------- ##

# On a laminate every method must reproduce the exact solution to laminate_tolerance, relative to each phase's
# largest stress. It is about 25 times the largest difference measured at convergence_tolerance = 1e-6. The exact
# solution is found by Newton to laminate_solver_tolerance (relative traction jump) within
# laminate_solver_max_iterations.
laminate_tolerance          = 1e-5
laminate_solver_tolerance   = 1e-11
laminate_solver_max_iterations = 50
# In elasticity a mixed partition's stress and tangent must match their closed form (notes/2_ls_model.md section 4.4)
# to this relative tolerance (the split itself converges to mixed_partition_tolerance).
mixed_partition_check_tolerance = 1e-9

## ------- Folders ------- ##

cache_folder = pathlib.Path(__file__).parent / "cache"
output_folder = pathlib.Path(__file__).parent / "output"

## ------- Input Checks ------- ##

def check_inputs():
    # Stops on an input the code cannot run with. check.py calls it again after setting its own inputs.
    if min(element_number_per_side, element_number_along_z, partition_number_per_side, strain_increment_count) < 1:
        raise ValueError("The element, partition and strain increment counts must all be at least 1.")
    if element_number_per_side % partition_number_per_side != 0:
        raise ValueError("The number of elements per side must be a multiple of the number of partitions per side.")

    if inclusion_shape in ("circle", "sphere"):
        if not 0 < inclusion_radius < domain_side_length / 2:
            raise ValueError("The inclusion's radius must be positive and below half the domain side, so neighbouring "
                             f"inclusions do not touch. Got {inclusion_radius}.")
    elif inclusion_shape in ("laminate", "laminate_z"):
        if not 0 < layer_thickness < domain_side_length:
            raise ValueError("The fiber layer's thickness must be positive and below the domain side, so matrix "
                             f"surrounds it. Got {layer_thickness}.")
    else:
        raise ValueError("inclusion_shape must be \"circle\", \"laminate\", \"laminate_z\" or \"sphere\". Got "
                         f"{inclusion_shape!r}.")

    # The radial return divides by 3G + H, so at or below zero it has no admissible solution.
    for phase, elastic_modulus, poisson_ratio, yield_stress, hardening_modulus in (
            ("matrix", elastic_modulus_matrix, poisson_ratio_matrix, matrix_yield_stress, matrix_hardening_modulus),
            ("inclusion", elastic_modulus_inclusion, poisson_ratio_inclusion, inclusion_yield_stress,
             inclusion_hardening_modulus)):
        shear_modulus = elastic_modulus / (2 * (1 + poisson_ratio))
        if np.isfinite(yield_stress) and 3 * shear_modulus + hardening_modulus <= 0:
            raise ValueError(f"The {phase} hardening modulus must be above -3G = {-3 * shear_modulus:.4g} Pa, or the "
                             f"return map has no admissible solution. Got {hardening_modulus:.4g} Pa.")

    if load_path_shape not in ("monotonic", "cyclic"):
        raise ValueError(f"load_path_shape must be \"monotonic\" or \"cyclic\". Got {load_path_shape!r}.")
    if reference_stiffness not in ("matrix", "voigt", "homogenized"):
        raise ValueError("reference_stiffness must be \"matrix\", \"voigt\" or \"homogenized\". "
                         f"Got {reference_stiffness!r}.")
    if not reference_stiffness_scale > 0:
        raise ValueError(f"reference_stiffness_scale must be positive. Got {reference_stiffness_scale}.")
    if spectral_ls_reference_stiffness not in ("midpoint", "matrix"):
        raise ValueError("spectral_ls_reference_stiffness must be \"midpoint\" or \"matrix\". "
                         f"Got {spectral_ls_reference_stiffness!r}.")
    for name, minimum, maximum in (("von Mises", von_mises_min_MPa, von_mises_max_MPa),
                                   ("plastic strain", plastic_strain_min_percent, plastic_strain_max_percent)):
        if minimum is not None and maximum is not None and minimum >= maximum:
            raise ValueError(f"The {name} colour-scale minimum must be below its maximum.")

check_inputs()
