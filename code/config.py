# Every setting of the code, and the only file to edit to change a run. Every other module reads these as
# config.<name>. Lengths are in m, stresses and moduli in Pa.
#
#   python code/run.py      solve the problem below with every method switched on, print a timing table, and save it
#                           with the figures in code/output/
#   python code/check.py    check the code on small problems with known answers (about a minute)

import pathlib
import numpy as np

## ------- Geometry and Mesh ------- ##

# A periodic square cell of side domain_side_length, meshed with element_number_per_side² × element_number_along_z
# hexahedra and divided into partition_number_per_side² partitions, each a column of elements through the thickness.
# element_number_per_side must be a multiple of partition_number_per_side. inclusion_shape is one of
#   "square"    a centred square inclusion of side inclusion_side_length
#   "circle"    a centred circular inclusion of radius inclusion_radius
#   "laminate"  a centred inclusion layer of thickness inclusion_side_length, with interfaces normal to x
# Each partition has one material law, so the run stops if a partition holds both phases: a square or laminate needs
# its edges on partition edges (0.5e-3 does on 32 partitions), and a circle needs one partition per element
# (partition_number_per_side = element_number_per_side).
domain_side_length          = 1e-3
inclusion_shape             = "square"
inclusion_side_length       = 0.5e-3
inclusion_radius            = 0.35e-3
element_number_per_side     = 64
element_number_along_z      = 1
partition_number_per_side   = 32

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

# The reference stiffness C0 of CP and Spectral LS. It is part of their model, so it changes their answer; TFA does
# not use it.
#   "matrix"       the matrix stiffness
#   "voigt"        the Voigt average <L> over the partitions
#   "homogenized"  the homogenized stiffness of the composite, <L E> over the partitions; it needs the FE influence
#                  function E, so it adds an FE solve even when TFA is off
reference_stiffness         = "matrix"

## ------- Newton–GMRES Solver ------- ##

# Every method solves each load step by Newton's method, each Newton correction by GMRES, with the same settings.
# A step has converged when its relative residual (the strain residual over the imposed strain, 3_tfa_baseline.md
# section 12) is below convergence_tolerance; the imposed strain is never taken smaller than
# residual_strain_scale_floor, which matters near zero imposed strain.
convergence_tolerance       = 1e-6
residual_strain_scale_floor = 1e-4
newton_max_iterations       = 50
# GMRES stops when its linear residual falls below newton_krylov_tolerance times the Newton residual. It restarts
# every newton_krylov_restart iterations, at most newton_krylov_max_restarts times (None: scipy's default, ten times
# the number of unknowns).
newton_krylov_tolerance     = 1e-3
newton_krylov_restart       = 50
newton_krylov_max_restarts  = None
# A method gives up the rest of the load path when its relative residual exceeds its divergence limit times the
# larger of 1 and the step's first relative residual. ls_divergence_limit is for CP and Spectral LS.
ls_divergence_limit         = 1e3
tfa_divergence_limit        = 1.0

## ------- Offline Step ------- ##

# The FE influence functions E and P are solved this many load columns at a time (memory against speed). They are
# cached in code/cache/, so a configuration that was already solved loads them instead.
offline_columns_per_solve   = 100

## ------- Figures ------- ##

# Colour-scale limits (MPa) of the von Mises stress maps, so that maps from different runs can share a scale. None
# takes that limit from the plotted maps.
von_mises_min_MPa           = None
von_mises_max_MPa           = None

## ------- Checks (check.py) ------- ##

# On a laminate every method must reproduce the exact solution to laminate_tolerance, relative to each phase's
# largest stress. It is about 25 times the largest difference measured at convergence_tolerance = 1e-6. The exact
# solution is found by Newton to laminate_solver_tolerance (relative traction jump) within
# laminate_solver_max_iterations.
laminate_tolerance          = 1e-5
laminate_solver_tolerance   = 1e-11
laminate_solver_max_iterations = 50

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

    if inclusion_shape in ("square", "laminate"):
        if not 0 < inclusion_side_length < domain_side_length:
            raise ValueError("The inclusion's side (the layer's thickness for the laminate) must be positive and below "
                             f"the domain side, so matrix surrounds it. Got {inclusion_side_length}.")
    elif inclusion_shape == "circle":
        if not 0 < inclusion_radius < domain_side_length / 2:
            raise ValueError("The circular inclusion's radius must be positive and below half the domain side, so "
                             f"neighbouring inclusions do not touch. Got {inclusion_radius}.")
    else:
        raise ValueError(f"inclusion_shape must be \"square\", \"circle\" or \"laminate\". Got {inclusion_shape!r}.")

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
    if von_mises_min_MPa is not None and von_mises_max_MPa is not None and von_mises_min_MPa >= von_mises_max_MPa:
        raise ValueError("The von Mises colour-scale minimum must be below its maximum.")

check_inputs()
