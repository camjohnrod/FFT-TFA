# Every setting of the code, and the only file to edit to change a run. Every other module reads these as
# config.<name>. Lengths are in m, stresses and moduli in Pa.
#
#   python code/run.py      solve the problem below with every method switched on, print a timing table, and save it
#                           with the figures in code/output/
#   python code/check.py    check the code on small problems with known answers (about 15 seconds)
#
# From the settings changed most to those changed least:
#   Problem         the inclusion, the grids, the load path
#   Methods         which run, their material states (clusters), CP's reference stiffness C0
#   Figures         colour scales of the maps
#   Materials       the two phases
#   Solver details  CP and Spectral LS side by side, k-means, TFA's offline step, the shared Newton–GMRES settings
#   Checks          check.py's tolerances
# A setting of one method carries its prefix (cp_, spectral_ls_, tfa_, k_means_); the same setting for two methods
# sits as a pair. The three grids mean different things to each method:
#
#                 elements                     partitions                         clusters
#   CP            its kernel, offline only     its lattice: sets its online cost  cp_clusters
#   Spectral LS   its lattice: sets all cost   not used                           one per element
#   TFA           its FE mesh: offline cost    only if tfa_clusters uses them     tfa_clusters
#   k-means       its features, offline        not used                           for cp/tfa_clusters = "k-means"
#
#   elements    element_number_per_side (and element_number_along_z): the fine mesh, which resolves the geometry
#   partitions  partition_number_per_side: CP's uniform lattice of blocks of elements
#   clusters    the material states: regions of one phase with one uniform plastic eigenstrain each
# So CP's online cost is set by its partitions and clusters, not by the mesh.

import os
import pathlib

# The number of CPU threads every method may use, for its dense products (BLAS: TFA's P) and its FFTs (CP and Spectral
# LS) alike, so no method gets more cores than another. It is set here, before numpy loads, which is why run.py and
# check.py import config first; a session that imports numpy before config keeps numpy's default.
thread_count                = 1

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = str(thread_count)

import numpy as np  # noqa: E402, after the thread count is set

## ------- Problem ------- ##

# A periodic cubic cell of side domain_side_length. inclusion_shape is one of
#   "circle"      a centred circular fiber of radius inclusion_radius, running through the thickness
#   "laminate"    a centred fiber layer of thickness layer_thickness, with interfaces normal to x (check.py uses it: its
#                 solution is known exactly)
#   "laminate_z"  the same layer with interfaces normal to z (check.py's test of the 3D kernel)
#   "sphere"      a centred spherical inclusion of radius inclusion_radius
inclusion_shape             = "sphere"
inclusion_radius            = 0.35e-3
layer_thickness             = 0.5e-3
domain_side_length          = 1e-3

# The grids (table above). The mesh resolves the geometry: a shape whose geometry does not vary along z (circle,
# laminate) gets element_number_per_side² × element_number_along_z hexahedra (one layer is exact) and
# partition_number_per_side² partitions, columns through the thickness; any other shape element_number_per_side³
# hexahedra and partition_number_per_side³ partitions. partition_number_per_side must divide element_number_per_side.
element_number_per_side     = 18
element_number_along_z      = 6
partition_number_per_side   = 9

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

# CP, the method under development, always runs: the partition-averaged Lippmann–Schwinger (LS) equation on the
# partitions, which interact through a homogeneous reference medium C0 by an FFT convolution (notes/2_ls_model.md),
# one polarization σ − C0 ε per partition. The comparisons can be switched off:
#   include_spectral_ls  Spectral LS, the same equation with one partition per element: the Moulinec–Suquet spectral
#                        method, the finest lattice the mesh allows
#   include_tfa          TFA, the baseline (notes/3_tfa_baseline.md): clusters interact through the composite's exact
#                        FE influence functions E and P, built offline on the mesh and cached in code/cache/
include_spectral_ls         = True
include_tfa                 = True

# The material states (clusters) of CP and of TFA, chosen independently (Spectral LS keeps one per element), each
#   "partition phases"  each partition's part in each phase: one cluster for a single-phase partition, two for a mixed
#                       one
#   "k-means"           k-means clusters of the elements' elastic strain concentration (Liu–Bessa–Liu 2016, section
#                       2.2.1), found in each phase apart, so a cluster may span many partitions; CP keeps its
#                       partitions, and the clusters set which eigenstrain each part of a partition carries (LS-25)
# Methods both on k-means share the same clusters: at most k_means_matrix_clusters in the matrix and
# k_means_inclusion_clusters in the inclusion (fewer when a phase has fewer distinct strain concentrations, or k-means
# leaves one empty). An elastic phase's eigenstrain is always zero, so one inclusion cluster is enough unless it
# yields: more only show its stress in more detail. TFA's P has (6 × its clusters)² doubles: with partition phases
# 0.3 GB at 32² partitions and 4.8 GB at 64², far less with k-means.
cp_clusters                 = "partition phases"
tfa_clusters                = "partition phases"
k_means_matrix_clusters     = 16
k_means_inclusion_clusters  = 1

# CP's reference stiffness, part of its model:
#
#   C0 = cp_reference_stiffness_scale × (the stiffness named by cp_reference_stiffness)
#
# so "matrix" with 0.01 gives one hundredth of the matrix stiffness. cp_reference_stiffness is one of
#   "matrix"       the matrix stiffness
#   "voigt"        the Voigt average <L> over the clusters
#   "homogenized"  the homogenized stiffness of the composite, <L E> over the clusters; it needs the FE influence
#                  function E, so it adds an FE solve even when TFA is off
# One C0 serves the whole model (LS-7 and LS-25): the interaction between partitions (the kernel P0), and how a
# partition's strain is shared among its parts at equal polarization (the polarization is constant in every
# partition). C0 must stay below the tangent stiffness of every cluster sharing a partition all along the load path,
# or the clusters' eigenstrains (LS-26) have no solution: a yielding phase's shear tangent drops to about
# G H / (3G + H), and to zero without hardening. A smaller C0 shares the strain closer to equal stress. Spectral LS's
# reference stiffness is only a solver setting (Solver details).
cp_reference_stiffness      = "matrix"
cp_reference_stiffness_scale = 0.05

## ------- Figures ------- ##

# Colour-scale limits of the von Mises stress maps (MPa) and the accumulated plastic strain maps (%), so that maps from
# different runs can share a scale. None takes that limit from the plotted maps; for plastic strain, from the Spectral
# LS map alone (every map when Spectral LS is off), so the other methods are read on its scale. With
# von_mises_matrix_only the von Mises maps of every method show the matrix alone, the inclusion greyed out, so limits
# left at None come from the matrix, whose stresses the much stiffer inclusion's would otherwise dwarf.
von_mises_min_MPa           = None
von_mises_max_MPa           = 2.0
von_mises_matrix_only       = False
plastic_strain_min_percent  = None
plastic_strain_max_percent  = None

## ------- Materials ------- ##

# Isotropic elasticity and J2 plasticity with linear isotropic hardening. A yield stress of np.inf keeps a phase
# elastic. A phase that can yield needs a hardening modulus above -3G.
elastic_modulus_matrix      = 100e6
elastic_modulus_inclusion   = 10e9
poisson_ratio_matrix        = 0.19
poisson_ratio_inclusion     = 0.3
matrix_yield_stress         = 1.0e6
inclusion_yield_stress      = np.inf
matrix_hardening_modulus    = 10e6
inclusion_hardening_modulus = 0.0

## ------- Solver Details: CP and Spectral LS ------- ##

# Spectral LS's reference stiffness, a solver setting only: its solution does not depend on it (its kernel is a
# projection), unlike CP's C0. One of
#   "midpoint"  Moulinec–Suquet's rule (1998, section 3.1): the phases' Lamé constants averaged, (L_m + L_f) / 2
#   "matrix"    the matrix stiffness
spectral_ls_reference_stiffness = "midpoint"

# The Fourier preconditioner of each one's GMRES: the best Fourier-diagonal approximation of its Newton Jacobian, i.e.
# the Jacobian of a homogeneous cell with the mean current tangent, rebuilt at every Newton correction
# (notes/2_ls_model.md section 7.4). It changes how many GMRES iterations a Newton correction takes, not the
# correction, so not the answer. It pays off for CP, whose small C0 stretches the Jacobian's spectrum; for Spectral LS
# it changes nothing but costs one more FFT per GMRES iteration, since its reference medium already plays that role.
cp_fourier_preconditioner   = True
spectral_ls_fourier_preconditioner = False

# CP's eigenstrains of clusters sharing a partition are coupled (LS-26), and solved at every residual evaluation by
# Newton until ‖L (μ − μ(ε))‖ is below cp_eigenstrain_tolerance times the largest cluster stress, within
# cp_eigenstrain_max_iterations.
cp_eigenstrain_tolerance    = 1e-12
cp_eigenstrain_max_iterations = 50

## ------- Solver Details: k-means and TFA ------- ##

# k-means runs k_means_iterations Lloyd iterations from a k-means++ start seeded by k_means_seed, on features from six
# elastic Spectral LS solves on the mesh.
k_means_iterations          = 50
k_means_seed                = 0

# TFA's E and P are solved this many load columns at a time (memory against speed); E also serves CP's homogenized C0.
# Building them factorizes the FE stiffness once, which in 3D limits the mesh (a 24³ mesh works with 7 GB of memory,
# 36³ does not).
fe_columns_per_solve        = 100

## ------- Solver Details: Newton–GMRES (every method) ------- ##

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
# Each Newton correction goes through a backtracking line search: the full step if it lowers the relative residual by
# at least the fraction line_search_sufficient_decrease of the step length, otherwise the step times
# line_search_step_reduction, at most line_search_max_reductions times (the shortest step is taken if none does).
# Where full Newton steps work, the line search changes nothing.
line_search_sufficient_decrease = 1e-4
line_search_step_reduction  = 0.5
line_search_max_reductions  = 10
# A method gives up the rest of the load path when its relative residual exceeds its divergence limit times the
# larger of 1 and the step's first relative residual: ls_divergence_limit for CP and Spectral LS, tfa_divergence_limit
# for TFA.
ls_divergence_limit         = 1e3
tfa_divergence_limit        = 1.0

## ------- Checks (check.py) ------- ##

# On a laminate every method must reproduce the exact solution to laminate_tolerance, relative to each phase's
# largest stress. It is about 25 times the largest difference measured at convergence_tolerance = 1e-6. The exact
# solution is found by Newton to laminate_solver_tolerance (relative traction jump) within
# laminate_solver_max_iterations.
laminate_tolerance          = 1e-5
laminate_solver_tolerance   = 1e-11
laminate_solver_max_iterations = 50
# CP's piece law must match LS-25 computed independently, piece by piece (equal polarization in a partition, strains
# averaging to the partition's; notes/2_ls_model.md section 4.5), to this relative tolerance.
piece_law_check_tolerance   = 1e-9


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
    if cp_reference_stiffness not in ("matrix", "voigt", "homogenized"):
        raise ValueError("cp_reference_stiffness must be \"matrix\", \"voigt\" or \"homogenized\". "
                         f"Got {cp_reference_stiffness!r}.")
    if not cp_reference_stiffness_scale > 0:
        raise ValueError(f"cp_reference_stiffness_scale must be positive. Got {cp_reference_stiffness_scale}.")
    for method, setting in (("cp_clusters", cp_clusters), ("tfa_clusters", tfa_clusters)):
        if setting not in ("partition phases", "k-means"):
            raise ValueError(f"{method} must be \"partition phases\" or \"k-means\". Got {setting!r}.")
    if min(k_means_matrix_clusters, k_means_inclusion_clusters, k_means_iterations) < 1:
        raise ValueError("The k-means cluster counts and iteration count must all be at least 1.")
    if spectral_ls_reference_stiffness not in ("midpoint", "matrix"):
        raise ValueError("spectral_ls_reference_stiffness must be \"midpoint\" or \"matrix\". "
                         f"Got {spectral_ls_reference_stiffness!r}.")
    for name, minimum, maximum in (("von Mises", von_mises_min_MPa, von_mises_max_MPa),
                                   ("plastic strain", plastic_strain_min_percent, plastic_strain_max_percent)):
        if minimum is not None and maximum is not None and minimum >= maximum:
            raise ValueError(f"The {name} colour-scale minimum must be below its maximum.")

check_inputs()
