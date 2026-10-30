# Every setting of the code, and the only file to edit to change a run. Every other module reads these as
# config.<name>. Lengths are in m, stresses and moduli in Pa.
#
#   python code/run.py      solve the problem below with every method switched on, print a timing table, and save it
#                           with the figures in code/output/
#   python code/check.py    check the code on small problems with known answers (about 15 seconds)
#
# From the settings changed most to those changed least:
#   Problem         the inclusion, the grids, the load path
#   Methods         which run, their material states (clusters), FFT-TFA's neighbour blocks, CP's reference stiffness
#   Figures         colour scales of the maps
#   Materials       the two phases
#   Solver details  CP and Spectral LS side by side, k-means, TFA's offline step, the shared Newton–GMRES settings
#   Checks          check.py's tolerances
# A setting of one method carries its prefix (fft_tfa_, spectral_ls_, tfa_, cp_, k_means_); the same setting for
# several methods sits as a group. The three grids mean different things to each method:
#
#                 elements                        partitions                         clusters
#   FFT-TFA       its kernel and its partitions'  its FFT lattice: sets its online   fft_tfa_clusters
#                 corrections, offline only       cost with its clusters
#   Spectral LS   its lattice: sets all cost      not used                           one per element
#   TFA           its FE mesh: offline cost       only if tfa_clusters uses them     tfa_clusters
#   CP            its kernel, offline only        its lattice: sets its online cost  cp_clusters
#   k-means       its features, offline           only for "partition k-means"       for the k-means settings
#   reference     its own finer mesh (reference_element_number_per_side), solved once and cached
#
#   elements    element_number_per_side (and element_number_along_z): the fine mesh, which resolves the geometry
#   partitions  partition_number_per_side (and partition_number_along_z): a uniform lattice of blocks of elements
#   clusters    the material states: regions of one phase with one uniform plastic eigenstrain each
# So FFT-TFA's online cost is set by its partitions and clusters, not by the mesh.

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
#   "fiber"       a centred circular fiber of radius inclusion_radius, running through the thickness
#   "laminate"    a centred fiber layer of thickness layer_thickness, with interfaces normal to x (check.py uses it: its
#                 solution is known exactly)
#   "laminate_z"  the same layer with interfaces normal to z (check.py's test of the 3D kernel)
#   "sphere"      a centred spherical inclusion of radius inclusion_radius
#   "random fibers"  fiber_count equal circular fibers running through the thickness, together fiber_volume_fraction
#                 of the cell, placed by random sequential addition: uniformly random centres (seeded by fiber_seed),
#                 each kept if its fiber stays fiber_gap_elements element widths clear of every fiber kept so far,
#                 measured across the periodic cell (a fiber crossing a side of the cell continues on the opposite
#                 side). It jams near 55 % of the cell for discs, less with a gap: placing stops with an error after
#                 fiber_placement_attempts random centres.
inclusion_shape             = "random fibers"
inclusion_radius            = 0.35e-3
layer_thickness             = 0.5e-3
fiber_count                 = 30
fiber_volume_fraction       = 0.35
fiber_gap_elements          = 2
fiber_seed                  = 0
fiber_placement_attempts    = 100000
domain_side_length          = 1e-3

# The grids (table above). The mesh resolves the geometry: a shape whose geometry does not vary along z (fiber,
# laminate, random fibers) gets element_number_per_side² × element_number_along_z hexahedra (one layer is exact) and
# partition_number_per_side² × partition_number_along_z partitions (1: columns through the thickness); any other shape
# element_number_per_side³ hexahedra and partition_number_per_side³ partitions, and the two along-z counts are not
# used. partition_number_per_side must divide element_number_per_side, and partition_number_along_z
# element_number_along_z. More partition layers give the same answer as columns, except for FFT-TFA without neighbour
# blocks, whose truncation then changes it (by a few %): keep columns for it.
element_number_per_side     = 128
element_number_along_z      = 1
partition_number_per_side   = 8
partition_number_along_z    = 1

# The macroscopic strain, in engineering notation (ε11, ε22, ε33, γ12, γ23, γ13), reaches max_macro_strain in
# strain_increment_count equal steps. load_path_shape is
#   "monotonic"  0 -> max, strain_increment_count steps
#   "cyclic"     0 -> +max -> -max -> 0, 4 × strain_increment_count steps: unloading, reverse yielding and passing
#                through zero imposed strain
# Examples: np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0]) uniaxial, np.array([0.03, 0.018, 0.001, 0.0, 0.0, 0.0]) mixed
# tension, np.array([0.015, 0.020, 0.0, 0.03, 0.0, 0.0]) mixed with shear.
max_macro_strain            = np.array([0.015, 0.020, 0.0, 0.03, 0.0, 0.0])
strain_increment_count      = 30
load_path_shape             = "cyclic"

## ------- Methods ------- ##

# FFT-TFA, the method under development, always runs (notes/2_ls_model.md section 10): TFA's clusters, one uniform
# eigenstrain each, interacting through the Lippmann–Schwinger (LS) equation of a homogeneous medium of the matrix's
# stiffness, applied by an FFT on the partitions plus exact corrections between nearby clusters. The comparisons can
# be switched off:
#   include_spectral_ls  Spectral LS, the LS equation on the elements: the Moulinec–Suquet spectral method, the finest
#                        lattice the mesh allows
#   include_tfa          TFA, the baseline (notes/3_tfa_baseline.md): clusters interact through the composite's exact
#                        FE influence functions E and P, built offline on the mesh and cached in code/cache/
#   include_cp           CP, the earlier method: the partition-averaged LS equation with one polarization σ − C0 ε per
#                        partition and its own reference stiffness C0 (below)
include_spectral_ls         = True
include_tfa                 = True
include_cp                  = False

# A reference to compare the methods with, not one of them: Spectral LS on a finer mesh of
# reference_element_number_per_side elements per side (a multiple of element_number_per_side; one element layer for a
# shape constant along z, cubes otherwise), on the same geometry, materials and load path. It is solved once and
# cached in code/cache/ with its macroscopic stress at every load step and its fields at the last one; any change of
# those inputs solves it again. It shows in the figures, and is not timed against the methods. A first solve is long
# (about 30 min for 540² elements and 120 load steps, 15 min for 96³ and 10 steps): run it in the background.
include_reference           = True
reference_element_number_per_side = 256

# The material states (clusters) of FFT-TFA, TFA and CP, chosen independently (Spectral LS keeps one per element), each
#   "partition phases"   each partition's part in each phase: one cluster for a single-phase partition, two for a
#                        mixed one
#   "partition k-means"  k-means clusters inside each partition's part in each phase, on the elements' elastic strain
#                        concentration: up to k_means_matrix_clusters_per_partition in its matrix part and
#                        k_means_inclusion_clusters_per_partition in its inclusion part; one each is "partition phases"
#   "k-means"            k-means clusters on the same features over the whole cell (Liu–Bessa–Liu 2016, section
#                        2.2.1), found in each phase apart, so a cluster may span many partitions: at most
#                        k_means_matrix_clusters in the matrix and k_means_inclusion_clusters in the inclusion
# A part or phase gets fewer clusters when it has fewer distinct strain concentrations, or k-means leaves one empty.
# Methods on the same setting share the same clusters. An elastic phase's eigenstrain is always zero, so one inclusion
# cluster is enough unless it yields: more only show its stress in more detail. TFA's P has (6 × its clusters)²
# doubles: with partition phases 0.3 GB at 32² partitions and 4.8 GB at 64², far less with k-means.
fft_tfa_clusters            = "partition k-means"
tfa_clusters                = "partition k-means"
cp_clusters                 = "k-means"
k_means_matrix_clusters     = 20
k_means_inclusion_clusters  = 1
k_means_matrix_clusters_per_partition = 6
k_means_inclusion_clusters_per_partition = 1

# FFT-TFA keeps the exact interaction between the clusters of each partition. With fft_tfa_neighbour_blocks it keeps
# it also between each partition's clusters and those of its neighbouring partitions (8 in-plane for columns, 26 in
# 3D), at about 9 or 27 times the cost of those corrections. Off, the partitions' FFT alone couples them.
fft_tfa_neighbour_blocks    = True

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
# different runs can share a scale. None takes that limit from the finest field: the reference's map when it is on,
# otherwise Spectral LS's, so the other methods are read on its scale (values beyond it take the end colour, marked on
# the colour bar); from every map when both are off. With von_mises_matrix_only the von Mises maps of every method
# show the matrix alone, the inclusion greyed out, so limits left at None come from the matrix, whose stresses the much
# stiffer inclusion's would otherwise dwarf.
von_mises_min_MPa           = None
von_mises_max_MPa           = None
von_mises_matrix_only       = True
plastic_strain_min_percent  = None
plastic_strain_max_percent  = None

# With the reference on, each method's macroscopic stress error is plotted (macro_stress_error.png) and summarized
# (errors.txt) for every stress component the load path produces: those whose reference value reaches
# loaded_stress_fraction of the largest reference component somewhere along the path. The others are zero up to
# round-off, or a small coupling (the shear of a cell without mirror symmetry) whose relative error says little.
loaded_stress_fraction      = 0.1

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
# FFT-TFA's reference medium is the matrix, so its residual already is that strain, a strain error like TFA's.
convergence_tolerance       = 1e-6
residual_strain_scale_floor = 1e-4
newton_max_iterations       = 50
# GMRES stops when its linear residual falls below newton_krylov_tolerance times the Newton residual. It restarts
# every newton_krylov_restart iterations, at most newton_krylov_max_restarts times (None: scipy's default, ten times
# the number of unknowns).
newton_krylov_tolerance     = 1e-3
newton_krylov_restart       = 100
newton_krylov_max_restarts  = None
# The reference's GMRES restarts more often: it keeps reference_krylov_restart strain fields, 48 bytes per element
# each (at 96³ elements, 100 of them would take 4 GB). The restart changes how many iterations a correction takes,
# not the answer.
reference_krylov_restart    = 30
# Each Newton correction goes through a backtracking line search: the full step if it lowers the relative residual by
# at least the fraction line_search_sufficient_decrease of the step length, otherwise the step times
# line_search_step_reduction, at most line_search_max_reductions times (the shortest step is taken if none does).
# Where full Newton steps work, the line search changes nothing.
line_search_sufficient_decrease = 1e-4
line_search_step_reduction  = 0.5
line_search_max_reductions  = 10
# A method gives up the rest of the load path when its relative residual exceeds its divergence limit times the
# larger of 1 and the step's first relative residual: ls_divergence_limit for the LS methods (FFT-TFA, Spectral LS,
# CP), tfa_divergence_limit for TFA.
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
# FFT-TFA's interaction with every partition in reach must match the element-level LS operator, computed independently
# from the elements, to this relative tolerance (notes/2_ls_model.md section 10.6; measured differences are below
# 1e-14).
interaction_check_tolerance = 1e-12
# Every method, stopped at convergence_tolerance, must lie within stopping_accuracy_tolerance (relative, as
# laminate_tolerance) of its own solution at the tight stopping_check_tolerance, so that no method gains speed by
# stopping less accurately than another. It is about 14 times the largest difference measured at
# convergence_tolerance = 1e-6 (7e-7; every method measured lies between 1e-8 and 7e-7).
stopping_check_tolerance    = 1e-11
stopping_accuracy_tolerance = 1e-5


## ------- Folders ------- ##

cache_folder = pathlib.Path(__file__).parent / "cache"
output_folder = pathlib.Path(__file__).parent / "output"

## ------- Input Checks ------- ##

def check_inputs():
    # Stops on an input the code cannot run with. check.py calls it again after setting its own inputs.
    if min(element_number_per_side, element_number_along_z, partition_number_per_side, partition_number_along_z,
           strain_increment_count) < 1:
        raise ValueError("The element, partition and strain increment counts must all be at least 1.")
    if element_number_per_side % partition_number_per_side != 0:
        raise ValueError("The number of elements per side must be a multiple of the number of partitions per side.")
    if include_reference and (reference_element_number_per_side % element_number_per_side != 0
                              or reference_krylov_restart < 1):
        raise ValueError("reference_element_number_per_side must be a multiple of element_number_per_side, and "
                         "reference_krylov_restart at least 1.")
    if (inclusion_shape in ("fiber", "laminate", "random fibers")
            and element_number_along_z % partition_number_along_z != 0):
        raise ValueError("The number of elements along z must be a multiple of the number of partitions along z.")

    if inclusion_shape in ("fiber", "sphere"):
        if not 0 < inclusion_radius < domain_side_length / 2:
            raise ValueError("The inclusion's radius must be positive and below half the domain side, so neighbouring "
                             f"inclusions do not touch. Got {inclusion_radius}.")
    elif inclusion_shape in ("laminate", "laminate_z"):
        if not 0 < layer_thickness < domain_side_length:
            raise ValueError("The fiber layer's thickness must be positive and below the domain side, so matrix "
                             f"surrounds it. Got {layer_thickness}.")
    elif inclusion_shape == "random fibers":
        if min(fiber_count, fiber_placement_attempts) < 1 or fiber_gap_elements < 0:
            raise ValueError("fiber_count and fiber_placement_attempts must be at least 1, fiber_gap_elements at "
                             "least 0.")
        if not 0 < fiber_volume_fraction < 1:
            raise ValueError(f"fiber_volume_fraction must lie between 0 and 1. Got {fiber_volume_fraction}.")
        # A fiber and its own periodic image must keep the gap too.
        fiber_diameter = 2 * domain_side_length * np.sqrt(fiber_volume_fraction / (np.pi * fiber_count))
        if fiber_diameter + fiber_gap_elements * domain_side_length / element_number_per_side >= domain_side_length:
            raise ValueError("A fiber's diameter plus the gap must be below the domain side, so a fiber does not touch "
                             "its own periodic image.")
    else:
        raise ValueError("inclusion_shape must be \"fiber\", \"laminate\", \"laminate_z\", \"sphere\" or "
                         f"\"random fibers\". Got {inclusion_shape!r}.")

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
    for method, setting in (("fft_tfa_clusters", fft_tfa_clusters), ("tfa_clusters", tfa_clusters),
                            ("cp_clusters", cp_clusters)):
        if setting not in ("partition phases", "partition k-means", "k-means"):
            raise ValueError(f"{method} must be \"partition phases\", \"partition k-means\" or \"k-means\". "
                             f"Got {setting!r}.")
    if min(k_means_matrix_clusters, k_means_inclusion_clusters, k_means_matrix_clusters_per_partition,
           k_means_inclusion_clusters_per_partition, k_means_iterations) < 1:
        raise ValueError("The k-means cluster counts and iteration count must all be at least 1.")
    if spectral_ls_reference_stiffness not in ("midpoint", "matrix"):
        raise ValueError("spectral_ls_reference_stiffness must be \"midpoint\" or \"matrix\". "
                         f"Got {spectral_ls_reference_stiffness!r}.")
    if not 0 < loaded_stress_fraction <= 1:
        raise ValueError(f"loaded_stress_fraction must lie in (0, 1]. Got {loaded_stress_fraction}.")
    for name, minimum, maximum in (("von Mises", von_mises_min_MPa, von_mises_max_MPa),
                                   ("plastic strain", plastic_strain_min_percent, plastic_strain_max_percent)):
        if minimum is not None and maximum is not None and minimum >= maximum:
            raise ValueError(f"The {name} colour-scale minimum must be below its maximum.")

check_inputs()
