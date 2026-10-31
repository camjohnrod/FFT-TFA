# Every setting, and the only file to edit to change a run: other modules read them as config.<name>. Lengths are in
# m, stresses and moduli in Pa.
#
#   python code/run.py      solve the problem below with every method switched on; table and figures to code/output/
#   python code/check.py    check the code on small problems with known answers (about a minute)
#
# Sections run from the settings changed most to those changed least: Problem, Methods, Figures, Materials, Solver
# details, Checks. A setting of one method carries its prefix (fft_tfa_, spectral_ls_, tfa_, cp_, k_means_); the same
# setting for several methods sits as a group. The three grids mean different things to each method:
#
#                 elements                      partitions                         clusters
#   FFT-TFA       its kernel and corrections,   its FFT lattice: with its          fft_tfa_clusters
#                 offline only                  clusters, sets its online cost
#   Spectral LS   its lattice, unless it has    not used                           one per element
#                 its own: sets all cost
#   TFA           its FE mesh: offline cost     only if tfa_clusters uses them     tfa_clusters
#   CP            its kernel, offline only      its lattice: sets its online cost  cp_clusters
#   k-means       its features, offline         only for "partition k-means"       the k_means_ counts
#   reference     its own finer mesh (reference_elements_per_side), solved once and cached
#
#   elements    elements_per_side (and elements_along_z): the fine mesh, which resolves the geometry
#   partitions  partitions_per_side (and partitions_along_z): a uniform lattice of blocks of elements
#   clusters    the material states: regions of one phase with one uniform plastic eigenstrain each

import os
import pathlib

# CPU threads for every method's dense products (BLAS) and FFTs alike. Set here, before numpy loads, which is why
# run.py and check.py import config first.
thread_count                = 1

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = str(thread_count)

import numpy as np  # noqa: E402, after the thread count is set

## ------- Problem ------- ##

# A periodic cubic cell holding one inclusion_shape, each shape's own settings below it.
domain_side_length          = 1e-3
inclusion_shape             = "random fibers"
# "fiber": a centred circular fiber through the thickness. "sphere": a centred sphere.
inclusion_radius            = 0.35e-3
# "laminate": a centred fiber layer with interfaces normal to x; "laminate_z": normal to z. check.py uses both, since
# their solution is known exactly.
layer_thickness             = 0.5e-3
# "random fibers": fiber_count equal fibers through the thickness, together fiber_volume_fraction of the cell, at
# random centres (fiber_seed), each kept if it stays fiber_gap_elements element widths clear of the others across the
# periodic cell. Placing jams near 55 % of the cell, less with a gap.
fiber_count                 = 50
fiber_volume_fraction       = 0.35
fiber_gap_elements          = 2
fiber_seed                  = 0

# The grids (table above). A shape constant along z (fiber, laminate, random fibers) gets elements_per_side² ×
# elements_along_z elements (one layer is exact) and partitions_per_side² × partitions_along_z partitions (1: columns);
# any other shape gets cubes and ignores the along-z counts. Partition counts must divide element counts. Partition
# layers change only FFT-TFA without neighbour blocks (by a few %): keep columns for it. Spectral LS runs on
# spectral_ls_elements_per_side elements per side instead of the mesh's, coarser or finer, unless None (the k-means
# features stay on the mesh).
elements_per_side           = 128
elements_along_z            = 1
partitions_per_side         = 4
partitions_along_z          = 1
spectral_ls_elements_per_side = 64

# The macroscopic strain (ε11, ε22, ε33, γ12, γ23, γ13) reaches max_macro_strain in strain_increment_count equal steps.
# load_path_shape is
#   "monotonic"  0 -> max
#   "cyclic"     0 -> +max -> -max -> 0, 4 × strain_increment_count steps: unloading and reverse yielding
max_macro_strain            = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
strain_increment_count      = 60
load_path_shape             = "monotonic"

## ------- Methods ------- ##

# FFT-TFA, the method under development, always runs (notes/2_ls_model.md section 10): TFA's clusters interacting
# through the Lippmann–Schwinger (LS) equation of the matrix, by an FFT on the partitions plus exact corrections
# between nearby clusters. The comparisons:
#   include_spectral_ls  the LS equation on the elements (Moulinec–Suquet), the finest lattice the mesh allows
#   include_tfa          the baseline (notes/3_tfa_baseline.md): clusters interact through the FE influence functions
#                        E and P, built offline and cached in code/cache/
#   include_cp           the earlier method: the partition-averaged LS equation, one polarization per partition
include_spectral_ls         = True
include_tfa                 = False
include_cp                  = False

# A reference, not a method: Spectral LS on a finer mesh of reference_elements_per_side per side (a multiple of
# elements_per_side), on the same problem. Solved once and cached (macroscopic stress at every step, fields at the
# last), shown in the figures, not timed. A first solve takes about 30 min at 540² elements and 120 steps: run it in
# the background.
include_reference           = True
reference_elements_per_side = 512

# The clusters (material states) of FFT-TFA, TFA and CP, chosen independently (Spectral LS keeps one per element):
#   "partition phases"   each partition's part in each phase
#   "partition k-means"  k-means inside each partition's part in each phase: up to
#                        k_means_matrix_clusters_per_partition in the matrix part and
#                        k_means_inclusion_clusters_per_partition in the inclusion part (one each: "partition phases")
#   "k-means"            k-means over the whole cell, phase by phase (Liu–Bessa–Liu 2016, section 2.2.1): up to
#                        k_means_matrix_clusters and k_means_inclusion_clusters, each possibly spanning many partitions
# k-means groups the elements by elastic strain concentration. Methods on the same setting share clusters. An elastic
# inclusion needs one cluster (more only resolve its stress). TFA's P takes (6 × its clusters)² doubles.
fft_tfa_clusters            = "partition k-means"
tfa_clusters                = "partition k-means"
cp_clusters                 = "k-means"
k_means_matrix_clusters     = 20
k_means_inclusion_clusters  = 1
k_means_matrix_clusters_per_partition = 20
k_means_inclusion_clusters_per_partition = 1

# FFT-TFA keeps the exact interaction among each partition's clusters, and with fft_tfa_neighbour_blocks also with
# those of the neighbouring partitions (8 for columns, 26 in 3D), at about 9 or 27 times the cost of those corrections.
fft_tfa_neighbour_blocks    = True

# CP's reference stiffness C0 is cp_reference_stiffness_scale times
#   "matrix"       the matrix stiffness
#   "voigt"        <L> over the clusters
#   "homogenized"  <L E> over the clusters, which needs TFA's E: an FE solve even with TFA off
# C0 is part of CP's model (LS-7, LS-25): it sets the interaction between partitions and how a partition's strain is
# shared among its clusters (closer to equal stress for a smaller C0). It must stay below the tangent of every cluster
# sharing a partition, or LS-26 has no solution; a yielding phase's shear tangent drops to G H / (3G + H).
cp_reference_stiffness      = "matrix"
cp_reference_stiffness_scale = 0.05

## ------- Figures ------- ##

# Colour-scale ranges (min, max) of the von Mises stress maps (MPa) and the accumulated plastic strain maps (%), to
# share a scale across runs. An end left at None comes from the finest field (the reference, else Spectral LS, else
# every map); values beyond an end take its colour. von_mises_matrix_only greys out the inclusion, whose far larger
# stresses would otherwise set the scale.
von_mises_range_MPa         = (None, None)
von_mises_matrix_only       = True
plastic_strain_range_percent = (None, None)

# With the reference on, the macroscopic stress errors (macro_stress_error.png, errors.txt) cover each stress component
# whose reference value reaches loaded_stress_fraction of the largest one somewhere along the path.
loaded_stress_fraction      = 0.25

## ------- Materials ------- ##

# Isotropic elasticity and J2 plasticity with linear isotropic hardening. A yield stress of np.inf keeps a phase
# elastic; a phase that yields needs a hardening modulus above -3G.
# matrix_elastic_modulus      = 100e6
# matrix_poisson_ratio        = 0.19
# matrix_yield_stress         = 1.0e6
# matrix_hardening_modulus    = 10e6
# inclusion_elastic_modulus   = 10e9
# inclusion_poisson_ratio     = 0.3
# inclusion_yield_stress      = np.inf
# inclusion_hardening_modulus = 0.0

matrix_elastic_modulus      = 2.1e9
matrix_poisson_ratio        = 0.3
matrix_yield_stress         = 29e6
matrix_hardening_modulus    = 139e6
inclusion_elastic_modulus   = 72e9
inclusion_poisson_ratio     = 0.22
inclusion_yield_stress      = np.inf
inclusion_hardening_modulus = 0.0

## ------- Solver details ------- ##

# Spectral LS's reference stiffness, a solver setting only (its kernel is a projection, so its answer does not depend
# on it): "midpoint", the phases' Lamé constants averaged (Moulinec–Suquet 1998, section 3.1), or "matrix".
spectral_ls_reference_stiffness = "midpoint"

# The Fourier preconditioner of each Newton correction's Krylov solver (notes/2_ls_model.md section 7.4) changes the
# iteration count, not the answer. It pays off for CP, whose small C0 stretches the spectrum; for Spectral LS it only
# costs one more FFT per iteration.
spectral_ls_fourier_preconditioner = False
cp_fourier_preconditioner   = True

# TFA's E and P are solved this many load columns at a time (memory against speed). Their FE factorization limits 3D
# meshes: 24³ works with 7 GB of memory, 36³ does not.
fe_columns_per_solve        = 100

# CP solves the coupled eigenstrains of clusters sharing a partition (LS-26) by Newton at every residual evaluation,
# until ‖L (μ − μ(ε))‖ is below cp_eigenstrain_tolerance times the largest cluster stress.
cp_eigenstrain_tolerance    = 1e-12
cp_eigenstrain_max_iterations = 50

# k-means: Lloyd iterations from a seeded k-means++ start, on features from six elastic Spectral LS solves, smoothed
# within each phase (fe.smooth_within_phases).
k_means_iterations          = 50
k_means_seed                = 0

# Random fibers: placing stops with an error after this many random centres.
fiber_placement_attempts    = 100000

# Every method solves each load step by Newton, each Newton correction by a Krylov method, with these settings: GMRES,
# or conjugate gradients for Spectral LS, whose Newton system is symmetric (notes/2_ls_model.md section 7.7). A step has
# converged when its relative residual (the strain residual over the imposed strain, never over less than
# residual_strain_scale_floor) is below convergence_tolerance. The residual of CP and Spectral LS is a strain of their
# reference medium, so it is judged as the correction an all-matrix cell would need (notes/2_ls_model.md section 7.5),
# and the tolerance means the same for any reference stiffness. FFT-TFA's reference medium is the matrix already.
convergence_tolerance       = 1e-6
residual_strain_scale_floor = 1e-4
newton_max_iterations       = 50
# Both Krylov methods stop below krylov_tolerance times the Newton residual. GMRES restarts every gmres_restart
# iterations, at most gmres_max_restarts times (None: scipy's default); CG takes at most spectral_ls_cg_max_iterations
# (None: ten times the unknowns, as scipy's).
krylov_tolerance            = 1e-3
gmres_restart               = 50
gmres_max_restarts          = None
spectral_ls_cg_max_iterations = None
# Backtracking line search: the full step if it lowers the relative residual by line_search_sufficient_decrease times
# the step length, else the step times line_search_step_reduction, at most line_search_max_reductions times.
line_search_sufficient_decrease = 1e-4
line_search_step_reduction  = 0.5
line_search_max_reductions  = 10
# A method abandons the load path when its relative residual exceeds its limit times the larger of 1 and the step's
# first relative residual: ls_divergence_limit for FFT-TFA, Spectral LS and CP, tfa_divergence_limit for TFA.
ls_divergence_limit         = 1e3
tfa_divergence_limit        = 1.0

## ------- Checks (check.py) ------- ##

# Every method must match a laminate's exact solution to laminate_tolerance, relative to each phase's largest stress
# (about 25 times the largest difference measured). The exact solution is found by Newton to laminate_solver_tolerance
# (relative traction jump) within laminate_solver_max_iterations.
laminate_tolerance          = 1e-5
laminate_solver_tolerance   = 1e-11
laminate_solver_max_iterations = 50
# CP's piece law must match LS-25 computed independently (notes/2_ls_model.md section 4.5), to this relative tolerance.
piece_law_check_tolerance   = 1e-9
# FFT-TFA's interaction at full reach must match the element-level LS operator, to this relative tolerance
# (notes/2_ls_model.md section 10.6; measured below 1e-14).
interaction_check_tolerance = 1e-12
# Every method stopped at convergence_tolerance must lie within stopping_accuracy_tolerance (relative, as
# laminate_tolerance) of its own solution at stopping_check_tolerance, so none gains speed by stopping less accurately
# (about 14 times the largest difference measured).
stopping_check_tolerance    = 1e-11
stopping_accuracy_tolerance = 1e-5

## ------- Folders ------- ##

cache_folder = pathlib.Path(__file__).parent / "cache"
output_folder = pathlib.Path(__file__).parent / "output"

## ------- Input Checks ------- ##

def check_inputs():
    # Stops on an input the code cannot run with. check.py calls it again after setting its own inputs.
    if min(elements_per_side, elements_along_z, partitions_per_side, partitions_along_z, strain_increment_count) < 1:
        raise ValueError("The element, partition and strain increment counts must all be at least 1.")
    if elements_per_side % partitions_per_side != 0:
        raise ValueError("The number of elements per side must be a multiple of the number of partitions per side.")
    if spectral_ls_elements_per_side is not None and spectral_ls_elements_per_side < 1:
        raise ValueError("spectral_ls_elements_per_side must be None or at least 1.")
    if include_reference and reference_elements_per_side % elements_per_side != 0:
        raise ValueError("reference_elements_per_side must be a multiple of elements_per_side.")
    if spectral_ls_cg_max_iterations is not None and spectral_ls_cg_max_iterations < 1:
        raise ValueError("spectral_ls_cg_max_iterations must be None or at least 1.")
    if inclusion_shape in ("fiber", "laminate", "random fibers") and elements_along_z % partitions_along_z != 0:
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
        if fiber_diameter + fiber_gap_elements * domain_side_length / elements_per_side >= domain_side_length:
            raise ValueError("A fiber's diameter plus the gap must be below the domain side, so a fiber does not touch "
                             "its own periodic image.")
    else:
        raise ValueError("inclusion_shape must be \"fiber\", \"laminate\", \"laminate_z\", \"sphere\" or "
                         f"\"random fibers\". Got {inclusion_shape!r}.")

    # The radial return divides by 3G + H, so at or below zero it has no admissible solution.
    for phase, elastic_modulus, poisson_ratio, yield_stress, hardening_modulus in (
            ("matrix", matrix_elastic_modulus, matrix_poisson_ratio, matrix_yield_stress, matrix_hardening_modulus),
            ("inclusion", inclusion_elastic_modulus, inclusion_poisson_ratio, inclusion_yield_stress,
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
    for name, (minimum, maximum) in (("von_mises_range_MPa", von_mises_range_MPa),
                                     ("plastic_strain_range_percent", plastic_strain_range_percent)):
        if minimum is not None and maximum is not None and minimum >= maximum:
            raise ValueError(f"{name}: the minimum must be below the maximum.")

check_inputs()
