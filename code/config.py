# Every setting, and the only file to edit to change a run: other modules read them as config.<name>. Lengths are in
# m, stresses and moduli in Pa.
#
#   python code/run.py      solve the problem below with every method switched on; table and figures to code/output/
#   python code/check.py    check the code on small problems with known answers (about a minute)
#
# Sections run from the settings changed most to those changed least: Problem, Methods, Figures, Materials, Solver
# details, Checks. A setting of one method carries its prefix (fft_tfa_, spectral_ls_, tfa_, k_means_); the same
# setting for several methods sits as a group. The three grids mean different things to each method:
#
#                 elements                      partitions                         clusters
#   FFT-TFA       its kernel and corrections,   its FFT lattice: with its          the k_means_ counts
#                 offline only                  clusters, sets its online cost
#   Spectral LS   its lattice, unless it has    not used                           one per element
#                 its own: sets all cost
#   TFA           its FE mesh: offline cost     its clusters lie in them           the k_means_ counts
#   k-means       its features, offline         it runs inside each                the k_means_ counts
#   reference     its own finer mesh (reference_elements_per_side), solved once and cached
#
#   elements    elements_per_side: the fine mesh, which resolves the geometry
#   partitions  partitions_per_side: a uniform lattice of blocks of elements
#   clusters    the material states: regions of one phase in one partition, with one uniform plastic eigenstrain each

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
# random centres (fiber_seed), each kept if it stays fiber_gap (a fraction of the domain side, so the placing does not
# depend on any mesh) clear of the others across the periodic cell. That placing jams near 55 % of the cell (fibers
# with half the gap around each); past it, the remaining fibers go at random and all are pushed apart until none
# overlap (cell.get_fiber_centres), which reaches about 80 %.
fiber_count                 = 100
fiber_volume_fraction       = 0.50
fiber_gap                   = 1 / 128
fiber_seed                  = 1

# The grids (table above). A shape constant along z (fiber, laminate, random fibers) gets elements_per_side² elements in
# one layer, which is exact, and partitions_per_side² partitions, columns through the thickness; any other shape gets
# cubes, the same counts along x, y and z. Partition counts must divide element counts. Spectral LS runs on
# spectral_ls_elements_per_side elements per side instead of the mesh's, coarser or finer, unless None (the k-means
# features stay on the mesh).
elements_per_side           = 128
partitions_per_side         = 8
spectral_ls_elements_per_side = 128

# The macroscopic strain (ε11, ε22, ε33, γ12, γ23, γ13) reaches max_macro_strain in strain_increment_count equal steps.
# load_path_shape is
#   "monotonic"  0 -> max
#   "cyclic"     0 -> +max -> -max -> 0, 4 × strain_increment_count steps: unloading and reverse yielding
max_macro_strain            = np.array([0.025, 0.025, 0.0, 0.0, 0.0, 0.0])
strain_increment_count      = 50
load_path_shape             = "cyclic"

## ------- Methods ------- ##

# FFT-TFA, the method under development, always runs (notes/2_ls_model.md section 10): TFA's clusters interacting
# through the Lippmann–Schwinger (LS) equation of the matrix, by an FFT on the partitions plus exact corrections
# between nearby clusters. The comparisons:
#   include_spectral_ls  the LS equation on the elements (Moulinec–Suquet), the finest lattice the mesh allows
#   include_tfa          the baseline (notes/3_tfa_baseline.md): clusters interact through the FE influence functions
#                        E and P, built offline and cached in code/cache/
include_spectral_ls         = True
include_tfa                 = False

# A reference, not a method: Spectral LS on a finer mesh of reference_elements_per_side per side (a multiple of
# elements_per_side), on the same problem. Solved once and cached (macroscopic stress at every step, fields at the
# last), shown in the figures, not timed. A first solve takes about 30 min at 540² elements and 120 steps: run it in
# the background.
include_reference           = True
reference_elements_per_side = 512

# The clusters (material states) of FFT-TFA and TFA: k-means inside each partition's part in each phase, on the elements'
# elastic strain concentration, up to k_means_matrix_clusters_per_partition in the matrix part and
# k_means_inclusion_clusters_per_partition in the inclusion part. One each needs no k-means: each partition's part in
# each phase is one cluster. An elastic inclusion needs one cluster (more only resolve its stress). TFA's P takes
# (6 × its clusters)² doubles.
k_means_matrix_clusters_per_partition = 8
k_means_inclusion_clusters_per_partition = 1

# FFT-TFA keeps the exact interaction among each partition's clusters, and with fft_tfa_neighbour_blocks also with
# those of the neighbouring partitions (8 for columns, 26 in 3D), at about 9 or 27 times the cost of those corrections.
fft_tfa_neighbour_blocks    = True

## ------- Figures ------- ##

# Colour-scale ranges (min, max) of the von Mises stress maps (MPa) and the accumulated plastic strain maps (%), to
# share a scale across runs. An end left at None comes from the finest field (the reference, else Spectral LS, else
# every map); values beyond an end take its colour. field_maps_matrix_only greys out the inclusion in both maps
# (its far larger stresses would otherwise set the stress scale).
von_mises_range_MPa         = (None, None)
field_maps_matrix_only      = True
plastic_strain_range_percent = (None, None)

# With the reference on, the macroscopic stress errors (macro_stress_error.png, errors.txt) cover each stress component
# whose reference value reaches loaded_stress_fraction of the largest one somewhere along the path.
loaded_stress_fraction      = 0.05

## ------- Materials ------- ##

# Isotropic elasticity and J2 plasticity with linear isotropic hardening, the hardening modulus zero or above (Spectral
# LS's conjugate gradients need a positive definite tangent, notes/2_ls_model.md section 7.7). A yield stress of np.inf
# keeps a phase elastic.
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

# TFA's E and P are solved this many load columns at a time (memory against speed). Their FE factorization limits 3D
# meshes: 24³ works with 7 GB of memory, 36³ does not.
fe_columns_per_solve        = 100

# k-means: Lloyd iterations from a seeded k-means++ start, on features from six elastic Spectral LS solves.
k_means_iterations          = 50
k_means_seed                = 0

# Random fibers: random sequential addition gives up after fiber_placement_attempts random centres; pushing apart
# then moves overlapping pairs to (1 + fiber_push_margin) times their minimum distance, and stops with an error after
# fiber_push_passes passes.
fiber_placement_attempts    = 100000
fiber_push_margin           = 1e-3
fiber_push_passes           = 10000

# Every method solves each load step by Newton, each Newton correction by a Krylov method, with these settings: GMRES,
# or conjugate gradients for Spectral LS, whose Newton system is symmetric (notes/2_ls_model.md section 7.7). A step has
# converged when its relative residual (the strain residual over the imposed strain, never over less than
# residual_strain_scale_floor) is below convergence_tolerance. The residual of Spectral LS is a strain of its reference
# medium, so it is judged as the correction an all-matrix cell would need (notes/2_ls_model.md section 7.5), and the
# tolerance means the same for any reference stiffness. FFT-TFA's reference medium is the matrix already.
convergence_tolerance       = 1e-6
residual_strain_scale_floor = 1e-4
newton_max_iterations       = 50
# Both Krylov methods stop below krylov_tolerance times the Newton residual. GMRES restarts every gmres_restart
# iterations, at most gmres_max_restarts times (None: scipy's default); CG takes at most spectral_ls_cg_max_iterations
# (None: ten times the unknowns, as scipy's).
krylov_tolerance            = 1e-6
gmres_restart               = 50
gmres_max_restarts          = None
spectral_ls_cg_max_iterations = None
# Backtracking line search: the full step if it lowers the relative residual by line_search_sufficient_decrease times
# the step length, else the step times line_search_step_reduction, at most line_search_max_reductions times.
line_search_sufficient_decrease = 1e-4
line_search_step_reduction  = 0.5
line_search_max_reductions  = 10
# A method abandons the load path when its relative residual exceeds its limit times the larger of 1 and the step's
# first relative residual: ls_divergence_limit for FFT-TFA and Spectral LS, tfa_divergence_limit for TFA.
ls_divergence_limit         = 1e3
tfa_divergence_limit        = 1.0

## ------- Checks (check.py) ------- ##

# Every method must match a laminate's exact solution to laminate_tolerance, relative to each phase's largest stress
# (about 25 times the largest difference measured). The exact solution is found by Newton to laminate_solver_tolerance
# (relative traction jump) within laminate_solver_max_iterations.
laminate_tolerance          = 1e-5
laminate_solver_tolerance   = 1e-11
laminate_solver_max_iterations = 50
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
    if min(elements_per_side, partitions_per_side, strain_increment_count) < 1:
        raise ValueError("The element, partition and strain increment counts must all be at least 1.")
    if elements_per_side % partitions_per_side != 0:
        raise ValueError("The number of elements per side must be a multiple of the number of partitions per side.")
    if spectral_ls_elements_per_side is not None and spectral_ls_elements_per_side < 1:
        raise ValueError("spectral_ls_elements_per_side must be None or at least 1.")
    if include_reference and reference_elements_per_side % elements_per_side != 0:
        raise ValueError("reference_elements_per_side must be a multiple of elements_per_side.")
    if spectral_ls_cg_max_iterations is not None and spectral_ls_cg_max_iterations < 1:
        raise ValueError("spectral_ls_cg_max_iterations must be None or at least 1.")

    if inclusion_shape in ("fiber", "sphere"):
        if not 0 < inclusion_radius < domain_side_length / 2:
            raise ValueError("The inclusion's radius must be positive and below half the domain side, so neighbouring "
                             f"inclusions do not touch. Got {inclusion_radius}.")
    elif inclusion_shape in ("laminate", "laminate_z"):
        if not 0 < layer_thickness < domain_side_length:
            raise ValueError("The fiber layer's thickness must be positive and below the domain side, so matrix "
                             f"surrounds it. Got {layer_thickness}.")
    elif inclusion_shape == "random fibers":
        if min(fiber_count, fiber_placement_attempts, fiber_push_passes) < 1 or fiber_gap < 0:
            raise ValueError("fiber_count, fiber_placement_attempts and fiber_push_passes must be at least 1, "
                             "fiber_gap at least 0.")
        # At zero, pushed pairs only approach their minimum distance.
        if not fiber_push_margin > 0:
            raise ValueError(f"fiber_push_margin must be positive. Got {fiber_push_margin}.")
        if not 0 < fiber_volume_fraction < 1:
            raise ValueError(f"fiber_volume_fraction must lie between 0 and 1. Got {fiber_volume_fraction}.")
        # A fiber and its own periodic image must keep the gap too.
        fiber_diameter = 2 * domain_side_length * np.sqrt(fiber_volume_fraction / (np.pi * fiber_count))
        if fiber_diameter + fiber_gap * domain_side_length >= domain_side_length:
            raise ValueError("A fiber's diameter plus the gap must be below the domain side, so a fiber does not touch "
                             "its own periodic image.")
    else:
        raise ValueError("inclusion_shape must be \"fiber\", \"laminate\", \"laminate_z\", \"sphere\" or "
                         f"\"random fibers\". Got {inclusion_shape!r}.")

    if min(matrix_hardening_modulus, inclusion_hardening_modulus) < 0:
        raise ValueError("The hardening moduli must be zero or above: softening is not supported.")

    if load_path_shape not in ("monotonic", "cyclic"):
        raise ValueError(f"load_path_shape must be \"monotonic\" or \"cyclic\". Got {load_path_shape!r}.")
    if min(k_means_matrix_clusters_per_partition, k_means_inclusion_clusters_per_partition, k_means_iterations) < 1:
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
