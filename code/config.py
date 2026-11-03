# Every setting, and the only file to edit to change a run: other modules read them as config.<name>.
#
#   python code/run.py                 solve what is listed below; results and figures to a new folder in code/output/
#   python code/run.py preview         say what that would need, draw the cross-section, solve nothing
#   python code/run.py replot latest   redraw a saved run's figures with the Figures settings below, solve nothing
#   python code/run.py cache           list what is cached in code/cache/
#   python code/check.py               check the code on small problems with known answers (about a minute)
#
# Sections run from the settings changed most to those changed least. The three grids mean different things to each
# method:
#
#                 elements                      partitions                         clusters
#   FFT-TFA       its kernel and corrections,   its FFT lattice: with its          matrix_clusters and
#                 offline only                  clusters, sets its online cost     inclusion_clusters per partition
#   Spectral LS   its own lattice               not used                           one per element
#                 (elements): sets all cost
#   TFA           its FE mesh: offline cost     its clusters lie in them           as FFT-TFA
#   k-means       its features, offline         it runs inside each                as FFT-TFA
#   reference     its own finer mesh (reference_elements_per_side), solved once and cached
#
#   elements    elements_per_side: the fine mesh, which resolves the geometry
#   partitions  a uniform lattice of blocks of elements
#   clusters    the material states: regions of one phase in one partition, with one uniform plastic eigenstrain each

import os
import pathlib

# CPU threads for every method's dense products (BLAS) and FFTs alike. Set here, before numpy loads, which is why
# run.py and check.py import config first.
thread_count                = 1

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = str(thread_count)

import numpy as np  # noqa: E402, after the thread count is set

# Units: lengths in m, stresses and moduli in Pa.
GPa, MPa = 1e9, 1e6

def strain(e11=0, e22=0, e33=0, g12=0, g23=0, g13=0):
    # A macroscopic strain by its components, the shears as engineering shear (twice the tensor shear), others zero.
    return np.array([e11, e22, e33, g12, g23, g13], dtype=float)

## ------- What to Run ------- ##

# Names the run's folder in code/output/, which also holds these settings and the code's version.
run_name                    = "biaxial cyclic"

# What to compare: any number of entries per method, each a line in the table and in the figures; an empty list switches
# a method off. Later entries of a method are drawn darker, so list them coarse to fine. With several entries of a
# method and the reference on, error_against_time.png shows the trade-off.
#   FFT-TFA      the method under development (notes/2_ls_model.md section 10): TFA's clusters interacting through the
#                Lippmann–Schwinger (LS) equation of the matrix, by an FFT on the partitions plus exact corrections
#                between nearby clusters.
#                partitions          per side, must divide elements_per_side
#                matrix_clusters     per partition's part in the matrix, by k-means
#                inclusion_clusters  per partition's part in the inclusion; an elastic inclusion needs one (more only
#                                    resolve its stress). One of each needs no k-means.
#                reach               partitions over which the exact interaction is kept: 0 each partition's own
#                                    clusters, 1 its neighbours too (8 for columns, 26 in 3D, about 9 or 27 times the
#                                    cost of the corrections), "full" every partition, where the interaction is exact
#   Spectral LS  the LS equation on its own elements (Moulinec–Suquet).
#                elements            per side, coarser or finer than elements_per_side
#                C0                  its reference stiffness, a solver setting only (its kernel is a projection, so its
#                                    answer does not depend on it): "midpoint", the phases' stiffnesses averaged
#                                    (Moulinec–Suquet 1998, section 3.1), or "matrix", the same equation as FFT-TFA's,
#                                    which takes about as long at contrast 34 and 17 % longer at contrast 100
#   TFA          the baseline (notes/3_tfa_baseline.md): clusters interact through the FE influence functions E and P,
#                built offline and cached in code/cache/. P takes (6 × clusters)² doubles.
#                partitions, matrix_clusters, inclusion_clusters  as FFT-TFA
fft_tfa     = [dict(partitions=8, matrix_clusters=8, inclusion_clusters=1, reach=1)]
spectral_ls = [dict(elements=128, C0="midpoint")]
tfa         = []

# A reference, not a method: Spectral LS on a finer mesh of reference_elements_per_side per side (a multiple of
# elements_per_side), on the same problem. Solved once and cached, shown in the figures, not timed. A first solve takes
# about 30 min at 540² elements and 120 steps: run it in the background. reference_C0 is its reference stiffness, and
# that of the elastic solves giving k-means its features.
include_reference           = True
reference_elements_per_side = 512
reference_C0                = "midpoint"

# Each method runs timing_repeats times, in turn, keeping its fastest time (timing noise between runs reaches 30 %).
# Each entry, and the reference, runs in its own process, stopped by an error above memory_limit_GB (None: no limit),
# so that a run too large fails instead of taking the machine down; the others carry on.
timing_repeats              = 1
memory_limit_GB             = 5

## ------- Problem ------- ##

# The fine mesh of every FFT-TFA and TFA entry. A shape constant along z (fiber, laminate, random fibers) has
# elements_per_side² elements in one layer, which is exact, and columns of partitions through the thickness; any other
# shape has the same counts along x, y and z.
elements_per_side           = 128

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

# The phases, by a name from material_sets: isotropic elasticity and J2 plasticity with linear isotropic hardening, the
# hardening zero or above (Spectral LS's conjugate gradients need a positive definite tangent, notes/2_ls_model.md
# section 7.7). A yield stress of np.inf keeps a phase elastic.
materials                   = "glass/PA"
material_sets = {
    "glass/PA": dict(matrix=dict(E=2.1 * GPa, poisson=0.3, yield_stress=29 * MPa, hardening=139 * MPa),
                     inclusion=dict(E=72 * GPa, poisson=0.22, yield_stress=np.inf, hardening=0)),
    "contrast 100": dict(matrix=dict(E=100 * MPa, poisson=0.19, yield_stress=1 * MPa, hardening=10 * MPa),
                         inclusion=dict(E=10 * GPa, poisson=0.3, yield_stress=np.inf, hardening=0)),
}

# The macroscopic strain, by a name from load_paths: straight lines from zero through each turning point in turn.
# strain_increment_count equal steps reach the first; the other lines keep that step length. Fields are saved at every
# turning point.
load                        = "biaxial cyclic"
strain_increment_count      = 50
load_paths = {
    "biaxial cyclic": [strain(e11=0.02, e22=0.02), strain(e11=-0.02, e22=-0.02), strain()],
    "uniaxial": [strain(e11=0.05)],
    "uniaxial cyclic": [strain(e11=0.02), strain(e11=-0.02), strain()],
    "along the fibers": [strain(e33=0.02, g13=0.03)],
    "tension, then shear": [strain(e11=0.02), strain(e11=0.02, g12=0.03)],
}

## ------- Figures ------- ##

# replot uses these settings on a saved run. The maps show the fields at turning point map_turning_point of the load
# path (1: the first; "last": the last, which on a cyclic path is back at zero strain, so residual stress), and each
# entry that reached it.
map_turning_point           = "last"
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

## ------- Solver Details ------- ##

# TFA's E and P are solved this many load columns at a time (memory against speed), with this fill-reducing ordering of
# the sparse factorization. It limits 3D meshes: 24³ works with 7 GB of memory, 36³ does not.
fe_columns_per_solve        = 100
fe_factorization_ordering   = "MMD_AT_PLUS_A"

# k-means: Lloyd iterations from a seeded start (scipy's minit), on features from six elastic Spectral LS solves.
k_means_iterations          = 50
k_means_seed                = 0
k_means_start               = "++"

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

def check_entries(name, entries, needed_settings):
    for entry in entries:
        if set(entry) != set(needed_settings):
            raise ValueError(f"Each entry of {name} needs exactly {', '.join(needed_settings)}. Got {entry}.")

def check_inputs():
    # Stops on an input the code cannot run with, naming the setting to change. check.py calls it again after setting
    # its own inputs.
    if not (fft_tfa or spectral_ls or tfa):
        raise ValueError("Nothing to run: fft_tfa, spectral_ls and tfa are all empty.")
    check_entries("fft_tfa", fft_tfa, ("partitions", "matrix_clusters", "inclusion_clusters", "reach"))
    check_entries("spectral_ls", spectral_ls, ("elements", "C0"))
    check_entries("tfa", tfa, ("partitions", "matrix_clusters", "inclusion_clusters"))
    for entry in fft_tfa + tfa:
        if elements_per_side % entry["partitions"] != 0:
            raise ValueError(f"partitions={entry['partitions']} must divide elements_per_side={elements_per_side}.")
        if min(entry["matrix_clusters"], entry["inclusion_clusters"]) < 1:
            raise ValueError(f"The cluster counts must be at least 1. Got {entry}.")
    for entry in fft_tfa:
        if entry["reach"] != "full" and not (isinstance(entry["reach"], int) and entry["reach"] >= 0):
            raise ValueError(f"reach must be \"full\" or an integer from 0. Got {entry}.")
    for C0 in [reference_C0] + [entry["C0"] for entry in spectral_ls]:
        if C0 not in ("midpoint", "matrix"):
            raise ValueError(f"A reference stiffness must be \"midpoint\" or \"matrix\". Got {C0!r}.")
    if include_reference and reference_elements_per_side % elements_per_side != 0:
        raise ValueError("reference_elements_per_side must be a multiple of elements_per_side.")

    if inclusion_shape in ("fiber", "sphere"):
        if not 0 < inclusion_radius < domain_side_length / 2:
            raise ValueError("inclusion_radius must be positive and below half the domain side, so neighbouring "
                             f"inclusions do not touch. Got {inclusion_radius}.")
    elif inclusion_shape in ("laminate", "laminate_z"):
        if not 0 < layer_thickness < domain_side_length:
            raise ValueError(f"layer_thickness must lie between zero and the domain side. Got {layer_thickness}.")
    elif inclusion_shape == "random fibers":
        if not 0 < fiber_volume_fraction < 1:
            raise ValueError(f"fiber_volume_fraction must lie between 0 and 1. Got {fiber_volume_fraction}.")
        # A fiber and its own periodic image must keep the gap too.
        fiber_diameter = 2 * domain_side_length * np.sqrt(fiber_volume_fraction / (np.pi * fiber_count))
        if fiber_diameter + fiber_gap * domain_side_length >= domain_side_length:
            raise ValueError("A fiber's diameter plus fiber_gap must be below the domain side, so a fiber does not "
                             "touch its own periodic image.")
    else:
        raise ValueError("inclusion_shape must be \"fiber\", \"laminate\", \"laminate_z\", \"sphere\" or "
                         f"\"random fibers\". Got {inclusion_shape!r}.")

    if materials not in material_sets:
        raise ValueError(f"materials must be one of material_sets: {', '.join(material_sets)}. Got {materials!r}.")
    if min(phase["hardening"] for phase in material_sets[materials].values()) < 0:
        raise ValueError("The hardening moduli must be zero or above: softening is not supported.")
    if load not in load_paths:
        raise ValueError(f"load must be one of load_paths: {', '.join(load_paths)}. Got {load!r}.")
    if not np.any(load_paths[load][0]):
        raise ValueError("A load path's first turning point must not be zero: it sets the step length.")
    if map_turning_point != "last" and not 1 <= map_turning_point <= len(load_paths[load]):
        raise ValueError(f"map_turning_point must be \"last\" or from 1 to {len(load_paths[load])}, this load path's "
                         f"turning points. Got {map_turning_point!r}.")
    if strain_increment_count < 1 or timing_repeats < 1:
        raise ValueError("strain_increment_count and timing_repeats must be at least 1.")
    for name, (minimum, maximum) in (("von_mises_range_MPa", von_mises_range_MPa),
                                     ("plastic_strain_range_percent", plastic_strain_range_percent)):
        if minimum is not None and maximum is not None and minimum >= maximum:
            raise ValueError(f"{name}: the minimum must be below the maximum.")

check_inputs()
