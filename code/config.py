# All settings. Change them here, then run from the repository's folder:
#
#   python code/run.py preview    see what the run would need; solves nothing
#   python code/run.py            run it; results and figures go to a new folder in code/output/
#
# Also: run.py replot latest (redraw a saved run), run.py cache (list code/cache/), check.py (test the code).

import math
import os
import pathlib

GPa, MPa = 1e9, 1e6
never = math.inf                 # the yield stress of a phase that stays elastic

def strain(e11=0, e22=0, e33=0, g12=0, g23=0, g13=0):
    # A macroscopic strain by its components, the shears as engineering shear (twice the tensor shear), others zero.
    return [e11, e22, e33, g12, g23, g13]

## ======= 1. The Problem ======= ##

# One name from each list in section 4, where you can add your own.
geometry                    = "100 fibers"
materials                   = "glass/PA"
load_path                   = "biaxial cyclic"

steps_to_first_turning_point = 50      # the load path's other lines keep the same step length
elements_per_side           = 128      # the fine mesh that resolves the geometry; FFT-TFA and TFA both use it

## ======= 2. What to Compare ======= ##

# Any number of entries per method, each a line in the table and in the figures; an empty list switches a method off.
# Later entries of a method are drawn darker, so list them coarse to fine.

# FFT-TFA, the method under development (notes/2_ls_model.md section 5): clusters interacting through an FFT on the
# partitions plus exact corrections between nearby clusters.
#   partitions          per side: the blocks of elements its FFT runs on; must divide elements_per_side
#   matrix_clusters     per partition's matrix part, by k-means
#   inclusion_clusters  per partition's inclusion part; an elastic inclusion needs one
#   reach               how many partitions away the exact corrections go: 0 the partition itself, 1 its neighbours too
#                       (about 9 times the corrections' cost in 2D, 27 in 3D), "full" all of them
fft_tfa     = [dict(partitions=8, matrix_clusters=8, inclusion_clusters=1, reach=1)]

# Spectral LS, the standard Moulinec–Suquet method, one cluster per element.
#   elements            per side: its own lattice, coarser or finer than elements_per_side
#   C0                  its reference stiffness, which does not change its answer: "midpoint" of the phases, or "matrix"
#                       (about as fast at contrast 34, 17 % slower at contrast 100)
spectral_ls = [dict(elements=128, C0="midpoint")]

# TFA, the baseline (notes/3_tfa_baseline.md): clusters interacting through FE influence functions, built once and
# cached. Its settings mean the same as FFT-TFA's. It stores (6 × clusters)² doubles, which preview reports.
tfa         = []

# The fine solution everything is compared with: Spectral LS, elements a multiple of elements_per_side. Solved once and
# cached, not timed; a first solve can take 30 min, so run it in the background. None: no reference.
reference   = dict(elements=512, C0="midpoint")

## ======= 3. Results and Computer ======= ##

results_name                = "test"   # names the results folder, nothing else

# The maps show the fields at this turning point of the load path: 1 is the first; "last" on a cyclic path is back at
# zero strain, so residual stress. replot uses this section's figure settings on a saved run.
map_turning_point           = "last"
field_maps_matrix_only      = True     # grey out the inclusion, whose large stresses would set the colour scale
# Colour scales (min, max), to share one across runs. None: from the finest field (the reference, else Spectral LS).
von_mises_range_MPa         = (None, None)
plastic_strain_range_percent = (None, None)
# The stress errors cover each component whose reference value reaches this fraction of the largest one.
loaded_stress_fraction      = 0.05

thread_count                = 1        # for every method alike
memory_limit_GB             = 5        # an entry needing more stops with an error, the others carry on; None: no limit
timing_repeats              = 1        # runs per entry, in turn, keeping the fastest; timing noise reaches 30 %

## ======= 4. Lists to Choose From ======= ##

all_geometries = {
    # Fibers through the thickness, together volume_fraction of the cell, at random centres (seed), each kept if it
    # stays gap (a fraction of the cell's side) clear of the others across the periodic cell. That placing jams near 55 %
    # of the cell (fibers with half the gap around each); past it, all are pushed apart until none overlap, up to about
    # 80 %.
    "100 fibers":    dict(shape="random fibers", count=100, volume_fraction=0.50, gap=1 / 128, seed=1),
    "one fiber":     dict(shape="fiber", radius=0.35e-3),          # centred, through the thickness
    "sphere":        dict(shape="sphere", radius=0.35e-3),         # centred
    "layer":         dict(shape="laminate", thickness=0.5e-3),     # centred, interfaces normal to x; exact solution known
    "layer along z": dict(shape="laminate_z", thickness=0.5e-3),   # centred, interfaces normal to z; exact solution known
}

# Isotropic elasticity and J2 plasticity with linear isotropic hardening, zero or above.
all_materials = {
    "glass/PA": dict(matrix=dict(E=2.1 * GPa, poisson=0.3, yield_stress=29 * MPa, hardening=139 * MPa),
                     inclusion=dict(E=72 * GPa, poisson=0.22, yield_stress=never, hardening=0)),
    "contrast 100": dict(matrix=dict(E=100 * MPa, poisson=0.19, yield_stress=1 * MPa, hardening=10 * MPa),
                         inclusion=dict(E=10 * GPa, poisson=0.3, yield_stress=never, hardening=0)),
}

# Straight lines from zero through each turning point in turn. Fields are saved at every turning point.
all_load_paths = {
    "biaxial cyclic":      [strain(e11=0.02, e22=0.02), strain(e11=-0.02, e22=-0.02), strain()],
    "uniaxial":            [strain(e11=0.05)],
    "uniaxial cyclic":     [strain(e11=0.02), strain(e11=-0.02), strain()],
    "along the fibers":    [strain(e33=0.02, g13=0.03)],
    "tension, then shear": [strain(e11=0.02), strain(e11=0.02, g12=0.03)],
}

## ======= 5. Solver Details (rarely changed) ======= ##

domain_side_length          = 1e-3     # the periodic cubic cell, in m

# Every method solves each load step by Newton, each correction by a Krylov method: GMRES, or conjugate gradients for
# Spectral LS, whose Newton system is symmetric (notes/2_ls_model.md section 4.2). A step has converged when its
# relative residual (the strain residual over the imposed strain, never over less than residual_strain_scale_floor) is
# below convergence_tolerance. Spectral LS's residual is a strain of its reference medium, so it is judged as the
# correction an all-matrix cell would need (notes/2_ls_model.md section 4.3), and the tolerance means the same for any
# reference stiffness. FFT-TFA's reference medium is the matrix already.
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

# k-means: Lloyd iterations from a seeded start (scipy's minit), on features from six elastic Spectral LS solves with
# this reference stiffness.
k_means_iterations          = 50
k_means_seed                = 0
k_means_start               = "++"
k_means_features_C0         = "midpoint"

# TFA's E and P are solved this many load columns at a time (memory against speed), with this fill-reducing ordering of
# the sparse factorization. It limits 3D meshes: 24³ works with 7 GB of memory, 36³ does not.
fe_columns_per_solve        = 100
fe_factorization_ordering   = "MMD_AT_PLUS_A"

# Random fibers: random sequential addition gives up after fiber_placement_attempts random centres; pushing apart
# then moves overlapping pairs to (1 + fiber_push_margin) times their minimum distance, and stops with an error after
# fiber_push_passes passes.
fiber_placement_attempts    = 100000
fiber_push_margin           = 1e-3
fiber_push_passes           = 10000

## ======= 6. Checks (check.py) ======= ##

# Every method must match a laminate's exact solution to laminate_tolerance, relative to each phase's largest stress
# (about 25 times the largest difference measured). The exact solution is found by Newton to laminate_solver_tolerance
# (relative traction jump) within laminate_solver_max_iterations.
laminate_tolerance          = 1e-5
laminate_solver_tolerance   = 1e-11
laminate_solver_max_iterations = 50
# FFT-TFA's interaction at full reach must match the element-level LS operator, to this relative tolerance
# (notes/2_ls_model.md section 5.6; measured below 1e-14).
interaction_check_tolerance = 1e-12
# Every method stopped at convergence_tolerance must lie within stopping_accuracy_tolerance (relative, as
# laminate_tolerance) of its own solution at stopping_check_tolerance, so none gains speed by stopping less accurately
# (about 14 times the largest difference measured).
stopping_check_tolerance    = 1e-11
stopping_accuracy_tolerance = 1e-5

## ======= Behind the Scenes ======= ##

cache_folder = pathlib.Path(__file__).parent / "cache"
output_folder = pathlib.Path(__file__).parent / "output"

# numpy must not load before this, so run.py and check.py import config first, and config never imports numpy.
for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = str(thread_count)

def check_entries(name, entries, needed_settings):
    for entry in entries:
        if set(entry) != set(needed_settings):
            raise ValueError(f"Each entry of {name} needs exactly {', '.join(needed_settings)}. Got {entry}.")

def check_geometry():
    if geometry not in all_geometries:
        raise ValueError(f"geometry must be one of all_geometries: {', '.join(all_geometries)}. Got {geometry!r}.")
    settings = all_geometries[geometry]
    shape = settings["shape"]
    if shape in ("fiber", "sphere"):
        if not 0 < settings["radius"] < domain_side_length / 2:
            raise ValueError(f"{geometry!r}: radius must be positive and below half the domain side, so neighbouring "
                             f"inclusions do not touch. Got {settings['radius']}.")
    elif shape in ("laminate", "laminate_z"):
        if not 0 < settings["thickness"] < domain_side_length:
            raise ValueError(f"{geometry!r}: thickness must lie between zero and the domain side. "
                             f"Got {settings['thickness']}.")
    elif shape == "random fibers":
        if not 0 < settings["volume_fraction"] < 1:
            raise ValueError(f"{geometry!r}: volume_fraction must lie between 0 and 1. "
                             f"Got {settings['volume_fraction']}.")
        # A fiber and its own periodic image must keep the gap too.
        diameter = 2 * domain_side_length * math.sqrt(settings["volume_fraction"] / (math.pi * settings["count"]))
        if diameter + settings["gap"] * domain_side_length >= domain_side_length:
            raise ValueError(f"{geometry!r}: a fiber's diameter plus gap must be below the domain side, so a fiber does "
                             "not touch its own periodic image.")
    else:
        raise ValueError(f"{geometry!r}: shape must be \"random fibers\", \"fiber\", \"sphere\", \"laminate\" or "
                         f"\"laminate_z\". Got {shape!r}.")

def check_inputs():
    # Stops on an input the code cannot run with, naming the setting to change. check.py calls it again after setting
    # its own inputs.
    check_geometry()
    if materials not in all_materials:
        raise ValueError(f"materials must be one of all_materials: {', '.join(all_materials)}. Got {materials!r}.")
    if min(phase["hardening"] for phase in all_materials[materials].values()) < 0:
        raise ValueError("The hardening moduli must be zero or above: softening is not supported.")
    if load_path not in all_load_paths:
        raise ValueError(f"load_path must be one of all_load_paths: {', '.join(all_load_paths)}. Got {load_path!r}.")
    if not any(all_load_paths[load_path][0]):
        raise ValueError("A load path's first turning point must not be zero: it sets the step length.")
    if steps_to_first_turning_point < 1 or timing_repeats < 1:
        raise ValueError("steps_to_first_turning_point and timing_repeats must be at least 1.")

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
    if reference is not None:
        check_entries("reference", [reference], ("elements", "C0"))
        if reference["elements"] % elements_per_side != 0:
            raise ValueError("reference's elements must be a multiple of elements_per_side.")
    for C0 in [entry["C0"] for entry in spectral_ls] + [k_means_features_C0] + ([reference["C0"]] if reference else []):
        if C0 not in ("midpoint", "matrix"):
            raise ValueError(f"A reference stiffness C0 must be \"midpoint\" or \"matrix\". Got {C0!r}.")

    if map_turning_point != "last" and not 1 <= map_turning_point <= len(all_load_paths[load_path]):
        raise ValueError(f"map_turning_point must be \"last\" or from 1 to {len(all_load_paths[load_path])}, this load "
                         f"path's turning points. Got {map_turning_point!r}.")
    for name, (minimum, maximum) in (("von_mises_range_MPa", von_mises_range_MPa),
                                     ("plastic_strain_range_percent", plastic_strain_range_percent)):
        if minimum is not None and maximum is not None and minimum >= maximum:
            raise ValueError(f"{name}: the minimum must be below the maximum.")

check_inputs()
