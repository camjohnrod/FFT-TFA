# Every setting of the code lives here. The usual workflow: choose the problem (section 1) and the methods to compare
# (section 2), then, from the repository's folder,
#
#   python code/run.py preview    what the run would need, and a drawing of each method's cross-section; solves nothing
#   python code/run.py            run it; results and figures go to a new folder in code/output/
#
# Also: run.py replot latest (redraw a saved run), run.py cache (list code/cache/), check.py (test the code).

import math
import os
import pathlib

GPa, MPa = 1e9, 1e6
never = math.inf                 # the yield stress of a phase that stays elastic

def strain(e11=0, e22=0, e33=0, g12=0, g23=0, g13=0):
    """A macroscopic strain by its components, the shears as engineering shear (twice the tensor shear), others zero."""
    return [e11, e22, e33, g12, g23, g13]

## ======= 1. The Problem ======= ##

# One name from each list in section 4, where you can add your own.
geometry                    = "100 fibers"
materials                   = "glass/PA"
load_path                   = "biaxial cyclic"  # a list of turning points, each a strain; the first must not be zero

steps_to_first_turning_point = 25      # the load path's other lines keep the same step length

## ======= 2. What to Compare ======= ##

# Any number of entries per method, each a line in the table and in the figures; an empty list switches a method off.
# Later entries of a method are drawn darker, so list them coarse to fine.

# FFT-TFA and TFA both group the elements of one fine mesh, which resolves the geometry, into clusters. Spectral LS and
# the reference set their own elements.
clustered_mesh              = 256       # elements per side

# FFT-TFA, the method under development (notes/0_fft_tfa_lecture.md): clusters interacting exactly inside each
# partition, and through an FFT on the partitions beyond, each partition summarized by polynomial moments.
#   partitions          per side: the blocks of elements its FFT runs on; must divide clustered_mesh. Best at most about
#                       half an inclusion across
#   matrix_clusters     per partition's matrix part, by k-means
#   inclusion_clusters  per partition's inclusion part; an elastic inclusion needs one
#   degree              of the polynomials summarizing each partition: 0 its mean only; higher converges to the exact
#                       interaction, "full" is exact. About 2–4 in 2D, 2 in 3D, where the kernels grow fast (preview)
fft_tfa     = [dict(partitions=4,   matrix_clusters=4, inclusion_clusters=1, degree=3), 
               dict(partitions=8,   matrix_clusters=4, inclusion_clusters=1, degree=3), 
               dict(partitions=16,  matrix_clusters=4, inclusion_clusters=1, degree=3), 
               dict(partitions=32,  matrix_clusters=4, inclusion_clusters=1, degree=3), 
               dict(partitions=64,  matrix_clusters=4, inclusion_clusters=1, degree=3)]
    
# Spectral LS, the standard Moulinec–Suquet method, one cluster per element.
#   elements            per side: its own mesh, coarser or finer than clustered_mesh
#   C0                  its reference stiffness, which does not change its answer: "midpoint" of the phases, or "matrix"
#                       (about as fast at contrast 34, 17 % slower at contrast 100)
spectral_ls = [dict(elements=64,   C0="midpoint"), 
               dict(elements=128,  C0="midpoint"), 
               dict(elements=256,  C0="midpoint")]

# TFA, the baseline (notes/3_tfa_baseline.md): clusters interacting through FE influence functions, built once and
# cached. Its settings mean the same as FFT-TFA's. It stores (6 × clusters)² doubles, which preview reports.
tfa         = []

# The fine solution everything is compared with: Spectral LS, elements a multiple of clustered_mesh. Solved once and
# cached, not timed; a first solve can take 30 min, so run it in the background. None: no reference.
reference   = dict(elements=512, C0="midpoint")

## ======= 3. Results, Figures and Computer ======= ##

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
    # stays gap (a fraction of the cell's side) clear of the others across the periodic cell. That placing jams near
    # 55 % of the cell (fibers with half the gap around each); past it, all are pushed apart until none overlap, up to
    # about 80 %.
    "100 fibers":    dict(shape="random fibers", count=100, volume_fraction=0.50, gap=1 / 128, seed=1),
    "one fiber":     dict(shape="fiber", radius=0.35e-3),          # centred, through the thickness
    "sphere":        dict(shape="sphere", radius=0.35e-3),         # centred
    "layer":         dict(shape="laminate", thickness=0.5e-3),     # centred, interfaces normal to x; exact solution
    "layer along z": dict(shape="laminate_z", thickness=0.5e-3),   # centred, interfaces normal to z; exact solution
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
    "biaxial":             [strain(e11=0.025, e22=0.025)],
    "biaxial cyclic":      [strain(e11=0.02, e22=0.02), strain(e11=-0.02, e22=-0.02), strain()],
    "uniaxial":            [strain(e11=0.05)],
    "uniaxial cyclic":     [strain(e11=0.02), strain(e11=-0.02), strain()],
    "along the fibers":    [strain(e33=0.02, g13=0.03)],
    "tension, then shear": [strain(e11=0.02), strain(e11=0.02, g12=0.03)],
    "uniaxial, then biaxial": [strain(e11=0.02), strain(e11=-0.02, e22=0.02), strain()],
}

## ======= 5. Solver Details (rarely changed) ======= ##

cell_side                   = 1e-3     # the side of the periodic cubic cell, in m

# Newton–Krylov, shared by every method: each load step is solved by Newton, each correction by a Krylov method, GMRES
# or conjugate gradients for Spectral LS, whose Newton system is symmetric (notes/2_ls_model.md section 4.2). A step has
# converged when its relative residual (the strain residual over the imposed strain, never over less than
# residual_strain_scale_floor) is below convergence_tolerance. Spectral LS's residual is a strain of its reference
# medium, so it is judged as the correction an all-matrix cell would need (notes/2_ls_model.md section 4.3), and the
# tolerance means the same for any reference stiffness. FFT-TFA's reference medium is the matrix already.
convergence_tolerance       = 1e-6
residual_strain_scale_floor = 1e-4
newton_max_iterations       = 50
# Both Krylov methods stop below krylov_tolerance times the Newton residual. GMRES restarts every gmres_restart
# iterations, at most gmres_max_restarts times (None: scipy's default); CG takes at most cg_max_iterations
# (None: ten times the unknowns, as scipy's).
krylov_tolerance            = 1e-2
gmres_restart               = 50
gmres_max_restarts          = None
cg_max_iterations           = None
# Backtracking line search: the full step if it lowers the relative residual by line_search_sufficient_decrease times
# the step length, else the step times line_search_step_reduction, at most line_search_max_reductions times.
line_search_sufficient_decrease = 1e-4
line_search_step_reduction  = 0.5
line_search_max_reductions  = 10
# A method abandons the load path when its relative residual exceeds its limit times the larger of 1 and the step's
# first relative residual: divergence_limit for FFT-TFA and Spectral LS, tfa_divergence_limit for TFA.
divergence_limit            = 1e3
tfa_divergence_limit        = 1.0

# k-means: Lloyd iterations from a seeded start (scipy's minit), on features from six elastic Spectral LS solves with
# this reference stiffness.
k_means_iterations          = 50
k_means_seed                = 0
k_means_start               = "++"
k_means_features_C0         = "midpoint"

# TFA's E and P are solved this many load columns at a time (memory against speed), with this fill-reducing ordering of
# the sparse factorization. It limits 3D meshes: 24³ works with 7 GB of memory, 36³ does not.
tfa_columns_per_solve       = 100
tfa_factorization_ordering  = "MMD_AT_PLUS_A"

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
# FFT-TFA's interaction at full degree must match the element-level LS operator, and a uniform polarization must cause
# no strain at any degree, both to this relative tolerance (measured below 1e-14).
interaction_check_tolerance = 1e-12
# Every method stopped at convergence_tolerance must lie within stopping_accuracy_tolerance (relative, as
# laminate_tolerance) of its own solution at tight_convergence_tolerance, so none gains speed by stopping less
# accurately (about 14 times the largest difference measured).
tight_convergence_tolerance = 1e-11
stopping_accuracy_tolerance = 1e-5

## ======= Behind the Scenes ======= ##

# Nothing to change below: the folders, the thread count, and the checks of the inputs above, which run on import.

cache_folder = pathlib.Path(__file__).parent / "cache"
output_folder = pathlib.Path(__file__).parent / "output"

# numpy must not load before this, so run.py and check.py import config first, and config never imports numpy.
for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = str(thread_count)

def check_entries(name, entries, needed_settings):
    """Every entry of a method's list must hold exactly that method's settings."""
    for entry in entries:
        if set(entry) != set(needed_settings):
            raise ValueError(f"Each entry of {name} needs exactly {', '.join(needed_settings)}. Got {entry}.")

def check_geometry():
    """The geometry config.py names must exist, and its sizes must fit in the cell."""
    if geometry not in all_geometries:
        raise ValueError(f"geometry must be one of all_geometries: {', '.join(all_geometries)}. Got {geometry!r}.")
    settings = all_geometries[geometry]
    shape = settings["shape"]
    if shape in ("fiber", "sphere"):
        if not 0 < settings["radius"] < cell_side / 2:
            raise ValueError(f"{geometry!r}: radius must be positive and below half the cell's side, so neighbouring "
                             f"inclusions do not touch. Got {settings['radius']}.")
    elif shape in ("laminate", "laminate_z"):
        if not 0 < settings["thickness"] < cell_side:
            raise ValueError(f"{geometry!r}: thickness must lie between zero and the cell's side. "
                             f"Got {settings['thickness']}.")
    elif shape == "random fibers":
        if not 0 < settings["volume_fraction"] < 1:
            raise ValueError(f"{geometry!r}: volume_fraction must lie between 0 and 1. "
                             f"Got {settings['volume_fraction']}.")
        # A fiber and its own periodic image must keep the gap too.
        diameter = 2 * cell_side * math.sqrt(settings["volume_fraction"] / (math.pi * settings["count"]))
        if diameter + settings["gap"] * cell_side >= cell_side:
            raise ValueError(f"{geometry!r}: a fiber's diameter plus gap must be below the cell's side, so a fiber "
                             "does not touch its own periodic image.")
    else:
        raise ValueError(f"{geometry!r}: shape must be \"random fibers\", \"fiber\", \"sphere\", \"laminate\" or "
                         f"\"laminate_z\". Got {shape!r}.")

def check_inputs():
    """Stops on an input the code cannot run with, naming the setting to change. check.py calls it again after setting
    its own inputs."""
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
    check_entries("fft_tfa", fft_tfa, ("partitions", "matrix_clusters", "inclusion_clusters", "degree"))
    check_entries("spectral_ls", spectral_ls, ("elements", "C0"))
    check_entries("tfa", tfa, ("partitions", "matrix_clusters", "inclusion_clusters"))
    for entry in fft_tfa + tfa:
        if clustered_mesh % entry["partitions"] != 0:
            raise ValueError(f"partitions={entry['partitions']} must divide clustered_mesh={clustered_mesh}.")
        if min(entry["matrix_clusters"], entry["inclusion_clusters"]) < 1:
            raise ValueError(f"The cluster counts must be at least 1. Got {entry}.")
    for entry in fft_tfa:
        if entry["degree"] != "full" and not (isinstance(entry["degree"], int) and entry["degree"] >= 0):
            raise ValueError(f"degree must be \"full\" or an integer from 0. Got {entry}.")
    if reference is not None:
        check_entries("reference", [reference], ("elements", "C0"))
        if reference["elements"] % clustered_mesh != 0:
            raise ValueError("reference's elements must be a multiple of clustered_mesh.")
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
