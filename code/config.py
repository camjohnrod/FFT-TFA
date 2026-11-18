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
geometry                    = "100 fibers, 35 %"   # the geometry's name, which sets its shape and size
materials                   = "glass/PA, hot, elastic"
load_path                   = "transverse to 1 %, then shear"  # turning points, each a strain; the first must not be zero
# The directions the load path never strains: "fixed", their strain stays zero (strain-controlled, the cell cannot
# contract sideways); "free", their stress stays zero instead (a path along one component is then uniaxial stress, a
# tensile test; lecture, appendix F).
unloaded_directions         = "free"

steps_to_first_turning_point = 15      # the load path's other lines keep the same step length

## ======= 2. What to Compare ======= ##

# Any number of entries per method, each a line in the table; an empty list switches a method off. A list of entries is
# one scheme: the error-against-time plot joins them in one line, in the order listed (so list them coarse to fine), and
# the other figures draw its last. To compare schemes, give a list of lists, one per scheme, for example:
#   fft_tfa = [[dict(partitions=p, matrix_clusters=4, inclusion_clusters=1, degree=2) for p in (4, 8, 16)],
#              [dict(partitions=p, matrix_clusters=4, inclusion_clusters=1, degree=3) for p in (4, 8, 16)]]

# FFT-TFA, SCA and TFA group the elements of a fine mesh, which resolves the geometry, into clusters: this one, unless
# an entry sets its own elements. Spectral LS and the reference set their own elements.
clustered_mesh              = 1024        # elements per side
# How every method's k-means clusters are found: "plain", one k-means into all of a part's clusters; or "nested",
# halving the cluster with the largest spread until there are enough, so clusters at a larger count split those at a
# smaller one. For clusters over the whole cell (SCA's), nested is far faster to compute (seconds against minutes for
# many clusters) and about 15 % less accurate at equal count; for clusters inside small partitions it saves nothing.
k_means_method              = "plain"

# The element kernel Γ0 of every LS method: FFT-TFA, SCA, Spectral LS, the reference and the k-means features, so all
# share one discretization. "Willot", the rotated staggered grid (Willot 2015): smaller errors at equal resolution and
# fewer oscillations at interfaces; or "Moulinec-Suquet", the original, by trigonometric polynomials. TFA uses finite
# elements whatever this says.
element_kernel              = "Willot"

# FFT-TFA, the method under development (notes/fft_tfa_lecture.md): clusters interacting exactly inside each
# partition, and through an FFT on the partitions beyond, each partition summarized by polynomial moments.
#   partitions          per side: the blocks of elements its FFT runs on; must divide its mesh's elements. Best at most
#                       about half an inclusion across
#   matrix_clusters     per partition's matrix part, by k-means; 1 (with inclusion_clusters 1): each part is one
#                       cluster, with no k-means and so no offline solves
#   inclusion_clusters  per partition's inclusion part, by k-means: more follow its polarization more closely
#   degree              of the polynomials summarizing each partition: 0 its mean only, which stalls as partitions
#                       shrink; 1 or 2 is usually enough; higher converges to the exact interaction, "full" is exact.
#                       The kernels grow fast with it, in 3D above all (preview)
#   elements            optional, per side: the mesh its clusters group, instead of clustered_mesh. On a Spectral LS
#                       entry's elements, both solve the same image. For example, one scheme per mesh:
#                       [[dict(elements=e, partitions=p, ...) for p in (8, 16)] for e in (64, 128)]
fft_tfa = [dict(elements=clustered_mesh, partitions=p, matrix_clusters=1, inclusion_clusters=1, degree=1) for p in (16, 32, 64,  128, 256)]

# SCA, self-consistent clustering analysis (Liu, Bessa & Liu 2016; lecture, appendix E): the clustered equation FFT-TFA
# approximates, solved exactly, on clusters over the whole cell. It stores (6 × clusters)² doubles, which preview
# reports.
#   matrix_clusters     of the matrix, by k-means over the whole cell
#   inclusion_clusters  of the inclusion, likewise
#   C0                  its reference stiffness, which changes its answer: "matrix", which makes it FFT-TFA on one
#                       partition; or "self-consistent", isotropic, its two Lamé constants refitted at every load
#                       step (Liu, Bessa & Liu's Box II), which stores D in two (three) parts besides D itself
#   elements            optional, as for FFT-TFA
# For example: sca = [dict(matrix_clusters=c, inclusion_clusters=c, C0="matrix") for c in (8, 32, 128)]
sca         = []

# Spectral LS, the standard method of Moulinec and Suquet, one cluster per element, on element_kernel.
#   elements            per side: its own mesh, coarser or finer than clustered_mesh
#   C0                  its reference stiffness, which does not change its answer: "midpoint" of the phases, or "matrix"
#                       (about as fast at contrast 34, 17 % slower at contrast 100)
spectral_ls = [dict(elements=e, C0="midpoint") for e in (64, 128, 256)]

# TFA, the baseline (lecture, appendix C): clusters interacting through FE influence functions, built once and
# cached. Its settings mean the same as FFT-TFA's. It stores (6 × clusters)² doubles, which preview reports.
# For example: tfa = [dict(partitions=8, matrix_clusters=2, inclusion_clusters=1)]
tfa         = []

# The fine solution everything is compared with: Spectral LS, elements a multiple of clustered_mesh. Equal to it, the
# reference is the exact answer on the very image the clustered methods group, so their errors are their models' alone;
# finer, their errors also hold their mesh's own (in elasticity it is stiff while clustering is soft, and the two can
# cancel). Solved once and cached, not timed, and solved again after any change of geometry, materials, load path or
# solver tolerance; a first solve can take 30 min, so run it in the background. None: no reference.
reference   = dict(elements=1024, C0="midpoint")

## ======= 3. Results, Figures and Computer ======= ##

results_name = "fibers35-elastic"   # names the results folder, nothing else

# The maps show the fields at this turning point of the load path: 1 is the first; "last" on a cyclic path is back at
# zero strain, so residual stress. replot uses this section's figure settings on a saved run.
map_turning_point           = "last"
field_maps_matrix_only      = True     # grey out the inclusion, so the matrix's values set the colour scale
# Each map's colour scale (min, max), to share one across runs. None: from the finest field (the reference, else
# Spectral LS).
von_mises_range_MPa         = (None, None)
plastic_strain_range_percent = (None, None)
# The per-component stress errors (summary.txt's second error table, macro_stress_error.png) cover each component
# whose reference value reaches this fraction of the largest one.
loaded_stress_fraction      = 0.05

thread_count                = 1        # for every method alike
memory_limit_GB             = 5        # an entry needing more stops with an error, the others carry on; None: no limit

## ======= 4. Lists to Choose From ======= ##

all_geometries = {
    # Fibers through the thickness, together volume_fraction of the cell, at random centres (seed), each kept if it
    # stays gap (a fraction of the cell's side) clear of the others across the periodic cell. That placing jams near
    # 55 % of the cell (fibers with half the gap around each); past it, all are pushed apart until none overlap, up to
    # about 80 %.
    "10 fibers":     dict(shape="random fibers", count=10, volume_fraction=0.30, gap=2 / 128, seed=0),
    "20 fibers":     dict(shape="random fibers", count=20, volume_fraction=0.30, gap=2 / 128, seed=0),
    "100 fibers, 50 %": dict(shape="random fibers", count=100, volume_fraction=0.50, gap=2 / 256, seed=0),
    "100 fibers, 35 %": dict(shape="random fibers", count=100, volume_fraction=0.35, gap=2 / 256, seed=0),
    # Spheres placed alike, in 3D: random placing jams near 38 % of the cell (spheres with half the gap around each);
    # past it, all are pushed apart until none overlap.
    "16 spheres":    dict(shape="random spheres", count=16, volume_fraction=0.25, gap=2 / 64, seed=0),
    "one fiber":     dict(shape="fiber", radius=0.35e-3),          # centred, through the thickness
    "sphere":        dict(shape="sphere", radius=0.35e-3),         # centred
    "layer":         dict(shape="laminate", thickness=0.5e-3),     # centred, interfaces normal to x; exact solution
    "layer along z": dict(shape="laminate_z", thickness=0.5e-3),   # centred, interfaces normal to z; exact solution
}

# Isotropic elasticity and J2 plasticity with linear isotropic hardening, zero or above.
all_materials = {
    "glass/PA": dict(matrix=dict(E=2.1 * GPa, poisson=0.3, yield_stress=29 * MPa, hardening=139 * MPa),
                     inclusion=dict(E=72 * GPa, poisson=0.22, yield_stress=never, hardening=0)),
    # Provisional, to see the effect of contrast: glass/PA with the matrix softened as polyamide is when hot (above its
    # glass transition) or moist, E about 0.8 GPa, so contrast 90 instead of 34. The matrix's E, yield stress and
    # hardening are glass/PA's scaled by one factor, so its stress-strain curve keeps its shape and only the contrast
    # changes. To be replaced by a cited hot or conditioned PA6 (Poisson's ratio then nearer 0.4).
    "glass/PA, hot": dict(matrix=dict(E=0.8 * GPa, poisson=0.3, yield_stress=29 * MPa * 0.8 / 2.1,
                                      hardening=139 * MPa * 0.8 / 2.1),
                          inclusion=dict(E=72 * GPa, poisson=0.22, yield_stress=never, hardening=0)),
    # The same with the matrix kept elastic: every method's error is then its model's elastic error.
    "glass/PA, hot, elastic": dict(matrix=dict(E=0.8 * GPa, poisson=0.3, yield_stress=never, hardening=0),
                                  inclusion=dict(E=72 * GPa, poisson=0.22, yield_stress=never, hardening=0)),
    # Synthetic, for numerical tests only: contrast 100 with a Poisson's ratio no polymer has.
    "contrast 100": dict(matrix=dict(E=100 * MPa, poisson=0.19, yield_stress=1 * MPa, hardening=10 * MPa),
                         inclusion=dict(E=10 * GPa, poisson=0.3, yield_stress=never, hardening=0)),
}

# Straight lines from zero through each turning point in turn. Fields are saved at every turning point.
all_load_paths = {
    "biaxial":                [strain(e11=0.025, e22=0.025)],
    "biaxial cyclic":         [strain(e11=0.02, e22=0.02), strain(e11=-0.02, e22=-0.02), strain()],
    "transverse":             [strain(e11=0.025)],
    "along the fibers":       [strain(e33=0.05)],
    "transverse to 2.5 %, then shear": [strain(e11=0.025), strain(e11=0.025, g12=0.015)],
    "transverse to 1 %, then shear":   [strain(e11=0.01), strain(e11=0.01, g12=0.01)],
}

## ======= 5. Solver Details (rarely changed) ======= ##

cell_side                   = 1e-3     # the side of the periodic cubic cell, in m

# Newton–Krylov, shared by every method: each load step is solved by Newton, each correction by a Krylov method, GMRES
# or conjugate gradients for Spectral LS, whose Newton system is symmetric (lecture, appendix B.2). A step has
# converged when its relative residual (the strain residual over the imposed strain, never over less than
# residual_strain_scale_floor) is below convergence_tolerance. Spectral LS's residual is a strain of its reference
# medium, so it is judged as the correction an all-matrix cell would need (lecture, appendix B.3), and the
# tolerance means the same for any reference stiffness. FFT-TFA's reference medium is the matrix already; SCA's
# residual is judged as it is, and check.py checks that it stops as accurately as the others.
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
# first relative residual: divergence_limit for FFT-TFA, SCA and Spectral LS, tfa_divergence_limit for TFA.
divergence_limit            = 1e3
tfa_divergence_limit        = 1.0

# SCA's self-consistent C0: after each solve of a load step, its Lamé constants are refitted to the step's macroscopic
# increments, and the step solved again until both change by less than sca_reference_tolerance (relative;
# Liu–Bessa–Liu 2016 use 0.1 %), at most sca_max_reference_updates times.
sca_reference_tolerance     = 1e-3
sca_max_reference_updates   = 10
# A load step counts as pure shear, its λ kept from the last step, when its volume change is below
# sca_pure_shear_tolerance times its shear. The fixed kernels D is assembled from must rebuild Γ0 of a test medium to
# sca_kernel_parts_tolerance (relative), and D is assembled sca_assembly_rows rows at a time (memory against speed).
sca_pure_shear_tolerance    = 1e-12
sca_kernel_parts_tolerance  = 1e-10
sca_assembly_rows           = 600

# FFT-TFA's corrections (FT-2) count each partition's pairs of elements by offset, on a window twice the partition
# across, for this many window elements at a time (memory against speed: 2²⁰ needs about 50 MB).
fft_tfa_pair_count_elements = 2**20

# k-means (each k-means of k_means_method): Lloyd iterations from a seeded start (scipy's minit), on features from six
# elastic Spectral LS solves with this reference stiffness.
k_means_iterations          = 50
k_means_seed                = 0
k_means_start               = "++"
k_means_features_C0         = "midpoint"

# TFA's E and P are solved this many load columns at a time (memory against speed), with this fill-reducing ordering of
# the sparse factorization. It limits 3D meshes: 24³ works with 7 GB of memory, 36³ does not.
tfa_columns_per_solve       = 100
tfa_factorization_ordering  = "MMD_AT_PLUS_A"

# Random fibers and spheres: random sequential addition gives up after placement_attempts random centres; pushing
# apart then moves overlapping pairs to (1 + push_margin) times their minimum distance, and stops with an error after
# push_passes passes.
placement_attempts          = 100000
push_margin                 = 1e-3
push_passes                 = 10000

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

def get_schemes(listed):
    """A method's schemes, each a list of entries: a list of entries is one scheme, a list of lists one per inner list."""
    if all(isinstance(entry, dict) for entry in listed):
        return [list(listed)] if listed else []
    if all(isinstance(scheme, list) and scheme and all(isinstance(entry, dict) for entry in scheme) for scheme in listed):
        return [list(scheme) for scheme in listed]
    raise ValueError("A method's entries must be a list of dicts (one scheme) or a list of non-empty lists of dicts "
                     f"(one per scheme). Got {listed}.")

def get_all_entries(listed):
    """Every entry of a method's schemes."""
    return [entry for scheme in get_schemes(listed) for entry in scheme]

def get_clustered_elements(entry):
    """The elements per side of the mesh an FFT-TFA, SCA or TFA entry's clusters group: its own, or clustered_mesh."""
    return entry.get("elements", clustered_mesh)

def check_entries(name, listed, needed_settings, optional_settings=()):
    """Every entry of a method must hold that method's settings, and no others but its optional ones, and no scheme may
    list one entry twice."""
    for scheme in get_schemes(listed):
        if any(scheme.count(entry) > 1 for entry in scheme):
            raise ValueError(f"A scheme of {name} lists an entry twice: {scheme}.")
    for entry in get_all_entries(listed):
        if not set(needed_settings) <= set(entry) <= set(needed_settings) | set(optional_settings):
            raise ValueError(f"Each entry of {name} needs exactly {', '.join(needed_settings)}"
                             + (f", and may hold {', '.join(optional_settings)}" if optional_settings else "")
                             + f". Got {entry}.")

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
    elif shape in ("random fibers", "random spheres"):
        if not 0 < settings["volume_fraction"] < 1:
            raise ValueError(f"{geometry!r}: volume_fraction must lie between 0 and 1. "
                             f"Got {settings['volume_fraction']}.")
        # An inclusion and its own periodic image must keep the gap too.
        share = settings["volume_fraction"] / settings["count"]
        diameter = 2 * cell_side * (math.sqrt(share / math.pi) if shape == "random fibers"
                                    else (3 * share / (4 * math.pi)) ** (1 / 3))
        if diameter + settings["gap"] * cell_side >= cell_side:
            raise ValueError(f"{geometry!r}: an inclusion's diameter plus gap must be below the cell's side, so it "
                             "does not touch its own periodic image.")
    else:
        raise ValueError(f"{geometry!r}: shape must be \"random fibers\", \"random spheres\", \"fiber\", \"sphere\", "
                         f"\"laminate\" or \"laminate_z\". Got {shape!r}.")

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
    if unloaded_directions not in ("fixed", "free"):
        raise ValueError(f"unloaded_directions must be \"fixed\" or \"free\". Got {unloaded_directions!r}.")
    if steps_to_first_turning_point < 1:
        raise ValueError("steps_to_first_turning_point must be at least 1.")

    if not (fft_tfa or sca or spectral_ls or tfa):
        raise ValueError("Nothing to run: fft_tfa, sca, spectral_ls and tfa are all empty.")
    if k_means_method not in ("plain", "nested"):
        raise ValueError(f"k_means_method must be \"plain\" or \"nested\". Got {k_means_method!r}.")
    if element_kernel not in ("Willot", "Moulinec-Suquet"):
        raise ValueError(f"element_kernel must be \"Willot\" or \"Moulinec-Suquet\". Got {element_kernel!r}.")
    check_entries("fft_tfa", fft_tfa, ("partitions", "matrix_clusters", "inclusion_clusters", "degree"), ("elements",))
    check_entries("sca", sca, ("matrix_clusters", "inclusion_clusters", "C0"), ("elements",))
    check_entries("spectral_ls", spectral_ls, ("elements", "C0"))
    check_entries("tfa", tfa, ("partitions", "matrix_clusters", "inclusion_clusters"), ("elements",))
    for entry in get_all_entries(fft_tfa) + get_all_entries(tfa):
        if get_clustered_elements(entry) % entry["partitions"] != 0:
            raise ValueError(f"partitions={entry['partitions']} must divide the mesh's "
                             f"{get_clustered_elements(entry)} elements per side. Got {entry}.")
    for entry in get_all_entries(fft_tfa) + get_all_entries(sca) + get_all_entries(tfa):
        if min(entry["matrix_clusters"], entry["inclusion_clusters"]) < 1:
            raise ValueError(f"The cluster counts must be at least 1. Got {entry}.")
    for entry in get_all_entries(sca):
        if entry["C0"] not in ("matrix", "self-consistent"):
            raise ValueError(f"SCA's C0 must be \"matrix\" or \"self-consistent\". Got {entry}.")
    for entry in get_all_entries(fft_tfa):
        if entry["degree"] != "full" and not (isinstance(entry["degree"], int) and entry["degree"] >= 0):
            raise ValueError(f"degree must be \"full\" or an integer from 0. Got {entry}.")
    if reference is not None:
        check_entries("reference", [reference], ("elements", "C0"))
        if reference["elements"] % clustered_mesh != 0:
            raise ValueError("reference's elements must be a multiple of clustered_mesh.")
    for C0 in [entry["C0"] for entry in get_all_entries(spectral_ls)] + [k_means_features_C0] + ([reference["C0"]] if reference else []):
        if C0 not in ("midpoint", "matrix"):
            raise ValueError(f"A reference stiffness C0 must be \"midpoint\" or \"matrix\". Got {C0!r}.")

    if map_turning_point != "last" and not 1 <= map_turning_point <= len(all_load_paths[load_path]):
        raise ValueError(f"map_turning_point must be \"last\" or from 1 to {len(all_load_paths[load_path])}, this load "
                         f"path's turning points. Got {map_turning_point!r}.")
    for name in ("von_mises_range_MPa", "plastic_strain_range_percent"):
        minimum, maximum = globals()[name]
        if minimum is not None and maximum is not None and minimum >= maximum:
            raise ValueError(f"{name}: the minimum must be below the maximum.")

check_inputs()
