"""The entry point: python code/run.py [command], with every setting in code/config.py.

Commands (defined first below):
  (none)            solve every entry config.py lists along the load path, save the run in a new folder of
                    code/output/, and draw its tables and figures from what was saved
  preview           what each entry would store, and a drawing of the cross-section; solves nothing
  replot [folder]   redraw a saved run (default: the latest) with config.py's figure settings
  cache             list what code/cache/ holds

A run, in order: the load path (get_load_path), the reference (get_reference, cached), every entry built and solved in
a process of its own (solve_all), then saved (save_run) and drawn (draw_run). Its folder holds:
  config.py                    the settings it ran with
  run.json, results.npz        what each entry did and cost, and its results (read them with get_saved_run): the
                               macroscopic stress at every step, and maps of the element layer at mid-thickness (each
                               element's cluster, partition and phase; its stress and accumulated plastic strain at
                               every turning point)
  summary.txt                  each entry's clusters, times, iterations and peak memory, and with the reference on,
                               its macroscopic stress errors
  stress_strain.png            macroscopic stress against applied strain, every entry and the reference
  macro_stress_error.png       with the reference on: each entry's macroscopic stress error in every loaded component
  error_against_time.png       with the reference on and several entries of a method: rms error against online time
  cross_section.png            each entry's elements, clusters and partitions
  plastic_strain.png           the accumulated plastic strain at config.map_turning_point, one map per entry
  plastic_strain_profile.png   the same along the middle element row, every entry on one plot
code/output/latest points at the last run.
"""

import datetime
import json
import multiprocessing
import re
import resource
import shutil
import sys
import time
import traceback
from typing import Callable, NamedTuple

import config  # first: it sets the thread count before numpy loads
import numpy as np
import tqdm

import plots
from cell import (Clusters, cluster_by_k_means, cluster_by_phase, get_cache_entries, get_cache_path, get_cached,
                  get_clusters, get_fiber_radius, get_geometry, get_geometry_parameters, get_inclusion_L, get_matrix_L,
                  get_memory_MB, get_mesh, is_shape_constant_along_z)
from fft_tfa_model import FFTTFASolver, get_polynomial_basis, resolve_degree
from ls_model import SpectralLSSolver, get_elastic_strain_concentration, get_element_clusters
from material import get_unloaded_plastic_state
from newton import SolverDidNotConverge
from tfa_model import TFASolver

## ------- Commands ------- ##

def run():
    """Solves the reference and every entry config.py lists, saves the run in a new folder of code/output/ and draws
    it."""
    load_path = get_load_path()
    results, reference = {}, None
    if config.reference is not None:
        value, failure_reason = run_in_own_process(get_reference, load_path)
        if value is None:
            sys.exit(f"The reference failed: {failure_reason}")
        results["Reference"], reference = value
    entry_results, entries = solve_all(load_path)
    folder = save_run(load_path, results | entry_results, entries, reference)
    draw_run(folder)
    print(f"Saved to {folder}")

def replot(name):
    """Redraws a saved run (a folder name in code/output/, or "latest") with config.py's figure settings."""
    folder = config.output_folder / name
    if not (folder / "results.npz").exists():
        sys.exit(f"No saved run at {folder}")
    draw_run(folder.resolve())
    print(f"Redrawn in {folder.resolve()}")

def preview():
    """Prints what config.py asks for and what each entry would store, and draws the cross-section, solving nothing but
    the elastic k-means features."""
    load_path = get_load_path()
    if get_geometry()["shape"] == "random fibers":
        diameter = 2 * get_fiber_radius() * config.clustered_mesh / config.cell_side
        print(f"Geometry {config.geometry!r}: {diameter:.3g} elements per fiber diameter")
    phases = config.all_materials[config.materials]
    print(f"Materials {config.materials!r}: contrast {phases['inclusion']['E'] / phases['matrix']['E']:.3g}")
    print(f"Load path {config.load_path!r}: {len(load_path.applied_macro_strain)} steps, turning points at steps "
          f"{', '.join(str(step + 1) for step in load_path.turning_steps)}")
    entries, arrays = [], {}
    for entry in get_entries():
        clusters = get_entry_clusters(entry)
        print(f"{entry.label}: {clusters.cluster_count} clusters, {describe_storage(entry, clusters)}")
        arrays |= get_cluster_maps(entry.label, clusters)
        entries.append({"label": entry.label, "method": entry.method, "settings": entry.settings,
                        "cluster_count": clusters.cluster_count})
    if config.reference is not None:
        is_cached = get_cache_path("reference", get_reference_parameters(load_path)).exists()
        print(f"Reference {config.reference['elements']} per side: "
              f"{'cached' if is_cached else 'NOT cached, a run would solve it first'}")
    config.output_folder.mkdir(parents=True, exist_ok=True)
    path = config.output_folder / "preview_cross_section.png"
    plots.plot_cross_section([get_cell_map(entry, arrays) for entry in entries], path)
    print(f"Cross-section drawn in {path}")

def describe_storage(entry, clusters):
    """What an entry stores, in MB: FFT-TFA's moment kernels (one 6p × 6p block per frequency, p the polynomials of its
    degree) and one 6 × 6 block per pair of clusters in the same partition (FT-3), TFA's dense P, Spectral LS's kernel
    and LS-24's operator."""
    mesh = clusters.mesh
    partition_shape = mesh.partition_shape
    frequencies = partition_shape[0] * partition_shape[1] * (partition_shape[2] // 2 + 1)     # rfftn's half
    kernel_MB = 36 * 16 * frequencies / 1024**2                     # one complex 6 × 6 block per frequency
    if entry.method == "Spectral LS":
        return f"kernels {2 * kernel_MB:.0f} MB"
    if entry.method == "TFA":
        return f"P {(6 * clusters.cluster_count)**2 * 8 / 1024**2:.0f} MB"
    block = mesh.partition_size
    polynomials = get_polynomial_basis(block, resolve_degree(entry.settings["degree"], mesh)).shape[-1]
    counts = np.bincount(clusters.cluster_partition, minlength=clusters.partition_count)
    # Offline, the largest arrays: the kernels in real space before their transform, and get_correction_blocks' pair
    # counts over twice a partition, for every pair of slots in every partition.
    kernels_built_MB = polynomials**2 * 36 * 8 * clusters.partition_count / 1024**2
    pair_counts_MB = clusters.partition_count * counts.max()**2 * 8 * np.prod(2 * block) / 1024**2
    return (f"kernels {polynomials**2 * kernel_MB:.0f} MB ({polynomials} polynomials), corrections "
            f"{np.sum(counts**2) * 36 * 8 / 1024**2:.0f} MB; offline needs more than "
            f"{max(kernels_built_MB, 2 * pair_counts_MB):.0f} MB")

def list_cache():
    """Lists the files in code/cache/: E and P, references."""
    rows = [("File", "MB", "Saved", "What")]
    for path, description in get_cache_entries():
        rows.append((path.name, f"{path.stat().st_size / 1024**2:.1f}",
                     datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"), description))
    print("\n".join(format_table(rows)))
    print(f"In {config.cache_folder}; delete what you no longer need, it is solved again when asked for.")

def main():
    """Runs the command given on the command line (none: run)."""
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        run()
    elif command == "preview":
        preview()
    elif command == "replot":
        replot(sys.argv[2] if len(sys.argv) > 2 else "latest")
    elif command == "cache":
        list_cache()
    else:
        sys.exit(f"Unknown command {command!r}: use none, preview, replot [folder] or cache.")

## ------- Entries ------- ##

class Entry(NamedTuple):
    """One line of the comparison: a method, what it is called, and its settings (config.fft_tfa and the others), in a
    scheme of its method: its name, its place among the method's schemes, and what tells the entry apart within it."""
    method: str
    label: str
    settings: dict
    scheme: str
    scheme_index: int
    point: str

def format_setting(name, value):
    """How a setting shows in an entry's label: partitions or elements per side as 8² or 8³, m=, i=, degree, C0."""
    if name in ("partitions", "elements"):
        return f"{value}{'²' if is_shape_constant_along_z() else '³'}"
    return {"matrix_clusters": f"m={value}", "inclusion_clusters": f"i={value}", "degree": f"degree {value}",
            "C0": f"C0 {value}"}[name]

def get_entries():
    """Every entry config.py lists, scheme by scheme, in the order of the table. A scheme is named by its method and the
    settings its entries share that differ between the method's schemes; an entry within it, by the settings that
    differ among the scheme's entries. A method with one scheme is named by the method alone."""
    entries = []
    for method, listed in (("FFT-TFA", config.fft_tfa), ("Spectral LS", config.spectral_ls), ("TFA", config.tfa)):
        schemes = config.get_schemes(listed)
        shared = [{name: value for name, value in scheme[0].items() if all(entry[name] == value for entry in scheme)}
                  for scheme in schemes]
        names = [" ".join([method, *(format_setting(name, value) for name, value in scheme_shared.items()
                                     if any(other.get(name) != value for other in shared))])
                 for scheme_shared in shared]
        if len(set(names)) < len(names):
            names = [f"{name} ({index + 1})" for index, name in enumerate(names)]
        for index, (scheme, scheme_shared, name) in enumerate(zip(schemes, shared, names)):
            for settings in scheme:
                point = " ".join(format_setting(setting, value) for setting, value in settings.items()
                                 if setting not in scheme_shared)
                label = name if not point else f"{name} {point}" if name == method else f"{name}, {point}"
                entries.append(Entry(method, label, settings, name, index, point))
    return entries

def get_element_cluster(mesh, settings):
    """Every element's cluster for an FFT-TFA or TFA entry: k-means on each element's elastic strain concentration (36
    numbers, from elastic Spectral LS on the same elements), or with one cluster per part, the parts themselves."""
    if settings["matrix_clusters"] == settings["inclusion_clusters"] == 1:
        return cluster_by_phase(mesh)
    features = get_elastic_strain_concentration(get_element_clusters(mesh.element_shape[2]),
                                                config.k_means_features_C0)
    return cluster_by_k_means(mesh, features.reshape(-1, 36), settings["matrix_clusters"],
                              settings["inclusion_clusters"])

def get_partition_clusters(settings):
    """The clusters of an FFT-TFA or TFA entry's settings, on the shared clustered mesh."""
    mesh = get_mesh(config.clustered_mesh, settings["partitions"])
    return get_clusters(mesh, get_element_cluster(mesh, settings))

def get_entry_clusters(entry):
    """An entry's clusters: one per element for Spectral LS, partition clusters for FFT-TFA and TFA."""
    if entry.method == "Spectral LS":
        return get_element_clusters(entry.settings["elements"])
    return get_partition_clusters(entry.settings)

def get_solver(entry):
    """An entry's solver, built from its clusters and its own settings."""
    clusters = get_entry_clusters(entry)
    if entry.method == "FFT-TFA":
        return FFTTFASolver(clusters, entry.settings["degree"])
    if entry.method == "Spectral LS":
        return SpectralLSSolver(clusters, entry.settings["C0"])
    return TFASolver(clusters)

## ------- Load Path ------- ##

class LoadPath(NamedTuple):
    applied_macro_strain: np.ndarray   # the macroscopic strain at every load step, one row per step
    turning_steps: np.ndarray          # the load step of each turning point

def get_load_path():
    """Straight lines through config.load_path's turning points, config.steps_to_first_turning_point steps to the first
    and the same step length on the others."""
    turning_points = np.array(config.all_load_paths[config.load_path], dtype=float)
    step_length = np.linalg.norm(turning_points[0]) / config.steps_to_first_turning_point
    lines, start = [], np.zeros(6)
    for point in turning_points:
        step_count = max(1, round(np.linalg.norm(point - start) / step_length))
        lines.append(start + np.outer(np.arange(1, step_count + 1) / step_count, point - start))
        start = point
    return LoadPath(np.vstack(lines), np.cumsum([len(line) for line in lines]) - 1)

class LoadPathResult(NamedTuple):
    """One solver's results at every load step: the stress's volume average over the cell and each step's iterations and
    time; at field_steps only, the stress and accumulated plastic strain per cluster (cell.Clusters), a fine lattice's
    fields at every step not fitting in memory. Steps past completed_step_count were not solved and hold zeros."""
    clusters: Clusters
    macro_stress: np.ndarray
    field_steps: np.ndarray
    cluster_stress: np.ndarray
    accumulated_plastic_strain: np.ndarray
    newton_iterations_per_step: np.ndarray
    krylov_iterations_per_step: np.ndarray
    solve_time_per_step: np.ndarray
    completed_step_count: int
    failure_reason: str

def run_load_path(label, solver, applied_macro_strain, field_steps):
    """Solves every load step in turn, each from the plastic history the previous one accepted. A step the solver cannot
    solve ends its load path there."""
    clusters = solver.clusters
    plastic_history = get_unloaded_plastic_state(clusters.cluster_count)
    step_count = len(applied_macro_strain)
    macro_stress = np.zeros((step_count, 6))
    cluster_stress = np.zeros((len(field_steps), clusters.cluster_count, 6))
    accumulated_plastic_strain = np.zeros((len(field_steps), clusters.cluster_count))
    newton_iterations_per_step = np.zeros(step_count, dtype=int)
    krylov_iterations_per_step = np.zeros(step_count, dtype=int)
    solve_time_per_step = np.zeros(step_count)
    completed_step_count, failure_reason = step_count, ""

    load_steps = tqdm.tqdm(applied_macro_strain, desc=f"{label:<12}", file=sys.stdout,
                           bar_format="{desc}: {percentage:3.0f}% |{bar:25}| {n_fmt}/{total_fmt} load steps "
                                      "[{elapsed} elapsed, {remaining} left]")
    for step, macro_strain in enumerate(load_steps):
        start_time = time.perf_counter()
        try:
            step_result = solver.solve_step(macro_strain, plastic_history)
        except SolverDidNotConverge as failure:
            completed_step_count, failure_reason = step, str(failure)
            break
        solve_time_per_step[step] = time.perf_counter() - start_time
        newton_iterations_per_step[step] = step_result.newton_iteration_count
        krylov_iterations_per_step[step] = step_result.krylov_iteration_count
        macro_stress[step] = clusters.average(step_result.stress)
        for position in np.flatnonzero(field_steps == step):
            cluster_stress[position] = step_result.stress
            accumulated_plastic_strain[position] = step_result.plastic_state.accumulated_plastic_strain
        plastic_history = step_result.plastic_state
    load_steps.close()
    return LoadPathResult(clusters, macro_stress, field_steps, cluster_stress, accumulated_plastic_strain,
                          newton_iterations_per_step, krylov_iterations_per_step, solve_time_per_step,
                          completed_step_count, failure_reason)

## ------- One Entry at a Time, Each in its Own Process ------- ##

def reset_peak_memory():
    with open("/proc/self/clear_refs", "w") as clear_refs:
        clear_refs.write("5")

def solve_entry(entry, load_path, field_steps):
    """An entry's LoadPathResult, and its costs: offline, building its solver (work loaded from the cache counted as
    when it was done); online, solving the load path; and the peak memory of each, above the process's start, which
    leaves out Python and its modules (about 100 MB)."""
    start_memory, start_time = get_memory_MB("VmRSS"), time.perf_counter()
    reset_peak_memory()
    solver = get_solver(entry)
    cached_seconds, cached_peak_MB = solver.offline_from_cache
    costs = {"offline_seconds": time.perf_counter() - start_time + cached_seconds,
             "peak_offline_MB": max(get_memory_MB("VmHWM") - start_memory, cached_peak_MB)}
    reset_peak_memory()
    result = run_load_path(entry.label, solver, load_path.applied_macro_strain, field_steps)
    costs["peak_online_MB"] = get_memory_MB("VmHWM") - start_memory
    return result, costs

def run_in_own_process(function, *arguments):
    """The value of function(*arguments), computed in a process of its own, so that its peak memory is its own and
    running out of memory (config.memory_limit_GB) stops only it. Returns (value, ""), or (None, why it failed)."""
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)

    def target():
        try:
            if config.memory_limit_GB is not None:
                limit = int(config.memory_limit_GB * 1024**3)
                _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
                resource.setrlimit(resource.RLIMIT_AS, (limit if hard_limit == resource.RLIM_INFINITY
                                                        else min(limit, hard_limit), hard_limit))
            sender.send((function(*arguments), ""))
        except MemoryError:
            sender.send((None, f"ran out of memory_limit_GB = {config.memory_limit_GB}"))
        except Exception:
            sender.send((None, traceback.format_exc()))

    process = context.Process(target=target)
    process.start()
    sender.close()
    try:
        value, failure_reason = receiver.recv()
    except EOFError:
        value, failure_reason = None, "its process stopped: probably out of memory, see memory_limit_GB"
    process.join()
    return value, failure_reason

def solve_all(load_path, field_steps=None):
    """Every entry's LoadPathResult by label (None where it could not be built), and each entry's line of run.json: what
    it is, how far it got and what it cost. Fields are kept at the given load steps, by default the turning points."""
    field_steps = load_path.turning_steps if field_steps is None else field_steps
    results, entries, outcomes = {}, [], {}
    for entry in get_entries():
        # An entry listed in several schemes is solved once.
        key = (entry.method, json.dumps(entry.settings, sort_keys=True))
        if key not in outcomes:
            outcomes[key] = run_in_own_process(solve_entry, entry, load_path, field_steps)
        value, failure_reason = outcomes[key]
        if value is None:
            print(f"{entry.label} failed: {failure_reason}")
            result, costs = None, {"offline_seconds": 0.0, "peak_offline_MB": 0.0, "peak_online_MB": 0.0}
        else:
            result, costs = value
            failure_reason = result.failure_reason
        solved = result is not None
        results[entry.label] = result
        entries.append({"label": entry.label, "method": entry.method, "settings": entry.settings,
                        "scheme": entry.scheme, "scheme_index": entry.scheme_index, "point": entry.point,
                        "cluster_count": result.clusters.cluster_count if solved else 0,
                        "completed_step_count": result.completed_step_count if solved else 0,
                        "newton_iterations": int(result.newton_iterations_per_step.sum()) if solved else 0,
                        "krylov_iterations": int(result.krylov_iterations_per_step.sum()) if solved else 0,
                        "online_seconds": float(result.solve_time_per_step.sum()) if solved else 0.0,
                        **costs, "failure_reason": failure_reason})
    return results, entries

## ------- Reference ------- ##

# The reference (config.reference): Spectral LS on its own finer mesh, solved along the same load path once and
# cached (cell.get_cached) under every input that changes its answer, like E and P.

# Increase whenever a code change alters the reference, so caches made by the old code are recomputed.
reference_cache_version = 3

def get_reference_parameters(load_path):
    """Every input that changes the reference's answer, its cache key. The keys' names are hashed into the cache file's
    name, so they keep their names even where the inputs were renamed."""
    elements = config.reference["elements"]
    return {"cache_version": reference_cache_version, "discretization": "Moulinec-Suquet",
            **get_geometry_parameters(), "elements_per_side": elements, "constant_along_z": is_shape_constant_along_z(),
            "L_matrix": get_matrix_L(), "L_inclusion": get_inclusion_L(),
            "yield_stress": [phase["yield_stress"] for phase in config.all_materials[config.materials].values()],
            "hardening": [phase["hardening"] for phase in config.all_materials[config.materials].values()],
            "applied_macro_strain": load_path.applied_macro_strain, "turning_steps": load_path.turning_steps,
            # The solver inputs that change the answer to the last bit.
            "convergence_tolerance": config.convergence_tolerance,
            "residual_strain_scale_floor": config.residual_strain_scale_floor,
            "krylov_tolerance": config.krylov_tolerance,
            "line_search": [config.line_search_sufficient_decrease, config.line_search_step_reduction,
                            config.line_search_max_reductions],
            "reference_C0": config.reference["C0"]}

def get_reference(load_path):
    """The reference's LoadPathResult, with fields at the turning points (its iterations and times are not kept), from
    the cache when it matches, otherwise solved now and cached; and its line of run.json."""
    clusters = get_element_clusters(config.reference["elements"])
    voxels = get_lattice_label(clusters.mesh.element_shape)
    step_count = len(load_path.applied_macro_strain)

    def compute():
        print(f"Solving the reference: Spectral LS on {voxels} voxels, {step_count} load steps")
        start_time = time.perf_counter()
        solver = SpectralLSSolver(clusters, config.reference["C0"])
        result = run_load_path("Reference", solver, load_path.applied_macro_strain, load_path.turning_steps)
        if result.completed_step_count < step_count:
            raise RuntimeError(f"The reference stopped at load step {result.completed_step_count + 1}: "
                               f"{result.failure_reason}")
        # The arrays keep the names older caches were saved with, so those still load.
        return {"material_ids": clusters.materials.phase,
                "macroscopic_stress": result.macro_stress, "stress": result.cluster_stress,
                "accumulated_plastic_strain": result.accumulated_plastic_strain,
                "build_seconds": time.perf_counter() - start_time}

    description = f"reference, {voxels} voxels, {config.geometry}, {config.materials}, {config.load_path}"
    arrays = get_cached("reference", get_reference_parameters(load_path), description, compute)
    no_iterations = np.zeros(step_count, dtype=int)
    result = LoadPathResult(clusters, arrays["macroscopic_stress"], load_path.turning_steps, arrays["stress"],
                            arrays["accumulated_plastic_strain"], no_iterations, no_iterations, np.zeros(step_count),
                            step_count, "")
    return result, {"settings": config.reference, "voxels": voxels, "build_seconds": float(arrays["build_seconds"])}

## ------- Saving a Run ------- ##

def get_lattice_label(lattice_shape):
    """A lattice of shape [z, y, x] as "N×N" when its cells are columns through the thickness, "N×N×N_z" otherwise."""
    counts = [lattice_shape[2], lattice_shape[1]] + ([lattice_shape[0]] if lattice_shape[0] > 1 else [])
    return "×".join(map(str, counts))

def get_mid_layer_clusters(clusters):
    """The cluster of every element on the element layer at mid-thickness, which the maps show. A finer mesh's layer
    there (the reference's) lies in a coarser one's, since its elements subdivide them."""
    shape = clusters.mesh.element_shape
    return clusters.element_cluster.reshape(shape)[shape[0] // 2]

def get_cluster_maps(label, clusters):
    """Each element's cluster, partition and phase on the mid-thickness layer: what the cross-section shows."""
    layer = get_mid_layer_clusters(clusters)
    return {f"{label}/element_cluster": layer, f"{label}/element_partition": clusters.cluster_partition[layer],
            f"{label}/element_phase": clusters.materials.phase[layer]}

def save_run(load_path, results, entries, reference):
    """A new folder in config.output_folder with the settings, run.json (the run, each entry's line from solve_all and
    the reference's from get_reference) and results.npz (every result, the reference's under "Reference"), and
    code/output/latest pointing at it."""
    date = datetime.datetime.now()
    folder = config.output_folder / f"{date:%Y-%m-%d_%H%M%S}_{re.sub(r'[^A-Za-z0-9.-]+', '-', config.results_name)}"
    folder.mkdir(parents=True)
    shutil.copy(config.__file__, folder / "config.py")
    info = {"results_name": config.results_name, "date": date.isoformat(timespec="seconds"),
            "geometry": config.geometry, "materials": config.materials, "load_path": config.load_path,
            "clustered_mesh": config.clustered_mesh, "side_length_mm": 1e3 * config.cell_side,
            "reference": reference, "entries": entries}
    (folder / "run.json").write_text(json.dumps(info, indent=1))

    arrays = {"applied_macro_strain": load_path.applied_macro_strain, "turning_steps": load_path.turning_steps}
    for label, result in results.items():
        if result is None:
            continue
        layer = get_mid_layer_clusters(result.clusters)
        arrays |= get_cluster_maps(label, result.clusters)
        # The fields at the turning points in single precision, since they are only drawn.
        arrays |= {f"{label}/macro_stress": result.macro_stress,
                   f"{label}/stress": result.cluster_stress[:, layer].astype(np.float32),
                   f"{label}/accumulated_plastic_strain": result.accumulated_plastic_strain[:, layer].astype(np.float32)}
    np.savez_compressed(folder / "results.npz", **arrays)

    latest = config.output_folder / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(folder.name)
    return folder

def get_saved_run(folder):
    """A saved run's run.json, and its results.npz as a dict of arrays."""
    with np.load(folder / "results.npz") as saved:
        return json.loads((folder / "run.json").read_text()), {name: saved[name] for name in saved.files}

## ------- Tables ------- ##

def format_table(rows):
    """Rows of cells as aligned text lines: the first column left-aligned, the others right-aligned."""
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return ["   ".join(cell.ljust(width) if column == 0 else cell.rjust(width)
                       for column, (cell, width) in enumerate(zip(row, widths))) for row in rows]

def get_timing_table(info, step_count):
    """Offline: building the entry's solver. Online: solving the load steps it completed, with its Newton and Krylov
    (GMRES, or CG for Spectral LS) iterations over them. Peaks: the largest memory it held in each. Then why an entry
    stopped early."""
    rows = [("Entry", "Clusters", "Offline (s)", "Online (s)", "Newton its", "Krylov its", "Steps", "Offline peak (MB)",
             "Online peak (MB)")]
    for entry in info["entries"]:
        rows.append((entry["label"], str(entry["cluster_count"]), f"{entry['offline_seconds']:.2f}",
                     f"{entry['online_seconds']:.2f}", str(entry["newton_iterations"]), str(entry["krylov_iterations"]),
                     f"{entry['completed_step_count']}/{step_count}", f"{entry['peak_offline_MB']:.0f}",
                     f"{entry['peak_online_MB']:.0f}"))
    lines = format_table(rows)
    lines += [f"{entry['label']} stopped at step {entry['completed_step_count'] + 1}: {entry['failure_reason']}"
              for entry in info["entries"] if entry["completed_step_count"] < step_count]
    if info["reference"] is not None:
        lines.append(f"Reference: Spectral LS on {info['reference']['voxels']} voxels, built in "
                     f"{info['reference']['build_seconds']:.0f} s (not timed against the entries)")
    return "\n".join(lines)

def get_loaded_components(reference_stress):
    """The stress components the load path produces: those whose reference value reaches config.loaded_stress_fraction
    of the largest reference component somewhere along the path."""
    largest_per_component = np.abs(reference_stress).max(axis=0)
    return np.flatnonzero(largest_per_component >= config.loaded_stress_fraction * largest_per_component.max())

def get_macro_stress_errors(info, arrays, components):
    """Each entry's signed macroscopic stress error in the given components at every step it solved, in % of each
    component's largest reference value along the path, so a smaller component is read on its own scale."""
    reference_stress = arrays["Reference/macro_stress"][:, components]
    scale = np.abs(reference_stress).max(axis=0)
    errors = {}
    for entry in info["entries"]:
        solved = entry["completed_step_count"]
        if solved > 0:
            stress = arrays[f"{entry['label']}/macro_stress"][:solved, components]
            errors[entry["label"]] = 100 * (stress - reference_stress[:solved]) / scale
    return errors

def get_root_mean_square_error(error):
    """An entry's one number: the root mean square over the path of each component's error, then over the components."""
    return np.sqrt(np.mean(np.mean(error**2, axis=0)))

def get_error_table(info, macro_stress_errors, components, step_count):
    """The errors of get_macro_stress_errors summarized per component: the largest magnitude over the steps an entry
    solved, and their root mean square."""
    rows = [("Entry", *(f"σ{plots.stress_component_subscripts[component]} max / rms (%)" for component in components),
             "Steps")]
    for label, error in macro_stress_errors.items():
        rows.append((label, *(f"{np.abs(error[:, column]).max():.2f} / {np.sqrt(np.mean(error[:, column]**2)):.2f}"
                              for column in range(len(components))), f"{len(error)}/{step_count}"))
    return "\n".join([f"Macroscopic stress error against the reference (Spectral LS on {info['reference']['voxels']} "
                      "voxels), per loaded component, in % of its largest reference value along the path:",
                      *format_table(rows)])

## ------- Figures ------- ##

class Field(NamedTuple):
    """A field drawn at config.map_turning_point: one map per entry (name.png) and a profile (name_profile.png)."""
    name: str                   # its figures' file names
    label: str                  # its colour bar's and axis's label
    get_values: Callable        # its value at every element, from the maps of stress (Pa) and accumulated plastic strain
    range_setting: str          # the config.py setting that holds its colour range

# The fields drawn. Another is one more line, for example the von Mises stress (material.get_von_mises_stress), with a
# von_mises_range_MPa = (None, None) in config.py:
#   Field("von_mises_stress", "von Mises stress (MPa)", lambda stress, _: get_von_mises_stress(stress) / 1e6,
#         "von_mises_range_MPa")
fields = [Field("plastic_strain", "Accumulated plastic strain (%)", lambda _, plastic_strain: 100 * plastic_strain,
                "plastic_strain_range_percent")]

def get_scheme_indices(info):
    """Every drawn label's place among its method's schemes, which sets its colour (the reference's is 0)."""
    return {"Reference": 0, **{entry["label"]: entry.get("scheme_index", 0) for entry in info["entries"]}}

def get_finest_entries(entries):
    """Each scheme's last entry that solved a load step: the one the line figures draw."""
    finest = {}
    for entry in entries:
        if entry["completed_step_count"] > 0:
            finest[entry["method"], entry.get("scheme", entry["label"])] = entry
    return list(finest.values())

def get_scheme_points(entries, errors):
    """The error-against-time plot's schemes, from the entries that completed the load path: one dict per scheme, its
    points in the order listed."""
    schemes = {}
    for entry in entries:
        key = entry["method"], entry.get("scheme", entry["method"])
        scheme = schemes.setdefault(key, {"method": entry["method"], "index": entry.get("scheme_index", 0),
                                          "name": key[1], "points": []})
        scheme["points"].append((entry.get("point", entry["label"]), entry["online_seconds"],
                                 get_root_mean_square_error(errors[entry["label"]])))
    return list(schemes.values())

def get_cell_map(entry, arrays):
    """An entry's discretization on its element grid at mid-thickness, with its settings: k-means clusters coloured in.
    """
    label, settings = entry["label"], entry["settings"]
    cluster_ids, element_phase = arrays[f"{label}/element_cluster"], arrays[f"{label}/element_phase"]
    is_k_means = max(settings.get("matrix_clusters", 1), settings.get("inclusion_clusters", 1)) > 1
    fill = cluster_ids if is_k_means else None
    clusters_line = f"{entry['cluster_count']} {'k-means' if is_k_means else 'partition phase'} clusters"
    if entry["method"] == "Spectral LS":
        return plots.CellMap(label, f"{settings['elements']} voxels per side, C0: {settings['C0']}",
                             {"voxel": cluster_ids}, element_phase)
    if entry["method"] == "TFA":
        return plots.CellMap(label, f"{settings['partitions']} partitions per side\n{clusters_line}",
                             {"cluster": cluster_ids}, element_phase, fill)
    return plots.CellMap(label, f"{settings['partitions']} partitions per side, degree {settings['degree']}\n"
                                f"{clusters_line}",
                         {"cluster": cluster_ids, "partition": arrays[f"{label}/element_partition"]},
                         element_phase, fill)

def get_color_limits(field_maps, value_min, value_max):
    """A map figure's colour limits: the given ones, and those left at None from the finest field's map, the reference's
    or else Spectral LS's when no map is finer, so the others are read against it (values beyond it are clipped);
    otherwise from every map. Elements without a value (NaN, the inclusion in a matrix-only map) are left out."""
    finest_side = max(len(field_map.values) for field_map in field_maps)
    scale_maps = ([field_map.values for field_map in field_maps if field_map.title == "Reference"]
                  or [field_map.values for field_map in field_maps
                      if field_map.title.startswith("Spectral LS") and len(field_map.values) == finest_side]
                  or [field_map.values for field_map in field_maps])
    return (min(map(np.nanmin, scale_maps)) if value_min is None else value_min,
            max(map(np.nanmax, scale_maps)) if value_max is None else value_max)

def get_field_maps(info, arrays, turning_point, field):
    """A field's map for the reference and every entry that reached the turning point, each on its own elements, and
    each one's map of its inclusion, by label."""
    turning_step = arrays["turning_steps"][turning_point]
    labels = (["Reference"] if info["reference"] is not None else []) + [
        entry["label"] for entry in info["entries"] if entry["completed_step_count"] > turning_step]
    field_maps = [plots.FieldMap(label, field.get_values(arrays[f"{label}/stress"][turning_point],
                                                         arrays[f"{label}/accumulated_plastic_strain"][turning_point]))
                  for label in labels]
    return field_maps, {label: arrays[f"{label}/element_phase"] == 1 for label in labels}

def get_matrix_only(field_maps, inclusion_maps):
    """The maps with their inclusion elements hidden (NaN) when config.field_maps_matrix_only is on."""
    if not config.field_maps_matrix_only:
        return field_maps
    return [plots.FieldMap(field_map.title, np.where(inclusion_maps[field_map.title], np.nan, field_map.values))
            for field_map in field_maps]

def draw_load_path_figures(folder, info, arrays, drawn):
    """The figures along the load path, each scheme's finest entry in the stress-strain and stress-error figures, every
    entry in the error-against-time plot; and with the reference on, the error table of every entry."""
    applied_macro_strain = arrays["applied_macro_strain"]
    step_count = len(applied_macro_strain)
    finest = get_finest_entries(drawn)
    scheme_indices = get_scheme_indices(info)
    stress_MPa = {entry["label"]: arrays[f"{entry['label']}/macro_stress"] / 1e6 for entry in finest}
    completed_step_counts = {entry["label"]: entry["completed_step_count"] for entry in finest}
    error_table = ""
    if info["reference"] is not None:
        reference_stress = arrays["Reference/macro_stress"]
        stress_MPa = {"Reference": reference_stress / 1e6, **stress_MPa}
        completed_step_counts["Reference"] = step_count
        components = get_loaded_components(reference_stress)
        errors = get_macro_stress_errors(info, arrays, components)
        error_table = get_error_table(info, errors, components, step_count)
        plots.plot_macro_stress_errors({entry["label"]: errors[entry["label"]] for entry in finest}, components,
                                       step_count, scheme_indices, folder / "macro_stress_error.png")
        schemes = get_scheme_points([entry for entry in drawn if entry["completed_step_count"] == step_count], errors)
        if any(len(scheme["points"]) > 1 for scheme in schemes):
            plots.plot_error_against_time(schemes, folder / "error_against_time.png")
    plots.plot_stress_strain(stress_MPa, completed_step_counts, applied_macro_strain, scheme_indices,
                             folder / "stress_strain.png")
    return error_table

def draw_field_figures(folder, info, arrays):
    """Every field's maps and profile at config.map_turning_point. A saved run may have fewer turning points than
    config.py's load path."""
    turning_steps = arrays["turning_steps"]
    turning_point = (len(turning_steps) if config.map_turning_point == "last"
                     else min(config.map_turning_point, len(turning_steps))) - 1
    title = (f"Turning point {turning_point + 1} of {len(turning_steps)}: load step {turning_steps[turning_point] + 1} "
             f"of {len(arrays['applied_macro_strain'])}")
    # The profile follows the middle row of the mesh's elements, which the reference's subdivide; a map on other
    # elements (Spectral LS's) shows its row at that height. The inclusion's stretches come from the finest map.
    profile_height = (info["clustered_mesh"] // 2 + 0.5) / info["clustered_mesh"]
    reached = [entry for entry in info["entries"] if entry["completed_step_count"] > turning_steps[turning_point]]
    profile_labels = {"Reference", *(entry["label"] for entry in get_finest_entries(reached))}
    for field in fields:
        field_maps, inclusion_maps = get_field_maps(info, arrays, turning_point, field)
        if not field_maps:
            print(f"No entry reached turning point {turning_point + 1}, so no maps: see config.map_turning_point.")
            return
        shown_maps = get_matrix_only(field_maps, inclusion_maps)
        plots.plot_field_maps(shown_maps, field.label, title,
                              *get_color_limits(shown_maps, *getattr(config, field.range_setting)),
                              folder / f"{field.name}.png")
        plots.plot_field_profiles([field_map for field_map in field_maps if field_map.title in profile_labels],
                                  profile_height, max(inclusion_maps.values(), key=len), field.label, title,
                                  info["side_length_mm"], get_scheme_indices(info), folder / f"{field.name}_profile.png")

def draw_run(folder):
    """Every table and figure of a saved run, from what it saved, with config.py's figure settings: the tables printed
    and written to summary.txt."""
    info, arrays = get_saved_run(folder)
    tables = [get_timing_table(info, len(arrays["applied_macro_strain"]))]
    drawn = [entry for entry in info["entries"] if entry["completed_step_count"] > 0]
    if drawn:
        tables.append(draw_load_path_figures(folder, info, arrays, drawn))
        plots.plot_cross_section([get_cell_map(entry, arrays) for entry in drawn], folder / "cross_section.png")
        draw_field_figures(folder, info, arrays)
    summary = "\n\n".join(table for table in tables if table)
    print(f"\n{summary}\n")
    (folder / "summary.txt").write_text(summary + "\n")

if __name__ == "__main__":
    main()
