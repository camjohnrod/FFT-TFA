# The entry point (config.py lists the commands). A run solves every entry config.py lists, one after the other along
# the load path, then saves them in a new folder of code/output/ and draws them from what it saved:
#   config.py, code_version.txt  the settings, and the code's commit and uncommitted changes
#   run.json, results.npz        what each entry did, and its results (read them with get_saved_run)
#   timing.txt                   each entry's clusters, times, Newton and Krylov iterations, peak memory
#   stress_strain.png            macroscopic stress against applied strain, every entry and the reference
#   macro_stress_error.png       with the reference on: each entry's macroscopic stress error in every loaded component
#                                along the load path, and errors.txt, its largest and root-mean-square value
#   error_against_time.png       with the reference on and several entries of a method: root-mean-square error against
#                                online time
#   cross_section.png            the cell on the element grid, with each entry's partitions, elements or clusters,
#                                k-means clusters coloured in
#   von_mises_stress.png         von Mises stress at a turning point (config.map_turning_point), one map per entry
#   plastic_strain.png           accumulated plastic strain there, one map per entry
#   plastic_strain_profile.png   the same along the middle element row, left to right, every entry on one plot
# The maps show the element layer at mid-thickness. code/output/runs.csv gets one row per entry of every run, and
# code/output/latest points at the last run.

import csv
import datetime
import json
import multiprocessing
import pathlib
import re
import resource
import shutil
import subprocess
import sys
import time
import traceback
from typing import NamedTuple

import config  # first: it sets the thread count before numpy loads
import numpy as np
import tqdm

import plots
from cell import (get_cache_entries, get_cache_path, get_cached, get_clusters, get_discretization, get_fiber_radius,
                  get_geometry_parameters, get_inclusion_L, get_matrix_L, get_memory_MB, get_periodic_mesh,
                  is_shape_constant_along_z, label_partition_k_means, label_partition_phases)
from fft_tfa_model import FFTTFASolver, get_interaction, get_neighbour_offsets, resolve_reach
from ls_model import SpectralLSSolver, get_elastic_strain_concentration, get_spectral_clusters, get_spectral_lattice
from material import get_unloaded_plastic_state, get_von_mises_stress
from newton import SolverDidNotConverge
from tfa_model import TFASolver, get_E_and_P

## ------- Entries ------- ##

class Entry(NamedTuple):
    # One line of the comparison: a method, what it is called, and its settings (config.fft_tfa and the others).
    method: str
    label: str
    settings: dict

def format_setting(name, value):
    if name in ("partitions", "elements"):
        return f"{value}{'²' if is_shape_constant_along_z() else '³'}"
    return {"matrix_clusters": f"m={value}", "inclusion_clusters": f"f={value}", "reach": f"reach {value}",
            "C0": f"C0 {value}"}[name]

def get_entries():
    # Every entry config.py lists, in the order of the table and the figures. A lone entry is called by its method;
    # several are told apart by the settings that differ among them.
    entries = []
    for method, listed in (("FFT-TFA", config.fft_tfa), ("Spectral LS", config.spectral_ls), ("TFA", config.tfa)):
        for settings in listed:
            differing = [format_setting(name, value) for name, value in settings.items()
                         if any(other[name] != value for other in listed)]
            entries.append(Entry(method, " ".join([method, *differing]), settings))
    return entries

def get_cluster_labels(mesh, settings):
    # Every element's cluster: k-means on each element's elastic strain concentration (36 numbers), from Spectral LS on
    # the same elements. One cluster per part needs no features.
    if settings["matrix_clusters"] == settings["inclusion_clusters"] == 1:
        return label_partition_phases(mesh)
    lattice = get_spectral_lattice(mesh.discretization.elements_per_side, config.reference_C0)
    return label_partition_k_means(mesh, get_elastic_strain_concentration(lattice).reshape(-1, 36),
                                   settings["matrix_clusters"], settings["inclusion_clusters"])

def get_partition_clusters(settings):
    # The mesh and clusters of an FFT-TFA or TFA entry.
    mesh = get_periodic_mesh(get_discretization(config.elements_per_side, settings["partitions"]))
    return mesh, get_clusters(mesh, get_cluster_labels(mesh, settings))

# Each builder returns the entry's solver, its offline seconds and any offline memory (MB) spent earlier, in a cached
# build.

def build_fft_tfa(settings):
    start_time = time.perf_counter()
    mesh, clusters = get_partition_clusters(settings)
    solver = FFTTFASolver(get_interaction(mesh, clusters, settings["reach"]))
    return solver, time.perf_counter() - start_time, 0.0

def build_spectral_ls(settings):
    start_time = time.perf_counter()
    solver = SpectralLSSolver(get_spectral_lattice(settings["elements"], settings["C0"]))
    return solver, time.perf_counter() - start_time, 0.0

def build_tfa(settings):
    # Its offline costs count E and P as they were when computed, even when they come from the cache.
    start_time = time.perf_counter()
    mesh, clusters = get_partition_clusters(settings)
    clusters_seconds = time.perf_counter() - start_time
    E, P, build_seconds, build_peak_MB = get_E_and_P(mesh, clusters)
    return TFASolver(E, P, clusters), clusters_seconds + build_seconds, build_peak_MB

builders = {"FFT-TFA": build_fft_tfa, "Spectral LS": build_spectral_ls, "TFA": build_tfa}

## ------- Load Path ------- ##

class LoadPath(NamedTuple):
    applied_macro_strain: np.ndarray   # the macroscopic strain at every load step, one row per step
    turning_steps: np.ndarray          # the load step of each turning point

def get_load_path():
    # Straight lines through config.load's turning points, config.strain_increment_count steps to the first and the same
    # step length on the others.
    turning_points = config.load_paths[config.load]
    step_length = np.linalg.norm(turning_points[0]) / config.strain_increment_count
    lines, start = [], np.zeros(6)
    for point in turning_points:
        step_count = max(1, round(np.linalg.norm(point - start) / step_length))
        lines.append(start + np.outer(np.arange(1, step_count + 1) / step_count, point - start))
        start = point
    return LoadPath(np.vstack(lines), np.cumsum([len(line) for line in lines]) - 1)

class LoadPathResult(NamedTuple):
    # One solver's results at every load step: the stress's volume average over the cell and each step's iterations and
    # time; at field_steps only, the stress and accumulated plastic strain per cluster (cell.Clusters), a fine lattice's
    # fields at every step not fitting in memory. Steps past completed_step_count were not solved and hold zeros.
    clusters: tuple
    macroscopic_stress: np.ndarray
    field_steps: np.ndarray
    cluster_stress: np.ndarray
    accumulated_plastic_strain: np.ndarray
    newton_iterations_per_step: np.ndarray
    krylov_iterations_per_step: np.ndarray
    solve_time_per_step: np.ndarray
    completed_step_count: int
    failure_reason: str

def run_load_path(label, solver, applied_macro_strain, field_steps):
    # Solves every load step in turn, each from the plastic history the previous one accepted. A step the solver cannot
    # solve ends its load path there.
    clusters = solver.clusters
    plastic_history = get_unloaded_plastic_state(clusters.cluster_count)
    step_count = len(applied_macro_strain)
    macroscopic_stress = np.zeros((step_count, 6))
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
        macroscopic_stress[step] = clusters.average(step_result.stress)
        for position in np.flatnonzero(field_steps == step):
            cluster_stress[position] = step_result.stress
            accumulated_plastic_strain[position] = step_result.plastic_state.accumulated_plastic_strain
        plastic_history = step_result.plastic_state
    load_steps.close()
    return LoadPathResult(clusters, macroscopic_stress, field_steps, cluster_stress, accumulated_plastic_strain,
                          newton_iterations_per_step, krylov_iterations_per_step, solve_time_per_step,
                          completed_step_count, failure_reason)

## ------- One Entry at a Time, Each in its Own Process ------- ##

def reset_peak_memory():
    with open("/proc/self/clear_refs", "w") as clear_refs:
        clear_refs.write("5")

class EntryResult(NamedTuple):
    # An entry's load path (None if it could not be built) and its costs. The peaks are above the process's start, so
    # they leave out Python and its modules, about 100 MB.
    entry: Entry
    result: LoadPathResult
    offline_seconds: float
    online_seconds: float
    peak_offline_MB: float
    peak_online_MB: float
    failure_reason: str

def solve_entry(entry, applied_macro_strain, field_steps):
    start_memory = get_memory_MB("VmRSS")
    reset_peak_memory()
    solver, offline_seconds, earlier_peak_MB = builders[entry.method](entry.settings)
    peak_offline_MB = max(get_memory_MB("VmHWM") - start_memory, earlier_peak_MB)
    reset_peak_memory()
    result = run_load_path(entry.label, solver, applied_macro_strain, field_steps)
    return EntryResult(entry, result, offline_seconds, result.solve_time_per_step.sum(), peak_offline_MB,
                       get_memory_MB("VmHWM") - start_memory, result.failure_reason)

def run_in_own_process(function, *arguments):
    # The value of function(*arguments), computed in a process of its own, so that its peak memory is its own and
    # running out of memory (config.memory_limit_GB) stops only it. Returns (value, ""), or (None, why it failed).
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
    # Every entry's EntryResult, by label: each from timing_repeats runs, in turn, with its fastest times. Fields are
    # kept at the given load steps, by default the turning points.
    field_steps = load_path.turning_steps if field_steps is None else field_steps
    results = {}
    for _ in range(config.timing_repeats):
        for entry in get_entries():
            result, failure_reason = run_in_own_process(solve_entry, entry, load_path.applied_macro_strain, field_steps)
            if result is None:
                print(f"{entry.label} failed: {failure_reason}")
                result = EntryResult(entry, None, 0.0, 0.0, 0.0, 0.0, failure_reason)
            previous = results.get(entry.label)
            if previous is not None and previous.result is not None:
                result = result._replace(offline_seconds=min(result.offline_seconds, previous.offline_seconds),
                                         online_seconds=min(result.online_seconds, previous.online_seconds))
            results[entry.label] = result
    return results

## ------- Reference ------- ##

# The reference (config.include_reference): Spectral LS on its own finer mesh, solved along the same load path once and
# cached (cell.get_cached) under every input that changes its answer, like E and P.

# Increase whenever a code change alters the reference, so caches made by the old code are recomputed.
reference_cache_version = 3

class Reference(NamedTuple):
    lattice_shape: tuple                  # its voxels, [z, y, x]
    material_ids: np.ndarray              # every voxel's phase
    macroscopic_stress: np.ndarray        # at every load step
    stress: np.ndarray                    # every voxel's, at every turning point
    accumulated_plastic_strain: np.ndarray
    build_seconds: float

def get_reference_parameters(load_path):
    elements = config.reference_elements_per_side
    return {"cache_version": reference_cache_version, "discretization": "Moulinec-Suquet",
            **get_geometry_parameters(), "elements_per_side": elements, "constant_along_z": is_shape_constant_along_z(),
            "L_matrix": get_matrix_L(), "L_inclusion": get_inclusion_L(),
            "yield_stress": [phase["yield_stress"] for phase in config.material_sets[config.materials].values()],
            "hardening": [phase["hardening"] for phase in config.material_sets[config.materials].values()],
            "applied_macro_strain": load_path.applied_macro_strain, "turning_steps": load_path.turning_steps,
            # The solver inputs that change the answer to the last bit.
            "convergence_tolerance": config.convergence_tolerance,
            "residual_strain_scale_floor": config.residual_strain_scale_floor,
            "krylov_tolerance": config.krylov_tolerance,
            "line_search": [config.line_search_sufficient_decrease, config.line_search_step_reduction,
                            config.line_search_max_reductions],
            "reference_C0": config.reference_C0}

def get_reference(load_path):
    # The reference from the cache when it matches, otherwise solved now and cached, with the seconds it took to build.
    elements = config.reference_elements_per_side
    discretization = get_discretization(elements, elements)

    def compute():
        voxel_label = get_lattice_label(discretization.partition_lattice_shape)
        print(f"Solving the reference: Spectral LS on {voxel_label} voxels, {len(load_path.applied_macro_strain)} load "
              "steps")
        start_time = time.perf_counter()
        lattice = get_spectral_lattice(elements, config.reference_C0)
        result = run_load_path("Reference", SpectralLSSolver(lattice), load_path.applied_macro_strain,
                               load_path.turning_steps)
        if result.completed_step_count < len(load_path.applied_macro_strain):
            raise RuntimeError(f"The reference stopped at load step {result.completed_step_count + 1}: "
                               f"{result.failure_reason}")
        return {"material_ids": lattice.clusters.materials.material_ids,
                "macroscopic_stress": result.macroscopic_stress, "stress": result.cluster_stress,
                "accumulated_plastic_strain": result.accumulated_plastic_strain,
                "build_seconds": time.perf_counter() - start_time}

    description = (f"reference, {get_lattice_label(discretization.partition_lattice_shape)} voxels, "
                   f"{config.inclusion_shape}, {config.materials}, {config.load}")
    arrays = get_cached("reference", get_reference_parameters(load_path), description, compute)
    return Reference(discretization.partition_lattice_shape, arrays["material_ids"], arrays["macroscopic_stress"],
                     arrays["stress"], arrays["accumulated_plastic_strain"], float(arrays["build_seconds"]))

## ------- Saving a Run ------- ##

def get_lattice_label(lattice_shape):
    # A lattice of shape [z, y, x] as "N×N" when its cells are columns through the thickness, "N×N×N_z" otherwise.
    counts = [lattice_shape[2], lattice_shape[1]] + ([lattice_shape[0]] if lattice_shape[0] > 1 else [])
    return "×".join(map(str, counts))

def get_entry_discretization(entry):
    if entry.method == "Spectral LS":
        return get_discretization(entry.settings["elements"], entry.settings["elements"])
    return get_discretization(config.elements_per_side, entry.settings["partitions"])

def get_entry_arrays(entry, clusters):
    # What the cross-section of an entry needs.
    return {f"{entry.label}/element_cluster": clusters.element_cluster.reshape(
                get_entry_discretization(entry).element_lattice_shape),
            f"{entry.label}/cluster_partition": clusters.cluster_partition,
            f"{entry.label}/cluster_material_ids": clusters.materials.material_ids}

def get_reference_layer(reference, voxel_values):
    # One value per reference voxel on its voxel layer at mid-thickness, which lies in the entries' mid-thickness layer,
    # since the reference's elements subdivide theirs.
    return voxel_values.reshape(reference.lattice_shape)[reference.lattice_shape[0] // 2]

def get_code_version():
    # The commit this code stands on, and its uncommitted changes, to reproduce a saved run.
    def git(*arguments):
        return subprocess.run(["git", *arguments], capture_output=True, text=True,
                              cwd=pathlib.Path(__file__).parent).stdout
    return f"commit {git('rev-parse', 'HEAD')}\nuncommitted changes:\n{git('status', '--short')}\n{git('diff')}"

def get_entry_info(entry_result):
    result = entry_result.result
    return {"label": entry_result.entry.label, "method": entry_result.entry.method,
            "settings": entry_result.entry.settings, "offline_seconds": entry_result.offline_seconds,
            "online_seconds": entry_result.online_seconds, "peak_offline_MB": entry_result.peak_offline_MB,
            "peak_online_MB": entry_result.peak_online_MB,
            "cluster_count": 0 if result is None else result.clusters.cluster_count,
            "completed_step_count": 0 if result is None else result.completed_step_count,
            "failure_reason": entry_result.failure_reason}

def get_run_info(results, reference):
    return {"run_name": config.run_name, "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "inclusion_shape": config.inclusion_shape, "materials": config.materials, "load": config.load,
            "elements_per_side": config.elements_per_side, "side_length_mm": 1e3 * config.domain_side_length,
            "reference": None if reference is None else {"voxels": get_lattice_label(reference.lattice_shape),
                                                         "build_seconds": reference.build_seconds},
            "entries": [get_entry_info(entry_result) for entry_result in results.values()]}

def save_run(load_path, results, reference):
    # A new folder in config.output_folder with the settings, the code's version and the results, and code/output/latest
    # pointing at it.
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder = config.output_folder / f"{date}_{re.sub(r'[^A-Za-z0-9.-]+', '-', config.run_name)}"
    folder.mkdir(parents=True)
    shutil.copy(config.__file__, folder / "config.py")
    (folder / "code_version.txt").write_text(get_code_version())
    (folder / "run.json").write_text(json.dumps(get_run_info(results, reference), indent=1))

    arrays = {"applied_macro_strain": load_path.applied_macro_strain, "turning_steps": load_path.turning_steps}
    for label, entry_result in results.items():
        result = entry_result.result
        if result is None:
            continue
        arrays |= get_entry_arrays(entry_result.entry, result.clusters)
        arrays |= {f"{label}/macroscopic_stress": result.macroscopic_stress,
                   f"{label}/cluster_stress": result.cluster_stress,
                   f"{label}/accumulated_plastic_strain": result.accumulated_plastic_strain,
                   f"{label}/newton_iterations_per_step": result.newton_iterations_per_step,
                   f"{label}/krylov_iterations_per_step": result.krylov_iterations_per_step}
    if reference is not None:
        arrays |= {"Reference/macroscopic_stress": reference.macroscopic_stress,
                   "Reference/material_ids": get_reference_layer(reference, reference.material_ids),
                   "Reference/von_mises_stress": np.array([get_reference_layer(reference, get_von_mises_stress(stress))
                                                           for stress in reference.stress], dtype=np.float32),
                   "Reference/accumulated_plastic_strain": np.array(
                       [get_reference_layer(reference, strain) for strain in reference.accumulated_plastic_strain],
                       dtype=np.float32)}
    np.savez_compressed(folder / "results.npz", **arrays)

    latest = config.output_folder / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(folder.name)
    return folder

def get_saved_run(folder):
    # A saved run's run.json, and its results.npz as a dict of arrays.
    with np.load(folder / "results.npz") as saved:
        return json.loads((folder / "run.json").read_text()), {name: saved[name] for name in saved.files}

def append_to_runs_csv(folder, info, errors):
    # One row per entry of the run in code/output/runs.csv, to compare runs at a glance.
    path = config.output_folder / "runs.csv"
    is_new = not path.exists()
    with open(path, "a", newline="") as runs:
        for entry in info["entries"]:
            rms, largest = errors.get(entry["label"], ("", ""))
            row = {"folder": folder.name, "date": info["date"], "run_name": info["run_name"],
                   "shape": info["inclusion_shape"], "materials": info["materials"], "load": info["load"],
                   "elements_per_side": info["elements_per_side"], "label": entry["label"],
                   "settings": json.dumps(entry["settings"]), "clusters": entry["cluster_count"],
                   "offline_s": f"{entry['offline_seconds']:.2f}", "online_s": f"{entry['online_seconds']:.2f}",
                   "peak_offline_MB": f"{entry['peak_offline_MB']:.0f}",
                   "peak_online_MB": f"{entry['peak_online_MB']:.0f}", "steps": entry["completed_step_count"],
                   "rms_error_percent": rms, "max_error_percent": largest}
            writer = csv.DictWriter(runs, row)
            if is_new:
                writer.writeheader()
                is_new = False
            writer.writerow(row)

## ------- Tables ------- ##

def format_table(rows):
    # Rows of cells as aligned text lines: the first column left-aligned, the others right-aligned.
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return ["   ".join(cell.ljust(width) if column == 0 else cell.rjust(width)
                       for column, (cell, width) in enumerate(zip(row, widths))) for row in rows]

def get_timing_table(info, arrays):
    # Offline: building the entry's operators. Online: solving the load steps it completed, with its Newton and Krylov
    # (GMRES, or CG for Spectral LS) iterations over them. Peaks: the largest memory it held in each. Then why an entry
    # stopped early.
    step_count = len(arrays["applied_macro_strain"])
    rows = [("Entry", "Clusters", "Offline (s)", "Online (s)", "Newton its", "Krylov its", "Steps", "Offline peak (MB)",
             "Online peak (MB)")]
    for entry in info["entries"]:
        label = entry["label"]
        newton = arrays.get(f"{label}/newton_iterations_per_step", np.zeros(1, dtype=int)).sum()
        krylov = arrays.get(f"{label}/krylov_iterations_per_step", np.zeros(1, dtype=int)).sum()
        rows.append((label, str(entry["cluster_count"]), f"{entry['offline_seconds']:.2f}",
                     f"{entry['online_seconds']:.2f}", str(newton), str(krylov),
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
    # The stress components the load path produces: those whose reference value reaches config.loaded_stress_fraction
    # of the largest reference component somewhere along the path.
    largest_per_component = np.abs(reference_stress).max(axis=0)
    return np.flatnonzero(largest_per_component >= config.loaded_stress_fraction * largest_per_component.max())

def get_macro_stress_errors(info, arrays, components):
    # Each entry's signed macroscopic stress error in the given components at every step it solved, in % of each
    # component's largest reference value along the path, so a smaller component is read on its own scale.
    reference_stress = arrays["Reference/macroscopic_stress"][:, components]
    scale = np.abs(reference_stress).max(axis=0)
    errors = {}
    for entry in info["entries"]:
        solved = entry["completed_step_count"]
        if solved > 0:
            stress = arrays[f"{entry['label']}/macroscopic_stress"][:solved, components]
            errors[entry["label"]] = 100 * (stress - reference_stress[:solved]) / scale
    return errors

def get_root_mean_square_error(error):
    # An entry's one number: the root mean square over the path of each component's error, then over the components.
    return np.sqrt(np.mean(np.mean(error**2, axis=0)))

def get_error_table(info, macro_stress_errors, components, step_count):
    # The errors of get_macro_stress_errors summarized per component: the largest magnitude over the steps an entry
    # solved, and their root mean square.
    rows = [("Entry", *(f"σ{plots.stress_component_subscripts[component]} max / rms (%)" for component in components),
             "Steps")]
    for label, error in macro_stress_errors.items():
        rows.append((label, *(f"{np.abs(error[:, column]).max():.2f} / {np.sqrt(np.mean(error[:, column]**2)):.2f}"
                              for column in range(len(components))), f"{len(error)}/{step_count}"))
    return "\n".join([f"Macroscopic stress error against the reference (Spectral LS on {info['reference']['voxels']} "
                      "voxels), per loaded component, in % of its largest reference value along the path:",
                      *format_table(rows)])

## ------- Figures ------- ##

def get_mid_layer(lattice_values):
    # The element layer at mid-thickness of values on a lattice [z, y, x].
    return lattice_values[len(lattice_values) // 2]

def get_cell_map(entry, arrays):
    # An entry's discretization on its element grid at mid-thickness, with its settings: k-means clusters coloured in.
    label, settings = entry["label"], entry["settings"]
    cluster_ids = get_mid_layer(arrays[f"{label}/element_cluster"])
    material_ids = arrays[f"{label}/cluster_material_ids"][cluster_ids]
    is_k_means = max(settings.get("matrix_clusters", 1), settings.get("inclusion_clusters", 1)) > 1
    fill = cluster_ids if is_k_means else None
    clusters_line = f"{entry['cluster_count']} {'k-means' if is_k_means else 'partition phase'} clusters"
    if entry["method"] == "Spectral LS":
        return plots.CellMap(label, f"{settings['elements']} voxels per side, C0: {settings['C0']}",
                             {"voxel": cluster_ids}, material_ids)
    partition_ids = arrays[f"{label}/cluster_partition"][cluster_ids]
    if entry["method"] == "TFA":
        return plots.CellMap(label, f"{settings['partitions']} partitions per side\n{clusters_line}",
                             {"cluster": cluster_ids}, material_ids, fill)
    return plots.CellMap(label, f"{settings['partitions']} partitions per side, reach {settings['reach']}\n"
                                f"{clusters_line}",
                         {"cluster": cluster_ids, "partition": partition_ids}, material_ids, fill)

def get_color_limits(field_maps, value_min, value_max):
    # A map figure's colour limits: the given ones, and those left at None from the finest field's map, the reference's
    # or else Spectral LS's when no map is finer, so the others are read against it (values beyond it are clipped);
    # otherwise from every map. Elements without a value (NaN, the inclusion in a matrix-only map) are left out.
    finest_side = max(len(field_map.values) for field_map in field_maps)
    scale_maps = ([field_map.values for field_map in field_maps if field_map.title == "Reference"]
                  or [field_map.values for field_map in field_maps
                      if field_map.title.startswith("Spectral LS") and len(field_map.values) == finest_side]
                  or [field_map.values for field_map in field_maps])
    return (min(map(np.nanmin, scale_maps)) if value_min is None else value_min,
            max(map(np.nanmax, scale_maps)) if value_max is None else value_max)

def get_field_maps(info, arrays, turning_point, get_cluster_values, reference_name, scale):
    # One plots.FieldMap per entry that reached the turning point, the reference's first, each on its own elements, and
    # each one's map of its inclusion, by title.
    field_maps, inclusion_maps = [], {}
    if info["reference"] is not None:
        field_maps.append(plots.FieldMap("Reference", scale * arrays[f"Reference/{reference_name}"][turning_point]))
        inclusion_maps["Reference"] = arrays["Reference/material_ids"] == 1
    for entry in info["entries"]:
        label = entry["label"]
        if entry["completed_step_count"] > arrays["turning_steps"][turning_point]:
            cluster_ids = get_mid_layer(arrays[f"{label}/element_cluster"])
            field_maps.append(plots.FieldMap(label, scale * get_cluster_values(label, turning_point)[cluster_ids]))
            inclusion_maps[label] = arrays[f"{label}/cluster_material_ids"][cluster_ids] == 1
    return field_maps, inclusion_maps

def get_matrix_only(field_maps, inclusion_maps):
    # The maps with their inclusion elements hidden (NaN) when config.field_maps_matrix_only is on.
    if not config.field_maps_matrix_only:
        return field_maps
    return [plots.FieldMap(field_map.title, np.where(inclusion_maps[field_map.title], np.nan, field_map.values))
            for field_map in field_maps]

def draw_load_path_figures(folder, info, arrays, drawn):
    # The tables and figures along the load path. Returns each entry's root-mean-square and largest error for runs.csv.
    applied_macro_strain = arrays["applied_macro_strain"]
    step_count = len(applied_macro_strain)
    stress_MPa = {entry["label"]: arrays[f"{entry['label']}/macroscopic_stress"] / 1e6 for entry in drawn}
    completed_step_counts = {entry["label"]: entry["completed_step_count"] for entry in drawn}
    summary = {}
    if info["reference"] is not None:
        reference_stress = arrays["Reference/macroscopic_stress"]
        stress_MPa = {"Reference": reference_stress / 1e6, **stress_MPa}
        completed_step_counts["Reference"] = step_count
        components = get_loaded_components(reference_stress)
        errors = get_macro_stress_errors(info, arrays, components)
        error_table = get_error_table(info, errors, components, step_count)
        print(f"{error_table}\n")
        (folder / "errors.txt").write_text(error_table + "\n")
        plots.plot_macro_stress_errors(errors, components, step_count, folder / "macro_stress_error.png")
        summary = {label: (f"{get_root_mean_square_error(error):.3f}", f"{np.abs(error).max():.3f}")
                   for label, error in errors.items()}
        complete = [entry for entry in drawn if entry["completed_step_count"] == step_count]
        methods = [entry["method"] for entry in complete]
        if any(methods.count(method) > 1 for method in methods):
            plots.plot_error_against_time(
                {entry["label"]: (entry["method"], entry["online_seconds"],
                                  get_root_mean_square_error(errors[entry["label"]])) for entry in complete},
                folder / "error_against_time.png")
    plots.plot_stress_strain(stress_MPa, completed_step_counts, applied_macro_strain, folder / "stress_strain.png")
    return summary

def draw_field_figures(folder, info, arrays):
    # The maps and the profile at config.map_turning_point. A saved run may have fewer turning points than config.py's
    # load path.
    turning_steps = arrays["turning_steps"]
    turning_point = (len(turning_steps) if config.map_turning_point == "last"
                     else min(config.map_turning_point, len(turning_steps))) - 1
    title = (f"Turning point {turning_point + 1} of {len(turning_steps)}: load step {turning_steps[turning_point] + 1} "
             f"of {len(arrays['applied_macro_strain'])}")

    def get_von_mises(label, position):
        return get_von_mises_stress(arrays[f"{label}/cluster_stress"][position])

    def get_plastic_strain(label, position):
        return arrays[f"{label}/accumulated_plastic_strain"][position]

    von_mises_maps, inclusion_maps = get_field_maps(info, arrays, turning_point, get_von_mises, "von_mises_stress",
                                                    1e-6)
    if not von_mises_maps:
        print(f"No entry reached turning point {turning_point + 1}, so no maps: see config.map_turning_point.")
        return
    von_mises_maps = get_matrix_only(von_mises_maps, inclusion_maps)
    plots.plot_field_maps(von_mises_maps, ("von Mises stress in the matrix (MPa)" if config.field_maps_matrix_only
                                           else "von Mises stress (MPa)"), title,
                          *get_color_limits(von_mises_maps, *config.von_mises_range_MPa),
                          folder / "von_mises_stress.png")
    plastic_strain_maps, _ = get_field_maps(info, arrays, turning_point, get_plastic_strain,
                                            "accumulated_plastic_strain", 100)
    plastic_strain_contours = get_matrix_only(plastic_strain_maps, inclusion_maps)
    plots.plot_field_maps(plastic_strain_contours, "Accumulated plastic strain (%)", title,
                          *get_color_limits(plastic_strain_contours, *config.plastic_strain_range_percent),
                          folder / "plastic_strain.png")
    # The profile follows the middle row of the mesh's elements, which the reference's subdivide; a map on other
    # elements (Spectral LS's) shows its row at that height. The inclusion's stretches come from the finest map.
    profile_height = (info["elements_per_side"] // 2 + 0.5) / info["elements_per_side"]
    plots.plot_field_profiles(plastic_strain_maps, profile_height, max(inclusion_maps.values(), key=len),
                              "Accumulated plastic strain (%)", title, info["side_length_mm"],
                              folder / "plastic_strain_profile.png")

def draw_run(folder):
    # Every table and figure of a saved run, from what it saved, with config.py's Figures settings. Returns its run.json
    # and each entry's errors for runs.csv.
    info, arrays = get_saved_run(folder)
    table = get_timing_table(info, arrays)
    print(f"\n{table}\n")
    (folder / "timing.txt").write_text(table + "\n")
    drawn = [entry for entry in info["entries"] if entry["completed_step_count"] > 0]
    if not drawn:
        return info, {}
    summary = draw_load_path_figures(folder, info, arrays, drawn)
    plots.plot_cross_section([get_cell_map(entry, arrays) for entry in drawn], folder / "cross_section.png")
    draw_field_figures(folder, info, arrays)
    return info, summary

## ------- Commands ------- ##

def run():
    load_path = get_load_path()
    reference = None
    if config.include_reference:
        reference, failure_reason = run_in_own_process(get_reference, load_path)
        if reference is None:
            sys.exit(f"The reference failed: {failure_reason}")
    results = solve_all(load_path)
    folder = save_run(load_path, results, reference)
    info, errors = draw_run(folder)
    append_to_runs_csv(folder, info, errors)
    print(f"Saved to {folder}")

def replot(name):
    folder = config.output_folder / name
    if not (folder / "results.npz").exists():
        sys.exit(f"No saved run at {folder}")
    draw_run(folder.resolve())
    print(f"Redrawn in {folder.resolve()}")

def get_stored_description(entry, clusters):
    # What an entry stores, in MB: FFT-TFA's kernel and one 6 × 6 block per pair of clusters in partitions within reach
    # (FT-3), TFA's dense P, Spectral LS's kernel and LS-24's operator.
    discretization = get_entry_discretization(entry)
    partition_shape = discretization.partition_lattice_shape
    kernel_MB = 36 * 16 * partition_shape[0] * partition_shape[1] * (partition_shape[2] // 2 + 1) / 1024**2
    if entry.method == "Spectral LS":
        return f"kernels {2 * kernel_MB:.0f} MB"
    if entry.method == "TFA":
        return f"P {(6 * clusters.cluster_count)**2 * 8 / 1024**2:.0f} MB"
    counts = np.bincount(clusters.cluster_partition, minlength=clusters.partition_count).reshape(partition_shape)
    offsets = get_neighbour_offsets(partition_shape, resolve_reach(entry.settings["reach"], partition_shape))
    block_count = sum(np.sum(counts * np.roll(counts, tuple(offset), axis=(0, 1, 2))) for offset in offsets)
    # The largest array of get_correction_blocks: pair counts over twice a partition, every pair of ranks and
    # partitions.
    pair_counts_MB = (clusters.partition_count * counts.max()**2 * 8 / 1024**2
                      * np.prod(2 * np.array(discretization.element_lattice_shape) // np.array(partition_shape)))
    return (f"kernel {kernel_MB:.0f} MB, corrections {block_count * 36 * 8 / 1024**2:.0f} MB; offline needs more than "
            f"{2 * pair_counts_MB:.0f} MB")

def preview():
    # What config.py asks for, and what it would store, solving nothing but the elastic k-means features.
    load_path = get_load_path()
    if config.inclusion_shape == "random fibers":
        diameter = 2 * get_fiber_radius() * config.elements_per_side / config.domain_side_length
        print(f"{config.fiber_count} random fibers, {diameter:.3g} elements per diameter")
    phases = config.material_sets[config.materials]
    print(f"Materials {config.materials!r}: contrast {phases['inclusion']['E'] / phases['matrix']['E']:.3g}")
    print(f"Load {config.load!r}: {len(load_path.applied_macro_strain)} steps, turning points at steps "
          f"{', '.join(str(step + 1) for step in load_path.turning_steps)}")
    info_entries, arrays = [], {}
    for entry in get_entries():
        clusters = (get_spectral_clusters(entry.settings["elements"]) if entry.method == "Spectral LS"
                    else get_partition_clusters(entry.settings)[1])
        print(f"{entry.label}: {clusters.cluster_count} clusters, {get_stored_description(entry, clusters)}")
        arrays |= get_entry_arrays(entry, clusters)
        info_entries.append({"label": entry.label, "method": entry.method, "settings": entry.settings,
                             "cluster_count": clusters.cluster_count})
    if config.include_reference:
        is_cached = get_cache_path("reference", get_reference_parameters(load_path)).exists()
        print(f"Reference {config.reference_elements_per_side} per side: "
              f"{'cached' if is_cached else 'NOT cached, a run would solve it first'}")
    config.output_folder.mkdir(parents=True, exist_ok=True)
    path = config.output_folder / "preview_cross_section.png"
    plots.plot_cross_section([get_cell_map(entry, arrays) for entry in info_entries], path)
    print(f"Cross-section drawn in {path}")

def list_cache():
    rows = [("File", "MB", "Saved", "What")]
    for path, description in get_cache_entries():
        rows.append((path.name, f"{path.stat().st_size / 1024**2:.1f}",
                     datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"), description))
    print("\n".join(format_table(rows)))
    print(f"In {config.cache_folder}; delete what you no longer need, it is solved again when asked for.")

def main():
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

if __name__ == "__main__":
    main()
