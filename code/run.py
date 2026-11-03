# Solves the problem in config.py with every method switched on there, step by step along the load path, then prints
# a timing table and saves it with the figures in code/output/:
#   timing.txt              each method's offline and online time, Newton and Krylov iterations, over the load path
#   stress_strain.png       macroscopic stress against applied strain, every method (and the reference, when it is on)
#   macro_stress_error.png  with the reference on: each method's macroscopic stress error in every loaded component
#                           along the load path, and errors.txt, its largest and root-mean-square value
#   cross_section.png       the fiber on the element grid, with each method's cells (partitions, elements, clusters),
#                           k-means clusters coloured in
#   von_mises_stress.png    von Mises stress at the last load step, one map per method (and the reference)
#   plastic_strain.png      accumulated plastic strain at the last load step, one map per method (and the reference)
#   plastic_strain_profile.png  the same along the middle element row, left to right, every method on one plot
# The maps show the element layer at mid-thickness.
#
# Run from anywhere: python code/run.py

import sys
import time
from typing import NamedTuple

import config  # first: it sets the thread count before numpy loads
import numpy as np
import tqdm

import plots
from cell import (Clusters, Mesh, get_cached, get_clusters, get_fiber_radius,
                get_geometry_parameters, get_inclusion_L, get_matrix_L, get_partition_discretization, get_periodic_mesh,
                get_spectral_discretization, get_spectral_ls_discretization, label_k_means,
                label_partition_k_means, label_partition_phases, smooth_within_phases)
from fft_tfa_model import FFTTFASolver, Interaction, get_interaction
from ls_model import SpectralLSSolver, get_elastic_strain_concentration, get_P0_transformed, get_spectral_C0
from material import get_unloaded_plastic_state, get_von_mises_stress
from newton import SolverDidNotConverge
from tfa_model import TFASolver, get_E_and_P

## ------- Problem ------- ##

# Everything a run needs, built from config.py, and what each method's offline step costs:
#   FFT-TFA      its clusters (config.fft_tfa_clusters) and its interaction on the partitions (fft_tfa_model);
#   Spectral LS  its element lattice (its own or the mesh's) and clusters, its own reference stiffness and kernel P0;
#   TFA          its clusters (config.tfa_clusters, its regions) and the FE influence functions E and P on them.
# k-means clusters are found on each element's elastic strain concentration, from Spectral LS's lattice. The reference
# (config.include_reference) comes from its cache, or is solved once (Reference, below).

class Lattice(NamedTuple):
    # What an LS method needs on its lattice: the clusters, the reference stiffness C0, and the kernel P̂0 as one
    # 6 × 6 block per frequency.
    clusters: Clusters
    C0: np.ndarray
    P0_transformed: np.ndarray

class Problem(NamedTuple):
    mesh: Mesh                   # the elements and the partitions
    fft_tfa: Interaction         # FFT-TFA's clusters and interaction
    spectral_lattice: Lattice    # Spectral LS's lattice, None when it is switched off
    tfa_clusters: Clusters       # TFA's regions, None when it is switched off
    E: np.ndarray                # TFA's influence functions, None when it is switched off
    P: np.ndarray
    reference: "Reference"       # the reference's results, None when it is switched off
    matrix_L: np.ndarray         # FFT-TFA's reference medium, the unit of the LS methods' convergence test (LS-24)
    offline_seconds: dict        # by method: everything it needs built, as if it ran alone

def get_spectral_lattice(matrix_L, inclusion_L, discretization):
    # A Spectral LS lattice, and its mesh: one partition per element (column) of the given discretization (Spectral
    # LS's, the mesh's for the k-means features, or the reference's).
    spectral_mesh = get_periodic_mesh(discretization)
    spectral_C0 = get_spectral_C0(matrix_L, inclusion_L)
    return spectral_mesh, Lattice(get_clusters(spectral_mesh, label_partition_phases(spectral_mesh)),
                                  spectral_C0,
                                  get_P0_transformed(spectral_mesh.discretization, spectral_C0))

def get_cluster_labels(setting, mesh, matrix_L, inclusion_L):
    # Every element's cluster for a clusters setting (config.fft_tfa_clusters and the others). The k-means settings
    # cluster each element's elastic strain concentration (36 numbers), from Spectral LS on the same elements, whatever
    # Spectral LS's own, smoothed within each phase (cell.smooth_within_phases).
    if setting == "partition phases":
        return label_partition_phases(mesh)
    spectral_mesh, spectral_lattice = get_spectral_lattice(
        matrix_L, inclusion_L, get_spectral_discretization(mesh.discretization.elements_per_side))
    concentration = get_elastic_strain_concentration(spectral_lattice, matrix_L)
    features = smooth_within_phases(mesh, concentration.reshape(-1, 36)[spectral_mesh.element_partition_ids])
    return label_k_means(mesh, features) if setting == "k-means" else label_partition_k_means(mesh, features)

def get_problem():
    # E and P come from the cache when it matches; their offline time is the time they took to compute.
    matrix_L, inclusion_L = get_matrix_L(), get_inclusion_L()

    start_time = time.perf_counter()
    mesh = get_periodic_mesh(get_partition_discretization())
    mesh_seconds = time.perf_counter() - start_time
    if config.inclusion_shape == "random fibers":
        fiber_radius = get_fiber_radius()
        print(f"Random fibers: radius {fiber_radius:.4g}, "
              f"{2 * fiber_radius * config.elements_per_side / config.domain_side_length:.3g} elements per diameter")
    # As the elements resolve it (their centres), which a curved shape's nominal fraction only approaches.
    print(f"Inclusion volume fraction on the elements: {np.mean(mesh.element_material_ids):.4g}")

    # The clusters of each setting in use, each built once, with its time.
    settings = {"FFT-TFA": config.fft_tfa_clusters}
    if config.include_tfa:
        settings["TFA"] = config.tfa_clusters
    clusters, cluster_seconds = {}, {}
    for setting in dict.fromkeys(settings.values()):
        start_time = time.perf_counter()
        clusters[setting] = get_clusters(mesh, get_cluster_labels(setting, mesh, matrix_L, inclusion_L))
        cluster_seconds[setting] = time.perf_counter() - start_time
        if setting != "partition phases":
            matrix_cluster_count = np.count_nonzero(clusters[setting].materials.material_ids == 0)
            print(f"{setting}: {matrix_cluster_count} matrix and "
                  f"{clusters[setting].cluster_count - matrix_cluster_count} inclusion clusters")
    offline_seconds = {method: mesh_seconds + cluster_seconds[setting] for method, setting in settings.items()}

    start_time = time.perf_counter()
    reach = 1 if config.fft_tfa_neighbour_blocks else 0
    fft_tfa = get_interaction(mesh, clusters[config.fft_tfa_clusters], matrix_L, reach)
    offline_seconds["FFT-TFA"] += time.perf_counter() - start_time

    spectral_lattice = None
    if config.include_spectral_ls:
        start_time = time.perf_counter()
        spectral_mesh, spectral_lattice = get_spectral_lattice(matrix_L, inclusion_L, get_spectral_ls_discretization())
        print(f"Spectral LS's C0: C0_1111 = {spectral_lattice.C0[0, 0] / 1e6:.4g} MPa")
        if config.spectral_ls_elements_per_side is not None:
            print(f"Inclusion volume fraction on Spectral LS's elements: "
                  f"{np.mean(spectral_mesh.element_material_ids):.4g}")
        offline_seconds["Spectral LS"] = time.perf_counter() - start_time

    tfa_clusters, E, P = None, None, None
    if config.include_tfa:
        tfa_clusters = clusters[config.tfa_clusters]
        E, P, build_seconds = get_E_and_P(mesh, tfa_clusters)
        offline_seconds["TFA"] += build_seconds["E"] + build_seconds["P"]

    reference = get_reference(matrix_L, inclusion_L) if config.include_reference else None
    return Problem(mesh, fft_tfa, spectral_lattice, tfa_clusters, E, P, reference, matrix_L, offline_seconds)

## ------- Load Path ------- ##

def get_applied_macro_strain():
    # The macroscopic strain at every load step, one row per step.
    increment_count = config.strain_increment_count
    if config.load_path_shape == "monotonic":
        load_fractions = np.linspace(1 / increment_count, 1, increment_count)
    else:
        load_fractions = np.concatenate([np.arange(1, increment_count + 1),
                                         increment_count - np.arange(1, 2 * increment_count + 1),
                                         np.arange(1 - increment_count, 1)]) / increment_count
    return np.outer(load_fractions, config.max_macro_strain)

def get_solvers(problem):
    # Every method config.py switches on, in the order of the table and the figures.
    solvers = {"FFT-TFA": FFTTFASolver(problem.fft_tfa)}
    if config.include_spectral_ls:
        solvers["Spectral LS"] = SpectralLSSolver(problem.spectral_lattice, problem.matrix_L)
    if config.include_tfa:
        solvers["TFA"] = TFASolver(problem.E, problem.P, problem.tfa_clusters)
    return solvers

class LoadPathResult(NamedTuple):
    # One method's results at every load step: the stress and accumulated plastic strain per cluster (cell.Clusters),
    # and the stress's volume average over the cell. Steps past completed_step_count were not solved and hold zeros.
    # With run_load_path's keep_every_step off, the per-cluster arrays hold one step only, the last one solved.
    clusters: tuple
    macroscopic_stress: np.ndarray
    cluster_stress_per_step: np.ndarray
    accumulated_plastic_strain_per_step: np.ndarray
    newton_iterations_per_step: np.ndarray
    krylov_iterations_per_step: np.ndarray
    solve_time_per_step: np.ndarray
    completed_step_count: int
    failure_reason: str

def run_load_path(method, solver, applied_macro_strain, keep_every_step=True):
    # Solves every load step in turn, each from the plastic history the previous one accepted. A step the method
    # cannot solve ends its load path there (the other methods carry on), unless it is the first step. Without
    # keep_every_step only the last step's per-cluster fields are kept, for a fine lattice whose fields at every step
    # would not fit in memory.
    clusters = solver.clusters
    cluster_count = clusters.cluster_count
    plastic_history = get_unloaded_plastic_state(cluster_count)

    step_count = len(applied_macro_strain)
    field_step_count = step_count if keep_every_step else 1
    macroscopic_stress = np.zeros((step_count, 6))
    cluster_stress_per_step = np.zeros((field_step_count, cluster_count, 6))
    accumulated_plastic_strain_per_step = np.zeros((field_step_count, cluster_count))
    newton_iterations_per_step = np.zeros(step_count, dtype=int)
    krylov_iterations_per_step = np.zeros(step_count, dtype=int)
    solve_time_per_step = np.zeros(step_count)
    completed_step_count = step_count
    failure_reason = ""

    load_steps = tqdm.tqdm(applied_macro_strain, desc=f"{method:<12}", file=sys.stdout,
                           bar_format="{desc}: {percentage:3.0f}% |{bar:25}| {n_fmt}/{total_fmt} load steps "
                                      "[{elapsed} elapsed, {remaining} left]")
    for step, macro_strain in enumerate(load_steps):
        start_time = time.perf_counter()
        try:
            step_result = solver.solve_step(macro_strain, plastic_history)
        except SolverDidNotConverge as failure:
            if step == 0:
                raise
            completed_step_count, failure_reason = step, str(failure)
            break
        solve_time_per_step[step] = time.perf_counter() - start_time

        newton_iterations_per_step[step] = step_result.newton_iteration_count
        krylov_iterations_per_step[step] = step_result.krylov_iteration_count
        macroscopic_stress[step] = clusters.average(step_result.stress)
        field_step = step if keep_every_step else 0
        cluster_stress_per_step[field_step] = step_result.stress
        accumulated_plastic_strain_per_step[field_step] = step_result.plastic_state.accumulated_plastic_strain
        plastic_history = step_result.plastic_state
    load_steps.close()

    return LoadPathResult(clusters, macroscopic_stress, cluster_stress_per_step, accumulated_plastic_strain_per_step,
                          newton_iterations_per_step, krylov_iterations_per_step, solve_time_per_step,
                          completed_step_count, failure_reason)

def solve_all(problem):
    # Every method's results, by method name.
    applied_macro_strain = get_applied_macro_strain()
    return {method: run_load_path(method, solver, applied_macro_strain)
            for method, solver in get_solvers(problem).items()}

## ------- Reference ------- ##

# The reference (config.include_reference): Spectral LS on its own finer mesh, solved along the same load path once and
# cached (cell.get_cached) under every input that changes its answer, like E and P.

# Increase whenever a code change alters the reference, so caches made by the old code are recomputed.
reference_cache_version = 2

class Reference(NamedTuple):
    lattice_shape: tuple                  # its voxels, [z, y, x]
    material_ids: np.ndarray              # every voxel's phase
    macroscopic_stress: np.ndarray        # at every load step
    stress: np.ndarray                    # every voxel's, at the last load step
    accumulated_plastic_strain: np.ndarray
    build_seconds: float

def get_reference(matrix_L, inclusion_L):
    # The reference from the cache when it matches, otherwise solved now and cached, with the seconds it took to build.
    discretization = get_spectral_discretization(config.reference_elements_per_side)
    applied_macro_strain = get_applied_macro_strain()
    parameters = {"cache_version": reference_cache_version, "discretization": "Moulinec-Suquet",
                  **get_geometry_parameters(),
                  "elements_per_side": discretization.elements_per_side,
                  "elements_along_z": discretization.elements_along_z,
                  "L_matrix": matrix_L, "L_inclusion": inclusion_L,
                  "yield_stress": [config.matrix_yield_stress, config.inclusion_yield_stress],
                  "hardening_modulus": [config.matrix_hardening_modulus, config.inclusion_hardening_modulus],
                  "applied_macro_strain": applied_macro_strain,
                  # The solver inputs that change the answer to the last bit.
                  "convergence_tolerance": config.convergence_tolerance,
                  "residual_strain_scale_floor": config.residual_strain_scale_floor,
                  "krylov_tolerance": config.krylov_tolerance,
                  "line_search": [config.line_search_sufficient_decrease, config.line_search_step_reduction,
                                  config.line_search_max_reductions],
                  "spectral_ls_reference_stiffness": config.spectral_ls_reference_stiffness}
    names = ("reference_material_ids", "reference_macroscopic_stress", "reference_stress",
             "reference_plastic_strain")

    def compute():
        voxel_label = get_lattice_label(discretization.partition_lattice_shape, "×")
        print(f"Solving the reference: Spectral LS on {voxel_label} voxels, {len(applied_macro_strain)} load steps")
        start_time = time.perf_counter()
        _, lattice = get_spectral_lattice(matrix_L, inclusion_L, discretization)
        solver = SpectralLSSolver(lattice, matrix_L)
        result = run_load_path("Reference", solver, applied_macro_strain, keep_every_step=False)
        if result.completed_step_count < len(applied_macro_strain):
            raise RuntimeError(f"The reference stopped at load step {result.completed_step_count + 1}: "
                               f"{result.failure_reason}")
        arrays = dict(zip(names, (lattice.clusters.materials.material_ids, result.macroscopic_stress,
                                  result.cluster_stress_per_step[0], result.accumulated_plastic_strain_per_step[0])))
        seconds = time.perf_counter() - start_time
        return arrays, {name: seconds for name in names}

    arrays, build_seconds = get_cached(names, parameters, compute)
    return Reference(discretization.partition_lattice_shape, *(arrays[name] for name in names),
                     build_seconds[names[0]])

## ------- Output ------- ##

def format_table(rows):
    # Rows of cells as aligned text lines: the first column left-aligned, the others right-aligned.
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return ["   ".join(cell.ljust(width) if column == 0 else cell.rjust(width)
                       for column, (cell, width) in enumerate(zip(row, widths))) for row in rows]

def get_timing_table(results, offline_seconds, step_count):
    # Offline: building the method's operators (get_problem). Online: solving the load steps it completed, with its
    # Newton and Krylov (GMRES, or CG for Spectral LS) iterations over them. Then why a method stopped early.
    rows = [("Method", "Offline (s)", "Online (s)", "Newton its", "Krylov its", "Steps")]
    for method, result in results.items():
        rows.append((method, f"{offline_seconds[method]:.2f}", f"{result.solve_time_per_step.sum():.2f}",
                     f"{result.newton_iterations_per_step.sum()}", f"{result.krylov_iterations_per_step.sum()}",
                     f"{result.completed_step_count}/{step_count}"))
    lines = format_table(rows)
    lines += [f"{method} stopped at step {result.completed_step_count + 1}: {result.failure_reason}"
              for method, result in results.items() if result.completed_step_count < step_count]
    return "\n".join(lines)

def get_loaded_components(reference):
    # The stress components the load path produces: those whose reference value reaches config.loaded_stress_fraction
    # of the largest reference component somewhere along the path.
    largest_per_component = np.abs(reference.macroscopic_stress).max(axis=0)
    return np.flatnonzero(largest_per_component >= config.loaded_stress_fraction * largest_per_component.max())

def get_macro_stress_errors(results, reference, components):
    # Each method's signed macroscopic stress error in the given components at every step it solved, in % of each
    # component's largest reference value along the path, so a smaller component is read on its own scale.
    reference_stress = reference.macroscopic_stress[:, components]
    scale = np.abs(reference_stress).max(axis=0)
    return {method: 100 * (result.macroscopic_stress[:result.completed_step_count, components]
                           - reference_stress[:result.completed_step_count]) / scale
            for method, result in results.items()}

def get_error_table(macro_stress_errors, components, reference, step_count):
    # The errors of get_macro_stress_errors summarized per component: the largest magnitude over the steps a method
    # solved, and their root mean square.
    voxel_label = get_lattice_label(reference.lattice_shape, "×")
    rows = [("Method", *(f"σ{plots.stress_component_subscripts[component]} max / rms (%)"
                         for component in components), "Steps")]
    for method, error in macro_stress_errors.items():
        rows.append((method, *(f"{np.abs(error[:, column]).max():.2f} / {np.sqrt(np.mean(error[:, column]**2)):.2f}"
                               for column in range(len(components))), f"{len(error)}/{step_count}"))
    return "\n".join([f"Macroscopic stress error against the reference (Spectral LS on {voxel_label} voxels), per "
                      "loaded component, in % of its largest reference value along the path:",
                      *format_table(rows)])

def get_layer_map(element_values, discretization):
    # The values of the discretization's element layer at mid-thickness on its element grid [j, i].
    side = discretization.elements_per_side
    middle_layer = discretization.elements_along_z // 2
    return element_values[middle_layer * side**2:(middle_layer + 1) * side**2].reshape(side, side)

def get_element_map(clusters, cluster_values, discretization):
    # get_layer_map of one value per cluster, each element taking its cluster's value.
    return get_layer_map(cluster_values[clusters.element_cluster], discretization)

def get_method_discretization(problem, method):
    # The elements a method's clusters lie on: Spectral LS's own, the mesh's for the others.
    return get_spectral_ls_discretization() if method == "Spectral LS" else problem.mesh.discretization

def get_lattice_label(lattice_shape, times="$\\times$"):
    # A lattice of shape [z, y, x] as "N×N" when its cells are columns through the thickness, "N×N×N_z" otherwise,
    # with the given multiplication sign (mathtext for the figures, "×" for text).
    counts = [lattice_shape[2], lattice_shape[1]] + ([lattice_shape[0]] if lattice_shape[0] > 1 else [])
    return times.join(map(str, counts))

def get_cluster_map(clusters, discretization):
    # Every element's cluster on the mid-thickness layer.
    return get_element_map(clusters, np.arange(clusters.cluster_count), discretization)

def get_clusters_label(clusters, setting):
    # A clusters setting as the cross-section states it, e.g. "40 k-means clusters".
    if setting == "partition phases":
        return f"{clusters.cluster_count} clusters: partitions' phases"
    return f"{clusters.cluster_count} {setting} clusters"

def get_fft_tfa_cell_map(clusters, setting, discretization):
    # FFT-TFA's partitions and clusters, k-means clusters coloured in.
    partition_label = get_lattice_label(discretization.partition_lattice_shape)
    cluster_map = get_cluster_map(clusters, discretization)
    settings = ", neighbour blocks" if config.fft_tfa_neighbour_blocks else ""
    return plots.CellMap("FFT-TFA", f"{partition_label} partitions{settings}\n{get_clusters_label(clusters, setting)}",
                         {"cluster": cluster_map,
                          "partition": get_layer_map(clusters.piece_partition[clusters.element_piece], discretization)},
                         None if setting == "partition phases" else cluster_map)

def get_cell_maps(problem, methods):
    # Each method's discretization on the element grid of the mid-thickness layer, with its settings: FFT-TFA's
    # partitions and clusters, Spectral LS's voxels (one element each, on its own elements with the phases they
    # see), TFA's clusters; k-means clusters coloured in.
    discretization = problem.mesh.discretization
    cell_maps = {"FFT-TFA": get_fft_tfa_cell_map(problem.fft_tfa.clusters, config.fft_tfa_clusters, discretization)}
    if problem.spectral_lattice is not None:
        spectral_clusters = problem.spectral_lattice.clusters
        spectral_discretization = get_spectral_ls_discretization()
        voxel_label = get_lattice_label(spectral_discretization.partition_lattice_shape)
        cell_maps["Spectral LS"] = plots.CellMap(
            "Spectral LS", f"{voxel_label} voxels, C0: {config.spectral_ls_reference_stiffness}",
            {"voxel": get_cluster_map(spectral_clusters, spectral_discretization)},
            element_material_ids=get_element_map(spectral_clusters, spectral_clusters.materials.material_ids,
                                                 spectral_discretization))
    if problem.tfa_clusters is not None:
        tfa_clusters = problem.tfa_clusters
        cluster_map = get_cluster_map(tfa_clusters, discretization)
        cell_maps["TFA"] = plots.CellMap(
            "TFA", get_clusters_label(tfa_clusters, config.tfa_clusters), {"cluster": cluster_map},
            None if config.tfa_clusters == "partition phases" else cluster_map)
    return [cell_maps[method] for method in methods]

def get_color_limits(field_maps, value_min, value_max):
    # A map figure's colour limits: the given ones, and those left at None from the finest field's map, the
    # reference's or else Spectral LS's when no map is finer, so the other methods are read against it (values beyond
    # it are clipped); otherwise from every map. Elements without a value (NaN, the inclusion in a matrix-only map) are
    # left out.
    finest_side = max(len(field_map.values) for field_map in field_maps)
    scale_maps = ([field_map.values for field_map in field_maps if field_map.title == "Reference"]
                  or [field_map.values for field_map in field_maps
                      if field_map.title == "Spectral LS" and len(field_map.values) == finest_side]
                  or [field_map.values for field_map in field_maps])
    return (min(map(np.nanmin, scale_maps)) if value_min is None else value_min,
            max(map(np.nanmax, scale_maps)) if value_max is None else value_max)

def get_reference_map(reference, voxel_values):
    # One value per reference voxel on its voxel layer at mid-thickness, which lies in the methods' (get_layer_map),
    # since the reference's elements subdivide theirs.
    return voxel_values.reshape(reference.lattice_shape)[reference.lattice_shape[0] // 2]

def get_matrix_only(field_maps, inclusion_maps):
    # The maps with their inclusion elements hidden (NaN) when config.field_maps_matrix_only is on, each by its own
    # inclusion map (keyed by title).
    if not config.field_maps_matrix_only:
        return field_maps
    return [plots.FieldMap(field_map.title, np.where(inclusion_maps[field_map.title], np.nan, field_map.values))
            for field_map in field_maps]

def save_figures(problem, results, applied_macro_strain, macro_stress_errors=None, loaded_components=None):
    clusters, reference = problem.fft_tfa.clusters, problem.reference
    plots.plot_cross_section(get_element_map(clusters, clusters.materials.material_ids, problem.mesh.discretization),
                             get_cell_maps(problem, results), config.output_folder / "cross_section.png")
    macroscopic_stress_MPa = {method: result.macroscopic_stress / 1e6 for method, result in results.items()}
    completed_step_counts = {method: result.completed_step_count for method, result in results.items()}
    if reference is not None:
        macroscopic_stress_MPa = {"Reference": reference.macroscopic_stress / 1e6, **macroscopic_stress_MPa}
        completed_step_counts["Reference"] = len(applied_macro_strain)
    plots.plot_stress_strain(macroscopic_stress_MPa, completed_step_counts, applied_macro_strain,
                             config.max_macro_strain, config.output_folder / "stress_strain.png")
    if macro_stress_errors is not None:
        plots.plot_macro_stress_errors(macro_stress_errors, loaded_components, len(applied_macro_strain),
                                       config.output_folder / "macro_stress_error.png")

    # The maps compare the same load: the last step every method completed. With field_maps_matrix_only the inclusion's
    # elements have no value (NaN), so the maps grey them out and take their limits from the matrix. The reference
    # keeps only its last step's fields, so it shows when every method reached that step. Each method's map lies on
    # its own elements (Spectral LS's may differ from the mesh's).
    step_count = min(result.completed_step_count for result in results.values())
    show_reference = reference is not None and step_count == len(applied_macro_strain)
    discretizations = {method: get_method_discretization(problem, method) for method in results}
    inclusion_elements = get_element_map(clusters, clusters.materials.material_ids, problem.mesh.discretization) == 1
    inclusion_maps = {method: get_element_map(result.clusters, result.clusters.materials.material_ids,
                                              discretizations[method]) == 1 for method, result in results.items()}
    von_mises_maps = [plots.FieldMap(method, get_element_map(
        result.clusters, get_von_mises_stress(result.cluster_stress_per_step[step_count - 1]) / 1e6,
        discretizations[method])) for method, result in results.items()]
    if show_reference:
        inclusion_maps["Reference"] = get_reference_map(reference, reference.material_ids) == 1
        von_mises_maps.insert(0, plots.FieldMap(
            "Reference", get_reference_map(reference, get_von_mises_stress(reference.stress) / 1e6)))
    von_mises_maps = get_matrix_only(von_mises_maps, inclusion_maps)
    plots.plot_field_maps(von_mises_maps,
                          "von Mises stress in the matrix (MPa)" if config.field_maps_matrix_only
                          else "von Mises stress (MPa)", plots.von_mises_colormap,
                          step_count, len(applied_macro_strain),
                          *get_color_limits(von_mises_maps, *config.von_mises_range_MPa),
                          config.output_folder / "von_mises_stress.png")
    plastic_strain_maps = [plots.FieldMap(method, get_element_map(
        result.clusters, 100 * result.accumulated_plastic_strain_per_step[step_count - 1], discretizations[method]))
        for method, result in results.items()]
    if show_reference:
        plastic_strain_maps.insert(0, plots.FieldMap(
            "Reference", get_reference_map(reference, 100 * reference.accumulated_plastic_strain)))
    # The profile follows the middle row of the mesh's elements, which the reference's subdivide; a map on other
    # elements (Spectral LS's) shows its row at that height.
    profile_height = (config.elements_per_side // 2 + 0.5) / config.elements_per_side
    # The contour greys the inclusion as the stress contour does; the profile keeps every value.
    plastic_strain_contours = get_matrix_only(plastic_strain_maps, inclusion_maps)
    plots.plot_field_maps(plastic_strain_contours, "Accumulated plastic strain (%)", plots.plastic_strain_colormap,
                          step_count, len(applied_macro_strain),
                          *get_color_limits(plastic_strain_contours, *config.plastic_strain_range_percent),
                          config.output_folder / "plastic_strain.png")
    inclusion_map = (get_reference_map(reference, reference.material_ids) == 1 if show_reference
                     else inclusion_elements)
    plots.plot_field_profiles(plastic_strain_maps, profile_height, inclusion_map, "Accumulated plastic strain (%)",
                              step_count, len(applied_macro_strain), 1e3 * config.domain_side_length,
                              config.output_folder / "plastic_strain_profile.png")

def main():
    problem = get_problem()
    applied_macro_strain = get_applied_macro_strain()
    results = solve_all(problem)

    table = get_timing_table(results, problem.offline_seconds, len(applied_macro_strain))
    if problem.reference is not None:
        voxel_label = get_lattice_label(problem.reference.lattice_shape, "×")
        table += (f"\nReference: Spectral LS on {voxel_label} voxels, built in {problem.reference.build_seconds:.0f} s "
                  "(not timed against the methods)")
    print(f"\n{table}\n")
    config.output_folder.mkdir(exist_ok=True)
    (config.output_folder / "timing.txt").write_text(table + "\n")
    saved_files = ["timing.txt", "stress_strain.png", "cross_section.png", "von_mises_stress.png",
                   "plastic_strain.png", "plastic_strain_profile.png"]
    loaded_components, macro_stress_errors = None, None
    if problem.reference is not None:
        loaded_components = get_loaded_components(problem.reference)
        macro_stress_errors = get_macro_stress_errors(results, problem.reference, loaded_components)
        error_table = get_error_table(macro_stress_errors, loaded_components, problem.reference,
                                      len(applied_macro_strain))
        print(f"{error_table}\n")
        (config.output_folder / "errors.txt").write_text(error_table + "\n")
        saved_files += ["errors.txt", "macro_stress_error.png"]
    save_figures(problem, results, applied_macro_strain, macro_stress_errors, loaded_components)
    print(f"{', '.join(saved_files)} saved to {config.output_folder}")

if __name__ == "__main__":
    main()
