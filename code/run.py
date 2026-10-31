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
from fe import (Clusters, Discretization, Mesh, get_cached, get_clusters, get_E_and_P, get_geometry_parameters,
                get_inclusion_L, get_matrix_L, get_partition_discretization, get_periodic_mesh,
                get_spectral_discretization, get_spectral_ls_discretization, is_shape_constant_along_z, label_k_means,
                label_partition_k_means, label_partition_phases, smooth_within_phases)
from fft_tfa_model import FFTTFASolver, Interaction, get_interaction
from ls_model import (CPSolver, SpectralLSSolver, get_cp_reference_L, get_elastic_strain_concentration,
                      get_named_stiffness, get_P0_transformed, get_spectral_reference_L)
from material import get_flow_stress, get_unloaded_plastic_state, get_von_mises_stress
from newton import LoadPathAbandoned, MaterialFullySoftened
from tfa_model import get_tfa_solver

## ------- Problem ------- ##

# Everything a run needs, built from config.py, and what each method's offline step costs:
#   FFT-TFA      its clusters (config.fft_tfa_clusters) and its interaction on the partitions (fft_tfa_model);
#   Spectral LS  its element lattice (its own or the mesh's) and clusters, its own reference stiffness and kernel P0;
#   TFA          its clusters (config.tfa_clusters, its regions) and the FE influence functions E and P on them;
#   CP           the partitions and its clusters (config.cp_clusters), C0, the partition-averaged kernel P0.
# k-means clusters are found on each element's elastic strain concentration, from Spectral LS's lattice. The reference
# (config.include_reference) comes from its cache, or is solved once (Reference, below).

class Lattice(NamedTuple):
    # What an LS method needs on its lattice: the clusters, the reference stiffness C0, and the kernel P̂0 as one
    # 6 × 6 block per frequency.
    clusters: Clusters
    reference_L: np.ndarray
    P0_transformed: np.ndarray

class Problem(NamedTuple):
    mesh: Mesh                   # the elements and the partitions
    fft_tfa: Interaction         # FFT-TFA's clusters and interaction
    spectral_lattice: Lattice    # Spectral LS's lattice, None when it is switched off
    tfa_clusters: Clusters       # TFA's regions, None when it is switched off
    E: np.ndarray                # TFA's influence functions, None when no method needs them
    P: np.ndarray
    cp_lattice: Lattice          # CP's lattice, None when it is switched off
    reference: "Reference"       # the reference's results, None when it is switched off
    matrix_L: np.ndarray         # FFT-TFA's reference medium, the unit of the LS methods' convergence test (LS-24)
    offline_seconds: dict        # by method: everything it needs built, as if it ran alone

def get_spectral_lattice(matrix_L, inclusion_L, discretization):
    # A Spectral LS lattice, and its mesh: one partition per element (column) of the given discretization (Spectral
    # LS's, the mesh's for the k-means features, or the reference's).
    spectral_mesh = get_periodic_mesh(discretization)
    spectral_reference_L = get_spectral_reference_L(matrix_L, inclusion_L)
    return spectral_mesh, Lattice(get_clusters(spectral_mesh, label_partition_phases(spectral_mesh)),
                                  spectral_reference_L,
                                  get_P0_transformed(spectral_mesh.discretization, spectral_reference_L))

def get_cluster_labels(setting, mesh, matrix_L, inclusion_L):
    # Every element's cluster for a clusters setting (config.fft_tfa_clusters and the others). The k-means settings
    # cluster each element's elastic strain concentration (36 numbers), from Spectral LS on the same elements, whatever
    # Spectral LS's own, smoothed within each phase (fe.smooth_within_phases).
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
    # As the elements resolve it (their centres), which a curved shape's nominal fraction only approaches.
    print(f"Inclusion volume fraction on the elements: {np.mean(mesh.element_material_ids):.4g}")

    # The clusters of each setting in use, each built once, with its time.
    settings = {"FFT-TFA": config.fft_tfa_clusters}
    if config.include_tfa:
        settings["TFA"] = config.tfa_clusters
    if config.include_cp:
        settings["CP"] = config.cp_clusters
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
        print(f"Spectral LS's C0: C0_1111 = {spectral_lattice.reference_L[0, 0] / 1e6:.4g} MPa")
        if config.spectral_ls_elements_per_side is not None:
            print(f"Inclusion volume fraction on Spectral LS's elements: "
                  f"{np.mean(spectral_mesh.element_material_ids):.4g}")
        offline_seconds["Spectral LS"] = time.perf_counter() - start_time

    # E is needed by TFA and by CP's homogenized reference stiffness, P by TFA only; both on TFA's clusters when TFA
    # runs.
    tfa_clusters = clusters[config.tfa_clusters] if config.include_tfa else None
    E, P, build_seconds = None, None, {}
    cp_needs_E = config.include_cp and config.cp_reference_stiffness == "homogenized"
    E_clusters = tfa_clusters if config.include_tfa else clusters.get(config.cp_clusters)
    if config.include_tfa or cp_needs_E:
        E, P, build_seconds = get_E_and_P(mesh, E_clusters, include_P=config.include_tfa)
    if config.include_tfa:
        offline_seconds["TFA"] += build_seconds["E"] + build_seconds["P"]

    cp_lattice = None
    if config.include_cp:
        start_time = time.perf_counter()
        cp_clusters = clusters[config.cp_clusters]
        named_clusters = E_clusters if config.cp_reference_stiffness == "homogenized" else cp_clusters
        reference_L = get_cp_reference_L(get_named_stiffness(named_clusters, E, matrix_L))
        print(f"CP's C0: C0_1111 = {reference_L[0, 0] / 1e6:.4g} MPa")
        cp_lattice = Lattice(cp_clusters, reference_L, get_P0_transformed(mesh.discretization, reference_L))
        offline_seconds["CP"] += time.perf_counter() - start_time + (build_seconds["E"] if cp_needs_E else 0)

    reference = get_reference(matrix_L, inclusion_L) if config.include_reference else None
    return Problem(mesh, fft_tfa, spectral_lattice, tfa_clusters, E, P, cp_lattice, reference, matrix_L,
                   offline_seconds)

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
        solvers["Spectral LS"] = SpectralLSSolver(problem.spectral_lattice, problem.matrix_L,
                                                  config.spectral_ls_fourier_preconditioner)
    if config.include_tfa:
        solvers["TFA"] = get_tfa_solver(problem.E, problem.P, problem.tfa_clusters)
    if config.include_cp:
        solvers["CP"] = CPSolver(problem.cp_lattice, problem.matrix_L, config.cp_fourier_preconditioner)
    return solvers

class LoadPathResult(NamedTuple):
    # One method's results at every load step: the stress and accumulated plastic strain per cluster (fe.Clusters),
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

def check_flow_stress_is_positive(materials, plastic_state):
    flow_stress = get_flow_stress(materials, plastic_state.accumulated_plastic_strain)
    if np.any(flow_stress <= 0):
        raise MaterialFullySoftened(f"{np.count_nonzero(flow_stress <= 0)} clusters softened to zero flow stress "
                                    f"(lowest {flow_stress.min():.3g} Pa)")

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
            check_flow_stress_is_positive(clusters.materials, step_result.plastic_state)
        except LoadPathAbandoned as failure:
            if step == 0:
                raise
            completed_step_count, failure_reason = step, str(failure)
            break
        solve_time_per_step[step] = time.perf_counter() - start_time

        newton_iterations_per_step[step] = len(step_result.residual_history) - 1
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
# cached (fe.get_cached) under every input that changes its answer, like E and P.

# Increase whenever a code change alters the reference, so caches made by the old code are recomputed.
reference_cache_version = 2

class Reference(NamedTuple):
    lattice_shape: tuple                  # its voxels, [z, y, x]
    material_ids: np.ndarray              # every voxel's phase
    macroscopic_stress: np.ndarray        # at every load step
    stress: np.ndarray                    # every voxel's, at the last load step
    accumulated_plastic_strain: np.ndarray
    build_seconds: float

def get_reference_discretization():
    # The reference's elements, one partition (voxel) each: one element layer for a shape constant along z, which is
    # exact there (fe.shapes_constant_along_z), cubes otherwise.
    elements = config.reference_elements_per_side
    layers = 1 if is_shape_constant_along_z() else elements
    return Discretization(elements, layers, elements, layers)

def get_reference(matrix_L, inclusion_L):
    # The reference from the cache when it matches, otherwise solved now and cached, with the seconds it took to build.
    discretization = get_reference_discretization()
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
                  "spectral_ls_reference_stiffness": config.spectral_ls_reference_stiffness,
                  "spectral_ls_fourier_preconditioner": config.spectral_ls_fourier_preconditioner}
    names = ("reference_material_ids", "reference_macroscopic_stress", "reference_stress",
             "reference_plastic_strain")

    def compute():
        voxel_label = get_lattice_label(discretization.partition_lattice_shape, "×")
        print(f"Solving the reference: Spectral LS on {voxel_label} voxels, {len(applied_macro_strain)} load steps")
        start_time = time.perf_counter()
        _, lattice = get_spectral_lattice(matrix_L, inclusion_L, discretization)
        solver = SpectralLSSolver(lattice, matrix_L, config.spectral_ls_fourier_preconditioner)
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
    # Offline: building the method's operators (fe.get_problem). Online: solving the load steps it completed, with its
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
    # The values of the discretization's element layer at mid-thickness on its element grid [j, i]. For a shape
    # constant along z every layer is the same.
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

def get_reference_label():
    # CP's C0 as config.py states it, e.g. "0.01 × matrix", or "matrix" at scale 1.
    if config.cp_reference_stiffness_scale == 1:
        return config.cp_reference_stiffness
    return f"{config.cp_reference_stiffness_scale:g} × {config.cp_reference_stiffness}"

def get_cluster_map(clusters, discretization):
    # Every element's cluster on the mid-thickness layer.
    return get_element_map(clusters, np.arange(clusters.cluster_count), discretization)

def get_clusters_label(clusters, setting):
    # A clusters setting as the cross-section states it, e.g. "40 k-means clusters".
    if setting == "partition phases":
        return f"{clusters.cluster_count} clusters: partitions' phases"
    return f"{clusters.cluster_count} {setting} clusters"

def get_partition_cell_map(method, clusters, setting, settings, discretization):
    # A method on the partitions (FFT-TFA, CP): its partitions and clusters, k-means clusters coloured in.
    partition_label = get_lattice_label(discretization.partition_lattice_shape)
    cluster_map = get_cluster_map(clusters, discretization)
    return plots.CellMap(method, f"{partition_label} partitions{settings}\n{get_clusters_label(clusters, setting)}",
                         {"cluster": cluster_map,
                          "partition": get_layer_map(clusters.piece_partition[clusters.element_piece], discretization)},
                         None if setting == "partition phases" else cluster_map)

def get_cell_maps(problem, methods):
    # Each method's discretization on the element grid of the mid-thickness layer, with its settings: FFT-TFA's and
    # CP's partitions and clusters, Spectral LS's voxels (one element each, on its own elements with the phases they
    # see), TFA's clusters; k-means clusters coloured in.
    discretization = problem.mesh.discretization
    cell_maps = {"FFT-TFA": get_partition_cell_map("FFT-TFA", problem.fft_tfa.clusters, config.fft_tfa_clusters,
                                                   ", neighbour blocks" if config.fft_tfa_neighbour_blocks else "",
                                                   discretization)}
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
    if problem.cp_lattice is not None:
        cell_maps["CP"] = get_partition_cell_map("CP", problem.cp_lattice.clusters, config.cp_clusters,
                                                 f", C0: {get_reference_label()}", discretization)
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

    # The maps compare the same load: the last step every method completed. With von_mises_matrix_only the inclusion's
    # elements have no von Mises value (NaN), so the maps grey them out and take their limits from the matrix. The
    # reference keeps only its last step's fields, so it shows when every method reached that step. Each method's map
    # lies on its own elements (Spectral LS's may differ from the mesh's).
    step_count = min(result.completed_step_count for result in results.values())
    show_reference = reference is not None and step_count == len(applied_macro_strain)
    discretizations = {method: get_method_discretization(problem, method) for method in results}
    inclusion_elements = get_element_map(clusters, clusters.materials.material_ids, problem.mesh.discretization) == 1
    von_mises_maps = []
    if show_reference:
        values = get_reference_map(reference, get_von_mises_stress(reference.stress) / 1e6)
        if config.von_mises_matrix_only:
            values = np.where(get_reference_map(reference, reference.material_ids) == 1, np.nan, values)
        von_mises_maps.append(plots.FieldMap("Reference", values))
    for method, result in results.items():
        values = get_element_map(result.clusters,
                                 get_von_mises_stress(result.cluster_stress_per_step[step_count - 1]) / 1e6,
                                 discretizations[method])
        if config.von_mises_matrix_only:
            values = np.where(get_element_map(result.clusters, result.clusters.materials.material_ids,
                                              discretizations[method]) == 1, np.nan, values)
        von_mises_maps.append(plots.FieldMap(method, values))
    plots.plot_field_maps(von_mises_maps,
                          "von Mises stress in the matrix (MPa)" if config.von_mises_matrix_only
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
    plots.plot_field_maps(plastic_strain_maps, "Accumulated plastic strain (%)", plots.plastic_strain_colormap,
                          step_count, len(applied_macro_strain),
                          *get_color_limits(plastic_strain_maps, *config.plastic_strain_range_percent),
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
