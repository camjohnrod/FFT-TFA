# Solves the problem in config.py with every method switched on there, step by step along the load path, then prints
# a timing table and saves it with the figures in code/output/:
#   timing.txt              each method's offline and online time, Newton and GMRES iterations, over the load path
#   stress_strain.png       macroscopic stress against applied strain, every method
#   cross_section.png       the fiber on the element grid, with each method's cells (partitions, elements, clusters)
#   von_mises_stress.png    von Mises stress at the last load step, one map per method
#   plastic_strain.png      accumulated plastic strain at the last load step, one map per method
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
from fe import (Clusters, get_clusters, get_E_and_P, get_inclusion_L, get_matrix_L, get_partition_discretization,
                get_periodic_mesh, get_spectral_discretization, is_shape_constant_along_z)
from ls_model import LSSolver, get_cp_reference_L, get_named_stiffness, get_P0_transformed, get_spectral_reference_L
from material import get_flow_stress, get_unloaded_plastic_state, get_von_mises_stress
from newton import LoadPathAbandoned, MaterialFullySoftened
from tfa_model import get_tfa_solver

## ------- Problem ------- ##

# Everything a run needs, built from config.py, and what each method's offline step costs:
#   CP           the partitions and their clusters, C0, the partition-averaged kernel P0;
#   Spectral LS  the element lattice and its clusters, its own reference stiffness and kernel P0;
#   TFA          the partitions' clusters (its regions) and the FE influence functions E and P on them.

class Lattice(NamedTuple):
    # What an LS method needs on its lattice: the clusters, the reference stiffness C0, and the kernel P̂0 as one
    # 6 × 6 block per frequency.
    clusters: Clusters
    reference_L: np.ndarray
    P0_transformed: np.ndarray

class Problem(NamedTuple):
    partitions: Lattice          # CP's lattice; its clusters are also TFA's regions
    spectral_lattice: Lattice    # Spectral LS's lattice, None when it is switched off
    E: np.ndarray                # TFA's influence functions, None when no method needs them
    P: np.ndarray
    matrix_L: np.ndarray         # the unit in which the LS methods judge convergence (LS-24)
    offline_seconds: dict        # by method: everything it needs built, as if it ran alone

def get_problem():
    # E and P come from the cache when it matches; their offline time is the time they took to compute.
    matrix_L, inclusion_L = get_matrix_L(), get_inclusion_L()

    start_time = time.perf_counter()
    mesh = get_periodic_mesh(get_partition_discretization())
    clusters = get_clusters(mesh)
    partition_seconds = time.perf_counter() - start_time

    # E is needed by TFA and by the homogenized reference stiffness, P by TFA only.
    E, P, build_seconds = None, None, {}
    if config.include_tfa_model or config.reference_stiffness == "homogenized":
        E, P, build_seconds = get_E_and_P(mesh, clusters, include_P=config.include_tfa_model)

    start_time = time.perf_counter()
    reference_L = get_cp_reference_L(get_named_stiffness(clusters, E, matrix_L))
    print(f"CP's C0: C0_1111 = {reference_L[0, 0] / 1e6:.4g} MPa")
    partitions = Lattice(clusters, reference_L, get_P0_transformed(mesh.discretization, reference_L))
    offline_seconds = {"CP": partition_seconds + time.perf_counter() - start_time
                             + (build_seconds["E"] if config.reference_stiffness == "homogenized" else 0)}

    spectral_lattice = None
    if config.include_spectral_ls:
        start_time = time.perf_counter()
        spectral_mesh = get_periodic_mesh(get_spectral_discretization())
        spectral_reference_L = get_spectral_reference_L(matrix_L, inclusion_L)
        print(f"Spectral LS's C0: C0_1111 = {spectral_reference_L[0, 0] / 1e6:.4g} MPa")
        spectral_lattice = Lattice(get_clusters(spectral_mesh), spectral_reference_L,
                                   get_P0_transformed(spectral_mesh.discretization, spectral_reference_L))
        offline_seconds["Spectral LS"] = time.perf_counter() - start_time

    if config.include_tfa_model:
        offline_seconds["TFA"] = partition_seconds + build_seconds["E"] + build_seconds["P"]

    return Problem(partitions, spectral_lattice, E, P, matrix_L, offline_seconds)

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
    solvers = {"CP": LSSolver(problem.partitions, problem.matrix_L, config.cp_fourier_preconditioner)}
    if config.include_spectral_ls:
        solvers["Spectral LS"] = LSSolver(problem.spectral_lattice, problem.matrix_L,
                                          config.spectral_ls_fourier_preconditioner)
    if config.include_tfa_model:
        solvers["TFA"] = get_tfa_solver(problem.E, problem.P, problem.partitions.clusters)
    return solvers

class LoadPathResult(NamedTuple):
    # One method's results at every load step: the stress and accumulated plastic strain per cluster (fe.Clusters),
    # and the stress's volume average over the cell. Steps past completed_step_count were not solved and hold zeros.
    clusters: tuple
    macroscopic_stress: np.ndarray
    cluster_stress_per_step: np.ndarray
    accumulated_plastic_strain_per_step: np.ndarray
    newton_iterations_per_step: np.ndarray
    gmres_iterations_per_step: np.ndarray
    solve_time_per_step: np.ndarray
    completed_step_count: int
    failure_reason: str

def check_flow_stress_is_positive(materials, plastic_state):
    flow_stress = get_flow_stress(materials, plastic_state.accumulated_plastic_strain)
    if np.any(flow_stress <= 0):
        raise MaterialFullySoftened(f"{np.count_nonzero(flow_stress <= 0)} clusters softened to zero flow stress "
                                    f"(lowest {flow_stress.min():.3g} Pa)")

def run_load_path(method, solver, applied_macro_strain):
    # Solves every load step in turn, each from the plastic history the previous one accepted. A step the method
    # cannot solve ends its load path there (the other methods carry on), unless it is the first step.
    clusters = solver.clusters
    cluster_count = len(clusters.partition)
    plastic_history = get_unloaded_plastic_state(cluster_count)

    step_count = len(applied_macro_strain)
    macroscopic_stress = np.zeros((step_count, 6))
    cluster_stress_per_step = np.zeros((step_count, cluster_count, 6))
    accumulated_plastic_strain_per_step = np.zeros((step_count, cluster_count))
    newton_iterations_per_step = np.zeros(step_count, dtype=int)
    gmres_iterations_per_step = np.zeros(step_count, dtype=int)
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
        gmres_iterations_per_step[step] = step_result.gmres_iteration_count
        macroscopic_stress[step] = clusters.average(step_result.stress)
        cluster_stress_per_step[step] = step_result.stress
        accumulated_plastic_strain_per_step[step] = step_result.plastic_state.accumulated_plastic_strain
        plastic_history = step_result.plastic_state
    load_steps.close()

    return LoadPathResult(clusters, macroscopic_stress, cluster_stress_per_step, accumulated_plastic_strain_per_step,
                          newton_iterations_per_step, gmres_iterations_per_step, solve_time_per_step,
                          completed_step_count, failure_reason)

def solve_all(problem):
    # Every method's results, by method name.
    applied_macro_strain = get_applied_macro_strain()
    return {method: run_load_path(method, solver, applied_macro_strain)
            for method, solver in get_solvers(problem).items()}

## ------- Output ------- ##

def get_timing_table(results, offline_seconds, step_count):
    # Offline: building the method's operators (fe.get_problem). Online: solving the load steps it completed, with its
    # Newton and GMRES iterations over them. Then why a method stopped early.
    rows = [("Method", "Offline (s)", "Online (s)", "Newton its", "GMRES its", "Steps")]
    for method, result in results.items():
        rows.append((method, f"{offline_seconds[method]:.2f}", f"{result.solve_time_per_step.sum():.2f}",
                     f"{result.newton_iterations_per_step.sum()}", f"{result.gmres_iterations_per_step.sum()}",
                     f"{result.completed_step_count}/{step_count}"))
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    lines = ["   ".join(cell.ljust(width) if column == 0 else cell.rjust(width)
                        for column, (cell, width) in enumerate(zip(row, widths))) for row in rows]
    lines += [f"{method} stopped at step {result.completed_step_count + 1}: {result.failure_reason}"
              for method, result in results.items() if result.completed_step_count < step_count]
    return "\n".join(lines)

def get_element_map(clusters, cluster_values):
    # One value per element on the element grid [j, i] of the element layer at mid-thickness, each element taking its
    # cluster's value. For a shape constant along z every layer is the same.
    side = config.element_number_per_side
    middle_layer = len(clusters.element_cluster) // side**2 // 2
    layer_elements = slice(middle_layer * side**2, (middle_layer + 1) * side**2)
    return cluster_values[clusters.element_cluster[layer_elements]].reshape(side, side)

def get_lattice_label(count_per_side):
    # "N×N" for a shape constant along z (cells are columns through the thickness), "N×N×N" otherwise.
    dimension = 2 if is_shape_constant_along_z() else 3
    return "$\\times$".join([str(count_per_side)] * dimension)

def get_reference_label():
    # CP's C0 as config.py states it, e.g. "0.01 × matrix", or "matrix" at scale 1.
    if config.reference_stiffness_scale == 1:
        return config.reference_stiffness
    return f"{config.reference_stiffness_scale:g} × {config.reference_stiffness}"

def get_cell_maps(problem, methods):
    # Each method's cells on the element grid of the mid-thickness layer, with its settings: CP's partitions, Spectral
    # LS's elements, TFA's clusters (CP's partitions, the mixed ones split by phase).
    clusters = problem.partitions.clusters
    partitions, elements = config.partition_number_per_side, config.element_number_per_side
    mixed_count = np.count_nonzero(clusters.volume_fraction < 1) // 2
    cell_maps = {
        "CP": plots.CellMap("CP", f"{get_lattice_label(partitions)} partitions, {mixed_count} mixed, "
                                  f"C0: {get_reference_label()}", get_element_map(clusters, clusters.partition)),
        "Spectral LS": plots.CellMap("Spectral LS",
                                     f"{get_lattice_label(elements)} elements, "
                                     f"C0: {config.spectral_ls_reference_stiffness}",
                                     np.arange(elements**2).reshape(elements, elements)),
        "TFA": plots.CellMap("TFA", f"{len(clusters.partition)} clusters: partitions, mixed ones split by phase",
                             get_element_map(clusters, np.arange(len(clusters.partition))))}
    return [cell_maps[method] for method in methods]

def save_figures(problem, results, applied_macro_strain):
    clusters = problem.partitions.clusters
    plots.plot_cross_section(get_element_map(clusters, clusters.materials.material_ids),
                             get_cell_maps(problem, results), config.output_folder / "cross_section.png")
    plots.plot_stress_strain({method: result.macroscopic_stress / 1e6 for method, result in results.items()},
                             {method: result.completed_step_count for method, result in results.items()},
                             applied_macro_strain, config.max_macro_strain,
                             config.output_folder / "stress_strain.png")

    # The maps compare the same load: the last step every method completed.
    step_count = min(result.completed_step_count for result in results.values())
    von_mises_maps = [plots.FieldMap(method, get_element_map(
        result.clusters, get_von_mises_stress(result.cluster_stress_per_step[step_count - 1]) / 1e6))
        for method, result in results.items()]
    plots.plot_field_maps(von_mises_maps, "von Mises stress (MPa)", step_count, len(applied_macro_strain),
                          config.von_mises_min_MPa, config.von_mises_max_MPa,
                          config.output_folder / "von_mises_stress.png")
    plastic_strain_maps = [plots.FieldMap(method, get_element_map(
        result.clusters, 100 * result.accumulated_plastic_strain_per_step[step_count - 1]))
        for method, result in results.items()]
    # Unset plastic strain limits come from Spectral LS's map, the finest field, so the other maps are read against it
    # (values beyond it are clipped); from every map if Spectral LS did not run.
    scale_maps = ([field_map.values for field_map in plastic_strain_maps if field_map.title == "Spectral LS"]
                  or [field_map.values for field_map in plastic_strain_maps])
    plastic_strain_min = config.plastic_strain_min_percent
    plastic_strain_max = config.plastic_strain_max_percent
    plots.plot_field_maps(plastic_strain_maps, "Accumulated plastic strain (%)", step_count,
                          len(applied_macro_strain),
                          min(map(np.min, scale_maps)) if plastic_strain_min is None else plastic_strain_min,
                          max(map(np.max, scale_maps)) if plastic_strain_max is None else plastic_strain_max,
                          config.output_folder / "plastic_strain.png")

def main():
    problem = get_problem()
    applied_macro_strain = get_applied_macro_strain()
    results = solve_all(problem)

    table = get_timing_table(results, problem.offline_seconds, len(applied_macro_strain))
    print(f"\n{table}\n")
    config.output_folder.mkdir(exist_ok=True)
    (config.output_folder / "timing.txt").write_text(table + "\n")
    save_figures(problem, results, applied_macro_strain)
    print(f"timing.txt, stress_strain.png, cross_section.png, von_mises_stress.png and plastic_strain.png saved to "
          f"{config.output_folder}")

if __name__ == "__main__":
    main()
