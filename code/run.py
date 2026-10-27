# Solves the problem in config.py with every method switched on there, step by step along the load path, then prints
# a timing table and saves it with the figures in code/output/:
#   timing.txt              each method's offline and online time, Newton and GMRES iterations, over the load path
#   stress_strain.png       macroscopic stress against applied strain, every method
#   cross_section.png       the fiber on the element grid, with each method's cells (partitions, elements, clusters)
#   von_mises_stress.png    von Mises stress at the last load step, one map per method
#   plastic_strain.png      accumulated plastic strain at the last load step, one map per method
#
# Run from anywhere: python code/run.py

import sys
import time
from typing import NamedTuple

import config  # first: it sets the thread count before numpy loads
import numpy as np
import tqdm

import plots
from fe import get_problem
from ls_model import get_ls_solver
from material import get_flow_stress, get_unloaded_plastic_state, get_von_mises_stress
from newton import LoadPathAbandoned, MaterialFullySoftened
from tfa_model import get_tfa_solver

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
    solvers = {"CP": get_ls_solver(problem.partitions, problem.reference_compliance)}
    if config.include_spectral_ls:
        solvers["Spectral LS"] = get_ls_solver(problem.spectral_lattice, problem.reference_compliance)
    if config.include_tfa_model:
        solvers["TFA"] = get_tfa_solver(problem.E, problem.P, problem.partitions.clusters)
    return solvers

## ------- Load Path ------- ##

class LoadPathResult(NamedTuple):
    # One method's results at every load step: the stress and accumulated plastic strain per cluster (fe.Clusters), and
    # the stress's volume average over the cell. Steps past completed_step_count were not solved and hold zeros.
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
    # One value per element column on the element grid [j, i], each element taking its cluster's value (the first
    # element layer: the geometry and fields do not vary through the thickness).
    side = config.element_number_per_side
    return cluster_values[clusters.element_cluster[:side**2]].reshape(side, side)

def get_cell_maps(problem, methods):
    # Each method's cells on the element grid, with its settings: CP's partitions, Spectral LS's elements, TFA's
    # clusters (CP's partitions, the mixed ones split by phase).
    clusters = problem.partitions.clusters
    partitions, elements = config.partition_number_per_side, config.element_number_per_side
    mixed_count = np.count_nonzero(clusters.volume_fraction < 1) // 2
    cell_maps = {
        "CP": plots.CellMap("CP", f"{partitions}$\\times${partitions} partitions, {mixed_count} mixed, "
                                  f"C0: {config.reference_stiffness}", get_element_map(clusters, clusters.partition)),
        "Spectral LS": plots.CellMap("Spectral LS",
                                     f"{elements}$\\times${elements} elements, C0: {config.reference_stiffness}",
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
