# Steps a solver along the prescribed macroscopic strain path, carrying plastic history between steps, and runs
# every solver in interleaved repeats for a fair timing comparison.

import time
from typing import NamedTuple
import numpy as np

import config
from convergence import LoadPathAbandoned, MaterialFullySoftened
from material import PlasticState, get_flow_stress
from online_timing import online_time_groups, online_timer

def check_flow_stress_is_positive(partition_materials, plastic_state):
    # Linear softening drives the flow stress to zero at finite plastic strain. Past that point the return map
    # flips the sign of the deviatoric stress, which the solvers would otherwise accept as a converged answer.
    flow_stress = get_flow_stress(partition_materials, plastic_state.accumulated_plastic_strain)
    if np.any(flow_stress <= 0):
        raise MaterialFullySoftened(f"{np.count_nonzero(flow_stress <= 0)} partitions softened to zero flow stress "
                                    f"(lowest {flow_stress.min():.3g} Pa)")

class LoadPathResult(NamedTuple):
    applied_macro_strain: np.ndarray
    macroscopic_stress: np.ndarray
    iterations_per_step: np.ndarray
    solve_time_per_step: np.ndarray
    group_time_per_step: np.ndarray
    group_calls_per_step: np.ndarray
    final_stress: np.ndarray
    final_plastic_strain: np.ndarray
    # Steps actually solved. Entries past this are untouched zeros, so totals stay honest but anything plotted
    # has to be sliced or the curves fall to zero at the abandoned steps.
    completed_step_count: int
    failure_reason: str

def get_macroscopic_stress(stress):
    # ER-5. Every partition has the same volume, so the volume-weighted sum is a plain mean.
    return np.mean(stress, axis=0)

def run_strain_path(solver_name, solver, E, P, partition_materials):
    plastic_history = PlasticState(np.zeros((config.partition_count, 6)), np.zeros(config.partition_count))

    load_fractions = np.linspace(1 / config.strain_increment_count, 1, config.strain_increment_count)
    applied_macro_strain = np.outer(load_fractions, config.max_macro_strain)
    macroscopic_stress = np.zeros((config.strain_increment_count, 6))
    iterations_per_step = np.zeros(config.strain_increment_count, dtype=int)
    solve_time_per_step = np.zeros(config.strain_increment_count)
    group_time_per_step = np.zeros((config.strain_increment_count, len(online_time_groups)))
    group_calls_per_step = np.zeros((config.strain_increment_count, len(online_time_groups)), dtype=int)
    completed_step_count = config.strain_increment_count
    failure_reason = ""

    for step, macro_strain in enumerate(applied_macro_strain):
        online_timer.reset()
        start_time = time.perf_counter()
        try:
            step_stress, step_plastic_state, residual_history = solver(E, P, macro_strain, partition_materials,
                                                                       plastic_history)
            check_flow_stress_is_positive(partition_materials, step_plastic_state)
        except LoadPathAbandoned as failure:
            # Failing on the first step leaves no partial result worth keeping.
            if step == 0:
                raise
            completed_step_count, failure_reason = step, str(failure)
            print(f"  {solver_name}: abandoned the load path at step {step + 1} of {config.strain_increment_count}. "
                  f"{failure_reason}.")
            break
        solve_time_per_step[step] = time.perf_counter() - start_time
        group_time_per_step[step] = online_timer.time_per_group
        group_calls_per_step[step] = online_timer.calls_per_group
        iterations_per_step[step] = len(residual_history)
        macroscopic_stress[step] = get_macroscopic_stress(step_stress)
        stress, plastic_history = step_stress, step_plastic_state

    return LoadPathResult(applied_macro_strain, macroscopic_stress, iterations_per_step, solve_time_per_step,
                          group_time_per_step, group_calls_per_step, stress, plastic_history.plastic_strain,
                          completed_step_count, failure_reason)

def run_interleaved_repeats(get_solvers_for_one_path, E, P, partition_materials):
    # Every solver runs once per repeat, and the one that runs first rotates between repeats. Taking the fastest
    # repeat per step removes random noise, but not a bias that recurs every repeat, such as the first solver
    # paying the first-touch cost of the dense P. Solvers are built fresh for every load path, so none carries
    # state from one path into the next.
    solver_names = list(get_solvers_for_one_path())
    results = {solver_name: [] for solver_name in solver_names}
    for repeat in range(config.timing_repeat_count):
        for position in range(len(solver_names)):
            solver_name = solver_names[(repeat + position) % len(solver_names)]
            solver = get_solvers_for_one_path()[solver_name]
            results[solver_name].append(run_strain_path(solver_name, solver, E, P, partition_materials))
    return results
