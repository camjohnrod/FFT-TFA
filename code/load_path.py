# Steps a solver along the prescribed macroscopic strain path, carrying plastic history between steps, and runs
# every solver in interleaved repeats for a fair timing comparison.

import time
from typing import Callable, NamedTuple
import numpy as np

import config
from convergence import LoadPathAbandoned, MaterialFullySoftened, get_relative_residual
from material import get_flow_stress, get_unloaded_plastic_state, get_yielding_partitions
from online_timing import online_time_groups, online_timer

class Solver(NamedTuple):
    # solve_step(E, P, macro_strain, partition_materials, plastic_history) returns a StepResult.
    # get_model_residual(E, P, macro_strain, strain, partition_materials, plastic_history) recomputes the residual of
    # the equation the solver solves, from scratch, for the independent check after every step.
    solve_step: Callable
    get_model_residual: Callable

def check_flow_stress_is_positive(partition_materials, plastic_state):
    # Linear softening drives the flow stress to zero at finite plastic strain. Past that point the return map
    # flips the sign of the deviatoric stress, which the solvers would otherwise accept as a converged answer.
    flow_stress = get_flow_stress(partition_materials, plastic_state.accumulated_plastic_strain)
    if np.any(flow_stress <= 0):
        raise MaterialFullySoftened(f"{np.count_nonzero(flow_stress <= 0)} partitions softened to zero flow stress "
                                    f"(lowest {flow_stress.min():.3g} Pa)")

def check_residual_independently(solver_name, step, recomputed_residual, macro_strain):
    # A solver only reports convergence from its own bookkeeping, so the model residual is recomputed from scratch at
    # the strain it returned. Allowing only round-off above the convergence tolerance, anything else is a bug.
    relative_residual = get_relative_residual(recomputed_residual, macro_strain)
    if not relative_residual <= config.convergence_tolerance + config.independent_check_round_off:
        raise RuntimeError(f"{solver_name} reported convergence at step {step + 1}, but the residual recomputed from "
                           f"scratch is {relative_residual:.3e} (tolerance {config.convergence_tolerance:.0e}).")

class LoadPathResult(NamedTuple):
    macroscopic_stress: np.ndarray
    # Every partition's stress, per step, for the von Mises stress maps.
    partition_stress_per_step: np.ndarray
    # Residual evaluations per step, the same unit for every solver: a fixed point's corrections plus one, or a
    # Newton solver's Newton steps plus one.
    iterations_per_step: np.ndarray
    # Taken from the material state rather than the iteration count, which marks elastic steps only for solvers that
    # update non-yielding partitions in closed form.
    yielding_per_step: np.ndarray
    solve_time_per_step: np.ndarray
    group_time_per_step: np.ndarray
    # Steps actually solved. Entries past this are untouched zeros, so totals stay honest but anything plotted
    # has to be sliced or the curves fall to zero at the abandoned steps.
    completed_step_count: int
    failure_reason: str

def get_macroscopic_stress(stress):
    # ER-5. Every partition has the same volume, so the volume-weighted sum is a plain mean.
    return np.mean(stress, axis=0)

def run_strain_path(solver_name, solver, E, P, partition_materials):
    plastic_history = get_unloaded_plastic_state()

    step_count = config.load_step_count
    macroscopic_stress = np.zeros((step_count, 6))
    partition_stress_per_step = np.zeros((step_count, config.partition_count, 6))
    iterations_per_step = np.zeros(step_count, dtype=int)
    yielding_per_step = np.zeros(step_count, dtype=bool)
    solve_time_per_step = np.zeros(step_count)
    group_time_per_step = np.zeros((step_count, len(online_time_groups)))
    completed_step_count = step_count
    failure_reason = ""

    for step, macro_strain in enumerate(config.applied_macro_strain):
        online_timer.reset()
        start_time = time.perf_counter()
        try:
            step_result = solver.solve_step(E, P, macro_strain, partition_materials, plastic_history)
            check_flow_stress_is_positive(partition_materials, step_result.plastic_state)
        except LoadPathAbandoned as failure:
            # Failing on the first step leaves no partial result worth keeping.
            if step == 0:
                raise
            completed_step_count, failure_reason = step, str(failure)
            print(f"  {solver_name}: abandoned the load path at step {step + 1} of {step_count}: {failure_reason}.")
            break
        solve_time_per_step[step] = time.perf_counter() - start_time
        group_time_per_step[step] = online_timer.time_per_group

        # The step's timings are recorded above, and the timer is reset before the next step, so the independent
        # check below adds nothing to any timing.
        recomputed_residual = solver.get_model_residual(E, P, macro_strain, step_result.strain, partition_materials,
                                                        plastic_history)
        check_residual_independently(solver_name, step, recomputed_residual, macro_strain)
        iterations_per_step[step] = len(step_result.residual_history)
        yielding_per_step[step] = np.any(get_yielding_partitions(step_result.plastic_state, plastic_history))
        macroscopic_stress[step] = get_macroscopic_stress(step_result.stress)
        partition_stress_per_step[step] = step_result.stress
        plastic_history = step_result.plastic_state

    return LoadPathResult(macroscopic_stress, partition_stress_per_step, iterations_per_step, yielding_per_step,
                          solve_time_per_step, group_time_per_step, completed_step_count, failure_reason)

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
