# Turns load-path results into the printed comparison, the plots and the solver agreement check. Times are the
# fastest repeat per step, summed over the steps every solver completed.

import numpy as np

import config
import plots
from material import get_deviatoric_stress
from online_timing import get_online_timer_overhead_per_call

## ------- Timing Report ------- ##

def get_completed_step_count(solver_results):
    return min(result.completed_step_count for result in solver_results)

def get_fastest_repeat_per_step(solver_results, field):
    fastest_repeat_per_step = np.argmin([result.solve_time_per_step for result in solver_results], axis=0)
    return np.array([getattr(result, field) for result in solver_results])[fastest_repeat_per_step,
                                                                           np.arange(config.strain_increment_count)]

def get_solver_label(solver_name, solver_results, step_count):
    iteration_count = get_fastest_repeat_per_step(solver_results, "iterations_per_step")[:step_count].sum()
    solve_time = get_fastest_repeat_per_step(solver_results, "solve_time_per_step")[:step_count].sum()
    label = (f"{solver_name}: {iteration_count} iterations, {solve_time:.2f} s, "
             f"{1000 * solve_time / iteration_count:.2f} ms per iteration")
    last_result = solver_results[-1]
    if last_result.completed_step_count < config.strain_increment_count:
        label += f"  [stopped at step {last_result.completed_step_count + 1}: {last_result.failure_reason}]"
    return label

def get_time_per_iteration_per_group(solver_results, step_count):
    solve_time = get_fastest_repeat_per_step(solver_results, "solve_time_per_step")[:step_count].sum()
    time_per_group = get_fastest_repeat_per_step(solver_results, "group_time_per_step")[:step_count].sum(axis=0)
    iteration_count = get_fastest_repeat_per_step(solver_results, "iterations_per_step")[:step_count].sum()
    return np.append(time_per_group, solve_time - time_per_group.sum()) / iteration_count

def get_timer_overhead_fraction(solver_results, timer_overhead_per_call, step_count):
    solve_time = get_fastest_repeat_per_step(solver_results, "solve_time_per_step")[:step_count].sum()
    timed_call_count = get_fastest_repeat_per_step(solver_results, "group_calls_per_step")[:step_count].sum()
    return timer_overhead_per_call * timed_call_count / solve_time

## ------- Main ------- ##

def print_comparison(results, comparable_step_count):
    print(f"relaxation_factor = {config.relaxation_factor}")
    compared_steps = (f"the {comparable_step_count} of {config.strain_increment_count} steps every solver completed"
                      if comparable_step_count < config.strain_increment_count
                      else f"all {config.strain_increment_count} steps")
    print(f"solve times are the minimum per load step over {config.timing_repeat_count} interleaved runs of each "
          f"solver, summed over {compared_steps}")
    for solver_name, solver_results in results.items():
        print(get_solver_label(solver_name, solver_results, comparable_step_count))

    timer_overhead_per_call = get_online_timer_overhead_per_call()
    overhead_labels = []
    for solver_name, solver_results in results.items():
        overhead_fraction = get_timer_overhead_fraction(solver_results, timer_overhead_per_call, comparable_step_count)
        overhead_labels.append(f"{100 * overhead_fraction:.3f} % ({solver_name})")
    print(f"online timer overhead: {', '.join(overhead_labels)} of online time")

def get_most_complete_solver_name(results):
    # The solver that got furthest along the load path, whose stresses are plotted. The first listed wins a tie.
    return max(results, key=lambda solver_name: get_completed_step_count(results[solver_name]))

def get_solver_steps(solver_results):
    return plots.SolverSteps(iterations_per_step=solver_results[-1].iterations_per_step,
                             solve_time_per_step=get_fastest_repeat_per_step(solver_results, "solve_time_per_step"),
                             completed_step_count=get_completed_step_count(solver_results))

def save_load_path_plots(results, comparable_step_count, plotted_solver_name):
    plots.plot_load_path_summary(
        get_deviatoric_stress(results[plotted_solver_name][-1].macroscopic_stress) / 1e6, plotted_solver_name,
        {solver_name: get_solver_steps(solver_results) for solver_name, solver_results in results.items()},
        plots.get_applied_strain_description(config.max_macro_strain, config.strain_increment_count),
        config.output_folder / "load_path_summary.png")
    plots.plot_time_per_iteration_breakdown(
        {solver_name: 1000 * get_time_per_iteration_per_group(solver_results, comparable_step_count)
         for solver_name, solver_results in results.items()},
        config.output_folder / "time_per_iteration_breakdown.png")

def check_solver_agreement(results, comparable_step_count):
    # Both solvers solve the same equations, so their macroscopic stresses must agree to within solver tolerance.
    standard_stress = results["Standard"][-1].macroscopic_stress[:comparable_step_count]
    fft_stress = results["FFT-preconditioned"][-1].macroscopic_stress[:comparable_step_count]
    stress_scale = max(np.abs(standard_stress).max(), np.finfo(float).tiny)
    stress_difference = np.abs(fft_stress - standard_stress).max() / stress_scale

    passed = stress_difference <= config.solver_agreement_tolerance
    print(f"[{'PASS' if passed else 'FAIL'}] stress agreement between solvers: {stress_difference:.2e} relative "
          f"(tolerance {config.solver_agreement_tolerance:.0e}), over {comparable_step_count} of "
          f"{config.strain_increment_count} steps.")
    if not passed:
        raise RuntimeError("the standard and FFT-preconditioned solvers disagree on the macroscopic stress.")
