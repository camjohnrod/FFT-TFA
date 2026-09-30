# Turns load-path results into the printed comparison, the saved results, the plots and the solver agreement check.
# Times are the fastest repeat per step, summed over the steps every solver completed. Every output file is written to
# config.output_folder and named after the entry point's run name.

import numpy as np

import config
import plots
from material import get_deviatoric_stress
from online_timing import get_online_timer_overhead_per_call, online_time_groups

## ------- Timing Report ------- ##

def get_completed_step_count(solver_results):
    return min(result.completed_step_count for result in solver_results)

def get_comparable_step_count(results):
    # Timings and stresses are compared only over the steps every solver completed.
    return min(get_completed_step_count(solver_results) for solver_results in results.values())

def get_fastest_repeat_per_step(solver_results, field):
    fastest_repeat_per_step = np.argmin([result.solve_time_per_step for result in solver_results], axis=0)
    return np.array([getattr(result, field) for result in solver_results])[fastest_repeat_per_step,
                                                                           np.arange(config.strain_increment_count)]

def get_total_over_steps(solver_results, field, step_count):
    return get_fastest_repeat_per_step(solver_results, field)[:step_count].sum(axis=0)

def get_solver_label(solver_name, solver_results, step_count):
    iteration_count = get_total_over_steps(solver_results, "iterations_per_step", step_count)
    solve_time = get_total_over_steps(solver_results, "solve_time_per_step", step_count)
    label = (f"{solver_name}: {iteration_count} iterations, {solve_time:.2f} s, "
             f"{1000 * solve_time / iteration_count:.2f} ms per iteration")
    last_result = solver_results[-1]
    if last_result.completed_step_count < config.strain_increment_count:
        label += f"  [stopped at step {last_result.completed_step_count + 1}: {last_result.failure_reason}]"
    return label

def get_time_per_iteration_per_group(solver_results, step_count):
    # Per outer iteration: one fixed-point iteration, or one Newton step including its whole Krylov solve. Operator
    # applications made inside another group, as a Newton step's Krylov solve does, are taken out of that group and
    # added to the induced-strain group, so all operator work shows as one segment and no time is counted twice.
    # The last entry, "Other", is the solve time no timed group covers.
    solve_time = get_total_over_steps(solver_results, "solve_time_per_step", step_count)
    nested_operator_time_per_group = get_total_over_steps(solver_results, "nested_operator_time_per_group_per_step",
                                                          step_count)
    time_per_group = get_total_over_steps(solver_results, "group_time_per_step", step_count)
    time_per_group = time_per_group - nested_operator_time_per_group
    time_per_group[online_time_groups.index("induced_strain")] += nested_operator_time_per_group.sum()
    iteration_count = get_total_over_steps(solver_results, "iterations_per_step", step_count)
    return np.append(time_per_group, solve_time - time_per_group.sum()) / iteration_count

def get_timer_overhead_fraction(solver_results, timer_overhead_per_call, step_count):
    solve_time = get_total_over_steps(solver_results, "solve_time_per_step", step_count)
    timed_call_count = get_total_over_steps(solver_results, "group_calls_per_step", step_count).sum()
    return timer_overhead_per_call * timed_call_count / solve_time

## ------- Printed Comparison and Checks ------- ##

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

def print_independent_check(results):
    # run_strain_path already stops the run if any step fails the check, so reaching here means every step passed.
    worst_residual = max(result.recomputed_residual_per_step[:result.completed_step_count].max()
                         for solver_results in results.values() for result in solver_results)
    print(f"[PASS] independent residual check: every completed step of every run is converged when its residual is "
          f"recomputed from scratch (worst {worst_residual:.2e}, tolerance {config.fixed_point_tolerance:.0e}).")

def check_solver_agreement(results, solver_names, comparable_step_count):
    # These solvers solve the same equations, so each one's macroscopic stress must match the first one's to within
    # solver tolerance.
    reference_name, *other_names = solver_names
    reference_stress = results[reference_name][-1].macroscopic_stress[:comparable_step_count]
    stress_scale = max(np.abs(reference_stress).max(), np.finfo(float).tiny)
    for solver_name in other_names:
        stress = results[solver_name][-1].macroscopic_stress[:comparable_step_count]
        stress_difference = np.abs(stress - reference_stress).max() / stress_scale
        passed = stress_difference <= config.solver_agreement_tolerance
        print(f"[{'PASS' if passed else 'FAIL'}] stress agreement, {solver_name} vs {reference_name}: "
              f"{stress_difference:.2e} relative (tolerance {config.solver_agreement_tolerance:.0e}), over "
              f"{comparable_step_count} of {config.strain_increment_count} steps.")
        if not passed:
            raise RuntimeError(f"{solver_name} and {reference_name} disagree on the macroscopic stress.")

## ------- Output Files ------- ##

def remove_old_outputs(run_name):
    # A file this run does not write would otherwise linger from an older configuration and look current. Only this
    # entry point's own files are removed.
    config.output_folder.mkdir(exist_ok=True)
    for path in config.output_folder.glob(f"{run_name}_*"):
        path.unlink()

def save_results(results, run_name):
    # Each solver's per-step results, with the problem they solve, so another entry point can check its own solvers
    # against these, but only when every problem parameter matches.
    arrays = {f"problem/{name}": np.asarray(getattr(config, name)) for name in config.problem_parameter_names}
    arrays["fixed_point_tolerance"] = np.asarray(config.fixed_point_tolerance)
    for solver_name, solver_results in results.items():
        result = solver_results[-1]
        arrays[f"{solver_name}/macroscopic_stress"] = result.macroscopic_stress
        arrays[f"{solver_name}/iterations_per_step"] = result.iterations_per_step
        arrays[f"{solver_name}/final_plastic_strain"] = result.final_plastic_strain
        arrays[f"{solver_name}/completed_step_count"] = np.asarray(result.completed_step_count)
    results_path = config.output_folder / f"{run_name}_results.npz"
    np.savez(results_path, **arrays)
    print(f"results saved to {results_path}")

def save_cross_section_plot(partition_material_ids):
    # The same geometry for every entry point, so one shared file without a run-name prefix.
    plots.plot_cross_section(
        partition_material_ids.reshape(config.partition_number_per_side, config.partition_number_per_side),
        config.element_number_per_side, config.output_folder / "cross_section.png")

def get_most_complete_solver_name(results):
    # The solver that got furthest along the load path, whose stresses are plotted. The first listed wins a tie.
    return max(results, key=lambda solver_name: get_completed_step_count(results[solver_name]))

def get_elastic_steps(result):
    # Steps where no partition yielded. Steps past a stopped solver's last are not elastic, only unsolved.
    elastic_steps = ~result.yielding_per_step
    elastic_steps[result.completed_step_count:] = False
    return elastic_steps

def get_solver_steps(solver_results):
    return plots.SolverSteps(iterations_per_step=solver_results[-1].iterations_per_step,
                             solve_time_per_step=get_fastest_repeat_per_step(solver_results, "solve_time_per_step"),
                             completed_step_count=get_completed_step_count(solver_results))

def save_load_path_plots(results, comparable_step_count, stress_solver_name, run_name):
    stress_result = results[stress_solver_name][-1]
    plots.plot_load_path_summary(
        get_deviatoric_stress(stress_result.macroscopic_stress) / 1e6, stress_solver_name,
        {solver_name: get_solver_steps(solver_results) for solver_name, solver_results in results.items()},
        get_elastic_steps(stress_result),
        plots.get_applied_strain_description(config.max_macro_strain, config.strain_increment_count),
        config.output_folder / f"{run_name}_load_path_summary.png")
    plots.plot_time_per_iteration_breakdown(
        {solver_name: 1000 * get_time_per_iteration_per_group(solver_results, comparable_step_count)
         for solver_name, solver_results in results.items()},
        config.output_folder / f"{run_name}_time_breakdown.png")
