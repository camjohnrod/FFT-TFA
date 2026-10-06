# Turns load-path results into the printed comparison, the saved results, the plots and the checks on them: solver
# agreement, agreement with another entry point's saved results, the matched-stiffness control and the laminate's
# exact solution. The first section is what both entry points call; the rest is how it is done.
# Times are the fastest repeat per step, summed over the steps every solver completed. Every output file is written to
# config.output_folder and named after the entry point's run name.

from typing import NamedTuple
import numpy as np

import config
import plots
from laminate import get_laminate_phase_stress
from material import get_deviatoric_stress, get_equivalent_stress

## ------- What Every Entry Point Reports and Checks ------- ##

class SolverRoles(NamedTuple):
    # Each solver of an entry point by the role it plays, under the name it reports with; None for a role that is not
    # run. ls is partitioned LS, the method being tested. TFA, when included, is its model baseline, and its
    # FFT-preconditioned solver is checked against the plain one. ls_fine and ls_coarse are the same LS scheme without
    # partitions, on the element lattice and on a coarse mesh with one element per partition.
    ls: str
    tfa_baseline: str = None
    tfa_fft: str = None
    ls_fine: str = None
    ls_coarse: str = None

def get_solver_roles(ls, tfa_baseline, tfa_fft, ls_fine=None, ls_coarse=None):
    # The roles of an entry point's solvers: TFA's only with config.include_tfa_model, and the non-partitioned LS
    # ones only with config.include_non_partitioned_ls, for an entry point that names them (MAIN_Newton.py).
    solver_roles = SolverRoles(ls)
    if config.include_tfa_model:
        solver_roles = solver_roles._replace(tfa_baseline=tfa_baseline, tfa_fft=tfa_fft)
    if config.include_non_partitioned_ls and ls_fine is not None:
        solver_roles = solver_roles._replace(ls_fine=ls_fine, ls_coarse=ls_coarse)
    return solver_roles

def report_problem(problem):
    print(f"inclusion volume fraction on the element grid: {problem.partition_materials.material_ids.mean():.4f}")
    save_cross_section_plot(problem.partition_materials.material_ids)

def report_results(results, solver_roles, run_name):
    # The printed comparison and every plot. Partitioned LS is the method being tested, against the TFA baseline when
    # there is one.
    comparable_step_count = get_comparable_step_count(results)
    print_comparison(results, comparable_step_count)
    if solver_roles.tfa_baseline is not None:
        print_model_difference(results, solver_roles.tfa_baseline, solver_roles.ls, comparable_step_count)
    save_load_path_plots(results, comparable_step_count, solver_roles.tfa_baseline, solver_roles.ls, run_name)
    save_von_mises_stress_plot(results, solver_roles, run_name)

def check_results(results, problem, solver_roles):
    # The checks on solved results that both entry points make. Each prints PASS or FAIL and stops the run on FAIL.
    comparable_step_count = get_comparable_step_count(results)
    if solver_roles.tfa_baseline is not None:
        check_solver_agreement(results, [solver_roles.tfa_baseline, solver_roles.tfa_fft], comparable_step_count)
        if config.matched_stiffness_control:
            check_matched_stiffness(results, solver_roles.tfa_baseline, solver_roles.ls, comparable_step_count)
    if solver_roles.ls_fine is not None and config.partition_number_per_side == config.element_number_per_side:
        # With one partition per element, the three LS lattices are the same lattice, so the three LS solvers are the
        # same solver and must agree.
        for solver_name in (solver_roles.ls_fine, solver_roles.ls_coarse):
            check_stress_agreement(f"partitions = elements, {solver_name} vs {solver_roles.ls}",
                                   get_stress_path(results, solver_name, comparable_step_count),
                                   get_stress_path(results, solver_roles.ls, comparable_step_count),
                                   config.solver_agreement_tolerance)
    if config.inclusion_shape == "laminate":
        check_laminate_solution(results, problem.partition_materials)

## ------- Timing Report ------- ##

def get_completed_step_count(solver_results):
    return min(result.completed_step_count for result in solver_results)

def get_comparable_step_count(results):
    # Timings and stresses are compared only over the steps every solver completed.
    return min(get_completed_step_count(solver_results) for solver_results in results.values())

def get_fastest_repeat_per_step(solver_results, field):
    fastest_repeat_per_step = np.argmin([result.solve_time_per_step for result in solver_results], axis=0)
    return np.array([getattr(result, field) for result in solver_results])[fastest_repeat_per_step,
                                                                           np.arange(config.load_step_count)]

def get_total_over_steps(solver_results, field, step_count):
    return get_fastest_repeat_per_step(solver_results, field)[:step_count].sum(axis=0)

def get_solver_label(solver_name, solver_results, step_count):
    iteration_count = get_total_over_steps(solver_results, "iterations_per_step", step_count)
    solve_time = get_total_over_steps(solver_results, "solve_time_per_step", step_count)
    label = (f"{solver_name}: {iteration_count} iterations, {solve_time:.2f} s, "
             f"{1000 * solve_time / iteration_count:.2f} ms per iteration")
    last_result = solver_results[-1]
    if last_result.completed_step_count < config.load_step_count:
        label += f"  [stopped at step {last_result.completed_step_count + 1}: {last_result.failure_reason}]"
    return label

def get_time_per_iteration_per_group(solver_results, step_count):
    # Per iteration as LoadPathResult counts them (residual evaluations), so a Newton iteration includes its whole
    # Krylov solve, whose operator products count as induced strain (online_timing). The last entry, "Other", is the
    # solve time no timed group covers.
    solve_time = get_total_over_steps(solver_results, "solve_time_per_step", step_count)
    time_per_group = get_total_over_steps(solver_results, "group_time_per_step", step_count)
    iteration_count = get_total_over_steps(solver_results, "iterations_per_step", step_count)
    return np.append(time_per_group, solve_time - time_per_group.sum()) / iteration_count

## ------- Printed Comparison and Checks ------- ##

def print_comparison(results, comparable_step_count):
    compared_steps = (f"the {comparable_step_count} of {config.load_step_count} steps every solver completed"
                      if comparable_step_count < config.load_step_count
                      else f"all {config.load_step_count} steps")
    print(f"solve times are the minimum per load step over {config.timing_repeat_count} interleaved runs of each "
          f"solver, summed over {compared_steps}")
    for solver_name, solver_results in results.items():
        print(get_solver_label(solver_name, solver_results, comparable_step_count))

def get_stress_path(results, solver_name, step_count):
    # A solver's macroscopic stress over its first step_count steps, from its last repeat: every repeat solves the
    # same steps to the same result, and only the timings differ.
    return results[solver_name][-1].macroscopic_stress[:step_count]

def get_relative_stress_difference(stress, baseline_stress):
    # The largest difference over every step and component, relative to the largest baseline stress.
    stress_scale = max(np.abs(baseline_stress).max(), np.finfo(float).tiny)
    return np.abs(stress - baseline_stress).max() / stress_scale

def print_check(description, relative_difference, tolerance, step_count):
    # Prints PASS or FAIL for a difference that must stay within tolerance, and stops the run on FAIL.
    passed = relative_difference <= tolerance
    print(f"[{'PASS' if passed else 'FAIL'}] {description}: {relative_difference:.2e} relative (tolerance "
          f"{tolerance:.0e}), over {step_count} of {config.load_step_count} steps.")
    if not passed:
        raise RuntimeError(f"failed: {description}.")

def check_stress_agreement(description, stress, baseline_stress, tolerance):
    # Two stress paths that solve the same equation.
    print_check(description, get_relative_stress_difference(stress, baseline_stress), tolerance, len(stress))

def check_solver_agreement(results, solver_names, comparable_step_count):
    # These solvers solve the same equations, so each one's macroscopic stress must match the first one's to within
    # solver tolerance.
    baseline_name, *other_names = solver_names
    for solver_name in other_names:
        check_stress_agreement(f"stress agreement, {solver_name} vs {baseline_name}",
                               get_stress_path(results, solver_name, comparable_step_count),
                               get_stress_path(results, baseline_name, comparable_step_count),
                               config.solver_agreement_tolerance)

def check_against_saved_results(results, solver_name, saved_run_name, saved_solver_name):
    # A solver in this entry point against the same model solved by another entry point's solver, from that one's
    # saved results. Both solve the same equation, so their macroscopic stresses must agree to solver tolerance. The
    # check is skipped, not failed, when there is no saved result or it was made for a different problem or
    # convergence tolerance: solver_agreement_tolerance only means something when both runs converged equally far.
    saved_path = config.output_folder / f"{saved_run_name}_results.npz"
    description = f"stress agreement, {solver_name} vs {saved_solver_name} saved by {saved_run_name}"
    if not saved_path.exists():
        print(f"[SKIPPED] {description}: {saved_path.name} not found. Run MAIN_{saved_run_name}.py first.")
        return
    with np.load(saved_path) as saved:
        # A parameter missing from the file was added after it was saved, so it counts as changed.
        changed_names = [name for name in config.problem_parameter_names
                         if f"problem/{name}" not in saved.files
                         or not np.array_equal(saved[f"problem/{name}"], np.asarray(getattr(config, name)))]
        if ("convergence_tolerance" not in saved.files
                or not np.array_equal(saved["convergence_tolerance"], config.convergence_tolerance)):
            changed_names.append("convergence_tolerance")
        if changed_names:
            print(f"[SKIPPED] {description}: it was saved for a different problem or tolerance "
                  f"({', '.join(changed_names)} changed). Rerun MAIN_{saved_run_name}.py.")
            return
        if f"{saved_solver_name}/macroscopic_stress" not in saved.files:
            print(f"[SKIPPED] {description}: {saved_path.name} has no {saved_solver_name} result. Rerun "
                  f"MAIN_{saved_run_name}.py with include_tfa_model = True.")
            return
        saved_stress = saved[f"{saved_solver_name}/macroscopic_stress"]
        saved_step_count = int(saved[f"{saved_solver_name}/completed_step_count"])

    step_count = min(get_completed_step_count(results[solver_name]), saved_step_count)
    check_stress_agreement(description, get_stress_path(results, solver_name, step_count), saved_stress[:step_count],
                           config.solver_agreement_tolerance)

## ------- LS Model Comparison ------- ##

def print_model_difference(results, tfa_solver_name, ls_solver_name, comparable_step_count):
    # The LS model's macroscopic stress against the TFA (actual E/P) model's: modelling error, not solver error.
    model_difference = get_relative_stress_difference(get_stress_path(results, ls_solver_name, comparable_step_count),
                                                      get_stress_path(results, tfa_solver_name, comparable_step_count))
    print(f"model discrepancy, different models ({ls_solver_name} vs {tfa_solver_name}): {model_difference:.2e} "
          "relative. This is a modelling difference, not a solver error.")

def check_matched_stiffness(results, tfa_solver_name, ls_solver_name, comparable_step_count):
    # With matched phase stiffness the LS and TFA models are the same equation, so here, and only here, the LS solver
    # must reproduce TFA to solver tolerance.
    check_stress_agreement(f"matched-stiffness control, {ls_solver_name} vs {tfa_solver_name}",
                           get_stress_path(results, ls_solver_name, comparable_step_count),
                           get_stress_path(results, tfa_solver_name, comparable_step_count),
                           config.matched_stiffness_tolerance)

## ------- Laminate Exact Solution ------- ##

def check_laminate_solution(results, partition_materials):
    # On a laminate every solver must reproduce the exact solution of laminate.py to config.laminate_tolerance, over
    # the steps it completed: its macroscopic stress, and the stress of every partition of its own lattice, which is
    # the stress of its phase. Partitions are judged phase by phase, each on its own phase's stress scale: on the
    # inclusion's, much larger, a matrix error would have to be about a hundred times bigger to show.
    exact_phase_stress = get_laminate_phase_stress(partition_materials)
    for solver_name, solver_results in results.items():
        result = solver_results[-1]
        step_count = result.completed_step_count
        partition_material_ids = result.partition_material_ids
        exact_partition_stress = exact_phase_stress[:, partition_material_ids]
        exact_macroscopic_stress = exact_partition_stress.mean(axis=1)
        check_stress_agreement(f"laminate exact solution, {solver_name}", result.macroscopic_stress[:step_count],
                               exact_macroscopic_stress[:step_count], config.laminate_tolerance)
        partition_difference = max(
            get_relative_stress_difference(result.partition_stress_per_step[:step_count, phase_partitions],
                                           exact_partition_stress[:step_count, phase_partitions])
            for phase_partitions in (partition_material_ids == 0, partition_material_ids == 1))
        print_check(f"laminate exact solution in every partition, {solver_name}", partition_difference,
                    config.laminate_tolerance, step_count)

## ------- Output Files ------- ##

def remove_old_outputs(run_name):
    # A file this run does not write would otherwise linger from an older configuration and look current. Only this
    # entry point's own files are removed.
    config.output_folder.mkdir(exist_ok=True)
    for path in config.output_folder.glob(f"{run_name}_*"):
        path.unlink()

def save_results(results, run_name):
    # Each solver's per-step results, with the problem they solve, so another entry point can check its own solvers
    # against these, but only when every problem parameter matches. check_code.py also compares whole files against
    # a saved baseline, which is what the iteration counts are kept for.
    arrays = {f"problem/{name}": np.asarray(getattr(config, name)) for name in config.problem_parameter_names}
    arrays["convergence_tolerance"] = np.asarray(config.convergence_tolerance)
    for solver_name, solver_results in results.items():
        result = solver_results[-1]
        arrays[f"{solver_name}/macroscopic_stress"] = result.macroscopic_stress
        arrays[f"{solver_name}/iterations_per_step"] = result.iterations_per_step
        arrays[f"{solver_name}/completed_step_count"] = np.asarray(result.completed_step_count)
    results_path = config.output_folder / f"{run_name}_results.npz"
    np.savez(results_path, **arrays)
    print(f"results saved to {results_path}")

def save_cross_section_plot(partition_material_ids):
    # The same geometry for every entry point, so one shared file without a run-name prefix.
    plots.plot_cross_section(
        partition_material_ids.reshape(config.partition_number_per_side, config.partition_number_per_side),
        config.element_number_per_side, config.output_folder / "cross_section.png")

def get_von_mises_stress(stress):
    return get_equivalent_stress(get_deviatoric_stress(stress))

def save_von_mises_stress_plot(results, solver_roles, run_name):
    # One map per LS solver, from the finest lattice to the coarsest, then TFA, each on its own lattice, at the last
    # step every mapped solver completed, so the maps compare the same load.
    mapped_solver_names = [name for name in (solver_roles.ls_fine, solver_roles.ls, solver_roles.ls_coarse,
                                             solver_roles.tfa_baseline) if name is not None]
    step_count = min(get_completed_step_count(results[solver_name]) for solver_name in mapped_solver_names)
    stress_maps = []
    for solver_name in mapped_solver_names:
        result = results[solver_name][-1]
        stress = result.partition_stress_per_step[step_count - 1]
        side = int(round(np.sqrt(len(stress))))
        # The coarse scheme's mesh has one element per partition; every other solver's is the main mesh.
        element_number_per_side = side if solver_name == solver_roles.ls_coarse else config.element_number_per_side
        stress_maps.append(plots.StressMap(solver_name, get_von_mises_stress(stress).reshape(side, side) / 1e6,
                                           result.partition_material_ids.reshape(side, side), element_number_per_side))
    plots.plot_von_mises_stress(stress_maps, step_count, config.load_step_count, config.von_mises_min_MPa,
                                config.von_mises_max_MPa, config.output_folder / f"{run_name}_von_mises_stress.png")

def get_elastic_steps(result):
    # Steps where no partition yielded. Steps past a stopped solver's last are not elastic, only unsolved.
    elastic_steps = ~result.yielding_per_step
    elastic_steps[result.completed_step_count:] = False
    return elastic_steps

def get_solver_steps(solver_results):
    return plots.SolverSteps(iterations_per_step=solver_results[-1].iterations_per_step,
                             solve_time_per_step=get_fastest_repeat_per_step(solver_results, "solve_time_per_step"),
                             completed_step_count=get_completed_step_count(solver_results))

def save_load_path_plots(results, comparable_step_count, baseline_solver_name, new_solver_name, run_name):
    # Stresses are plotted for the method being tested and, unless baseline_solver_name is None, the baseline. The
    # elastic steps shaded are the baseline's when there is one, since the LS model can yield at different steps from
    # TFA.
    plotted_solver_names = [new_solver_name] + ([baseline_solver_name] if baseline_solver_name is not None else [])
    plots.plot_load_path_summary(
        {solver_name: get_deviatoric_stress(results[solver_name][-1].macroscopic_stress) / 1e6
         for solver_name in plotted_solver_names},
        baseline_solver_name, new_solver_name,
        {solver_name: get_solver_steps(solver_results) for solver_name, solver_results in results.items()},
        get_elastic_steps(results[plotted_solver_names[-1]][-1]),
        plots.get_applied_strain_description(config.max_macro_strain, config.load_path_shape, config.load_step_count),
        config.output_folder / f"{run_name}_load_path_summary.png")
    plots.plot_time_per_iteration_breakdown(
        {solver_name: 1000 * get_time_per_iteration_per_group(solver_results, comparable_step_count)
         for solver_name, solver_results in results.items()},
        config.output_folder / f"{run_name}_time_breakdown.png")
