# The partition-averaged Lippmann-Schwinger (LS) model solved by damped fixed-point iteration, next to the actual E/P
# TFA model solved by the two fixed-point solvers of MAIN_TFA_FP.py. The two TFA solvers solve the same equation and
# must agree to solver tolerance. The LS model is a different model, so its difference from them is modelling error,
# printed as such; only under matched_stiffness_control are the two models the same equation, and then they must
# agree too. The problem and every solver setting are read from config.py.
#
# Run from anywhere: python code/MAIN_LS_FP.py
# Outputs, in code/output: LS_FP_load_path_summary.png, LS_FP_time_breakdown.png, LS_FP_results.npz and the shared
# cross_section.png.

import functools

import config
from load_path import run_interleaved_repeats
from ls_solvers import get_ls_fixed_point_solver, get_reference_relaxation_factor, get_reference_stability_limit
from offline import get_problem
from report import (check_matched_stiffness, check_solver_agreement, get_comparable_step_count,
                    get_most_complete_solver_name, print_comparison, print_independent_check, print_model_difference,
                    remove_old_outputs, save_cross_section_plot, save_load_path_plots, save_results)
from tfa_solvers import get_tfa_fixed_point_solvers
from verification import run_verification

run_name = "LS_FP"
ls_solver_name = "LS FP"

def get_solvers(problem, ls_relaxation):
    return {**get_tfa_fixed_point_solvers(problem.P0_transformed),
            ls_solver_name: get_ls_fixed_point_solver(problem.P0_transformed, problem.reference_compliance,
                                                      ls_relaxation)}

def main():
    remove_old_outputs(run_name)
    problem = get_problem()
    print(f"inclusion volume fraction on the partition grid: {problem.partition_material_ids.mean():.4f}")
    save_cross_section_plot(problem.partition_material_ids)
    if config.verification_enabled:
        run_verification(problem, print_ls_model_error=True)

    ls_relaxation = get_reference_relaxation_factor(problem.partition_materials, problem.reference_L)
    stability_limit = get_reference_stability_limit(problem.partition_materials, problem.reference_L)
    print(f"TFA fixed point: relaxation {config.relaxation_factor}")
    print(f"LS fixed point: relaxation {ls_relaxation:.4g}, stability limit {stability_limit:.4g}")
    results = run_interleaved_repeats(functools.partial(get_solvers, problem, ls_relaxation), problem.E, problem.P,
                                      problem.partition_materials)
    comparable_step_count = get_comparable_step_count(results)
    tfa_solver_names = [solver_name for solver_name in results if solver_name != ls_solver_name]
    # Stresses are plotted from the TFA solver that got furthest, since the LS solver solves a different model.
    actual_solver_name = get_most_complete_solver_name({solver_name: results[solver_name]
                                                        for solver_name in tfa_solver_names})
    print_comparison(results, comparable_step_count)
    print_model_difference(results, actual_solver_name, ls_solver_name, comparable_step_count)
    print_independent_check(results)
    save_results(results, run_name)
    save_load_path_plots(results, comparable_step_count, actual_solver_name, run_name)
    check_solver_agreement(results, tfa_solver_names, comparable_step_count)
    if config.matched_stiffness_control:
        check_matched_stiffness(results, tfa_solver_names[0], ls_solver_name, comparable_step_count)

if __name__ == "__main__":
    main()
