# The partition-averaged Lippmann-Schwinger (LS) model solved by damped fixed-point iteration (LS-19). With
# config.include_tfa_model, the actual E/P TFA model (ER-4) is solved next to it as the baseline, by fixed-point
# iteration with and without the FFT reference preconditioner (Formulation.md Strategy 1). The two TFA solvers solve
# the same equation and must agree to solver tolerance. LS is a different model, so its difference from TFA is
# modelling error, printed as such; only under matched_stiffness_control are the two models the same equation, and
# then they must agree too. The problem and every solver setting are read from config.py.
#
# Run from anywhere: python code/MAIN_FP.py
# Outputs, in code/output: FP_load_path_summary.png, FP_time_breakdown.png, FP_von_mises_stress.png, FP_results.npz
# and the shared cross_section.png.

import functools

import config
from load_path import run_interleaved_repeats
from ls_solvers import get_ls_fixed_point_solver, get_ls_relaxation_factor
from offline import get_problem
from report import (check_matched_stiffness, check_solver_agreement, get_comparable_step_count, print_comparison,
                    print_model_difference, remove_old_outputs, save_cross_section_plot, save_load_path_plots,
                    save_results, save_von_mises_stress_plot)
from tfa_solvers import get_tfa_fixed_point_solvers
from verification import run_verification

run_name = "FP"
ls_solver_name = "LS FP"
# With TFA included, the stress plot and the checks compare LS against the plain TFA solver, and the von Mises maps
# against the FFT-preconditioned one.
tfa_baseline_solver_name = "TFA Standard FP"
tfa_fft_solver_name = "TFA FFT FP"

def get_solvers(problem, ls_relaxation):
    solvers = {ls_solver_name: get_ls_fixed_point_solver(problem.P0_transformed, problem.reference_compliance,
                                                         ls_relaxation)}
    if config.include_tfa_model:
        solvers.update(get_tfa_fixed_point_solvers(problem.P0_transformed))
    return solvers

def main():
    remove_old_outputs(run_name)
    problem = get_problem()
    print(f"inclusion volume fraction on the partition grid: {problem.partition_material_ids.mean():.4f}")
    save_cross_section_plot(problem.partition_material_ids)
    if config.verification_enabled:
        run_verification(problem)

    ls_relaxation = get_ls_relaxation_factor(problem.partition_materials, problem.reference_L)
    print(f"LS fixed point: relaxation {ls_relaxation:.4g}")
    if config.include_tfa_model:
        print(f"TFA fixed point: relaxation {config.tfa_relaxation_factor}")
    results = run_interleaved_repeats(functools.partial(get_solvers, problem, ls_relaxation), problem.E, problem.P,
                                      problem.partition_materials)
    comparable_step_count = get_comparable_step_count(results)
    print_comparison(results, comparable_step_count)
    if config.include_tfa_model:
        print_model_difference(results, tfa_baseline_solver_name, ls_solver_name, comparable_step_count)
        save_load_path_plots(results, comparable_step_count, tfa_baseline_solver_name, ls_solver_name, run_name)
        save_von_mises_stress_plot(results, [tfa_fft_solver_name, ls_solver_name], problem.partition_material_ids,
                                   run_name)
        check_solver_agreement(results, [tfa_baseline_solver_name, tfa_fft_solver_name], comparable_step_count)
        if config.matched_stiffness_control:
            check_matched_stiffness(results, tfa_baseline_solver_name, ls_solver_name, comparable_step_count)
    else:
        save_load_path_plots(results, comparable_step_count, None, ls_solver_name, run_name)
        save_von_mises_stress_plot(results, [ls_solver_name], problem.partition_material_ids, run_name)
    # Saved only after every check has passed, so MAIN_Newton.py never cross-checks against a failed run.
    save_results(results, run_name)

if __name__ == "__main__":
    main()
