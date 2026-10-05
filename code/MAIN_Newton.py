# The partition-averaged Lippmann-Schwinger (LS) model solved by matrix-free Newton-GMRES on the LS-20 Jacobian. With
# config.include_tfa_model, the actual E/P TFA model (ER-4) is solved next to it as the baseline, by Newton-Krylov
# with GMRES, once plain and once right-preconditioned by the FFT reference M0 = I - P0 H_μ,0 (Formulation.md
# Strategy 2). The two TFA solvers solve the same equation and must agree to solver tolerance. LS is a different
# model, so its difference from TFA is modelling error, printed as such; only under matched_stiffness_control are the
# two models the same equation, and then they must agree too. Every solver is also checked against the fixed-point
# result of the same model saved by MAIN_FP.py, when the problem matches. The problem and every solver setting are
# read from config.py.
#
# Run from anywhere: python code/MAIN_Newton.py (after MAIN_FP.py, for the cross-checks)
# Outputs, in code/output: Newton_load_path_summary.png, Newton_time_breakdown.png, Newton_von_mises_stress.png,
# Newton_results.npz and the shared cross_section.png.

import functools

import config
from load_path import run_interleaved_repeats
from ls_solvers import get_ls_newton_solver
from offline import get_problem
from report import (check_against_saved_results, check_matched_stiffness, check_solver_agreement,
                    get_comparable_step_count, print_comparison, print_model_difference, remove_old_outputs,
                    save_cross_section_plot, save_load_path_plots, save_results, save_von_mises_stress_plot)
from tfa_solvers import get_tfa_newton_solvers
from verification import get_ls_jacobian_checks, get_tfa_jacobian_checks, run_verification

run_name = "Newton"
ls_solver_name = "LS Newton"
# With TFA included, every plot and check compares LS against the plain TFA solver, the baseline; the
# FFT-preconditioned TFA solver is checked against it.
tfa_baseline_solver_name = "TFA Newton"
tfa_fft_solver_name = "TFA FFT Newton"

def get_solvers(problem):
    solvers = {ls_solver_name: get_ls_newton_solver(problem.P0_transformed, problem.reference_compliance)}
    if config.include_tfa_model:
        solvers.update(get_tfa_newton_solvers(problem.P0_transformed))
    return solvers

def main():
    remove_old_outputs(run_name)
    problem = get_problem()
    print(f"inclusion volume fraction on the partition grid: {problem.partition_material_ids.mean():.4f}")
    save_cross_section_plot(problem.partition_material_ids)
    if config.verification_enabled:
        jacobian_checks = [get_ls_jacobian_checks]
        if config.include_tfa_model:
            jacobian_checks.append(get_tfa_jacobian_checks)
        run_verification(problem, jacobian_checks)

    print(f"Newton: GMRES rtol {config.newton_krylov_tolerance:.0e}, restart {config.newton_krylov_restart}, "
          f"at most {config.newton_max_steps} Newton steps")
    results = run_interleaved_repeats(functools.partial(get_solvers, problem), problem.E, problem.P,
                                      problem.partition_materials)
    comparable_step_count = get_comparable_step_count(results)
    print_comparison(results, comparable_step_count)
    if config.include_tfa_model:
        print_model_difference(results, tfa_baseline_solver_name, ls_solver_name, comparable_step_count)
        save_load_path_plots(results, comparable_step_count, tfa_baseline_solver_name, ls_solver_name, run_name)
        save_von_mises_stress_plot(results, [tfa_baseline_solver_name, ls_solver_name],
                                   problem.partition_material_ids, run_name)
        check_solver_agreement(results, [tfa_baseline_solver_name, tfa_fft_solver_name], comparable_step_count)
        if config.matched_stiffness_control:
            check_matched_stiffness(results, tfa_baseline_solver_name, ls_solver_name, comparable_step_count)
        for solver_name in (tfa_baseline_solver_name, tfa_fft_solver_name):
            check_against_saved_results(results, solver_name, "FP", "TFA Standard FP")
    else:
        save_load_path_plots(results, comparable_step_count, None, ls_solver_name, run_name)
        save_von_mises_stress_plot(results, [ls_solver_name], problem.partition_material_ids, run_name)
    check_against_saved_results(results, ls_solver_name, "FP", "LS FP")
    # Saved only after every check has passed, so a failed run leaves no results behind.
    save_results(results, run_name)

if __name__ == "__main__":
    main()
