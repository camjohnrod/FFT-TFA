# The partition-averaged Lippmann-Schwinger (LS) model solved by matrix-free Newton-GMRES on the LS-20 Jacobian, next
# to the actual E/P TFA model solved by the two Newton-Krylov solvers of MAIN_TFA_Newton.py. The two TFA solvers solve
# the same equation and must agree to solver tolerance. The LS model is a different model, so its difference from them
# is modelling error, printed as such; only under matched_stiffness_control are the two models the same equation, and
# then they must agree too. Every solver is also checked against the fixed-point result of the same model saved by
# MAIN_TFA_FP.py or MAIN_LS_FP.py, when the problem matches. The problem and every solver setting are read from
# config.py.
#
# Run from anywhere: python code/MAIN_LS_Newton.py
# Outputs, in code/output: LS_Newton_load_path_summary.png, LS_Newton_time_breakdown.png,
# LS_Newton_von_mises_stress.png, LS_Newton_results.npz and the shared cross_section.png.

import functools

import config
from load_path import run_interleaved_repeats
from ls_solvers import get_ls_newton_solver
from offline import get_problem
from report import (check_against_saved_results, check_matched_stiffness, check_solver_agreement,
                    get_comparable_step_count, print_comparison, print_independent_check, print_model_difference,
                    remove_old_outputs, save_cross_section_plot, save_load_path_plots, save_results,
                    save_von_mises_stress_plot)
from tfa_solvers import get_tfa_newton_solvers
from verification import get_ls_jacobian_checks, get_tfa_jacobian_checks, run_verification

run_name = "LS_Newton"
ls_solver_name = "LS Newton"
# The stress plot compares the LS model against the plain TFA solver of the same strategy.
reference_solver_name = "TFA Newton"
# The von Mises stress maps compare the LS model against the FFT-preconditioned TFA solver.
fft_tfa_solver_name = "TFA FFT Newton"

def get_solvers(problem):
    return {**get_tfa_newton_solvers(problem.P0_transformed),
            ls_solver_name: get_ls_newton_solver(problem.P0_transformed, problem.reference_compliance)}

def main():
    remove_old_outputs(run_name)
    problem = get_problem()
    print(f"inclusion volume fraction on the partition grid: {problem.partition_material_ids.mean():.4f}")
    save_cross_section_plot(problem.partition_material_ids)
    if config.verification_enabled:
        run_verification(problem, [get_tfa_jacobian_checks, get_ls_jacobian_checks], print_ls_model_error=True)

    print(f"Newton: GMRES rtol {config.newton_krylov_tolerance:.0e}, restart {config.newton_krylov_restart}, "
          f"at most {config.newton_max_steps} Newton steps")
    results = run_interleaved_repeats(functools.partial(get_solvers, problem), problem.E, problem.P,
                                      problem.partition_materials)
    comparable_step_count = get_comparable_step_count(results)
    tfa_solver_names = [solver_name for solver_name in results if solver_name != ls_solver_name]
    print_comparison(results, comparable_step_count)
    print_model_difference(results, reference_solver_name, ls_solver_name, comparable_step_count)
    print_independent_check(results)
    save_results(results, run_name)
    save_load_path_plots(results, comparable_step_count, reference_solver_name, ls_solver_name, run_name)
    save_von_mises_stress_plot(results, [fft_tfa_solver_name, ls_solver_name], problem.partition_material_ids, run_name)
    check_solver_agreement(results, tfa_solver_names, comparable_step_count)
    if config.matched_stiffness_control:
        check_matched_stiffness(results, tfa_solver_names[0], ls_solver_name, comparable_step_count)
    for solver_name in tfa_solver_names:
        check_against_saved_results(results, solver_name, "TFA_FP", "TFA Standard FP")
    check_against_saved_results(results, ls_solver_name, "LS_FP", "LS FP")

if __name__ == "__main__":
    main()
