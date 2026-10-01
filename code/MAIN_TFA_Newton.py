# The actual E/P TFA model (ER-4) solved by Newton-Krylov (Formulation.md Strategy 2): each Newton correction is
# solved by GMRES, once plain and once right-preconditioned by the FFT reference M0 = I - P0 H_μ,0. Both solve the
# same equation as the fixed-point solvers of MAIN_TFA_FP.py, so they must agree with each other and, when the problem
# matches, with that file's saved result. The problem and every solver setting are read from config.py.
#
# Run from anywhere: python code/MAIN_TFA_Newton.py
# Outputs, in code/output: TFA_Newton_load_path_summary.png, TFA_Newton_time_breakdown.png,
# TFA_Newton_von_mises_stress.png, TFA_Newton_results.npz and the shared cross_section.png.

import functools

import config
from load_path import run_interleaved_repeats
from offline import get_problem
from report import (check_against_saved_results, check_solver_agreement, get_comparable_step_count, print_comparison,
                    print_independent_check, remove_old_outputs, save_cross_section_plot, save_load_path_plots,
                    save_results, save_von_mises_stress_plot)
from tfa_solvers import get_tfa_newton_solvers
from verification import get_tfa_jacobian_checks, run_verification

run_name = "TFA_Newton"
# The stress plot compares the FFT-preconditioned solver against the plain one.
reference_solver_name = "TFA Newton"
new_solver_name = "TFA FFT Newton"

def main():
    remove_old_outputs(run_name)
    problem = get_problem()
    print(f"inclusion volume fraction on the partition grid: {problem.partition_material_ids.mean():.4f}")
    save_cross_section_plot(problem.partition_material_ids)
    if config.verification_enabled:
        run_verification(problem, [get_tfa_jacobian_checks])

    print(f"TFA Newton: GMRES rtol {config.newton_krylov_tolerance:.0e}, restart {config.newton_krylov_restart}, "
          f"at most {config.newton_max_steps} Newton steps")
    results = run_interleaved_repeats(functools.partial(get_tfa_newton_solvers, problem.P0_transformed),
                                      problem.E, problem.P, problem.partition_materials)
    comparable_step_count = get_comparable_step_count(results)
    print_comparison(results, comparable_step_count)
    print_independent_check(results)
    save_results(results, run_name)
    save_load_path_plots(results, comparable_step_count, reference_solver_name, new_solver_name, run_name)
    save_von_mises_stress_plot(results, [new_solver_name], problem.partition_material_ids, run_name)
    check_solver_agreement(results, list(results), comparable_step_count)
    for solver_name in results:
        check_against_saved_results(results, solver_name, "TFA_FP", "TFA Standard FP")

if __name__ == "__main__":
    main()
