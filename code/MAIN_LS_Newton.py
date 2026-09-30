# The partition-averaged Lippmann-Schwinger (LS) model solved by matrix-free Newton-GMRES on the LS-20 Jacobian. It
# solves the same equation as the LS fixed point of MAIN_LS_FP.py, so it is checked against that file's saved result
# when the problem matches. The TFA Newton solvers will join it here once they exist. The problem and every solver
# setting are read from config.py.
#
# Run from anywhere: python code/MAIN_LS_Newton.py
# Outputs, in code/output: LS_Newton_load_path_summary.png, LS_Newton_time_breakdown.png, LS_Newton_results.npz and
# the shared cross_section.png.

import functools

import config
from load_path import run_interleaved_repeats
from ls_solvers import get_ls_newton_solver
from offline import get_problem
from report import (check_against_saved_results, get_comparable_step_count, print_comparison,
                    print_independent_check, remove_old_outputs, save_cross_section_plot, save_load_path_plots,
                    save_results)
from verification import get_ls_jacobian_checks, run_verification

run_name = "LS_Newton"
ls_solver_name = "LS Newton"

def get_solvers(problem):
    return {ls_solver_name: get_ls_newton_solver(problem.P0_transformed, problem.reference_compliance)}

def main():
    remove_old_outputs(run_name)
    problem = get_problem()
    print(f"inclusion volume fraction on the partition grid: {problem.partition_material_ids.mean():.4f}")
    save_cross_section_plot(problem.partition_material_ids)
    if config.verification_enabled:
        run_verification(problem, get_ls_jacobian_checks)

    print(f"LS Newton: GMRES rtol {config.newton_krylov_tolerance:.0e}, restart {config.newton_krylov_restart}")
    results = run_interleaved_repeats(functools.partial(get_solvers, problem), problem.E, problem.P,
                                      problem.partition_materials)
    comparable_step_count = get_comparable_step_count(results)
    print_comparison(results, comparable_step_count)
    print_independent_check(results)
    save_results(results, run_name)
    save_load_path_plots(results, comparable_step_count, ls_solver_name, run_name)
    check_against_saved_results(results, ls_solver_name, "LS_FP", "LS FP")

if __name__ == "__main__":
    main()
