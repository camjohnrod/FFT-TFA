# The partition-averaged Lippmann-Schwinger (LS) model solved by matrix-free Newton-GMRES on the LS-20 Jacobian. With
# config.include_tfa_model, the actual E/P TFA model (ER-4) is solved next to it as the baseline, by Newton-Krylov
# with GMRES, once plain and once right-preconditioned by the FFT reference M0 = I - P0 H_μ,0 (3_tfa_baseline.md
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
from report import (check_against_saved_results, check_results, get_solver_roles, remove_old_outputs, report_problem,
                    report_results, save_results)
from tfa_solvers import get_tfa_newton_solvers
from verification import get_ls_jacobian_checks, get_tfa_jacobian_checks, run_verification

run_name = "Newton"
solver_roles = get_solver_roles(ls="LS Newton", tfa_baseline="TFA Newton", tfa_fft="TFA FFT Newton",
                                ls_fine="LS Newton, fine", ls_coarse="LS Newton, coarse")

def get_solvers(problem):
    solvers = {solver_roles.ls: get_ls_newton_solver(problem.P0_transformed, problem.reference_compliance,
                                                     problem.partition_materials)}
    if solver_roles.ls_fine is not None:
        # The same LS scheme on the fine and coarse lattices, with the same C0 (config.include_non_partitioned_ls).
        for solver_name, lattice in ((solver_roles.ls_fine, problem.fine_lattice),
                                     (solver_roles.ls_coarse, problem.coarse_lattice)):
            solvers[solver_name] = get_ls_newton_solver(lattice.P0_transformed, problem.reference_compliance,
                                                        lattice.partition_materials)
    if config.include_tfa_model:
        solvers.update(get_tfa_newton_solvers(problem.E, problem.P, problem.P0_transformed,
                                              problem.partition_materials))
    return solvers

def main():
    remove_old_outputs(run_name)
    problem = get_problem()
    report_problem(problem)
    if config.verification_enabled:
        jacobian_checks = [get_ls_jacobian_checks]
        if config.include_tfa_model:
            jacobian_checks.append(get_tfa_jacobian_checks)
        run_verification(problem, jacobian_checks)

    print(f"Newton: GMRES rtol {config.newton_krylov_tolerance:.0e}, restart {config.newton_krylov_restart}, "
          f"at most {config.newton_max_steps} Newton steps")
    results = run_interleaved_repeats(functools.partial(get_solvers, problem))
    report_results(results, solver_roles, run_name)
    check_results(results, problem, solver_roles)
    # Every solver against the fixed-point result of the same model, saved by MAIN_FP.py.
    check_against_saved_results(results, solver_roles.ls, "FP", "LS FP")
    if config.include_tfa_model:
        for solver_name in (solver_roles.tfa_baseline, solver_roles.tfa_fft):
            check_against_saved_results(results, solver_name, "FP", "TFA Standard FP")
    # Saved only after every check has passed, so a failed run leaves no results behind.
    save_results(results, run_name)

if __name__ == "__main__":
    main()
