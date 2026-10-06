# The partition-averaged Lippmann-Schwinger (LS) model solved by damped fixed-point iteration (LS-19). With
# config.include_tfa_model, the actual E/P TFA model (ER-4) is solved next to it as the baseline, by fixed-point
# iteration with and without the FFT reference preconditioner (Formulation.md Strategy 1). The two TFA solvers solve
# the same equation and must agree to solver tolerance. LS is a different model, so its difference from TFA is
# modelling error, printed as such; only under matched_stiffness_control are the two models the same equation, and
# then they must agree too. The problem and every solver setting are read from config.py. The non-partitioned LS
# solvers (config.include_non_partitioned_ls) run in MAIN_Newton.py only: the LS fixed point at fine resolution takes
# too long.
#
# Run from anywhere: python code/MAIN_FP.py
# Outputs, in code/output: FP_load_path_summary.png, FP_time_breakdown.png, FP_von_mises_stress.png, FP_results.npz
# and the shared cross_section.png.

import functools

import config
from load_path import run_interleaved_repeats
from ls_solvers import get_ls_fixed_point_solver, get_ls_relaxation_factor
from offline import get_problem
from report import (check_results, get_solver_roles, remove_old_outputs, report_problem, report_results,
                    save_results)
from tfa_solvers import get_tfa_fixed_point_solvers
from verification import run_verification

run_name = "FP"
solver_roles = get_solver_roles(ls="LS FP", tfa_baseline="TFA Standard FP", tfa_fft="TFA FFT FP")

def get_solvers(problem, ls_relaxation):
    solvers = {solver_roles.ls: get_ls_fixed_point_solver(problem.P0_transformed, problem.reference_compliance,
                                                          ls_relaxation, problem.partition_materials)}
    if config.include_tfa_model:
        solvers.update(get_tfa_fixed_point_solvers(problem.E, problem.P, problem.P0_transformed,
                                                   problem.partition_materials))
    return solvers

def main():
    remove_old_outputs(run_name)
    problem = get_problem()
    report_problem(problem)
    if config.verification_enabled:
        run_verification(problem)

    ls_relaxation = get_ls_relaxation_factor(problem.partition_materials, problem.reference_L)
    print(f"LS fixed point: relaxation {ls_relaxation:.4g}")
    if config.include_tfa_model:
        print(f"TFA fixed point: relaxation {config.tfa_relaxation_factor}")
    results = run_interleaved_repeats(functools.partial(get_solvers, problem, ls_relaxation))
    report_results(results, solver_roles, run_name)
    check_results(results, problem, solver_roles)
    # Saved only after every check has passed, so MAIN_Newton.py never cross-checks against a failed run.
    save_results(results, run_name)

if __name__ == "__main__":
    main()
