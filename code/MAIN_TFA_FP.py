# The actual E/P TFA model (ER-4) solved by fixed-point iteration, with and without the FFT reference preconditioner
# (Formulation.md Strategy 1). Both solve the same equation, so they must reach the same macroscopic stress; what is
# compared is their iterations and time. The problem and every solver setting are read from config.py.
#
# Run from anywhere: python code/MAIN_TFA_FP.py
# Outputs, in code/output: TFA_FP_load_path_summary.png, TFA_FP_time_breakdown.png, TFA_FP_results.npz and the
# shared cross_section.png.

import functools

from load_path import Solver, run_interleaved_repeats
from offline import get_problem
from report import (check_solver_agreement, get_comparable_step_count, get_most_complete_solver_name,
                    print_comparison, print_independent_check, remove_old_outputs, save_cross_section_plot,
                    save_load_path_plots, save_results)
from tfa_solvers import fft_preconditioned_richardson_iteration, get_actual_residual, standard_richardson_iteration

run_name = "TFA_FP"

def get_solvers(P0_transformed):
    return {"TFA Standard FP": Solver(standard_richardson_iteration, get_actual_residual),
            "TFA FFT FP": Solver(functools.partial(fft_preconditioned_richardson_iteration,
                                                   P0_transformed=P0_transformed), get_actual_residual)}

def main():
    remove_old_outputs(run_name)
    problem = get_problem()
    print(f"inclusion volume fraction on the partition grid: {problem.partition_material_ids.mean():.4f}")
    save_cross_section_plot(problem.partition_material_ids)

    results = run_interleaved_repeats(functools.partial(get_solvers, problem.P0_transformed), problem.E, problem.P,
                                      problem.partition_materials)
    comparable_step_count = get_comparable_step_count(results)
    print_comparison(results, comparable_step_count)
    print_independent_check(results)
    save_results(results, run_name)
    save_load_path_plots(results, comparable_step_count, get_most_complete_solver_name(results), run_name)
    check_solver_agreement(results, list(results), comparable_step_count)

if __name__ == "__main__":
    main()
