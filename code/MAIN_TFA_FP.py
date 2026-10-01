# The actual E/P TFA model (ER-4) solved by fixed-point iteration, with and without the FFT reference preconditioner
# (Formulation.md Strategy 1). Both solve the same equation, so they must reach the same macroscopic stress; what is
# compared is their iterations and time. The problem and every solver setting are read from config.py.
#
# Run from anywhere: python code/MAIN_TFA_FP.py
# Outputs, in code/output: TFA_FP_load_path_summary.png, TFA_FP_time_breakdown.png, TFA_FP_von_mises_stress.png,
# TFA_FP_results.npz and the shared cross_section.png.

import functools

import config
from load_path import run_interleaved_repeats
from offline import get_problem
from report import (check_solver_agreement, get_comparable_step_count, print_comparison, print_independent_check,
                    remove_old_outputs, save_cross_section_plot, save_load_path_plots, save_results,
                    save_von_mises_stress_plot)
from tfa_solvers import get_tfa_fixed_point_solvers

run_name = "TFA_FP"
# The stress plot compares the FFT-preconditioned solver against the plain one.
reference_solver_name = "TFA Standard FP"
new_solver_name = "TFA FFT FP"

def main():
    remove_old_outputs(run_name)
    problem = get_problem()
    print(f"inclusion volume fraction on the partition grid: {problem.partition_material_ids.mean():.4f}")
    save_cross_section_plot(problem.partition_material_ids)

    print(f"TFA fixed point: relaxation {config.tfa_relaxation_factor}")
    results = run_interleaved_repeats(functools.partial(get_tfa_fixed_point_solvers, problem.P0_transformed),
                                      problem.E, problem.P, problem.partition_materials)
    comparable_step_count = get_comparable_step_count(results)
    print_comparison(results, comparable_step_count)
    print_independent_check(results)
    save_results(results, run_name)
    save_load_path_plots(results, comparable_step_count, reference_solver_name, new_solver_name, run_name)
    save_von_mises_stress_plot(results, [new_solver_name], problem.partition_material_ids, run_name)
    check_solver_agreement(results, list(results), comparable_step_count)

if __name__ == "__main__":
    main()
