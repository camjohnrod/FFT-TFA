# Runs both reference solver methods in one process and checks they reach the same root.
# Builds one reference solver per method, so it compares fixed point against Newton on identical inputs.
# Agreement should sit at the convergence tolerance, not at machine precision: 6.9e-07 on macroscopic stress
# and 2.4e-06 on final plastic strain, against a 1e-6 relative residual tolerance.
#
# Reads the configuration from code/MAIN_Testing.py; edit the constants there, not here.
# Run from anywhere: python code/experiments/solver_cross_check.py

import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import MAIN_Testing as m

problem = m.get_problem()
relaxation = m.get_reference_relaxation_factor(problem.partition_materials, problem.reference_L)

results = {}
for method in ("newton", "fixed_point"):
    solver = m.ReferenceSolver(problem.P0_transformed, problem.reference_compliance, relaxation, method)
    results[method] = m.run_strain_path(f"reference {method}", solver, problem.E, problem.P,
                                        problem.partition_materials)
    print(f"{method:<12}: {results[method].iterations_per_step.sum():>6d} iterations, "
          f"{results[method].solve_time_per_step.sum():.2f} s")

newton, fixed_point = results["newton"], results["fixed_point"]
step_count = min(newton.completed_step_count, fixed_point.completed_step_count)
if step_count < m.strain_increment_count:
    print(f"\ncompared over the {step_count} steps both methods completed; final plastic strains are from "
          "different steps and are not compared")
stress_scale = np.abs(fixed_point.macroscopic_stress[:step_count]).max()
stress_difference = np.abs(newton.macroscopic_stress[:step_count] - fixed_point.macroscopic_stress[:step_count]).max()
print(f"\nsame root? macroscopic stress   : {stress_difference / stress_scale:.2e} relative")
if step_count == m.strain_increment_count:
    strain_scale = np.abs(fixed_point.final_plastic_strain).max()
    strain_difference = np.abs(newton.final_plastic_strain - fixed_point.final_plastic_strain).max()
    print(f"same root? final plastic strain : {strain_difference / strain_scale:.2e} relative")
