# Measures how the reference stiffness C0 trades model accuracy against basic-scheme convergence.
# For each candidate C0 it rebuilds P0 (six reference solves, cache bypassed), measures the elastic model error
# by direct dense solve, then runs the full nonlinear path and compares against the actual-E/P baseline.
# Result on 2026-09-28 at 27/9, contrast 100: the homogenized C0 was the most accurate of five candidates, and
# the classical midpoint-Lame choice was the worst by far (55 percent plastic error) despite converging fastest.
#
# Reads the configuration from code/MAIN_Testing.py; edit the constants there, not here.
# Run from anywhere: python code/experiments/c0_sweep.py

import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import MAIN_Testing as m

problem = m.get_problem()
materials = problem.partition_materials
exact_stiffness = problem.reference_L
inclusion_fraction = problem.partition_material_ids.mean()
baseline = m.run_strain_path("Standard", m.standard_richardson_iteration, problem.E, problem.P, materials)
A = m.get_partition_averaging_operator(problem.B, problem.mesh.element_nodes, problem.mesh.element_partition_ids)

candidates = {
    "matrix L_m":          problem.L_matrix,
    "homogenized (in use)": exact_stiffness,
    "Voigt average":       (1 - inclusion_fraction) * problem.L_matrix + inclusion_fraction * problem.L_inclusion,
    "midpoint Lame":       0.5 * (problem.L_matrix + problem.L_inclusion),
    "inclusion L_i":       problem.L_inclusion,
}

print(f"\n{'C0 choice':<21} {'C0_1111 MPa':>12} {'alpha':>8} {'elastic err':>12} {'plastic err':>12} {'iters':>8}")
for name, C0 in candidates.items():
    offset_blocks = m.get_P0_offset_blocks(problem.B, np.tile(C0, (m.element_count, 1, 1)), problem.mesh, A)
    P0_transformed = m.get_P0_transformed(offset_blocks)
    dense_P0 = m.get_dense_P0(offset_blocks)
    compliance = np.linalg.inv(C0)

    model_stiffness = m.get_model_elastic_stiffness(dense_P0, materials.L, compliance)
    elastic_error = np.abs(model_stiffness - exact_stiffness).max() / np.abs(exact_stiffness).max()

    smallest, largest = m.get_reference_stiffness_ratio_bounds(materials, C0)
    relaxation = m.reference_stability_safety * 2 / (smallest + largest)
    columns = f"{name:<21} {C0[0, 0] / 1e6:>12.1f} {relaxation:>8.4f} {elastic_error:>12.3e}"
    try:
        solver = m.ReferenceSolver(P0_transformed, compliance, relaxation, m.reference_solver_method)
        result = m.run_strain_path(name, solver, problem.E, problem.P, materials)
    except m.LoadPathAbandoned as failure:
        print(f"{columns} {'FAILED':>12} {'-':>8}  ({str(failure)[:60]})")
        continue
    # Compared over the steps both the candidate and the baseline completed.
    step_count = min(result.completed_step_count, baseline.completed_step_count)
    stress_scale = np.abs(baseline.macroscopic_stress[:step_count]).max()
    plastic_error = (np.abs(result.macroscopic_stress[:step_count] - baseline.macroscopic_stress[:step_count]).max()
                     / stress_scale)
    stopped = (f"  (stopped at step {result.completed_step_count + 1})"
               if result.completed_step_count < m.strain_increment_count else "")
    print(f"{columns} {plastic_error:>12.3e} {result.iterations_per_step.sum():>8d}{stopped}")
