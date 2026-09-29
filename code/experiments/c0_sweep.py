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

L_matrix = m.get_L(m.elastic_modulus_matrix, m.poisson_ratio_matrix)
L_inclusion = m.get_L(m.elastic_modulus_inclusion, m.poisson_ratio_inclusion)
partition_materials = m.get_partition_materials(L_matrix, L_inclusion)
mesh, B = m.get_periodic_mesh(), m.get_B()
L_per_element = m.get_value_per_material(L_matrix, L_inclusion, mesh.element_material_ids)
E, P, _ = m.get_offline_operators(mesh, B, L_per_element, L_matrix, L_inclusion, partition_materials.L)

exact_stiffness = m.get_homogenized_L(E, partition_materials.L)
inclusion_fraction = m.get_partition_material_ids().mean()
baseline = m.run_strain_path("standard", E, P, partition_materials)
stress_scale = np.abs(baseline.macroscopic_stress).max()

candidates = {
    "matrix L_m":          L_matrix,
    "homogenized (in use)": exact_stiffness,
    "Voigt average":       (1 - inclusion_fraction) * L_matrix + inclusion_fraction * L_inclusion,
    "midpoint Lame":       0.5 * (L_matrix + L_inclusion),
    "inclusion L_i":       L_inclusion,
}

print(f"\n{'C0 choice':<21} {'C0_1111 MPa':>12} {'alpha':>8} {'elastic err':>12} {'plastic err':>12} {'iters':>8}")
for name, C0 in candidates.items():
    offset_blocks = m.get_P0_offset_blocks(B, np.tile(C0, (m.element_count, 1, 1)), mesh.element_nodes,
                                           mesh.element_partition_ids)
    P0_transformed = m.get_P0_transformed(offset_blocks)
    dense_P0 = m.get_dense_P0(offset_blocks)
    compliance = np.linalg.inv(C0)

    model_stiffness = m.get_model_elastic_stiffness(dense_P0, partition_materials.L, compliance)
    elastic_error = np.abs(model_stiffness - exact_stiffness).max() / np.abs(exact_stiffness).max()

    smallest, largest = m.get_reference_stiffness_ratio_bounds(partition_materials, C0)
    relaxation = m.reference_stability_safety * 2 / (smallest + largest)
    try:
        result = m.run_strain_path("reference", E, P, partition_materials, P0_transformed, compliance, relaxation)
        plastic_error = (np.abs(result.macroscopic_stress - baseline.macroscopic_stress).max() / stress_scale)
        iterations = result.iterations_per_step.sum()
        print(f"{name:<21} {C0[0,0]/1e6:>12.1f} {relaxation:>8.4f} {elastic_error:>12.3e} "
              f"{plastic_error:>12.3e} {iterations:>8d}")
    except RuntimeError as error:
        print(f"{name:<21} {C0[0,0]/1e6:>12.1f} {relaxation:>8.4f} {elastic_error:>12.3e} "
              f"{'FAILED':>12} {'-':>8}  ({str(error)[:60]})")
