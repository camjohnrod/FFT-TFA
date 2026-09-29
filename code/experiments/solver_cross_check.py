# Runs both reference solver methods in one process and checks they reach the same root.
# Monkey-patches reference_solver_method, so it compares fixed point against Newton on identical inputs.
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

L_matrix = m.get_L(m.elastic_modulus_matrix, m.poisson_ratio_matrix)
L_inclusion = m.get_L(m.elastic_modulus_inclusion, m.poisson_ratio_inclusion)
partition_materials = m.get_partition_materials(L_matrix, L_inclusion)
mesh, B = m.get_periodic_mesh(), m.get_B()
L_per_element = m.get_value_per_material(L_matrix, L_inclusion, mesh.element_material_ids)
E, P, P0_transformed = m.get_offline_operators(mesh, B, L_per_element, L_matrix, L_inclusion, partition_materials.L)
reference_L = m.get_homogenized_L(E, partition_materials.L)
compliance = np.linalg.inv(reference_L)
relaxation = m.get_reference_relaxation_factor(partition_materials, reference_L)

results = {}
for method in ("newton", "fixed_point"):
    m.reference_solver_method = method
    results[method] = m.run_strain_path("reference", E, P, partition_materials, P0_transformed, compliance,
                                        relaxation)
    print(f"{method:<12}: {results[method].iterations_per_step.sum():>6d} iterations, "
          f"{results[method].solve_time_per_step.sum():.2f} s")

newton, fixed_point = results["newton"], results["fixed_point"]
stress_scale = np.abs(fixed_point.macroscopic_stress).max()
print(f"\nsame root? macroscopic stress   : {np.abs(newton.macroscopic_stress - fixed_point.macroscopic_stress).max() / stress_scale:.2e} relative")
strain_scale = np.abs(fixed_point.final_plastic_strain).max()
print(f"same root? final plastic strain : {np.abs(newton.final_plastic_strain - fixed_point.final_plastic_strain).max() / strain_scale:.2e} relative")
