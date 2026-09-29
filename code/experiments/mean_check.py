# Checks two invariants on a converged, partly plastic reference solve: the LS-9 mean strain
# consistency sum(c_B eps_B) = eps_bar, and that recomputing the residual from scratch reproduces the one the
# solver stopped on (guarding against stale cached state inside the iteration).
#
# Reads the configuration from code/MAIN_Testing.py; edit the constants there, not here.
# Run from anywhere: python code/experiments/mean_check.py

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
C0 = m.get_homogenized_L(E, partition_materials.L)
compliance = np.linalg.inv(C0)
relaxation = m.get_reference_relaxation_factor(partition_materials, C0)

macro_strain = m.max_macro_strain / 3        # partly plastic
plastic, accumulated = np.zeros((m.partition_count, 6)), np.zeros(m.partition_count)
stress, plastic, accumulated, residuals, strain = m.reference_iteration(
    macro_strain, np.tile(macro_strain, (m.partition_count, 1)), partition_materials, plastic, accumulated,
    P0_transformed, compliance, relaxation)

mean_strain = strain.mean(axis=0)
strain_scale = max(np.abs(macro_strain).max(), np.finfo(float).tiny)
print(f"converged in {len(residuals)} iterations, final relative residual {residuals[-1]:.2e}")
print(f"mean strain consistency  |<eps> - eps_bar| / |eps_bar| : "
      f"{np.abs(mean_strain - macro_strain).max() / strain_scale:.2e}")
print(f"yielding partitions: {int((accumulated > 0).sum())} of {m.partition_count}")

# residual recomputed from scratch, guarding against stale state inside the loop
b = np.tile(macro_strain, (m.partition_count, 1))
_, _, fresh_stress = m.get_plastic_eigenstrain(strain, partition_materials, np.zeros((m.partition_count, 6)),
                                               np.zeros(m.partition_count))
fresh_residual = strain - b - m.get_reference_induced_strain(P0_transformed, strain - fresh_stress @ compliance.T)
print(f"independently recomputed relative residual                : "
      f"{np.linalg.norm(fresh_residual) / np.linalg.norm(strain):.2e}")
