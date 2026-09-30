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

problem = m.get_problem()
materials, compliance = problem.partition_materials, problem.reference_compliance
relaxation = m.get_reference_relaxation_factor(materials, problem.reference_L)

macro_strain = m.max_macro_strain / 3        # partly plastic
no_plastic_history = m.PlasticState(np.zeros((m.partition_count, 6)), np.zeros(m.partition_count))
b = np.tile(macro_strain, (m.partition_count, 1))
stress, plastic_state, residuals, strain = m.reference_iteration(
    macro_strain, b, materials, no_plastic_history, problem.P0_transformed, compliance, relaxation,
    m.reference_solver_method)

mean_strain = strain.mean(axis=0)
strain_scale = max(np.abs(macro_strain).max(), np.finfo(float).tiny)
print(f"converged in {len(residuals)} iterations, final relative residual {residuals[-1]:.2e}")
print(f"mean strain consistency  |<eps> - eps_bar| / |eps_bar| : "
      f"{np.abs(mean_strain - macro_strain).max() / strain_scale:.2e}")
print(f"yielding partitions: {int((plastic_state.accumulated_plastic_strain > 0).sum())} of {m.partition_count}")

# residual recomputed from scratch, guarding against stale state inside the loop
_, fresh_stress = m.get_plastic_eigenstrain(strain, materials, no_plastic_history)
fresh_residual = strain - b - m.get_reference_induced_strain(problem.P0_transformed, strain - fresh_stress @ compliance.T)
print(f"independently recomputed relative residual                : "
      f"{m.get_relative_residual(fresh_residual, macro_strain):.2e}")
