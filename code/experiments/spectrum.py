# Builds the dense elastic Jacobian I - P0 (I - C0^-1 L) and reports its spectrum.
# Confirms the Jacobian is similar to a symmetric matrix (real spectrum) and gives the exact damped-Richardson
# stability limit alpha < 2 / lambda_max, which is what get_reference_stiffness_ratio_bounds approximates from
# 6x6 matrices at run time. At 27/9, contrast 100, the spectrum was [0.411, 50.87] so alpha < 0.0393.
#
# Reads the configuration from code/MAIN_Testing.py; edit the constants there, not here.
# Run from anywhere: python code/experiments/spectrum.py

import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.linalg
import MAIN_Testing as m

problem = m.get_problem()
C0 = problem.reference_L

print("eig(C0)       MPa:", np.round(np.linalg.eigvalsh(C0) / 1e6, 2))
print("eig(L_matrix) MPa:", np.round(np.linalg.eigvalsh(problem.L_matrix) / 1e6, 2))
print("eig(L_incl)   MPa:", np.round(np.linalg.eigvalsh(problem.L_inclusion) / 1e6, 2))

n = 6 * m.partition_count
dense_P0 = m.get_dense_P0(m.get_P0_offset_blocks_from_transformed(problem.P0_transformed))
C0_M = np.kron(np.eye(m.partition_count), C0)
L_M = scipy.linalg.block_diag(*problem.partition_materials.L)

J = np.eye(n) - dense_P0 @ (np.eye(n) - np.linalg.solve(C0_M, L_M))
eigenvalues = np.linalg.eigvals(J)
print(f"\nelastic J spectrum: real [{eigenvalues.real.min():.4g}, {eigenvalues.real.max():.4g}], "
      f"max|imag| {np.abs(eigenvalues.imag).max():.2e}")
print(f"stability limit alpha < 2/lambda_max = {2 / eigenvalues.real.max():.4g}")
optimal = 2 / (eigenvalues.real.min() + eigenvalues.real.max())
print(f"optimal alpha = 2/(lmin+lmax) = {optimal:.4g}, spectral radius there = "
      f"{abs(1 - optimal * eigenvalues.real.min()):.6f}")
