# Prints the elastic model error at the configured mesh, with no solver tolerance involved.
# Used for the M-refinement study: run it once per partition count and watch the error fall. Use a SQUARE
# inclusion aligned to partition boundaries, otherwise refining M also changes the voxelised geometry and the
# comparison is confounded. With that control the error converged at first order in partition size.
#
# Reads the configuration from code/MAIN_Testing.py; edit the constants there, not here.
# Run from anywhere: python code/experiments/elastic_only.py

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
offset_blocks = np.fft.ifftn(P0_transformed, axes=(0, 1)).real.reshape(6 * m.partition_count, 6)
dense_P0 = m.get_dense_P0(offset_blocks)
model = m.get_model_elastic_stiffness(dense_P0, partition_materials.L, np.linalg.inv(reference_L))
error = np.abs(model - reference_L).max() / np.abs(reference_L).max()
print(f"RESULT n_p={m.partition_number_per_side:>3} M={m.partition_count:>5} "
      f"volume_fraction={m.get_partition_material_ids().mean():.4f} "
      f"err={error:.4e} C11_exact={reference_L[0, 0] / 1e6:.3f} C11_model={model[0, 0] / 1e6:.3f}")
