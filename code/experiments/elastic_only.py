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

problem = m.get_problem()
reference_L = problem.reference_L
dense_P0 = m.get_dense_P0(m.get_P0_offset_blocks_from_transformed(problem.P0_transformed))
model = m.get_model_elastic_stiffness(dense_P0, problem.partition_materials.L, problem.reference_compliance)
error = np.abs(model - reference_L).max() / np.abs(reference_L).max()
print(f"RESULT n_p={m.partition_number_per_side:>3} M={m.partition_count:>5} "
      f"volume_fraction={problem.partition_material_ids.mean():.4f} "
      f"err={error:.4e} C11_exact={reference_L[0, 0] / 1e6:.3f} C11_model={model[0, 0] / 1e6:.3f}")
