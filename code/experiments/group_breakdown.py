# Prints per-timing-group wall time and call counts for all three solvers side by side.
# This is the diagnostic that exposed 52 percent of the Newton solver's time sitting unattributed in "other"
# (scipy GMRES scaffolding), which motivated splitting operator-application time out of its enclosing group.
#
# Reads the configuration from code/MAIN_Testing.py; edit the constants there, not here.
# Run from anywhere: python code/experiments/group_breakdown.py

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

runs = {"standard": ("standard", None, None), "fft": ("fft_preconditioned", P0_transformed, None),
        "reference": ("reference", P0_transformed, compliance)}
print(f"\n{'':<11}" + "".join(f"{g:>22}" for g in m.online_time_groups) + f"{'Other':>12}{'total':>10}")
for label, (name, p0, comp) in runs.items():
    result = m.run_strain_path(name, E, P, partition_materials, p0, comp, relaxation)
    group_time = result.group_time_per_step.sum(axis=0)
    total = result.solve_time_per_step.sum()
    print(f"{label:<11}" + "".join(f"{t:>21.3f}s" for t in group_time)
          + f"{total - group_time.sum():>11.3f}s{total:>9.2f}s")
    calls = result.group_calls_per_step.sum(axis=0)
    print(f"{'  calls':<11}" + "".join(f"{c:>22d}" for c in calls)
          + f"{'':>12}{result.iterations_per_step.sum():>9d} iters")
