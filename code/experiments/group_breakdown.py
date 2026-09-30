# Prints per-timing-group wall time and call counts for all three solvers side by side.
# This is the diagnostic that exposed 52 percent of the Newton solver's time sitting unattributed in "other"
# (scipy GMRES scaffolding), which motivated splitting operator-application time out of its enclosing group.
#
# Reads the configuration from code/MAIN_Testing.py; edit the constants there, not here.
# Run from anywhere: python code/experiments/group_breakdown.py

import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import MAIN_Testing as m

problem = m.get_problem()
relaxation = m.get_reference_relaxation_factor(problem.partition_materials, problem.reference_L)
solvers = m.get_solvers(problem.P0_transformed, problem.reference_compliance, relaxation)

label_width = max(len(solver_name) for solver_name in solvers) + 2
print(f"\n{'':<{label_width}}" + "".join(f"{group:>22}" for group in m.online_time_groups)
      + f"{'Other':>12}{'total':>10}")
for solver_name, solver in solvers.items():
    result = m.run_strain_path(solver_name, solver, problem.E, problem.P, problem.partition_materials)
    group_time = result.group_time_per_step.sum(axis=0)
    total = result.solve_time_per_step.sum()
    print(f"{solver_name:<{label_width}}" + "".join(f"{time:>21.3f}s" for time in group_time)
          + f"{total - group_time.sum():>11.3f}s{total:>9.2f}s")
    calls = result.group_calls_per_step.sum(axis=0)
    print(f"{'  calls':<{label_width}}" + "".join(f"{count:>22d}" for count in calls)
          + f"{'':>12}{result.iterations_per_step.sum():>9d} iters")
