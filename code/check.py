# Checks the code on two small problems, each solved by every method (CP, Spectral LS, TFA). Run it after any change
# to the code:
#
#   python code/check.py         run the checks (a few seconds)
#   python code/check.py save    run the checks and, if they all pass, save the results as the reference that later
#                                runs must reproduce exactly
#
# The problems:
#   laminate  a laminate loaded in tension and shear, then unloaded and loaded in reverse. Its exact solution (below)
#             is known, and every method must reproduce it in every partition.
#   square    a square inclusion under tension and shear, the kind of problem config.py sets up by default.
# On both, every method must complete every load step and, once a reference is saved, reproduce it bit for bit. So a
# change meant to leave the results alone (a cleanup) must pass, and one meant to change them must not: check its
# results, then save a new reference. The reference is code/cache/check_reference.npz, and holds bit for bit only at
# the config.thread_count it was saved with.
#
# The problems are set below, not read from config.py, so that the checks mean the same thing whatever config.py holds;
# only the check settings (laminate_*) come from config.py. config.py, the cache and code/output are not touched.

import contextlib
import io
import sys
import tempfile
import pathlib

import config  # first: it sets the thread count before numpy loads
import numpy as np

import run
from material import get_eigenstrain_sensitivity, get_plastic_eigenstrain, get_unloaded_plastic_state

reference_path = config.cache_folder / "check_reference.npz"

# Every config.py input that shapes the problems or the solvers, pinned to a fixed value: a 27 × 27 element mesh
# with 9 × 9 partitions, a yielding and hardening matrix, every method switched on.
pinned_inputs = {
    "domain_side_length": 1e-3, "inclusion_shape": "square", "inclusion_side_length": (5 / 9) * 1e-3,
    "inclusion_radius": 0.35e-3, "element_number_per_side": 27, "element_number_along_z": 1,
    "partition_number_per_side": 9,
    "elastic_modulus_inclusion": 10e9, "elastic_modulus_matrix": 100e6, "poisson_ratio_inclusion": 0.3,
    "poisson_ratio_matrix": 0.19, "inclusion_yield_stress": np.inf, "matrix_yield_stress": 1.0e6,
    "inclusion_hardening_modulus": 0.0, "matrix_hardening_modulus": 10e6,
    "max_macro_strain": np.array([0.015, 0.020, 0.0, 0.03, 0.0, 0.0]), "strain_increment_count": 12,
    "load_path_shape": "monotonic",
    "include_spectral_ls": True, "include_tfa_model": True, "reference_stiffness": "matrix",
    "convergence_tolerance": 1e-6, "residual_strain_scale_floor": 1e-4, "newton_max_iterations": 50,
    "newton_krylov_tolerance": 1e-3, "newton_krylov_restart": 50, "newton_krylov_max_restarts": None,
    "ls_divergence_limit": 1e3, "tfa_divergence_limit": 1.0, "offline_columns_per_solve": 100}

# What each problem changes from pinned_inputs.
check_problems = {
    # A layer 3 partitions thick, loaded so that every interface-normal component (ε11, γ12, γ13) takes part, over a
    # cyclic path: the only test of unloading, reverse yielding and zero imposed strain. 4 increments per quarter
    # still yield at both zero crossings. The homogenized C0 is the only test of that choice.
    "laminate": {"inclusion_shape": "laminate", "inclusion_side_length": (3 / 9) * 1e-3,
                 "max_macro_strain": np.array([0.015, 0.020, 0.0, 0.03, 0.0, 0.01]), "load_path_shape": "cyclic",
                 "strain_increment_count": 4, "reference_stiffness": "homogenized"},
    "square": {},
}

## ------- Exact Laminate Solution ------- ##

# An inclusion layer between matrix layers, with interfaces normal to x. Under a uniform macrostrain each phase has one
# uniform state, even in plasticity, so the whole cell reduces to two states, one per phase:
#   - compatibility keeps the strains along the interface, ε22, ε33 and γ23, equal to the macrostrain in both phases;
#   - equilibrium makes the tractions on the interface, σ11, σ12 and σ13, equal in both phases;
#   - the phase average of the strain is the macrostrain.
# That leaves the matrix phase's ε11, γ12 and γ13 as the only unknowns of a load step, found here by Newton.

# Voigt positions 0, 3, 5: the strain components ε11, γ12, γ13 may jump across an interface normal to x, while the
# stress components at the same positions, the tractions σ11, σ12, σ13, may not.
interface_normal_components = [0, 3, 5]

def get_phase_materials(partition_materials):
    # One row per phase, matrix first, copied from the first partition of each phase.
    rows = [np.flatnonzero(partition_materials.material_ids == material_id)[0] for material_id in (0, 1)]
    return partition_materials._make(field[rows] for field in partition_materials)

def get_phase_strains(macro_strain, matrix_normal_strain, inclusion_fraction):
    # The strain of each phase, matrix first, given the matrix's interface-normal components.
    phase_strains = np.tile(macro_strain, (2, 1))
    phase_strains[0, interface_normal_components] = matrix_normal_strain
    phase_strains[1, interface_normal_components] = ((macro_strain[interface_normal_components]
                                                      - (1 - inclusion_fraction) * matrix_normal_strain)
                                                     / inclusion_fraction)
    return phase_strains

def solve_laminate_step(macro_strain, matrix_normal_strain, phase_materials, phase_history, inclusion_fraction):
    # Newton on the interface tractions' jump, starting from the given matrix_normal_strain. The inclusion's
    # interface-normal strain moves by -(1 - f)/f times the matrix's, which gives the Jacobian below. Returns the
    # converged matrix_normal_strain and the phases' plastic state and stress.
    for _ in range(config.laminate_solver_max_iterations):
        phase_strains = get_phase_strains(macro_strain, matrix_normal_strain, inclusion_fraction)
        phase_state, phase_stress = get_plastic_eigenstrain(phase_strains, phase_materials, phase_history)
        traction_jump = (phase_stress[0] - phase_stress[1])[interface_normal_components]
        if np.abs(traction_jump).max() <= config.laminate_solver_tolerance * np.abs(phase_stress).max():
            return matrix_normal_strain, phase_state, phase_stress
        sensitivity = get_eigenstrain_sensitivity(phase_strains, phase_materials, phase_history)
        algorithmic_stiffness = phase_materials.L @ (np.eye(6) - sensitivity)
        normal_block = np.ix_(interface_normal_components, interface_normal_components)
        jacobian = (algorithmic_stiffness[0][normal_block]
                    + (1 - inclusion_fraction) / inclusion_fraction * algorithmic_stiffness[1][normal_block])
        matrix_normal_strain = matrix_normal_strain - np.linalg.solve(jacobian, traction_jump)
    raise RuntimeError(f"the laminate's exact solution did not converge at macrostrain {macro_strain}.")

def get_laminate_phase_stress(partition_materials, applied_macro_strain):
    # The exact stress of each phase at every load step, shape (load steps, 2, 6), matrix first.
    phase_materials = get_phase_materials(partition_materials)
    inclusion_fraction = partition_materials.material_ids.mean()
    phase_history = get_unloaded_plastic_state(2)
    matrix_normal_strain = np.zeros(len(interface_normal_components))
    previous_macro_strain = np.zeros(6)
    phase_stress = np.zeros((len(applied_macro_strain), 2, 6))
    for step, macro_strain in enumerate(applied_macro_strain):
        # Start from the previous step's matrix strain, moved by the macrostrain increment.
        matrix_normal_strain, phase_history, phase_stress[step] = solve_laminate_step(
            macro_strain, matrix_normal_strain + (macro_strain - previous_macro_strain)[interface_normal_components],
            phase_materials, phase_history, inclusion_fraction)
        previous_macro_strain = macro_strain
    return phase_stress

def get_relative_difference(stress, exact_stress):
    # The largest difference over every step and component, relative to the largest exact stress.
    return np.abs(stress - exact_stress).max() / np.abs(exact_stress).max()

def get_laminate_error(result, exact_phase_stress):
    # The larger of the macroscopic stress's difference and every partition's, partitions judged phase by phase on
    # their own phase's stress scale: on the inclusion's, about a hundred times larger, a matrix error would hide.
    exact_partition_stress = exact_phase_stress[:, result.material_ids]
    errors = [get_relative_difference(result.macroscopic_stress, exact_partition_stress.mean(axis=1))]
    for phase_partitions in (result.material_ids == 0, result.material_ids == 1):
        errors.append(get_relative_difference(result.partition_stress_per_step[:, phase_partitions],
                                              exact_partition_stress[:, phase_partitions]))
    return max(errors)

## ------- Checks ------- ##

def solve_check_problem(problem_inputs, cache_folder):
    # Every method's results on one check problem, with config's inputs replaced in memory and the solvers' printing
    # silenced.
    for name, value in {**pinned_inputs, **problem_inputs}.items():
        if not hasattr(config, name):
            raise AttributeError(f"config.py has no input {name}: update check.py's pinned_inputs or check_problems")
        setattr(config, name, value)
    config.cache_folder = cache_folder
    config.check_inputs()
    with contextlib.redirect_stdout(io.StringIO()):
        problem = run.get_problem()
        results = run.solve_all(problem)
    return problem, results

def report(passed, description):
    print(f"[{'PASS' if passed else 'FAIL'}] {description}")
    return passed

def get_result_arrays(problem_name, results):
    # Every array the reference holds, keyed "<problem>/<method>/<array>".
    arrays = {}
    for method, result in results.items():
        for field in ("macroscopic_stress", "partition_stress_per_step", "newton_iterations_per_step",
                      "gmres_iterations_per_step", "completed_step_count"):
            arrays[f"{problem_name}/{method}/{field}"] = np.asarray(getattr(result, field))
    return arrays

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode not in ("check", "save"):
        sys.exit(f"unknown option {mode!r}: use no option, or save")
    reference = None
    if reference_path.exists():
        with np.load(reference_path) as saved:
            reference = {name: saved[name] for name in saved.files}

    all_passed = True
    result_arrays = {}
    with tempfile.TemporaryDirectory() as cache_folder:
        for problem_name, problem_inputs in check_problems.items():
            problem, results = solve_check_problem(problem_inputs, pathlib.Path(cache_folder))
            step_count = len(run.get_applied_macro_strain())
            for method, result in results.items():
                all_passed &= report(result.completed_step_count == step_count,
                                     f"{problem_name}: {method} completed {result.completed_step_count} of "
                                     f"{step_count} load steps")
            if config.inclusion_shape == "laminate":
                exact_phase_stress = get_laminate_phase_stress(problem.partitions.partition_materials,
                                                               run.get_applied_macro_strain())
                for method, result in results.items():
                    error = get_laminate_error(result, exact_phase_stress)
                    all_passed &= report(error <= config.laminate_tolerance,
                                         f"{problem_name}: {method} matches the exact solution (largest relative "
                                         f"difference {error:.1e}, tolerance {config.laminate_tolerance:.0e})")
            arrays = get_result_arrays(problem_name, results)
            result_arrays.update(arrays)
            if reference is not None:
                changed = [name for name in arrays if name not in reference
                           or not np.array_equal(arrays[name], reference[name])]
                all_passed &= report(not changed, f"{problem_name}: identical to the saved reference"
                                     + "".join(f"\n       differs: {name}" for name in changed))

    if reference is None and mode == "check":
        print("No reference saved yet: run python code/check.py save to save one.")
    if mode == "save":
        if all_passed:
            reference_path.parent.mkdir(exist_ok=True)
            np.savez(reference_path, **result_arrays)
            print(f"Reference saved to {reference_path}")
        else:
            print("Reference not saved, since not every check passed.")
    print("All checks passed." if all_passed else "Some checks FAILED.")
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
