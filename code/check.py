# Checks the code on five small problems, each solved by every method (CP, Spectral LS, TFA). Run it after any
# change to the code:
#
#   python code/check.py         run the checks (a few seconds)
#   python code/check.py save    run the checks (except the comparison with the old reference) and, if they all
#                                pass, save the results as the reference that later runs must reproduce exactly
#
# The problems:
#   laminate      a laminate loaded in tension and shear, then unloaded and loaded in reverse. Its exact solution
#                 (below) is known, and every method must reproduce it in every cluster.
#   cut laminate  a laminate whose interfaces cut through partitions, so CP's partitions there hold both phases,
#                 loaded in transverse shear. Spectral LS and TFA must reproduce the exact solution in every cluster.
#                 CP shares a mixed partition's strain at equal polarization against C0, which is exact here only
#                 as C0 → 0 (equal stress), so its difference is reported but not judged.
#   circle        a circular fiber under tension and shear, the kind of problem config.py sets up by default, with a
#                 band of mixed partitions around the fiber.
#   z laminate    a laminate with interfaces normal to z, so the geometry varies along z: meshed and partitioned in 3D,
#                 the only test of the kernel's ξ_z frequencies. Every method must reproduce its exact solution.
#   sphere        a spherical inclusion, meshed and partitioned in 3D, with mixed partitions all around it.
# On all five, every method must complete every load step and, once a reference is saved, reproduce it bit for bit.
# Where CP has mixed partitions (cut laminate, circle, sphere), their elastic law must match its closed form (LS-22 in
# elasticity, notes/2_ls_model.md section 4.4): a wrong split or split tangent fails there.
# So a change meant to leave the results alone (a cleanup) must pass, and one meant to change them must not: check its
# results, then save a new reference. The reference is code/check_reference.npz, and holds bit for bit only at
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

import ls_model
import run
from material import get_eigenstrain_sensitivity, get_plastic_eigenstrain, get_unloaded_plastic_state

# Next to the code, not in the cache folder, so clearing the cache to rebuild E and P keeps it.
reference_path = pathlib.Path(__file__).parent / "check_reference.npz"

# Every config.py input that shapes the problems or the solvers, pinned to a fixed value: a 27 × 27 element mesh
# with 9 × 9 partitions, a yielding and hardening matrix, every method switched on.
pinned_inputs = {
    "domain_side_length": 1e-3, "inclusion_shape": "circle", "inclusion_radius": 0.35e-3,
    "layer_thickness": (3 / 9) * 1e-3, "element_number_per_side": 27, "element_number_along_z": 1,
    "partition_number_per_side": 9,
    "elastic_modulus_inclusion": 10e9, "elastic_modulus_matrix": 100e6, "poisson_ratio_inclusion": 0.3,
    "poisson_ratio_matrix": 0.19, "inclusion_yield_stress": np.inf, "matrix_yield_stress": 1.0e6,
    "inclusion_hardening_modulus": 0.0, "matrix_hardening_modulus": 10e6,
    "max_macro_strain": np.array([0.015, 0.020, 0.0, 0.03, 0.0, 0.0]), "strain_increment_count": 12,
    "load_path_shape": "monotonic",
    "include_spectral_ls": True, "include_tfa_model": True, "reference_stiffness": "matrix",
    "reference_stiffness_scale": 0.01, "spectral_ls_reference_stiffness": "midpoint", "cp_fourier_preconditioner": True,
    "spectral_ls_fourier_preconditioner": False,
    "convergence_tolerance": 1e-6, "residual_strain_scale_floor": 1e-4, "newton_max_iterations": 50,
    "newton_krylov_tolerance": 1e-3, "newton_krylov_restart": 50, "newton_krylov_max_restarts": None,
    "ls_divergence_limit": 1e3, "tfa_divergence_limit": 1.0, "offline_columns_per_solve": 100,
    "line_search_sufficient_decrease": 1e-4, "line_search_step_reduction": 0.5, "line_search_max_reductions": 10,
    "mixed_partition_tolerance": 1e-12, "mixed_partition_max_iterations": 50}

# What each problem changes from pinned_inputs.
check_problems = {
    # A layer 3 partitions thick, loaded so that every interface-normal component (ε11, γ12, γ13) takes part, over a
    # cyclic path: the only test of unloading, reverse yielding and zero imposed strain. 4 increments per quarter
    # still yield at both zero crossings. The homogenized C0 is the only test of that choice.
    "laminate": {"inclusion_shape": "laminate", "max_macro_strain": np.array([0.015, 0.020, 0.0, 0.03, 0.0, 0.01]),
                 "load_path_shape": "cyclic", "strain_increment_count": 4, "reference_stiffness": "homogenized"},
    # A layer 7 elements thick, so each interface cuts a partition (2/3 fiber). In transverse shear γ12 both phases
    # carry the same stress, so the equal-stress split is exact there and every method must match the exact solution.
    "cut laminate": {"inclusion_shape": "laminate", "layer_thickness": (7 / 27) * 1e-3,
                     "max_macro_strain": np.array([0.0, 0.0, 0.0, 0.03, 0.0, 0.0]), "load_path_shape": "cyclic",
                     "strain_increment_count": 4},
    "circle": {},
    # The laminate turned so its interfaces are normal to z, on a 9³ mesh with 3³ partitions (the layer is the middle
    # partition layer), loaded like the laminate with every interface-normal component (ε33, γ23, γ13) taking part.
    "z laminate": {"inclusion_shape": "laminate_z", "element_number_per_side": 9, "partition_number_per_side": 3,
                   "max_macro_strain": np.array([0.020, 0.0, 0.015, 0.0, 0.03, 0.01]), "load_path_shape": "cyclic",
                   "strain_increment_count": 4},
    # A sphere on a 12³ mesh with 4³ partitions, loaded with z components: the only test of mixed partitions in 3D.
    "sphere": {"inclusion_shape": "sphere", "element_number_per_side": 12, "partition_number_per_side": 4,
               "max_macro_strain": np.array([0.015, 0.0, 0.020, 0.0, 0.0, 0.03])},
}

# The methods whose model is not exact on a problem: compared with the exact solution, but not judged by it.
inexact_methods = {"cut laminate": ("CP",)}

## ------- Exact Laminate Solution ------- ##

# An inclusion layer between matrix layers. Under a uniform macrostrain each phase has one uniform state, even in
# plasticity, so the whole cell reduces to two states, one per phase:
#   - compatibility keeps the strains along the interface equal to the macrostrain in both phases;
#   - equilibrium makes the tractions on the interface equal in both phases;
#   - the phase average of the strain is the macrostrain.
# That leaves the matrix phase's three interface-normal strains as the only unknowns of a load step, found by Newton.

# By shape, the Voigt positions of the strain components that may jump across the interface, while the stress
# components at the same positions, the tractions, may not: ε11, γ12, γ13 for a normal along x; ε33, γ23, γ13 along z.
interface_normal_components_by_shape = {"laminate": [0, 3, 5], "laminate_z": [2, 4, 5]}

def get_phase_materials(materials):
    # One row per phase, matrix first, copied from the first cluster of each phase.
    return ls_model.take_rows(materials, [np.flatnonzero(materials.material_ids == phase)[0] for phase in (0, 1)])

def get_phase_strains(macro_strain, matrix_normal_strain, inclusion_fraction, interface_normal_components):
    # The strain of each phase, matrix first, given the matrix's interface-normal components.
    phase_strains = np.tile(macro_strain, (2, 1))
    phase_strains[0, interface_normal_components] = matrix_normal_strain
    phase_strains[1, interface_normal_components] = ((macro_strain[interface_normal_components]
                                                      - (1 - inclusion_fraction) * matrix_normal_strain)
                                                     / inclusion_fraction)
    return phase_strains

def solve_laminate_step(macro_strain, matrix_normal_strain, phase_materials, phase_history, inclusion_fraction,
                        interface_normal_components):
    # Newton on the interface tractions' jump, starting from the given matrix_normal_strain. The inclusion's
    # interface-normal strain moves by -(1 - f)/f times the matrix's, which gives the Jacobian below. Returns the
    # converged matrix_normal_strain and the phases' plastic state and stress.
    for _ in range(config.laminate_solver_max_iterations):
        phase_strains = get_phase_strains(macro_strain, matrix_normal_strain, inclusion_fraction,
                                          interface_normal_components)
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

def get_laminate_phase_stress(clusters, applied_macro_strain):
    # The exact stress of each phase at every load step, shape (load steps, 2, 6), matrix first.
    interface_normal_components = interface_normal_components_by_shape[config.inclusion_shape]
    phase_materials = get_phase_materials(clusters.materials)
    inclusion_fraction = clusters.average(clusters.materials.material_ids)
    phase_history = get_unloaded_plastic_state(2)
    matrix_normal_strain = np.zeros(len(interface_normal_components))
    previous_macro_strain = np.zeros(6)
    phase_stress = np.zeros((len(applied_macro_strain), 2, 6))
    for step, macro_strain in enumerate(applied_macro_strain):
        # Start from the previous step's matrix strain, moved by the macrostrain increment.
        matrix_normal_strain, phase_history, phase_stress[step] = solve_laminate_step(
            macro_strain, matrix_normal_strain + (macro_strain - previous_macro_strain)[interface_normal_components],
            phase_materials, phase_history, inclusion_fraction, interface_normal_components)
        previous_macro_strain = macro_strain
    return phase_stress

def get_relative_difference(stress, exact_stress):
    # The largest difference over every step and component, relative to the largest exact stress.
    return np.abs(stress - exact_stress).max() / np.abs(exact_stress).max()

def get_laminate_error(result, exact_phase_stress):
    # The larger of the macroscopic stress's difference and every cluster's, clusters judged phase by phase on their
    # own phase's stress scale: on the fiber's, about a hundred times larger, a matrix error would hide.
    material_ids = result.clusters.materials.material_ids
    exact_cluster_stress = exact_phase_stress[:, material_ids]
    exact_macroscopic_stress = result.clusters.average(np.moveaxis(exact_cluster_stress, 1, 0))
    errors = [get_relative_difference(result.macroscopic_stress, exact_macroscopic_stress)]
    for phase_clusters in (material_ids == 0, material_ids == 1):
        errors.append(get_relative_difference(result.cluster_stress_per_step[:, phase_clusters],
                                              exact_cluster_stress[:, phase_clusters]))
    return max(errors)

## ------- Mixed Partition Law ------- ##

def get_mixed_partition_law_error(clusters, reference_L):
    # In elasticity the equal-polarization split LS-22 has a closed form: σ_A = C*_A ε_A with
    # C*_A = C0 + [c_m (L_m − C0)⁻¹ + c_f (L_f − C0)⁻¹]⁻¹. The largest relative difference of the mixed partitions'
    # stress and tangent from it, at small random partition strains (well below yield); None without mixed partitions.
    matrix, inclusion = ls_model.get_mixed_partition_clusters(clusters)
    if len(matrix) == 0:
        return None
    strain = 1e-4 * np.random.default_rng(0).standard_normal((clusters.partition_count, 6))
    history = get_unloaded_plastic_state(len(clusters.partition))
    cluster_strain = ls_model.get_cluster_strain(strain, clusters, history, reference_L)
    _, cluster_stress = get_plastic_eigenstrain(cluster_strain, clusters.materials, history)
    mixed = clusters.partition[matrix]
    stress = ls_model.get_partition_stress(cluster_stress, clusters)[mixed]
    tangent = ls_model.get_partition_tangent(cluster_strain, clusters, history, reference_L)[mixed]

    c_m, c_f = clusters.volume_fraction[matrix, None, None], clusters.volume_fraction[inclusion, None, None]
    L = clusters.materials.L
    exact_tangent = reference_L + np.linalg.inv(c_m * np.linalg.inv(L[matrix] - reference_L)
                                                + c_f * np.linalg.inv(L[inclusion] - reference_L))
    exact_stress = np.einsum('pij,pj->pi', exact_tangent, strain[mixed])
    return max(get_relative_difference(stress, exact_stress), get_relative_difference(tangent, exact_tangent))

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
        for field in ("macroscopic_stress", "cluster_stress_per_step", "newton_iterations_per_step",
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
            if config.inclusion_shape in interface_normal_components_by_shape:
                exact_phase_stress = get_laminate_phase_stress(problem.partitions.clusters,
                                                               run.get_applied_macro_strain())
                for method, result in results.items():
                    error = get_laminate_error(result, exact_phase_stress)
                    if method in inexact_methods.get(problem_name, ()):
                        print(f"[INFO] {problem_name}: {method} differs from the exact solution by {error:.1e} "
                              "(not exact by design)")
                        continue
                    all_passed &= report(error <= config.laminate_tolerance,
                                         f"{problem_name}: {method} matches the exact solution (largest relative "
                                         f"difference {error:.1e}, tolerance {config.laminate_tolerance:.0e})")
            law_error = get_mixed_partition_law_error(problem.partitions.clusters, problem.partitions.reference_L)
            if law_error is not None:
                all_passed &= report(law_error <= config.mixed_partition_check_tolerance,
                                     f"{problem_name}: CP's mixed partitions match the elastic closed form (largest "
                                     f"relative difference {law_error:.1e}, tolerance "
                                     f"{config.mixed_partition_check_tolerance:.0e})")
            arrays = get_result_arrays(problem_name, results)
            result_arrays.update(arrays)
            if reference is not None and mode == "check":
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
