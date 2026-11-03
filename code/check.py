# Checks the code on small problems, each solved by every method (FFT-TFA, Spectral LS, TFA). Run it after any change to
# the code:
#
#   python code/check.py         run the checks (about a minute)
#   python code/check.py save    run the checks (except the comparison with the old results) and, if they all pass,
#                                save the results, which later runs must reproduce exactly
#
# The problems are in check_problems below. The checks:
# - On every problem, every method must complete every load step and, once results are saved, reproduce them bit for
#   bit.
# - On every problem, FFT-TFA's interaction at full reach (FT-2, FT-3) must equal the element-level LS operator averaged
#   over its clusters, computed here from the elements: a wrong offset, pair count, correction block or average fails.
# - On every laminate, every method must reproduce its exact solution in every cluster, except where inexact_methods
#   says its model is not exact there, and FFT-TFA at full reach must too: that separates its truncation, a model
#   error, from a wrong law or Newton solve.
# - On the laminate, FFT-TFA at full reach and TFA must also reproduce it with each partition's matrix part cut into two
#   clusters, since each phase's exact state is uniform: that tests clusters dividing a partition's phase.
# - On the fiber and the sphere, every method must stop at the same accuracy: its stresses at convergence_tolerance
#   within stopping_accuracy_tolerance of its own solution at stopping_check_tolerance. (Not with k-means clusters of a
#   few elements: there a cluster on the verge of yielding can converge elastic at one tolerance and plastic at the other,
#   for TFA too.)
# - On the fiber, the reference, solved at the working resolution, must equal Spectral LS bit for bit, and load back
#   from its cache unchanged.
# So a change meant to leave the results alone (a cleanup) must pass, and one meant to change them must not: check its
# results, then save them anew. They are in code/check_saved_results.npz, and hold bit for bit only at the
# config.thread_count they were saved with.
#
# The problems are set below, not read from config.py, so that the checks mean the same thing whatever config.py holds;
# only the check settings (Checks in config.py) come from config.py. config.py, the cache and code/output are not
# touched.

import contextlib
import io
import sys
import tempfile
import pathlib

import config  # first: it sets the thread count before numpy loads
from config import strain
import numpy as np

import cell
import fft_tfa_model
import ls_model
import run
import tfa_model
from material import get_eigenstrain_sensitivity, get_plastic_eigenstrain, get_unloaded_plastic_state

# Next to the code, not in the cache folder, so clearing the cache keeps it.
saved_results_path = pathlib.Path(__file__).parent / "check_saved_results.npz"

# The stiffly contrasted materials of every problem.
check_materials = dict(matrix=dict(E=100e6, poisson=0.19, yield_stress=1.0e6, hardening=10e6),
                       inclusion=dict(E=10e9, poisson=0.3, yield_stress=np.inf, hardening=0))

def get_entries(partitions, spectral_elements, matrix_clusters=1, reach=0):
    # One entry of each method.
    return {"fft_tfa": [dict(partitions=partitions, matrix_clusters=matrix_clusters, inclusion_clusters=1,
                             reach=reach)],
            "spectral_ls": [dict(elements=spectral_elements, C0="midpoint")],
            "tfa": [dict(partitions=partitions, matrix_clusters=matrix_clusters, inclusion_clusters=1)]}

def get_cyclic_load(max_strain):
    # Out to the given strain, to its opposite and back to zero: unloading, reverse yielding and zero imposed strain.
    opposite = [-component for component in max_strain]
    return {"all_load_paths": {"check": [max_strain, opposite, strain()]}, "steps_to_first_turning_point": 4}

def get_monotonic_load(max_strain):
    return {"all_load_paths": {"check": [max_strain]}, "steps_to_first_turning_point": 12}

# The problems' geometries.
check_geometries = {
    "fiber": dict(shape="fiber", radius=0.35e-3),
    "sphere": dict(shape="sphere", radius=0.35e-3),
    "laminate": dict(shape="laminate", thickness=(3 / 9) * 1e-3),
    "cut laminate": dict(shape="laminate", thickness=(7 / 27) * 1e-3),
    "z laminate": dict(shape="laminate_z", thickness=(3 / 9) * 1e-3),
    "random fibers": dict(shape="random fibers", count=6, volume_fraction=0.3, gap=1 / 27, seed=1),
}

# Every config.py input that shapes the problems or the solvers, pinned to a fixed value: a 27 × 27 element mesh, a
# yielding and hardening matrix.
pinned_inputs = {
    "domain_side_length": 1e-3, "all_geometries": check_geometries, "geometry": "fiber", "elements_per_side": 27,
    "all_materials": {"check": check_materials}, "materials": "check", "load_path": "check",
    **get_entries(partitions=9, spectral_elements=27),
    **get_monotonic_load(strain(e11=0.015, e22=0.020, g12=0.03)),
    "convergence_tolerance": 1e-6, "residual_strain_scale_floor": 1e-4, "newton_max_iterations": 50,
    "krylov_tolerance": 1e-3, "gmres_restart": 50, "gmres_max_restarts": None, "spectral_ls_cg_max_iterations": None,
    "ls_divergence_limit": 1e3, "tfa_divergence_limit": 1.0, "fe_columns_per_solve": 100,
    "fe_factorization_ordering": "MMD_AT_PLUS_A",
    "line_search_sufficient_decrease": 1e-4, "line_search_step_reduction": 0.5, "line_search_max_reductions": 10,
    "k_means_iterations": 50, "k_means_seed": 0, "k_means_start": "++", "fiber_placement_attempts": 100000,
    "fiber_push_margin": 1e-3, "fiber_push_passes": 10000, "reference": None, "k_means_features_C0": "midpoint",
    "timing_repeats": 1, "memory_limit_GB": None, "map_turning_point": "last"}

# What each problem changes from pinned_inputs.
check_problems = {
    # A layer 3 partitions thick, loaded so that every interface-normal component (ε11, γ12, γ13) takes part. Its exact
    # solution is known. 4 increments per quarter still yield at both zero crossings. Spectral LS runs on 9 elements per
    # side, the layer 3 of them thick: the only test of Spectral LS coarser than the mesh.
    "laminate": {"geometry": "laminate", **get_cyclic_load(strain(e11=0.015, e22=0.020, g12=0.03, g13=0.01)),
                 **get_entries(partitions=9, spectral_elements=9)},
    # A layer 7 elements thick, so each interface cuts a partition (2/3 fiber), so partitions there hold both phases. In
    # transverse shear γ12 both phases carry the same stress. FFT-TFA represents the exact polarization, but its
    # interaction is exact only within reach of each partition, and the cut partitions line up along the layer, so their
    # far-field error adds up instead of decaying (notes/2_ls_model.md section 5.5).
    "cut laminate": {"geometry": "cut laminate", **get_cyclic_load(strain(g12=0.03))},
    # A circular fiber under tension and shear, with a band of mixed partitions around it, and the reference switched
    # on, at the working resolution, so it must equal Spectral LS bit for bit.
    "fiber": {"reference": dict(elements=27, C0="midpoint")},
    # The laminate turned so its interfaces are normal to z, on a 9³ mesh with 3³ partitions (the layer is the middle
    # partition layer), loaded with every interface-normal component (ε33, γ23, γ13) taking part: the only test of the
    # kernel's ξ_z frequencies. Spectral LS runs on 18³ elements, the layer 6 of them thick: the only test of Spectral
    # LS finer than the mesh.
    "z laminate": {"geometry": "z laminate", "elements_per_side": 9,
                   **get_cyclic_load(strain(e11=0.020, e33=0.015, g23=0.03, g13=0.01)),
                   **get_entries(partitions=3, spectral_elements=18)},
    # A sphere on a 12³ mesh with 4³ partitions, loaded with z components: the only test of mixed partitions in 3D.
    "sphere": {"geometry": "sphere", "elements_per_side": 12,
               **get_monotonic_load(strain(e11=0.015, e33=0.020, g13=0.03)),
               **get_entries(partitions=4, spectral_elements=12)},
    # The sphere with k-means clusters, so a partition's matrix part holds several: the only test of k-means in 3D and
    # of FFT-TFA's neighbour blocks along a load path.
    "k-means sphere": {"geometry": "sphere", "elements_per_side": 12,
                       **get_monotonic_load(strain(e11=0.015, e33=0.020, g13=0.03)),
                       **get_entries(partitions=4, spectral_elements=12, matrix_clusters=2, reach=1)},
    # The shape config.py sets up by default: fibers at random positions, with k-means clusters.
    "random fibers": {"geometry": "random fibers",
                      **get_entries(partitions=9, spectral_elements=27, matrix_clusters=2)},
}

# The methods whose model is not exact on a problem: compared with the exact solution, but not judged by it.
inexact_methods = {"cut laminate": ("FFT-TFA",)}

# The problems on which every method's stopping accuracy is checked.
stopping_check_problems = ("fiber", "sphere")

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
    rows = [np.flatnonzero(materials.material_ids == phase)[0] for phase in (0, 1)]
    return materials._make(field[rows] for field in materials)

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
    interface_normal_components = interface_normal_components_by_shape[cell.get_geometry()["shape"]]
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

## ------- Differences ------- ##

def get_relative_difference(stress, other_stress):
    # The largest difference over every step and component, relative to the largest of the other stress.
    return np.abs(stress - other_stress).max() / np.abs(other_stress).max()

def get_stress_error(result, other_macroscopic_stress, other_cluster_stress):
    # The larger of the macroscopic stress's difference and every cluster's, clusters judged phase by phase on their own
    # phase's stress scale: on the fiber's, about a hundred times larger, a matrix error would hide.
    material_ids = result.clusters.materials.material_ids
    return max(get_relative_difference(result.macroscopic_stress, other_macroscopic_stress),
               *(get_relative_difference(result.cluster_stress[:, phase], other_cluster_stress[:, phase])
                 for phase in (material_ids == 0, material_ids == 1)))

def get_laminate_error(result, exact_phase_stress):
    exact_cluster_stress = exact_phase_stress[:, result.clusters.materials.material_ids]
    return get_stress_error(result, result.clusters.average(np.moveaxis(exact_cluster_stress, 1, 0)),
                            exact_cluster_stress)

## ------- FFT-TFA at Full Reach ------- ##

def get_interaction_error(mesh, clusters):
    # FFT-TFA's interaction at full reach (FT-2 to FT-4), against the element-level LS operator: random polarizations,
    # one per cluster, spread over its elements, strains −Γ0 * τ = −D0 τ by FFT on the element lattice (Moulinec–Suquet
    # kernel, C0 = L_m), averaged over each cluster's elements. The largest relative difference. At full reach FT-3
    # corrects every pair of partitions, so the two must agree to round-off; FT-4 then removes a mean that is zero to
    # round-off.
    discretization, matrix_L = mesh.discretization, cell.get_matrix_L()
    interaction = fft_tfa_model.get_interaction(mesh, clusters, "full")
    polarization = 1e6 * np.random.default_rng(0).standard_normal((clusters.cluster_count, 6))
    kernel = ls_model.get_transformed(ls_model.get_moulinec_suquet_kernel(discretization.element_lattice_shape,
                                                                          matrix_L),
                                      discretization.element_lattice_shape)
    element_strain = ls_model.apply_on_lattice(
        kernel, np.linalg.solve(matrix_L, polarization[clusters.element_cluster].T).T)
    element_average = np.zeros((clusters.cluster_count, 6))
    np.add.at(element_average, clusters.element_cluster, element_strain)
    element_average /= np.bincount(clusters.element_cluster)[:, None]
    return get_relative_difference(fft_tfa_model.get_strain_change(interaction, polarization), element_average)

def solve_load_path(solver):
    load_path = run.get_load_path()
    with contextlib.redirect_stdout(io.StringIO()):
        return run.run_load_path("check", solver, load_path.applied_macro_strain,
                                 np.arange(len(load_path.applied_macro_strain)))

def solve_at_full_reach(mesh, clusters):
    return solve_load_path(fft_tfa_model.FFTTFASolver(fft_tfa_model.get_interaction(mesh, clusters, "full")))

def get_split_clusters(mesh):
    # Each partition's part in each phase, its matrix part cut into two clusters along y by element row, in the order
    # cell.get_clusters expects: by partition, matrix first.
    side = mesh.discretization.elements_per_side
    block = mesh.discretization.elements_per_partition_side
    element_rows = (np.arange(mesh.discretization.element_count) // side) % side
    in_second_half = element_rows % block >= block // 2
    keys = 3 * mesh.element_partition_ids + np.where(mesh.element_material_ids == 1, 2, in_second_half)
    return cell.get_clusters(mesh, np.unique(keys, return_inverse=True)[1])

def solve_on_split_clusters(mesh):
    # FFT-TFA at full reach and TFA along the load path on get_split_clusters.
    clusters = get_split_clusters(mesh)
    E, P, *_ = tfa_model.get_E_and_P(mesh, clusters)
    return {"FFT-TFA at full reach": solve_at_full_reach(mesh, clusters),
            "TFA": solve_load_path(tfa_model.TFASolver(E, P, clusters))}

## ------- Checks ------- ##

def solve_check_problem(problem_inputs, cache_folder):
    # Every method's results by method, with config's inputs replaced in memory and the solvers' printing silenced, and
    # fields at every load step.
    for name, value in {**pinned_inputs, **problem_inputs}.items():
        if not hasattr(config, name):
            raise AttributeError(f"config.py has no input {name}: update check.py's pinned_inputs or check_problems")
        setattr(config, name, value)
    config.cache_folder = cache_folder
    config.check_inputs()
    load_path = run.get_load_path()
    with contextlib.redirect_stdout(io.StringIO()):
        results = run.solve_all(load_path, np.arange(len(load_path.applied_macro_strain)))
    return {label: entry_result.result for label, entry_result in results.items()}

def report(passed, description):
    print(f"[{'PASS' if passed else 'FAIL'}] {description}")
    return passed

def check_laminate(problem_name, results):
    # Every method, and FFT-TFA at full reach, against the laminate's exact solution.
    mesh, clusters = run.get_partition_clusters(config.fft_tfa[0])
    exact_phase_stress = get_laminate_phase_stress(clusters, run.get_load_path().applied_macro_strain)
    tolerance = config.laminate_tolerance
    all_passed = True
    for method, result in results.items():
        error = get_laminate_error(result, exact_phase_stress)
        if method in inexact_methods.get(problem_name, ()):
            print(f"[INFO] {problem_name}: {method} differs from the exact solution by {error:.1e} (not exact by "
                  "design)")
            continue
        all_passed &= report(error <= tolerance, f"{problem_name}: {method} matches the exact solution (largest "
                                                 f"relative difference {error:.1e}, tolerance {tolerance:.0e})")
    error = get_laminate_error(solve_at_full_reach(mesh, clusters), exact_phase_stress)
    all_passed &= report(error <= tolerance, f"{problem_name}: FFT-TFA at full reach matches the exact solution "
                                             f"(largest relative difference {error:.1e}, tolerance {tolerance:.0e})")
    if problem_name == "laminate":
        for method, result in solve_on_split_clusters(mesh).items():
            error = get_laminate_error(result, exact_phase_stress)
            all_passed &= report(error <= tolerance,
                                 f"{problem_name}: {method} on clusters dividing a partition's matrix part matches the "
                                 f"exact solution (largest relative difference {error:.1e}, tolerance {tolerance:.0e})")
    return all_passed

def check_reference(problem_name, spectral_ls_result):
    # The reference, solved at the working resolution, holds Spectral LS's macroscopic stress at every step and its
    # fields at the turning points, to the bit, and asking for it again loads the same arrays from its cache.
    load_path = run.get_load_path()
    with contextlib.redirect_stdout(io.StringIO()):
        reference, reloaded = run.get_reference(load_path), run.get_reference(load_path)
    turning_steps = load_path.turning_steps
    return report(np.array_equal(reference.macroscopic_stress, spectral_ls_result.macroscopic_stress)
                  and np.array_equal(reference.stress, spectral_ls_result.cluster_stress[turning_steps])
                  and np.array_equal(reference.accumulated_plastic_strain,
                                     spectral_ls_result.accumulated_plastic_strain[turning_steps])
                  and all(np.array_equal(solved, loaded) for solved, loaded in zip(reference, reloaded)),
                  f"{problem_name}: the reference at the working resolution equals Spectral LS bit for bit, and loads "
                  "back from its cache unchanged")

def check_stopping_accuracy(problem_name, problem_inputs, results, cache_folder):
    tolerance = config.stopping_check_tolerance
    tight_results = solve_check_problem({**problem_inputs, "convergence_tolerance": tolerance}, cache_folder)
    all_passed = True
    for method, result in results.items():
        tight = tight_results[method]
        error = get_stress_error(result, tight.macroscopic_stress, tight.cluster_stress)
        all_passed &= report(error <= config.stopping_accuracy_tolerance,
                             f"{problem_name}: {method} stops within {error:.1e} of its solution at tolerance "
                             f"{tolerance:.0e} (tolerance {config.stopping_accuracy_tolerance:.0e})")
    return all_passed

def get_result_arrays(problem_name, results):
    # Every array the saved results hold, keyed "<problem>/<method>/<array>".
    return {f"{problem_name}/{method}/{field}": np.asarray(getattr(result, field))
            for method, result in results.items()
            for field in ("macroscopic_stress", "cluster_stress", "newton_iterations_per_step",
                          "krylov_iterations_per_step", "completed_step_count")}

def check_problem(problem_name, problem_inputs, saved_results, cache_folder):
    # Every check of one problem: whether they all passed, and the problem's arrays to save.
    results = solve_check_problem(problem_inputs, cache_folder)
    step_count = len(run.get_load_path().applied_macro_strain)
    all_passed = True
    for method, result in results.items():
        all_passed &= report(result.completed_step_count == step_count,
                             f"{problem_name}: {method} completed {result.completed_step_count} of {step_count} "
                             "load steps")
    if cell.get_geometry()["shape"] in interface_normal_components_by_shape:
        all_passed &= check_laminate(problem_name, results)
    if config.reference is not None:
        all_passed &= check_reference(problem_name, results["Spectral LS"])
    mesh, clusters = run.get_partition_clusters(config.fft_tfa[0])
    interaction_error = get_interaction_error(mesh, clusters)
    all_passed &= report(interaction_error <= config.interaction_check_tolerance,
                         f"{problem_name}: FFT-TFA's interaction at full reach matches the element-level LS operator "
                         f"(largest relative difference {interaction_error:.1e}, tolerance "
                         f"{config.interaction_check_tolerance:.0e})")
    arrays = get_result_arrays(problem_name, results)
    if problem_name in stopping_check_problems:
        all_passed &= check_stopping_accuracy(problem_name, problem_inputs, results, cache_folder)
    if saved_results is not None:
        changed = [name for name in arrays if name not in saved_results
                   or not np.array_equal(arrays[name], saved_results[name])]
        all_passed &= report(not changed, f"{problem_name}: identical to the saved results"
                             + "".join(f"\n       differs: {name}" for name in changed))
    return all_passed, arrays

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode not in ("check", "save"):
        sys.exit(f"unknown option {mode!r}: use no option, or save")
    saved_results = None
    if mode == "check" and saved_results_path.exists():
        with np.load(saved_results_path) as saved:
            saved_results = {name: saved[name] for name in saved.files}

    all_passed = True
    result_arrays = {}
    with tempfile.TemporaryDirectory() as cache_folder:
        for problem_name, problem_inputs in check_problems.items():
            passed, arrays = check_problem(problem_name, problem_inputs, saved_results, pathlib.Path(cache_folder))
            all_passed &= passed
            result_arrays.update(arrays)

    if saved_results is None and mode == "check":
        print("No results saved yet: run python code/check.py save to save them.")
    if mode == "save":
        if all_passed:
            np.savez(saved_results_path, **result_arrays)
            print(f"Results saved to {saved_results_path}")
        else:
            print("Results not saved, since not every check passed.")
    print("All checks passed." if all_passed else "Some checks FAILED.")
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
