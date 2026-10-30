# Checks the code on seven small problems, each solved by every method (FFT-TFA, Spectral LS, TFA, CP). Run it after
# any change to the code:
#
#   python code/check.py         run the checks (about 30 seconds)
#   python code/check.py save    run the checks (except the comparison with the old reference) and, if they all
#                                pass, save the results as the reference that later runs must reproduce exactly
#
# The problems:
#   laminate      a laminate loaded in tension and shear, then unloaded and loaded in reverse. Its exact solution
#                 (below) is known, and every method must reproduce it in every cluster.
#   cut laminate  a laminate whose interfaces cut through partitions, so partitions there hold both phases, loaded in
#                 transverse shear. Spectral LS and TFA must reproduce the exact solution in every cluster. CP shares a
#                 mixed partition's strain at equal polarization against C0, which is exact here only as C0 → 0
#                 (equal stress); FFT-TFA represents the exact polarization, but its interaction is exact only
#                 within reach of each partition, and the cut partitions line up along the layer, so their far-field
#                 error adds up instead of decaying (notes/2_ls_model.md section 10.5). Both differences are reported
#                 but not judged.
#   fiber         a circular fiber under tension and shear, the kind of problem config.py sets up by default, with a
#                 band of mixed partitions around the fiber.
#   z laminate    a laminate with interfaces normal to z, so the geometry varies along z: meshed and partitioned in 3D,
#                 the only test of the kernel's ξ_z frequencies. Every method must reproduce its exact solution.
#   sphere        a spherical inclusion, meshed and partitioned in 3D, with mixed partitions all around it; FFT-TFA
#                 takes partition k-means clusters.
#   k-means laminate, k-means sphere
#                 the laminate and the sphere with k-means clusters for every method but Spectral LS, FFT-TFA with
#                 neighbour blocks on the sphere. On the laminate every method must still reproduce the exact
#                 solution, except FFT-TFA: its matrix clusters there share partitions, and its correction inside a
#                 partition then gives a uniform polarization a spurious strain (section 10.5), so its difference is
#                 reported but not judged.
# On every laminate, FFT-TFA is also solved with every partition within reach, where its interaction is exact
# (FT-3), so it must reproduce the exact solution: that separates its truncation, a model error, from a wrong law,
# region or Newton solve.
# On all seven, every method must complete every load step and, once a reference is saved, reproduce it bit for bit.
# On every problem, FFT-TFA's interaction with every partition within reach (FT-2, FT-3) must equal the element-level
# LS operator averaged over its regions and clusters, computed here from the elements: a wrong offset, pair count,
# correction block or average fails there. Where CP's partitions hold several pieces (cut laminate, fiber, sphere,
# k-means), CP's piece law must match LS-25 computed independently, piece by piece (notes/2_ls_model.md section 4.5).
# On the fiber and the k-means sphere, every method must stop at the same accuracy: its stresses at
# convergence_tolerance within stopping_accuracy_tolerance of its own solution at stopping_check_tolerance. (Not on
# problems with clusters of a few elements in one partition, such as the sphere's partition k-means: there a cluster
# on the verge of yielding can converge elastic at one tolerance and plastic at the other, for TFA too.)
# On the fiber, the reference (config.include_reference) is solved at the working resolution: it must equal Spectral
# LS bit for bit, and load back from its cache unchanged.
# So a change meant to leave the results alone (a cleanup) must pass, and one meant to change them must not: check its
# results, then save a new reference. The reference is code/check_reference.npz, and holds bit for bit only at
# the config.thread_count it was saved with.
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
import numpy as np

import fe
import fft_tfa_model
import ls_model
import run
from material import get_eigenstrain_sensitivity, get_plastic_eigenstrain, get_unloaded_plastic_state

# Next to the code, not in the cache folder, so clearing the cache to rebuild E and P keeps it.
reference_path = pathlib.Path(__file__).parent / "check_reference.npz"

# Every config.py input that shapes the problems or the solvers, pinned to a fixed value: a 27 × 27 element mesh
# with 9 × 9 partitions, a yielding and hardening matrix, every method switched on (CP too).
pinned_inputs = {
    "domain_side_length": 1e-3, "inclusion_shape": "fiber", "inclusion_radius": 0.35e-3,
    "layer_thickness": (3 / 9) * 1e-3, "element_number_per_side": 27, "element_number_along_z": 1,
    "partition_number_per_side": 9, "partition_number_along_z": 1,
    "elastic_modulus_inclusion": 10e9, "elastic_modulus_matrix": 100e6, "poisson_ratio_inclusion": 0.3,
    "poisson_ratio_matrix": 0.19, "inclusion_yield_stress": np.inf, "matrix_yield_stress": 1.0e6,
    "inclusion_hardening_modulus": 0.0, "matrix_hardening_modulus": 10e6,
    "max_macro_strain": np.array([0.015, 0.020, 0.0, 0.03, 0.0, 0.0]), "strain_increment_count": 12,
    "load_path_shape": "monotonic",
    "include_spectral_ls": True, "include_tfa": True, "include_cp": True, "fft_tfa_clusters": "partition phases",
    "fft_tfa_neighbour_blocks": False, "cp_reference_stiffness": "matrix",
    "cp_reference_stiffness_scale": 0.01, "spectral_ls_reference_stiffness": "midpoint",
    "cp_fourier_preconditioner": True,
    "spectral_ls_fourier_preconditioner": False,
    "convergence_tolerance": 1e-6, "residual_strain_scale_floor": 1e-4, "newton_max_iterations": 50,
    "newton_krylov_tolerance": 1e-3, "newton_krylov_restart": 50, "newton_krylov_max_restarts": None,
    "ls_divergence_limit": 1e3, "tfa_divergence_limit": 1.0, "fe_columns_per_solve": 100,
    "line_search_sufficient_decrease": 1e-4, "line_search_step_reduction": 0.5, "line_search_max_reductions": 10,
    "cp_eigenstrain_tolerance": 1e-12, "cp_eigenstrain_max_iterations": 50,
    "cp_clusters": "partition phases", "tfa_clusters": "partition phases", "k_means_matrix_clusters": 8,
    "k_means_inclusion_clusters": 1, "k_means_matrix_clusters_per_partition": 2,
    "k_means_inclusion_clusters_per_partition": 1, "k_means_iterations": 50, "k_means_seed": 0,
    "include_reference": False, "reference_element_number_per_side": 27, "reference_krylov_restart": 50}

# What each problem changes from pinned_inputs.
check_problems = {
    # A layer 3 partitions thick, loaded so that every interface-normal component (ε11, γ12, γ13) takes part, over a
    # cyclic path: the only test of unloading, reverse yielding and zero imposed strain. 4 increments per quarter
    # still yield at both zero crossings. The homogenized C0 is the only test of that choice.
    "laminate": {"inclusion_shape": "laminate", "max_macro_strain": np.array([0.015, 0.020, 0.0, 0.03, 0.0, 0.01]),
                 "load_path_shape": "cyclic", "strain_increment_count": 4, "cp_reference_stiffness": "homogenized"},
    # A layer 7 elements thick, so each interface cuts a partition (2/3 fiber). In transverse shear γ12 both phases
    # carry the same stress, so the equal-stress split is exact there and every method must match the exact solution.
    "cut laminate": {"inclusion_shape": "laminate", "layer_thickness": (7 / 27) * 1e-3,
                     "max_macro_strain": np.array([0.0, 0.0, 0.0, 0.03, 0.0, 0.0]), "load_path_shape": "cyclic",
                     "strain_increment_count": 4},
    # With the reference switched on, at the working resolution and Spectral LS's restart, so it must equal Spectral
    # LS bit for bit.
    "fiber": {"include_reference": True},
    # The laminate turned so its interfaces are normal to z, on a 9³ mesh with 3³ partitions (the layer is the middle
    # partition layer), loaded like the laminate with every interface-normal component (ε33, γ23, γ13) taking part.
    "z laminate": {"inclusion_shape": "laminate_z", "element_number_per_side": 9, "partition_number_per_side": 3,
                   "max_macro_strain": np.array([0.020, 0.0, 0.015, 0.0, 0.03, 0.01]), "load_path_shape": "cyclic",
                   "strain_increment_count": 4},
    # A sphere on a 12³ mesh with 4³ partitions, loaded with z components: the only test of mixed partitions in 3D,
    # and of FFT-TFA's partition k-means clusters along a load path.
    "sphere": {"inclusion_shape": "sphere", "element_number_per_side": 12, "partition_number_per_side": 4,
               "max_macro_strain": np.array([0.015, 0.0, 0.020, 0.0, 0.0, 0.03]),
               "fft_tfa_clusters": "partition k-means"},
    # The laminate with k-means clusters for FFT-TFA, CP and TFA, up to two per phase (the inclusion's strain
    # concentration is uniform, so it gets one). Each phase's exact state is uniform, so every method must still
    # reproduce it, however k-means divides a phase (FFT-TFA only at full reach); the matrix clusters cut through
    # partitions, so FFT-TFA's matrix regions span partitions.
    "k-means laminate": {"inclusion_shape": "laminate",
                         "max_macro_strain": np.array([0.015, 0.020, 0.0, 0.03, 0.0, 0.01]),
                         "load_path_shape": "cyclic", "strain_increment_count": 4, "fft_tfa_clusters": "k-means",
                         "cp_clusters": "k-means", "tfa_clusters": "k-means", "k_means_matrix_clusters": 2,
                         "k_means_inclusion_clusters": 2},
    # The sphere with k-means clusters for FFT-TFA, CP and TFA, so clusters span partitions, and some clusters
    # sharing a partition also fill others: the only test of LS-25's coupling across partitions, and of FFT-TFA's
    # neighbour blocks along a load path.
    "k-means sphere": {"inclusion_shape": "sphere", "element_number_per_side": 12, "partition_number_per_side": 4,
                       "max_macro_strain": np.array([0.015, 0.0, 0.020, 0.0, 0.0, 0.03]),
                       "fft_tfa_clusters": "k-means", "fft_tfa_neighbour_blocks": True, "cp_clusters": "k-means",
                       "tfa_clusters": "k-means"},
}

# The methods whose model is not exact on a problem: compared with the exact solution, but not judged by it.
inexact_methods = {"cut laminate": ("FFT-TFA", "CP"), "k-means laminate": ("FFT-TFA",)}

# The problems on which every method's stopping accuracy is checked.
stopping_check_problems = ("fiber", "k-means sphere")

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

## ------- Partition Law ------- ##

def get_partition_law_error(clusters, reference_L):
    # CP's piece law (LS-25) against its defining equations, at random partition strains and cluster eigenstrains,
    # computed here piece by piece, with no code of ls_model's: every piece p of cluster g in partition A carries the
    # partition's polarization, σ_p − C0 ε_p = τ_A with σ_p = L_g (ε_p − μ_g), so ε_p = (L_g − C0)⁻¹ (τ_A + L_g μ_g),
    # and the pieces' strains average to the partition strain, which gives τ_A. The solver's sparse matrices must then
    # give each cluster's average strain (E ε + F μ) and each partition's equivalent eigenstrain −C0⁻¹ τ_A
    # (D_e (ε − Λ μ)). The largest relative difference; None when no partition holds several pieces.
    if np.all(np.bincount(clusters.piece_partition) == 1):
        return None
    rng = np.random.default_rng(0)
    strain = 1e-3 * rng.standard_normal((clusters.partition_count, 6))
    eigenstrain = 1e-3 * rng.standard_normal((clusters.cluster_count, 6))

    share = clusters.piece_volume_fraction[:, None]
    piece_L = clusters.materials.L[clusters.piece_cluster]
    piece_compliance = np.linalg.inv(piece_L - reference_L)
    piece_eigenstrain = np.einsum('pij,pj->pi', piece_compliance @ piece_L, eigenstrain[clusters.piece_cluster])
    partition_compliance = np.zeros((clusters.partition_count, 6, 6))
    np.add.at(partition_compliance, clusters.piece_partition, share[:, :, None] * piece_compliance)
    eigenstrain_part = np.zeros_like(strain)
    np.add.at(eigenstrain_part, clusters.piece_partition, share * piece_eigenstrain)
    polarization = np.linalg.solve(partition_compliance, (strain - eigenstrain_part)[..., None])[..., 0]
    piece_strain = (np.einsum('pij,pj->pi', piece_compliance, polarization[clusters.piece_partition])
                    + piece_eigenstrain)
    cluster_strain = np.zeros_like(eigenstrain)
    np.add.at(cluster_strain, clusters.piece_cluster, share * piece_strain)
    cluster_strain /= clusters.volume[:, None]

    law = ls_model.get_piece_law(clusters, reference_L)
    return max(get_relative_difference(ls_model.get_cluster_strain(strain, eigenstrain, law), cluster_strain),
               get_relative_difference(ls_model.get_equivalent_eigenstrain(strain, eigenstrain, law),
                                       -np.linalg.solve(reference_L, polarization.T).T))

## ------- FFT-TFA's Interaction ------- ##

def get_interaction_error(mesh, clusters, matrix_L):
    # FFT-TFA's interaction with every partition within reach of every other (FT-2 to FT-4), against the element-level
    # LS operator: random polarizations, one per region, spread over its elements, strains −Γ0 * τ = −D0 τ by FFT on
    # the element lattice (Moulinec–Suquet kernel, C0 = L_m), averaged over each region's and each cluster's elements.
    # The largest relative difference in the regions' and the clusters' strains. At full reach FT-3 corrects every
    # pair of partitions, so the two must agree to round-off; FT-4 then removes a mean that is zero to round-off.
    discretization = mesh.discretization
    interaction = fft_tfa_model.get_interaction(mesh, clusters, matrix_L,
                                                max(discretization.partition_lattice_shape) // 2)
    region_count = len(interaction.region_cluster)
    polarization = 1e6 * np.random.default_rng(0).standard_normal((region_count, 6))
    kernel = ls_model.get_transformed(ls_model.get_moulinec_suquet_kernel(discretization.element_lattice_shape,
                                                                          matrix_L),
                                      discretization.element_lattice_shape)
    element_region = interaction.piece_region[clusters.element_piece]
    element_strain = ls_model.apply_on_lattice(kernel, np.linalg.solve(matrix_L, polarization[element_region].T).T)

    def get_element_average(element_group, group_count):
        total = np.zeros((group_count, 6))
        np.add.at(total, element_group, element_strain)
        return total / np.bincount(element_group, minlength=group_count)[:, None]

    region_strain = fft_tfa_model.get_strain_change(interaction, polarization)
    return max(get_relative_difference(region_strain, get_element_average(element_region, region_count)),
               get_relative_difference(interaction.region_to_cluster @ region_strain,
                                       get_element_average(clusters.element_cluster, clusters.cluster_count)))

def solve_at_full_reach(problem):
    # FFT-TFA's load path on the problem's clusters with every partition within reach of every other.
    discretization = problem.mesh.discretization
    interaction = fft_tfa_model.get_interaction(problem.mesh, problem.fft_tfa.clusters, problem.matrix_L,
                                                max(discretization.partition_lattice_shape) // 2)
    with contextlib.redirect_stdout(io.StringIO()):
        return run.run_load_path("FFT-TFA", fft_tfa_model.FFTTFASolver(interaction), run.get_applied_macro_strain())

## ------- Stopping Accuracy ------- ##

def get_stopping_error(result, tight_result):
    # The larger of the macroscopic stress's difference from the tight solution and every cluster's, clusters judged
    # phase by phase on their own phase's stress scale, as get_laminate_error does.
    material_ids = result.clusters.materials.material_ids
    errors = [get_relative_difference(result.macroscopic_stress, tight_result.macroscopic_stress)]
    for phase_clusters in (material_ids == 0, material_ids == 1):
        errors.append(get_relative_difference(result.cluster_stress_per_step[:, phase_clusters],
                                              tight_result.cluster_stress_per_step[:, phase_clusters]))
    return max(errors)

## ------- Reference ------- ##

def is_reference_spectral_ls(problem, spectral_ls_result):
    # Whether the reference, solved at the working resolution, holds Spectral LS's macroscopic stress at every step and
    # its fields at the last, to the bit, and whether asking for it again loads the same arrays from its cache.
    reference = problem.reference
    with contextlib.redirect_stdout(io.StringIO()):
        reloaded = run.get_reference(problem.matrix_L, fe.get_inclusion_L())
    return (np.array_equal(reference.macroscopic_stress, spectral_ls_result.macroscopic_stress)
            and np.array_equal(reference.stress, spectral_ls_result.cluster_stress_per_step[-1])
            and np.array_equal(reference.accumulated_plastic_strain,
                               spectral_ls_result.accumulated_plastic_strain_per_step[-1])
            and all(np.array_equal(saved, loaded) for saved, loaded in zip(reference, reloaded)))

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
                exact_phase_stress = get_laminate_phase_stress(problem.fft_tfa.clusters,
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
                error = get_laminate_error(solve_at_full_reach(problem), exact_phase_stress)
                all_passed &= report(error <= config.laminate_tolerance,
                                     f"{problem_name}: FFT-TFA at full reach matches the exact solution (largest "
                                     f"relative difference {error:.1e}, tolerance {config.laminate_tolerance:.0e})")
            if problem.reference is not None:
                all_passed &= report(is_reference_spectral_ls(problem, results["Spectral LS"]),
                                     f"{problem_name}: the reference at the working resolution equals Spectral LS "
                                     "bit for bit, and loads back from its cache unchanged")
            interaction_error = get_interaction_error(problem.mesh, problem.fft_tfa.clusters, problem.matrix_L)
            all_passed &= report(interaction_error <= config.interaction_check_tolerance,
                                 f"{problem_name}: FFT-TFA's interaction at full reach matches the element-level LS "
                                 f"operator (largest relative difference {interaction_error:.1e}, tolerance "
                                 f"{config.interaction_check_tolerance:.0e})")
            if problem_name in stopping_check_problems:
                _, tight_results = solve_check_problem(
                    {**problem_inputs, "convergence_tolerance": config.stopping_check_tolerance},
                    pathlib.Path(cache_folder))
                for method, result in results.items():
                    error = get_stopping_error(result, tight_results[method])
                    all_passed &= report(error <= config.stopping_accuracy_tolerance,
                                         f"{problem_name}: {method} stops within {error:.1e} of its solution at "
                                         f"tolerance {config.stopping_check_tolerance:.0e} (tolerance "
                                         f"{config.stopping_accuracy_tolerance:.0e})")
            law_error = get_partition_law_error(problem.cp_lattice.clusters, problem.cp_lattice.reference_L)
            if law_error is not None:
                all_passed &= report(law_error <= config.piece_law_check_tolerance,
                                     f"{problem_name}: CP's piece law matches LS-25 (largest relative difference "
                                     f"{law_error:.1e}, tolerance "
                                     f"{config.piece_law_check_tolerance:.0e})")
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
