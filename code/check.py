"""Checks the code on small problems, each solved by every method (FFT-TFA, SCA, Spectral LS, TFA). Run it after any
change to the code:

  python code/check.py         run the checks (under a minute)
  python code/check.py save    run the checks (except the comparison with the saved results) and, if they all pass,
                               save the results, which later runs must reproduce exactly

Each check prints PASS or FAIL with the number it judged. The problems are in check_problems below. The checks:
- On every problem, every method must complete every load step and, once results are saved, reproduce them bit for
  bit.
- On every problem, FFT-TFA's interaction at full degree (FT-2, FT-3), whose polynomials span every function on a
  partition, must equal the element-level LS operator averaged over its clusters, computed here from the elements: a
  wrong kernel, moment, pair count, correction block or average fails. So must SCA's D (SC-2), on its clusters over
  the whole cell, which makes SCA with C0 = L_m FFT-TFA on one partition. And at FFT-TFA's own degree, a uniform
  polarization must cause no strain (the patch test).
- On every laminate, every method must reproduce its exact solution in every cluster, except where inexact_methods
  says its model is not exact there, and FFT-TFA at full degree must too: that separates its far-field summary, a model
  error, from a wrong law or Newton solve.
- On the laminate, FFT-TFA at full degree and TFA must also reproduce it with each partition's matrix part cut into two
  clusters, since each phase's exact state is uniform: that tests clusters dividing a partition's phase.
- On every laminate, where SCA is exact whatever its C0, its self-consistent c after each load step must be SC-3
  applied to the exact solution's macroscopic increments. And SC-3 itself must match its definition, a least-squares
  fit on the 3 × 3 tensors, for random increments.
- On every problem, SCA with C0 = L_m and FFT-TFA on one partition, two codes for the same clustered equation, must
  agree along the load path to stopping_accuracy_tolerance.
- On the fiber and the sphere, every method must stop at the same accuracy: its stresses at convergence_tolerance
  within stopping_accuracy_tolerance of its own solution at tight_convergence_tolerance. (Not with k-means clusters
  of a few elements: there a cluster on the verge of yielding can converge elastic at one tolerance and plastic at the
  other, for TFA too.)
- On the fiber, the reference, solved at the working resolution, must equal Spectral LS bit for bit, and load back
  from its cache unchanged.
Every problem runs on Willot's element kernel, and the laminate and the fiber once more on Moulinec–Suquet's.
So a change meant to leave the results alone (a cleanup) must pass, and one meant to change them must not: check its
results, then save them anew. They are in code/check_saved_results.npz, and hold bit for bit only at the
config.thread_count they were saved with.

The problems are set below, not read from config.py, so that the checks mean the same thing whatever config.py holds;
only the check settings (section 6 of config.py) come from config.py. config.py, the cache and code/output are not
touched.
"""

import contextlib
import io
import itertools
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
import sca_model
import tfa_model
from material import get_eigenstrain_sensitivity, radial_return, get_unloaded_plastic_state

# Next to the code, not in the cache folder, so clearing the cache keeps it.
saved_results_path = pathlib.Path(__file__).parent / "check_saved_results.npz"

# The stiffly contrasted materials of every problem.
check_materials = dict(matrix=dict(E=100e6, poisson=0.19, yield_stress=1.0e6, hardening=10e6),
                       inclusion=dict(E=10e9, poisson=0.3, yield_stress=np.inf, hardening=0))

def get_check_entries(partitions, spectral_elements, matrix_clusters=1, degree=1):
    """One entry of each method."""
    return {"fft_tfa": [dict(partitions=partitions, matrix_clusters=matrix_clusters, inclusion_clusters=1,
                             degree=degree)],
            "sca": [dict(matrix_clusters=matrix_clusters, inclusion_clusters=1, C0="self-consistent")],
            "spectral_ls": [dict(elements=spectral_elements, C0="midpoint")],
            "tfa": [dict(partitions=partitions, matrix_clusters=matrix_clusters, inclusion_clusters=1)]}

def get_cyclic_load(max_strain):
    """Out to the given strain, to its opposite and back to zero: unloading, reverse yielding and zero imposed strain.
    """
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
    "random spheres": dict(shape="random spheres", count=4, volume_fraction=0.2, gap=1 / 12, seed=1),
}

# Every config.py input that shapes the problems or the solvers, pinned to a fixed value: a 27 × 27 element mesh, a
# yielding and hardening matrix.
pinned_inputs = {
    "cell_side": 1e-3, "all_geometries": check_geometries, "geometry": "fiber", "clustered_mesh": 27,
    "element_kernel": "Willot",
    "all_materials": {"check": check_materials}, "materials": "check", "load_path": "check",
    **get_check_entries(partitions=9, spectral_elements=27),
    **get_monotonic_load(strain(e11=0.015, e22=0.020, g12=0.03)),
    "convergence_tolerance": 1e-6, "residual_strain_scale_floor": 1e-4, "newton_max_iterations": 50,
    "krylov_tolerance": 1e-3, "gmres_restart": 50, "gmres_max_restarts": None, "cg_max_iterations": None,
    "divergence_limit": 1e3, "tfa_divergence_limit": 1.0, "tfa_columns_per_solve": 100,
    "sca_reference_tolerance": 1e-3, "sca_max_reference_updates": 10,
    "tfa_factorization_ordering": "MMD_AT_PLUS_A",
    "line_search_sufficient_decrease": 1e-4, "line_search_step_reduction": 0.5, "line_search_max_reductions": 10,
    "k_means_iterations": 50, "k_means_seed": 0, "k_means_start": "++", "placement_attempts": 100000,
    "push_margin": 1e-3, "push_passes": 10000, "reference": None, "k_means_features_C0": "midpoint",
    "memory_limit_GB": None, "map_turning_point": "last"}

# What each problem changes from pinned_inputs.
check_problems = {
    # A layer 3 partitions thick, loaded so that every interface-normal component (ε11, γ12, γ13) takes part. Its exact
    # solution is known. 4 increments per quarter still yield at both zero crossings. Spectral LS runs on 9 elements per
    # side, the layer 3 of them thick: the only test of Spectral LS coarser than the mesh.
    "laminate": {"geometry": "laminate", **get_cyclic_load(strain(e11=0.015, e22=0.020, g12=0.03, g13=0.01)),
                 **get_check_entries(partitions=9, spectral_elements=9)},
    # A layer 7 elements thick, so each interface cuts a partition (2/3 inclusion): partitions there hold both phases.
    # In transverse shear γ12 both phases carry the same stress. FFT-TFA represents the exact polarization, but at degree 1
    # its far field cannot follow the polarization's step across a cut partition (degree 2 can, on 3 elements), and the
    # cut partitions line up along the layer, so their far-field error adds up instead of decaying.
    "cut laminate": {"geometry": "cut laminate", **get_cyclic_load(strain(g12=0.03))},
    # A circular fiber under tension and shear, with a band of mixed partitions around it, and the reference switched
    # on, at the working resolution, so it must equal Spectral LS bit for bit.
    "fiber": {"reference": dict(elements=27, C0="midpoint")},
    # The laminate turned so its interfaces are normal to z, on a 9³ mesh with 3³ partitions (the layer is the middle
    # partition layer), loaded with every interface-normal component (ε33, γ23, γ13) taking part: the only test of the
    # kernel's ξ_z frequencies. Spectral LS runs on 18³ elements, the layer 6 of them thick: the only test of Spectral
    # LS finer than the mesh.
    "z laminate": {"geometry": "z laminate", "clustered_mesh": 9,
                   **get_cyclic_load(strain(e11=0.020, e33=0.015, g23=0.03, g13=0.01)),
                   **get_check_entries(partitions=3, spectral_elements=18)},
    # A sphere on a 12³ mesh with 4³ partitions, loaded with z components: the only test of mixed partitions in 3D.
    "sphere": {"geometry": "sphere", "clustered_mesh": 12,
               **get_monotonic_load(strain(e11=0.015, e33=0.020, g13=0.03)),
               **get_check_entries(partitions=4, spectral_elements=12)},
    # The sphere with k-means clusters, so a partition's matrix part holds several: the only test of k-means in 3D and
    # of FFT-TFA's degree 2 along a load path.
    "k-means sphere": {"geometry": "sphere", "clustered_mesh": 12,
                       **get_monotonic_load(strain(e11=0.015, e33=0.020, g13=0.03)),
                       **get_check_entries(partitions=4, spectral_elements=12, matrix_clusters=2, degree=2)},
    # The shape config.py sets up by default: fibers at random positions, with k-means clusters.
    "random fibers": {"geometry": "random fibers",
                      **get_check_entries(partitions=9, spectral_elements=27, matrix_clusters=2)},
    # Spheres at random positions on a 12³ mesh, with k-means clusters: the only test of random placement in 3D.
    "random spheres": {"geometry": "random spheres", "clustered_mesh": 12,
                       **get_monotonic_load(strain(e11=0.015, e33=0.020, g13=0.03)),
                       **get_check_entries(partitions=4, spectral_elements=12, matrix_clusters=2)},
}

# Two problems once more on Moulinec–Suquet's kernel, so both kernels stay tested: a laminate's exact solution, and
# the reference against Spectral LS.
for name in ("laminate", "fiber"):
    check_problems[f"{name}, Moulinec-Suquet"] = {**check_problems[name], "element_kernel": "Moulinec-Suquet"}

# The methods whose model is not exact on a problem: compared with the exact solution, but not judged by it.
inexact_methods = {"cut laminate": ("FFT-TFA",)}

# The problems on which every method's stopping accuracy is checked.
stopping_check_problems = ("fiber", "sphere")

## ------- Exact Laminate Solution ------- ##

# An inclusion layer between matrix layers. Under a uniform macro strain each phase has one uniform state, even in
# plasticity, so the whole cell reduces to two states, one per phase:
#   - compatibility keeps the strains along the interface equal to the macro strain in both phases;
#   - equilibrium makes the tractions on the interface equal in both phases;
#   - the phase average of the strain is the macro strain.
# That leaves the matrix phase's three interface-normal strains as the only unknowns of a load step, found by Newton.

# By shape, the Voigt positions of the strain components that may jump across the interface, while the stress
# components at the same positions, the tractions, may not: ε11, γ12, γ13 for a normal along x; ε33, γ23, γ13 along z.
interface_normal_components_by_shape = {"laminate": [0, 3, 5], "laminate_z": [2, 4, 5]}

def get_phase_materials(materials):
    """One row per phase, matrix first, copied from the first cluster of each phase."""
    rows = [np.flatnonzero(materials.phase == phase_id)[0] for phase_id in (0, 1)]
    return materials._make(field[rows] for field in materials)

def get_phase_strains(macro_strain, matrix_normal_strain, inclusion_fraction, interface_normal_components):
    """The strain of each phase, matrix first, given the matrix's interface-normal components."""
    phase_strains = np.tile(macro_strain, (2, 1))
    phase_strains[0, interface_normal_components] = matrix_normal_strain
    phase_strains[1, interface_normal_components] = ((macro_strain[interface_normal_components]
                                                      - (1 - inclusion_fraction) * matrix_normal_strain)
                                                     / inclusion_fraction)
    return phase_strains

def solve_laminate_step(macro_strain, matrix_normal_strain, phase_materials, phase_history, inclusion_fraction,
                        interface_normal_components):
    """Newton on the interface tractions' jump, starting from the given matrix_normal_strain. The inclusion's
    interface-normal strain moves by -(1 - f)/f times the matrix's, which gives the Jacobian below. Returns the
    converged matrix_normal_strain and the phases' plastic state and stress."""
    for _ in range(config.laminate_solver_max_iterations):
        phase_strains = get_phase_strains(macro_strain, matrix_normal_strain, inclusion_fraction,
                                          interface_normal_components)
        phase_state, phase_stress = radial_return(phase_strains, phase_materials, phase_history)
        traction_jump = (phase_stress[0] - phase_stress[1])[interface_normal_components]
        if np.abs(traction_jump).max() <= config.laminate_solver_tolerance * np.abs(phase_stress).max():
            return matrix_normal_strain, phase_state, phase_stress
        sensitivity = get_eigenstrain_sensitivity(phase_strains, phase_materials, phase_history)
        algorithmic_stiffness = phase_materials.L @ (np.eye(6) - sensitivity)
        normal_block = np.ix_(interface_normal_components, interface_normal_components)
        jacobian = (algorithmic_stiffness[0][normal_block]
                    + (1 - inclusion_fraction) / inclusion_fraction * algorithmic_stiffness[1][normal_block])
        matrix_normal_strain = matrix_normal_strain - np.linalg.solve(jacobian, traction_jump)
    raise RuntimeError(f"the laminate's exact solution did not converge at macro strain {macro_strain}.")

def get_laminate_phase_stress(clusters, applied_macro_strain):
    """The exact stress of each phase at every load step, shape (load steps, 2, 6), matrix first."""
    interface_normal_components = interface_normal_components_by_shape[cell.get_geometry()["shape"]]
    phase_materials = get_phase_materials(clusters.materials)
    inclusion_fraction = clusters.average(clusters.materials.phase)
    phase_history = get_unloaded_plastic_state(2)
    matrix_normal_strain = np.zeros(len(interface_normal_components))
    previous_macro_strain = np.zeros(6)
    phase_stress = np.zeros((len(applied_macro_strain), 2, 6))
    for step, macro_strain in enumerate(applied_macro_strain):
        # Start from the previous step's matrix strain, moved by the macro strain increment.
        matrix_normal_strain, phase_history, phase_stress[step] = solve_laminate_step(
            macro_strain, matrix_normal_strain + (macro_strain - previous_macro_strain)[interface_normal_components],
            phase_materials, phase_history, inclusion_fraction, interface_normal_components)
        previous_macro_strain = macro_strain
    return phase_stress

## ------- Differences ------- ##

def get_relative_difference(stress, other_stress):
    """The largest difference over every step and component, relative to the largest of the other stress."""
    return np.abs(stress - other_stress).max() / np.abs(other_stress).max()

def get_stress_error(result, other_macro_stress, other_cluster_stress):
    """The larger of the macroscopic stress's difference and every cluster's, clusters judged phase by phase on their
    own phase's stress scale: on the fiber's, about a hundred times larger, a matrix error would hide."""
    phase = result.clusters.materials.phase
    return max(get_relative_difference(result.macro_stress, other_macro_stress),
               *(get_relative_difference(result.cluster_stress[:, in_phase], other_cluster_stress[:, in_phase])
                 for in_phase in (phase == 0, phase == 1)))

def get_laminate_error(result, exact_phase_stress):
    exact_cluster_stress = exact_phase_stress[:, result.clusters.materials.phase]
    return get_stress_error(result, result.clusters.average(np.moveaxis(exact_cluster_stress, 1, 0)),
                            exact_cluster_stress)

## ------- Interactions, and the Patch Test ------- ##

def get_interaction_error(clusters, get_strain_change):
    """A method's interaction, get_strain_change(polarization) with C0 = L_m, against the element-level LS operator:
    random polarizations, one per cluster, spread over its elements, strains Γ0 * τ by FFT on the element lattice (the
    element kernel), averaged over each cluster's elements. The largest relative difference. FFT-TFA at full degree
    represents every partition's polarization exactly (FT-4 then removes a mean that is zero to round-off), and SCA's D
    is this operator averaged, so both must agree with it to round-off."""
    shape, matrix_L = clusters.mesh.element_shape, cell.get_matrix_L()
    polarization = 1e6 * np.random.default_rng(0).standard_normal((clusters.cluster_count, 6))
    kernel = ls_model.get_P0_transformed(shape, matrix_L)
    element_strain = ls_model.apply_on_lattice(
        kernel, np.linalg.solve(matrix_L, polarization[clusters.element_cluster].T).T)
    element_average = np.zeros((clusters.cluster_count, 6))
    np.add.at(element_average, clusters.element_cluster, element_strain)
    element_average /= np.bincount(clusters.element_cluster)[:, None]
    return get_relative_difference(get_strain_change(polarization), element_average)

def solve_load_path(solver):
    load_path = run.get_load_path()
    with contextlib.redirect_stdout(io.StringIO()):
        return run.run_load_path("check", solver, load_path.applied_macro_strain,
                                 np.arange(len(load_path.applied_macro_strain)))

def get_patch_test_error(clusters, degree):
    """The strain FFT-TFA's interaction gives a uniform polarization, which must cause none: its largest component,
    relative to the strain L_m⁻¹ τ the polarization stands for."""
    interaction = fft_tfa_model.get_interaction(clusters, degree)
    unit_strain = np.array(strain(e11=1.0, e22=0.5, g12=0.3))
    polarization = np.tile(cell.get_matrix_L() @ unit_strain, (clusters.cluster_count, 1))
    return np.abs(fft_tfa_model.get_strain_change(interaction, polarization)).max() / np.abs(unit_strain).max()

# Partition shapes [z, y, x] whose polynomial basis is checked: wider than the problems' partitions, since a basis built
# from monomials loses its high degrees to round-off only from about 8 elements across.
basis_check_shapes = [(1, 3, 3), (1, 8, 8), (1, 16, 16), (4, 4, 4), (8, 8, 8)]

def get_basis_error(block_shape, degree):
    """How far a partition's polynomial basis is from what FFT-TFA needs: orthonormal, the constant first, and as many
    polynomials as the products of axis degrees adding up to `degree` (at full degree, one per element). Returns the
    largest error of the first two, or infinity if the count is wrong."""
    basis = fft_tfa_model.get_polynomial_basis(np.array(block_shape), degree)
    basis = basis.reshape(-1, basis.shape[-1])
    axis_degrees = [range(min(degree, size - 1) + 1) for size in block_shape if size > 1]
    expected_count = sum(1 for e in itertools.product(*axis_degrees) if sum(e) <= degree)
    if basis.shape[1] != expected_count:
        return np.inf
    constant = np.full(len(basis), 1 / np.sqrt(len(basis)))
    return max(np.abs(basis.T @ basis - np.eye(basis.shape[1])).max(), np.abs(basis[:, 0] - constant).max())

def check_polynomial_basis():
    all_passed = True
    for block_shape in basis_check_shapes:
        full = int(np.sum(np.array(block_shape) - 1))
        for degree in (0, 2, full):
            error = get_basis_error(block_shape, degree)
            all_passed &= report_error(f"polynomial basis on {block_shape} elements at degree {degree}: "
                                       "orthonormal, constant first, complete", error,
                                       config.interaction_check_tolerance)
    return all_passed

def solve_at_full_degree(clusters):
    return solve_load_path(fft_tfa_model.FFTTFASolver(clusters, "full"))

def get_split_clusters(mesh):
    """Each partition's part in each phase, its matrix part cut into two clusters along y by element row, in the order
    cell.get_clusters expects: by partition, matrix first."""
    side, block = mesh.element_shape[2], mesh.partition_size[2]
    element_rows = (np.arange(mesh.element_count) // side) % side
    in_second_half = element_rows % block >= block // 2
    keys = 3 * mesh.element_partition + np.where(mesh.element_phase == 1, 2, in_second_half)
    return cell.get_clusters(mesh, np.unique(keys, return_inverse=True)[1])

def solve_on_split_clusters(mesh):
    """FFT-TFA at full degree and TFA along the load path on get_split_clusters."""
    clusters = get_split_clusters(mesh)
    return {"FFT-TFA at full degree": solve_at_full_degree(clusters),
            "TFA": solve_load_path(tfa_model.TFASolver(clusters))}

## ------- SCA: its Reference Stiffness, and FFT-TFA on One Partition ------- ##

def get_sca_clusters():
    """The clusters of the problem's SCA entry, over the whole cell."""
    return run.get_entry_clusters(next(entry for entry in run.get_entries() if entry.method == "SCA"))

def to_tensor(voigt, shear_factor):
    """A symmetric 3 × 3 tensor from Voigt components (11, 22, 33, 12, 23, 13), the shears times shear_factor: ½ for an
    engineering strain (ER-2), 1 for a stress."""
    (xx, yy, zz), (xy, yz, xz) = voigt[:3], shear_factor * voigt[3:]
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])

def get_reference_fit_error():
    """SC-3 against its definition, for random macroscopic increments: fit_reference_scale's c against the c of
    Δσ ≈ c Δε solved by least squares on the 3 × 3 tensors themselves, where every component counts once and no Voigt
    weight enters. The largest relative difference."""
    rng = np.random.default_rng(0)
    errors = []
    for _ in range(10):
        strain_increment, stress_increment = rng.standard_normal(6), rng.standard_normal(6)
        strain_tensor, stress_tensor = to_tensor(strain_increment, 0.5), to_tensor(stress_increment, 1.0)
        (scale,), *_ = np.linalg.lstsq(strain_tensor.reshape(-1, 1), stress_tensor.reshape(-1), rcond=None)
        errors.append(abs(sca_model.fit_reference_scale(strain_increment, stress_increment) - scale) / abs(scale))
    return max(errors)

def get_self_consistent_error(exact_phase_stress):
    """SCA's self-consistent loop on a laminate, where SCA is exact whatever its C0: after each load step, c must be
    SC-3 applied to the exact solution's macroscopic increments, since the loop stops where the fit to the step's own
    increments returns c. A wrong step start or average shows here. The largest relative difference."""
    clusters = get_sca_clusters()
    solver = sca_model.SCASolver(clusters, "self-consistent")
    exact_macro_stress = clusters.average(np.moveaxis(exact_phase_stress[:, clusters.materials.phase], 1, 0))
    plastic_history = get_unloaded_plastic_state(clusters.cluster_count)
    previous_macro_strain, previous_macro_stress, errors = np.zeros(6), np.zeros(6), []
    for macro_strain, macro_stress in zip(run.get_load_path().applied_macro_strain, exact_macro_stress):
        plastic_history = solver.solve_step(macro_strain, plastic_history).plastic_state
        exact_scale = sca_model.fit_reference_scale(macro_strain - previous_macro_strain,
                                                    macro_stress - previous_macro_stress)
        errors.append(abs(solver.reference_scale - exact_scale) / exact_scale)
        previous_macro_strain, previous_macro_stress = macro_strain, macro_stress
    return max(errors)

def get_one_partition_error():
    """SCA with C0 = L_m against FFT-TFA on one partition, both on SCA's clusters along the load path: the same
    clustered equation by two codes, SCA's dense D in increments (SC-1) and FFT-TFA's corrections in total form
    (FT-4), so they agree to the stopping accuracy. The larger of the macroscopic and the cluster stress differences."""
    clusters = get_sca_clusters()
    sca = solve_load_path(sca_model.SCASolver(clusters, "matrix"))
    fft_tfa = solve_load_path(fft_tfa_model.FFTTFASolver(clusters, 0))
    return get_stress_error(sca, fft_tfa.macro_stress, fft_tfa.cluster_stress)

## ------- Checks ------- ##

def solve_check_problem(problem_inputs, cache_folder):
    """Every method's results by method, with config's inputs replaced in memory and the solvers' printing silenced, and
    fields at every load step."""
    for name, value in {**pinned_inputs, **problem_inputs}.items():
        if not hasattr(config, name):
            raise AttributeError(f"config.py has no input {name}: update check.py's pinned_inputs or check_problems")
        setattr(config, name, value)
    config.cache_folder = cache_folder
    config.check_inputs()
    load_path = run.get_load_path()
    with contextlib.redirect_stdout(io.StringIO()):
        results, _ = run.solve_all(load_path, np.arange(len(load_path.applied_macro_strain)))
    return results

def report(passed, description):
    print(f"[{'PASS' if passed else 'FAIL'}] {description}")
    return passed

def report_error(description, error, tolerance):
    """PASS when the error is within the tolerance."""
    return report(error <= tolerance, f"{description} (largest relative difference {error:.1e}, tolerance "
                                      f"{tolerance:.0e})")

def check_laminate(problem_name, results):
    """Every method, and FFT-TFA at full degree, against the laminate's exact solution."""
    clusters = run.get_partition_clusters(config.fft_tfa[0])
    exact_phase_stress = get_laminate_phase_stress(clusters, run.get_load_path().applied_macro_strain)
    tolerance = config.laminate_tolerance
    all_passed = True
    for method, result in results.items():
        error = get_laminate_error(result, exact_phase_stress)
        if method in inexact_methods.get(problem_name, ()):
            print(f"[INFO] {problem_name}: {method} differs from the exact solution by {error:.1e} (not exact by "
                  "design)")
            continue
        all_passed &= report_error(f"{problem_name}: {method} matches the exact solution", error, tolerance)
    error = get_laminate_error(solve_at_full_degree(clusters), exact_phase_stress)
    all_passed &= report_error(f"{problem_name}: FFT-TFA at full degree matches the exact solution", error, tolerance)
    error = get_self_consistent_error(exact_phase_stress)
    all_passed &= report_error(f"{problem_name}: SCA's self-consistent c fits the exact solution's increments", error,
                               tolerance)
    if problem_name == "laminate":
        for method, result in solve_on_split_clusters(clusters.mesh).items():
            error = get_laminate_error(result, exact_phase_stress)
            all_passed &= report_error(f"{problem_name}: {method} on clusters dividing a partition's matrix part "
                                       "matches the exact solution", error, tolerance)
    return all_passed

def check_reference(problem_name, spectral_ls_result):
    """The reference, solved at the working resolution, holds Spectral LS's macroscopic stress at every step and its
    fields at the turning points, to the bit, and asking for it again loads the same arrays from its cache."""
    load_path = run.get_load_path()
    with contextlib.redirect_stdout(io.StringIO()):
        (reference, _), (reloaded, _) = run.get_reference(load_path), run.get_reference(load_path)
    turning_steps = load_path.turning_steps
    fields = ("macro_stress", "cluster_stress", "accumulated_plastic_strain")
    return report(np.array_equal(reference.macro_stress, spectral_ls_result.macro_stress)
                  and np.array_equal(reference.cluster_stress, spectral_ls_result.cluster_stress[turning_steps])
                  and np.array_equal(reference.accumulated_plastic_strain,
                                     spectral_ls_result.accumulated_plastic_strain[turning_steps])
                  and all(np.array_equal(getattr(reference, field), getattr(reloaded, field)) for field in fields),
                  f"{problem_name}: the reference at the working resolution equals Spectral LS bit for bit, and loads "
                  "back from its cache unchanged")

def check_stopping_accuracy(problem_name, problem_inputs, results, cache_folder):
    tolerance = config.tight_convergence_tolerance
    tight_results = solve_check_problem({**problem_inputs, "convergence_tolerance": tolerance}, cache_folder)
    all_passed = True
    for method, result in results.items():
        tight = tight_results[method]
        error = get_stress_error(result, tight.macro_stress, tight.cluster_stress)
        all_passed &= report_error(f"{problem_name}: {method} at convergence_tolerance against its solution at "
                                   f"{tolerance:.0e}", error, config.stopping_accuracy_tolerance)
    return all_passed

def get_result_arrays(problem_name, results):
    """Every array the saved results hold, keyed "<problem>/<method>/<array>"."""
    return {f"{problem_name}/{method}/{field}": np.asarray(getattr(result, field))
            for method, result in results.items()
            for field in ("macro_stress", "cluster_stress", "newton_iterations_per_step",
                          "krylov_iterations_per_step", "completed_step_count")}

def check_problem(problem_name, problem_inputs, saved_results, cache_folder):
    """Every check of one problem: whether they all passed, and the problem's arrays to save."""
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
    clusters = run.get_partition_clusters(config.fft_tfa[0])
    interaction = fft_tfa_model.get_interaction(clusters, "full")
    interaction_error = get_interaction_error(
        clusters, lambda polarization: fft_tfa_model.get_strain_change(interaction, polarization))
    all_passed &= report_error(f"{problem_name}: FFT-TFA's interaction at full degree matches the element-level LS "
                               "operator", interaction_error, config.interaction_check_tolerance)
    sca_clusters = get_sca_clusters()
    D = sca_model.get_interaction(sca_clusters, cell.get_matrix_L())
    interaction_error = get_interaction_error(
        sca_clusters, lambda polarization: sca_model.get_strain_change(D, polarization))
    all_passed &= report_error(f"{problem_name}: SCA's interaction with C0 = L_m matches the element-level LS operator",
                               interaction_error, config.interaction_check_tolerance)
    all_passed &= report_error(f"{problem_name}: SCA with C0 = L_m matches FFT-TFA on one partition along the load "
                               "path", get_one_partition_error(), config.stopping_accuracy_tolerance)
    degree = config.fft_tfa[0]["degree"]
    patch_error = get_patch_test_error(clusters, degree)
    all_passed &= report_error(f"{problem_name}: FFT-TFA at degree {degree} gives a uniform polarization no strain",
                               patch_error, config.interaction_check_tolerance)
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

    all_passed = check_polynomial_basis()
    all_passed &= report_error("SC-3: SCA's fit of c matches least squares on the tensors", get_reference_fit_error(),
                               config.interaction_check_tolerance)
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
