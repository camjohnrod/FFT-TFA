# Tolerance checks run before either entry point solves anything: the structure of the reference kernel
# P0, finite-difference checks of the Newton Jacobians, and, when the grid is small enough for dense matrices, P0
# against a direct reference solve and the LS elastic model error. Every check prints PASS or FAIL, and any failure
# stops the run.

import numpy as np
import scipy.linalg

import config
from lattice_fft import get_P0_offset_blocks_from_transformed
from ls_solvers import get_P0_induced_strain, get_ls_jacobian_blocks, get_ls_state
from material import (get_eigenstrain_sensitivity, get_plastic_eigenstrain, get_unloaded_plastic_state,
                      get_yielding_partitions)
from offline import get_homogenized_L, get_influence_functions, get_partition_averaging_operator
from tfa_solvers import get_actual_residual, get_induced_strain

def get_dense_P0(P0_offset_blocks):
    # Expands the offset kernel into the full P0 by P0^BA = p0[a_B - a_A]. get_grid_id puts i fastest, so the
    # C-order reshape indexes the lattice as [j, i] -- the same convention get_P0_transformed relies on.
    # offsets[receiver, source] is the periodic separation along one axis; blocks is then indexed
    # [j_receiver, i_receiver, j_source, i_source] and the transpose interleaves the 6-vector axes to give
    # dense[6 * receiver + component, 6 * source + component].
    offset_lattice = P0_offset_blocks.reshape(config.partition_number_per_side, config.partition_number_per_side, 6, 6)
    lattice_indices = np.arange(config.partition_number_per_side)
    offsets = (lattice_indices[:, None] - lattice_indices[None, :]) % config.partition_number_per_side
    blocks = offset_lattice[offsets[:, None, :, None], offsets[None, :, None, :]]
    return blocks.transpose(0, 1, 4, 2, 3, 5).reshape(6 * config.partition_count, 6 * config.partition_count)

def get_reference_kernel_checks(P0_transformed, reference_L):
    # P0_transformed holds half the frequencies. That covers them all: the block at -ξ is the complex conjugate of
    # the one at ξ, so it passes each check below exactly when the block at ξ does.
    zero_frequency_error = np.max(np.abs(P0_transformed[0, 0])) / np.max(np.abs(P0_transformed))

    D0_transformed = P0_transformed @ np.linalg.inv(reference_L)
    hermitian_difference = D0_transformed - np.conj(np.swapaxes(D0_transformed, -1, -2))
    hermitian_error = np.max(np.abs(hermitian_difference)) / np.max(np.abs(D0_transformed))

    eigenvalues = np.linalg.eigvals(P0_transformed)
    return [("zero-frequency block vanishes", zero_frequency_error, config.verification_tolerance),
            ("P0(xi) C0^-1 is Hermitian", hermitian_error, config.verification_tolerance),
            ("P0(xi) eigenvalues are real", np.max(np.abs(eigenvalues.imag)), config.verification_tolerance),
            ("P0(xi) eigenvalues lie in [0, 1]",
             max(-np.min(eigenvalues.real), np.max(eigenvalues.real) - 1, 0.0), config.verification_tolerance)]

def get_dense_reference_checks(problem, dense_P0):
    print(f"  dense reference influence functions ({6 + 6 * config.partition_count} load columns):")
    reference_L_per_element = np.tile(problem.reference_L, (config.element_count, 1, 1))
    A = get_partition_averaging_operator(problem.B, problem.mesh.element_nodes, problem.mesh.element_partition_ids)
    E_reference, P_reference = get_influence_functions(problem.B, reference_L_per_element, problem.mesh, A)

    identity_error = np.max(np.abs(E_reference.reshape(config.partition_count, 6, 6) - np.eye(6)))
    translation_error = np.max(np.abs(P_reference - dense_P0)) / np.max(np.abs(P_reference))

    partition_field = np.random.default_rng(0).standard_normal((config.partition_count, 6))
    dense_induced_strain = (dense_P0 @ partition_field.reshape(-1)).reshape(config.partition_count, 6)
    fft_induced_strain = get_P0_induced_strain(problem.P0_transformed, partition_field)
    convolution_error = (np.max(np.abs(fft_induced_strain - dense_induced_strain))
                         / np.max(np.abs(dense_induced_strain)))

    return [("reference E blocks equal the identity", identity_error, config.verification_tolerance),
            ("dense P0 equals the translated kernel", translation_error, config.verification_tolerance),
            ("FFT convolution equals the dense product", convolution_error, config.verification_tolerance)]

def get_jacobian_checks(model_name, equation_label, partition_materials, get_base_strain, get_residual,
                        apply_jacobian):
    # Central-difference check of a Newton Jacobian-vector product against the residual it differentiates, with no
    # plastic history. get_base_strain(ε̄) is the strain the check is taken at, get_residual(ε̄, ε) the model residual
    # and apply_jacobian(ε, v) the analytic product. These checks are the only test of the plastic branch of H_μ, so
    # the state has to be partly yielded, and the active yield set must not move across the perturbation or the
    # one-sided branch derivative is not the true one.
    no_plastic_history = get_unloaded_plastic_state()

    def get_yielding(trial_strain):
        plastic_state, _ = get_plastic_eigenstrain(trial_strain, partition_materials, no_plastic_history)
        return get_yielding_partitions(plastic_state, no_plastic_history)

    # Probe at the smallest of these load fractions that actually yields something. A fixed fraction is fragile:
    # at 0.4 of the default path nothing has yielded yet, which silently reduces this to an elastic-only test.
    load_fraction = next((fraction for fraction in (1.0, 2.0, 4.0, 8.0)
                          if get_yielding(get_base_strain(fraction * config.max_macro_strain)).any()), 1.0)
    macro_strain = load_fraction * config.max_macro_strain
    strain = get_base_strain(macro_strain)

    direction = np.random.default_rng(1).standard_normal((config.partition_count, 6))
    direction /= np.linalg.norm(direction)
    step = 1e-6 * np.linalg.norm(strain)

    analytic_product = apply_jacobian(strain, direction)
    finite_difference = (get_residual(macro_strain, strain + step * direction)
                         - get_residual(macro_strain, strain - step * direction)) / (2 * step)

    yielding = get_yielding(strain)
    active_set_moved = not (np.array_equal(get_yielding(strain - step * direction), yielding)
                            and np.array_equal(get_yielding(strain + step * direction), yielding))
    print(f"  {model_name} Jacobian check state: load fraction {load_fraction:g}, {int(yielding.sum())} of "
          f"{config.partition_count} partitions yielding")
    return [(f"{equation_label} Jacobian matches a central difference",
             np.max(np.abs(analytic_product - finite_difference)) / np.max(np.abs(analytic_product)),
             config.jacobian_check_tolerance),
            (f"{model_name} Jacobian check reaches the plastic branch", 0.0 if yielding.any() else 1.0, 0.5),
            (f"{model_name} Jacobian check active yield set is fixed", 1.0 if active_set_moved else 0.0, 0.5)]

def get_ls_jacobian_checks(problem):
    # The LS-20 product J v = v - P0 [(I - C0⁻¹ L_t) v], at the uniform strain ε̄ in every partition.
    partition_materials, P0_transformed = problem.partition_materials, problem.P0_transformed
    reference_compliance = problem.reference_compliance
    no_plastic_history = get_unloaded_plastic_state()

    def get_base_strain(macro_strain):
        return np.tile(macro_strain, (config.partition_count, 1))

    def get_residual(macro_strain, strain):
        _, _, residual = get_ls_state(strain, get_base_strain(macro_strain), P0_transformed,
                                             reference_compliance, partition_materials, no_plastic_history)
        return residual

    def apply_jacobian(strain, direction):
        jacobian_blocks = get_ls_jacobian_blocks(strain, partition_materials, no_plastic_history,
                                                        reference_compliance)
        return direction - get_P0_induced_strain(P0_transformed,
                                                        np.einsum('pij,pj->pi', jacobian_blocks, direction))

    return get_jacobian_checks("LS", "LS-20", partition_materials, get_base_strain, get_residual, apply_jacobian)

def get_tfa_jacobian_checks(problem):
    # The ER-18 product J v = v - P (H_μ v) of the TFA Newton solvers, at the elastic predictor E ε̄.
    partition_materials, E, P = problem.partition_materials, problem.E, problem.P
    no_plastic_history = get_unloaded_plastic_state()

    def get_base_strain(macro_strain):
        return (E @ macro_strain).reshape(config.partition_count, 6)

    def get_residual(macro_strain, strain):
        return get_actual_residual(E, P, macro_strain, strain, partition_materials, no_plastic_history)

    def apply_jacobian(strain, direction):
        sensitivity = get_eigenstrain_sensitivity(strain, partition_materials, no_plastic_history)
        return direction - get_induced_strain(P, np.einsum('pij,pj->pi', sensitivity, direction))

    return get_jacobian_checks("TFA", "ER-18", partition_materials, get_base_strain, get_residual, apply_jacobian)

def get_model_elastic_stiffness(dense_P0, L_per_partition, reference_compliance):
    # Solves the purely elastic reduced model [I + P0 (C0⁻¹ L - I)] e = 1 ⊗ I_6 directly, so the measured model
    # error carries no iterative solver tolerance, then homogenizes exactly as get_homogenized_L does.
    stiffness_ratio = scipy.linalg.block_diag(*(reference_compliance @ L_per_partition))
    identity = np.eye(6 * config.partition_count)
    partition_strain = np.linalg.solve(identity + dense_P0 @ (stiffness_ratio - identity),
                                       np.tile(np.eye(6), (config.partition_count, 1)))
    return np.mean(L_per_partition @ partition_strain.reshape(config.partition_count, 6, 6), axis=0)

def print_elastic_model_error(dense_P0, problem):
    exact_stiffness = get_homogenized_L(problem.E, problem.partition_materials.L)
    model_stiffness = get_model_elastic_stiffness(dense_P0, problem.partition_materials.L,
                                                  problem.reference_compliance)
    stiffness_error = np.max(np.abs(model_stiffness - exact_stiffness)) / np.max(np.abs(exact_stiffness))
    print("  elastic model error (constant-polarization approximation, no solver tolerance involved):")
    print(f"    homogenized stiffness    : {stiffness_error:.3e} relative, worst component")
    print(f"    C_1111 exact vs model MPa: {exact_stiffness[0, 0] / 1e6:.3f} vs {model_stiffness[0, 0] / 1e6:.3f}")

def run_verification(problem, extra_check_functions=()):
    # Each of extra_check_functions(problem) returns further (description, error, tolerance) checks from the entry
    # point, such as the Jacobian checks of its Newton solvers, printed and enforced with the rest. The LS elastic
    # model error is printed after them, from the dense P0, so only when the dense checks run.
    print("verification:")
    checks = get_reference_kernel_checks(problem.P0_transformed, problem.reference_L)
    for get_checks in extra_check_functions:
        checks += get_checks(problem)
    dense_P0 = None
    if config.partition_count <= config.verification_max_partitions:
        dense_P0 = get_dense_P0(get_P0_offset_blocks_from_transformed(problem.P0_transformed))
        checks += get_dense_reference_checks(problem, dense_P0)
    else:
        print(f"  dense checks skipped: {config.partition_count} partitions exceeds verification_max_partitions "
              f"({config.verification_max_partitions}). Lower partition_number_per_side to run them.")

    for description, error, tolerance in checks:
        print(f"  [{'PASS' if error <= tolerance else 'FAIL'}] {description:<45}: {error:.2e} "
              f"(tolerance {tolerance:.0e})")
    failed_checks = [description for description, error, tolerance in checks if not error <= tolerance]
    if failed_checks:
        raise RuntimeError(f"verification failed: {'; '.join(failed_checks)}.")

    if dense_P0 is not None:
        print_elastic_model_error(dense_P0, problem)
