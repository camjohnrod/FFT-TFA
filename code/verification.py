# Tolerance checks run before an LS entry point solves anything: the structure of the reference kernel P0, and,
# when the grid is small enough for dense matrices, P0 against a direct reference solve and the elastic model error.
# Every check prints PASS or FAIL, and any failure stops the run.

import numpy as np
import scipy.linalg

import config
from lattice_fft import get_P0_offset_blocks_from_transformed
from ls_solvers import get_reference_induced_strain
from offline import get_homogenized_L, get_influence_functions, get_partition_averaging_operator

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
    fft_induced_strain = get_reference_induced_strain(problem.P0_transformed, partition_field)
    convolution_error = (np.max(np.abs(fft_induced_strain - dense_induced_strain))
                         / np.max(np.abs(dense_induced_strain)))

    return [("reference E blocks equal the identity", identity_error, config.verification_tolerance),
            ("dense P0 equals the translated kernel", translation_error, config.verification_tolerance),
            ("FFT convolution equals the dense product", convolution_error, config.verification_tolerance)]

def get_model_elastic_stiffness(dense_P0, L_per_partition, reference_compliance):
    # Solves the purely elastic reduced model [I + P0 (C0^-1 L - I)] e = 1 x I_6 directly, so the measured model
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

def run_verification(problem, extra_checks=()):
    # extra_checks: further (description, error, tolerance) checks from the entry point, such as the Jacobian checks
    # of a Newton solver, printed and enforced with the rest.
    print("verification:")
    checks = get_reference_kernel_checks(problem.P0_transformed, problem.reference_L) + list(extra_checks)
    dense_P0 = None
    if config.partition_count <= config.verification_max_partitions:
        dense_P0 = get_dense_P0(get_P0_offset_blocks_from_transformed(problem.P0_transformed))
        checks += get_dense_reference_checks(problem, dense_P0)
    else:
        print(f"  dense checks skipped: {config.partition_count} partitions exceeds verification_max_partitions "
              f"({config.verification_max_partitions}). Lower partition_number_per_side to run them.")

    for description, error, tolerance in checks:
        print(f"  [{'PASS' if error <= tolerance else 'FAIL'}] {description:<43}: {error:.2e} "
              f"(tolerance {tolerance:.0e})")
    failed_checks = [description for description, error, tolerance in checks if not error <= tolerance]
    if failed_checks:
        raise RuntimeError(f"verification failed: {'; '.join(failed_checks)}.")

    if dense_P0 is not None:
        print_elastic_model_error(dense_P0, problem)
