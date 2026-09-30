# The online solvers for the actual E/P model (ER-4): standard and FFT-preconditioned fixed-point iteration.

import numpy as np

import config
from convergence import SolverDidNotConverge, get_relative_residual, has_converged
from lattice_fft import apply_on_partition_lattice
from material import get_eigenstrain_sensitivity, get_plastic_eigenstrain
from online_timing import timed_online

@timed_online("induced_strain")
def get_induced_strain(P, eigenstrain):
    return (P @ eigenstrain.reshape(-1)).reshape(config.partition_count, 6)

def get_partition_runs(partitions):
    # (first, stop) of every run of consecutive partitions in the mask, in partition order.
    edges = np.diff(np.concatenate(([0], partitions.astype(int), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))

@timed_online("induced_strain")
def get_induced_strain_increment(P, eigenstrain_increment, changed_partitions):
    # PΔμ from only the columns of P that belong to changed partitions. P is column-major, so each run of
    # consecutive changed partitions is one contiguous block of columns, used without a copy.
    increment = np.zeros(6 * config.partition_count)
    for first, stop in get_partition_runs(changed_partitions):
        increment += P[:, 6 * first:6 * stop] @ eigenstrain_increment[first:stop].reshape(-1)
    return increment.reshape(config.partition_count, 6)

def get_induced_strain_reusing_history(P, plastic_state, plastic_history, induced_strain_history):
    # Pμ = Pμₙ + PΔμ, where Δμ is zero outside the partitions yielding in this step, so only their columns of P
    # are needed. The mask is taken from Δμ itself, so every skipped column multiplies an exact zero.
    eigenstrain_increment = plastic_state.plastic_strain - plastic_history.plastic_strain
    changed_partitions = np.any(eigenstrain_increment != 0, axis=1)
    if not np.any(changed_partitions):
        return induced_strain_history
    return induced_strain_history + get_induced_strain_increment(P, eigenstrain_increment, changed_partitions)

def reset_elastic_partitions(strain, b, P, induced_strain_history, partition_materials, plastic_history):
    # Updates the material at the given strain, solves the partitions that stay elastic exactly, and returns the
    # residual r = ε - b - Pμ(ε) of ER-4, which is then zero in every elastic partition.
    #
    # An elastic partition's eigenstrain stays at its accepted value, so setting its strain to b + Pμ zeroes its
    # residual without changing μ. Yielding partitions keep their strain, so their μ does not change either. The
    # second material update catches elastic partitions that the reset pushed past yield: only then does μ, and
    # so Pμ, change.
    plastic_state, stress = get_plastic_eigenstrain(strain, partition_materials, plastic_history)
    induced_strain = get_induced_strain_reusing_history(P, plastic_state, plastic_history, induced_strain_history)
    yielding_partitions = plastic_state.accumulated_plastic_strain > plastic_history.accumulated_plastic_strain

    strain = strain.copy()
    strain[~yielding_partitions] = b[~yielding_partitions] + induced_strain[~yielding_partitions]

    plastic_state, stress = get_plastic_eigenstrain(strain, partition_materials, plastic_history)
    yielding_partitions_after_reset = (plastic_state.accumulated_plastic_strain
                                       > plastic_history.accumulated_plastic_strain)
    if not np.array_equal(yielding_partitions_after_reset, yielding_partitions):
        induced_strain = get_induced_strain_reusing_history(P, plastic_state, plastic_history, induced_strain_history)

    residual = strain - b - induced_strain
    return strain, plastic_state, stress, residual

@timed_online("correction_solve")
def get_standard_correction(residual):
    return -residual

def standard_richardson_iteration(E, P, macro_strain, partition_materials, plastic_history):
    b = (E @ macro_strain).reshape(config.partition_count, 6)
    induced_strain_history = get_induced_strain(P, plastic_history.plastic_strain)
    strain = b + induced_strain_history

    residual_history = []
    while True:
        strain, plastic_state, stress, residual = reset_elastic_partitions(
            strain, b, P, induced_strain_history, partition_materials, plastic_history)
        residual_history.append(get_relative_residual(residual, macro_strain))

        if has_converged(residual_history[-1], len(residual_history), "standard_richardson_iteration"):
            break

        strain = strain + config.relaxation_factor * get_standard_correction(residual)

    return stress, plastic_state, np.array(residual_history)

@timed_online("reference_sensitivity")
def get_reference_sensitivity(strain, partition_materials, plastic_history):
    # H_μ,0 (ER-8): the partition average of the actual sensitivity, one 6 × 6 block shared by every partition so
    # the reference stays a convolution.
    sensitivity = get_eigenstrain_sensitivity(strain, partition_materials, plastic_history)
    return np.mean(sensitivity, axis=0)

@timed_online("reference_inverse")
def get_reference_fourier_inverse(P0_transformed, reference_sensitivity):
    # Inverts M̂0(ξ) = I - P̂0(ξ) H_μ,0 (ER-15) at every frequency.
    M0_transformed = np.eye(6) - P0_transformed @ reference_sensitivity
    try:
        return np.linalg.inv(M0_transformed)
    except np.linalg.LinAlgError:
        raise SolverDidNotConverge("the reference operator M0 is singular at some frequency, so the FFT "
                                   "correction is undefined") from None

@timed_online("correction_solve")
def get_fft_correction(reference_fourier_inverse, residual):
    # ER-15 steps 2-4: solve M0 δε = -r one frequency at a time.
    return apply_on_partition_lattice(reference_fourier_inverse, -residual)

def fft_preconditioned_richardson_iteration(E, P, macro_strain, partition_materials, plastic_history,
                                            P0_transformed):
    b = (E @ macro_strain).reshape(config.partition_count, 6)
    induced_strain_history = get_induced_strain(P, plastic_history.plastic_strain)
    strain = b + induced_strain_history

    yielding_partitions_at_last_rebuild = None
    residual_history = []
    while True:
        strain, plastic_state, stress, residual = reset_elastic_partitions(
            strain, b, P, induced_strain_history, partition_materials, plastic_history)
        residual_history.append(get_relative_residual(residual, macro_strain))

        if has_converged(residual_history[-1], len(residual_history), "fft_preconditioned_richardson_iteration"):
            break

        # The reference is rebuilt only when the set of yielding partitions changes, not every iteration.
        yielding_partitions = plastic_state.accumulated_plastic_strain > plastic_history.accumulated_plastic_strain
        if not np.array_equal(yielding_partitions, yielding_partitions_at_last_rebuild):
            reference_sensitivity = get_reference_sensitivity(strain, partition_materials, plastic_history)
            reference_fourier_inverse = get_reference_fourier_inverse(P0_transformed, reference_sensitivity)
            yielding_partitions_at_last_rebuild = yielding_partitions

        strain = strain + config.relaxation_factor * get_fft_correction(reference_fourier_inverse, residual)

    return stress, plastic_state, np.array(residual_history)
