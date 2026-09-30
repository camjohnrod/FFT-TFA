# The online solvers for the actual E/P model (ER-4): standard and FFT-preconditioned fixed-point iteration
# (Formulation.md Strategy 1), and Newton-Krylov with and without the FFT reference preconditioner (Strategy 2).

import functools
import time
import numpy as np
import scipy.sparse.linalg

import config
from convergence import SolverDidNotConverge, StepResult, get_relative_residual, has_converged
from lattice_fft import apply_on_partition_lattice
from load_path import Solver
from material import get_eigenstrain_sensitivity, get_plastic_eigenstrain
from online_timing import online_timer, timed_online

@timed_online("induced_strain")
def get_induced_strain(P, eigenstrain):
    return (P @ eigenstrain.reshape(-1)).reshape(config.partition_count, 6)

def get_partition_runs(partitions):
    # (first, stop) of every run of consecutive partitions in the mask, in partition order.
    edges = np.diff(np.concatenate(([0], partitions.astype(int), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))

def apply_P_to_partitions(P, partition_field, partitions):
    # P times a partition field that is zero outside the given partitions, from only those partitions' columns of P.
    # P is column-major, so each run of consecutive partitions is one contiguous block of columns, used without a
    # copy.
    product = np.zeros(6 * config.partition_count)
    for first, stop in get_partition_runs(partitions):
        product += P[:, 6 * first:6 * stop] @ partition_field[first:stop].reshape(-1)
    return product.reshape(config.partition_count, 6)

@timed_online("induced_strain")
def get_induced_strain_increment(P, eigenstrain_increment, changed_partitions):
    # PΔμ, where Δμ is zero outside the changed partitions.
    return apply_P_to_partitions(P, eigenstrain_increment, changed_partitions)

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

def get_actual_residual(E, P, macro_strain, strain, partition_materials, plastic_history):
    # ER-4 recomputed from scratch at a converged strain, for the independent check in run_strain_path: a fresh
    # material update and the full product P μ, sharing none of the solvers' bookkeeping (the elastic reset, the
    # incremental P products, the reused history product).
    b = (E @ macro_strain).reshape(config.partition_count, 6)
    plastic_state, _ = get_plastic_eigenstrain(strain, partition_materials, plastic_history)
    return strain - b - get_induced_strain(P, plastic_state.plastic_strain)

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

    return StepResult(strain, stress, plastic_state, np.array(residual_history))

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

    return StepResult(strain, stress, plastic_state, np.array(residual_history))

@timed_online("reference_sensitivity")
def get_actual_sensitivity(strain, partition_materials, plastic_history):
    # H_μ (ER-8) at the current strain, one 6 x 6 block per partition, zero in partitions that stay elastic.
    return get_eigenstrain_sensitivity(strain, partition_materials, plastic_history)

@timed_online("correction_solve")
def get_newton_krylov_correction(P, sensitivity, residual, reference_fourier_inverse):
    # Solves J δε = -r (ER-18) by GMRES, with the matrix-free product J v = v - P (H_μ v) of ER-21. H_μ is zero
    # outside the yielding partitions, so each product reads only their columns of P. Given the reference inverse,
    # GMRES solves the right-preconditioned system J M0⁻¹ y = -r and returns δε = M0⁻¹ y: the same Newton correction
    # as ER-20, but GMRES then monitors the actual linear residual -r - J δε, as Formulation.md section 12 asks.
    # Timed as one correction solve; its P products are recorded as nested operator applications.
    yielding_partitions = np.any(sensitivity != 0, axis=(1, 2))

    def apply_jacobian(vector):
        start_time = time.perf_counter()
        coupled_strain = apply_P_to_partitions(P, np.einsum('pij,pj->pi', sensitivity, vector), yielding_partitions)
        online_timer.record_nested_operator_application(time.perf_counter() - start_time)
        return vector - coupled_strain

    def apply_preconditioner(vector):
        if reference_fourier_inverse is None:
            return vector
        return apply_on_partition_lattice(reference_fourier_inverse, vector)

    def apply_preconditioned_jacobian(flat_vector):
        vector = flat_vector.reshape(config.partition_count, 6)
        return apply_jacobian(apply_preconditioner(vector)).reshape(-1)

    system_size = 6 * config.partition_count
    operator = scipy.sparse.linalg.LinearOperator((system_size, system_size), matvec=apply_preconditioned_jacobian)
    preconditioned_correction, info = scipy.sparse.linalg.gmres(operator, -residual.reshape(-1),
                                                                rtol=config.newton_krylov_tolerance,
                                                                restart=config.newton_krylov_restart)
    if info != 0:
        raise SolverDidNotConverge(f"GMRES did not converge inside the TFA Newton step (info {info}). Loosen "
                                   "newton_krylov_tolerance or raise newton_krylov_restart")
    return apply_preconditioner(preconditioned_correction.reshape(config.partition_count, 6))

def newton_krylov_iteration(E, P, macro_strain, partition_materials, plastic_history, P0_transformed=None):
    # Strategy 2: Newton on the actual residual ER-4, each correction solved by GMRES. The residual is evaluated
    # exactly as in the fixed-point solvers, elastic reset included, from the same starting strain, so all TFA
    # solvers solve the same equation from the same start. With P0_transformed, GMRES is preconditioned by the FFT
    # reference M0 = I - P0 H_μ,0, with H_μ,0 and its rebuild rule exactly as in
    # fft_preconditioned_richardson_iteration.
    solver_name = "tfa_newton" if P0_transformed is None else "tfa_fft_newton"
    b = (E @ macro_strain).reshape(config.partition_count, 6)
    induced_strain_history = get_induced_strain(P, plastic_history.plastic_strain)
    strain = b + induced_strain_history

    reference_fourier_inverse = None
    yielding_partitions_at_last_rebuild = None
    residual_history = []
    while True:
        strain, plastic_state, stress, residual = reset_elastic_partitions(
            strain, b, P, induced_strain_history, partition_materials, plastic_history)
        residual_history.append(get_relative_residual(residual, macro_strain))

        if has_converged(residual_history[-1], len(residual_history), solver_name,
                         max_iterations=config.newton_max_steps):
            break

        sensitivity = get_actual_sensitivity(strain, partition_materials, plastic_history)
        if P0_transformed is not None:
            yielding_partitions = (plastic_state.accumulated_plastic_strain
                                   > plastic_history.accumulated_plastic_strain)
            if not np.array_equal(yielding_partitions, yielding_partitions_at_last_rebuild):
                reference_sensitivity = np.mean(sensitivity, axis=0)
                reference_fourier_inverse = get_reference_fourier_inverse(P0_transformed, reference_sensitivity)
                yielding_partitions_at_last_rebuild = yielding_partitions

        strain = strain + get_newton_krylov_correction(P, sensitivity, residual, reference_fourier_inverse)

    return StepResult(strain, stress, plastic_state, np.array(residual_history))

def get_tfa_newton_solvers(P0_transformed):
    # The two Newton-Krylov solvers for actual E/P, by the names every entry point and output file uses.
    return {"TFA Newton": Solver(newton_krylov_iteration, get_actual_residual),
            "TFA FFT Newton": Solver(functools.partial(newton_krylov_iteration, P0_transformed=P0_transformed),
                                     get_actual_residual)}

def get_tfa_fixed_point_solvers(P0_transformed):
    # The two fixed-point solvers for actual E/P, by the names every entry point and output file uses.
    return {"TFA Standard FP": Solver(standard_richardson_iteration, get_actual_residual),
            "TFA FFT FP": Solver(functools.partial(fft_preconditioned_richardson_iteration,
                                                   P0_transformed=P0_transformed), get_actual_residual)}
