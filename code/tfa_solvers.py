# The online solvers for the actual E/P model (ER-4): standard and FFT-preconditioned fixed-point iteration
# (Formulation.md Strategy 1), and Newton-Krylov with and without the FFT reference preconditioner (Strategy 2).

import functools
import numpy as np
import scipy.sparse.linalg

import config
from convergence import SolverDidNotConverge, StepResult, get_relative_residual, has_converged
from lattice_fft import apply_on_partition_lattice
from load_path import Solver
from material import get_eigenstrain_sensitivity, get_plastic_eigenstrain, get_yielding_partitions
from online_timing import timed_online

## ------- Shared by Every TFA Solver ------- ##

@timed_online("induced_strain")
def get_induced_strain(P, eigenstrain):
    return (P @ eigenstrain.reshape(-1)).reshape(config.partition_count, 6)

def get_partition_runs(partitions):
    # (first, stop) of every run of consecutive partitions in the mask, in partition order.
    edges = np.diff(np.concatenate(([0], partitions.astype(int), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))

@timed_online("induced_strain")
def apply_P_to_partitions(P, partition_field, partitions):
    # P times a partition field that is zero outside the given partitions, from only those partitions' columns of P.
    # P is column-major, so each run of consecutive partitions is one contiguous block of columns, used without a
    # copy.
    product = np.zeros(6 * config.partition_count)
    for first, stop in get_partition_runs(partitions):
        product += P[:, 6 * first:6 * stop] @ partition_field[first:stop].reshape(-1)
    return product.reshape(config.partition_count, 6)

def get_induced_strain_reusing_history(P, plastic_state, plastic_history, induced_strain_history):
    # Pμ = Pμₙ + PΔμ, where Δμ is zero outside the partitions yielding in this step, so only their columns of P
    # are needed. The mask is taken from Δμ itself, so every skipped column multiplies an exact zero.
    eigenstrain_increment = plastic_state.plastic_strain - plastic_history.plastic_strain
    changed_partitions = np.any(eigenstrain_increment != 0, axis=1)
    if not np.any(changed_partitions):
        return induced_strain_history
    return induced_strain_history + apply_P_to_partitions(P, eigenstrain_increment, changed_partitions)

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
    yielding_partitions = get_yielding_partitions(plastic_state, plastic_history)

    strain = strain.copy()
    strain[~yielding_partitions] = b[~yielding_partitions] + induced_strain[~yielding_partitions]

    plastic_state, stress = get_plastic_eigenstrain(strain, partition_materials, plastic_history)
    if not np.array_equal(get_yielding_partitions(plastic_state, plastic_history), yielding_partitions):
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

@timed_online("reference_inverse")
def get_reference_fourier_inverse(P0_transformed, reference_sensitivity):
    # Inverts M̂0(ξ) = I - P̂0(ξ) H_μ,0 (ER-15) at every frequency.
    M0_transformed = np.eye(6) - P0_transformed @ reference_sensitivity
    try:
        return np.linalg.inv(M0_transformed)
    except np.linalg.LinAlgError:
        raise SolverDidNotConverge("the reference operator M0 is singular at some frequency, so the FFT "
                                   "correction is undefined") from None

class FFTReference:
    # The FFT reference that preconditions both strategies: M0 = I - P0 H_μ,0 (ER-15), where H_μ,0 is the partition
    # mean of H_μ, one 6 × 6 block shared by every partition so that M0 stays a convolution. It is rebuilt only when
    # the set of yielding partitions changes, not every iteration, and a solver makes a fresh one for every load step.
    def __init__(self, P0_transformed):
        self.P0_transformed = P0_transformed
        self.yielding_partitions = None
        self.fourier_inverse = None

    def needs_rebuild(self, yielding_partitions):
        return not np.array_equal(yielding_partitions, self.yielding_partitions)

    def rebuild(self, yielding_partitions, sensitivity):
        self.fourier_inverse = get_reference_fourier_inverse(self.P0_transformed, np.mean(sensitivity, axis=0))
        self.yielding_partitions = yielding_partitions

    def solve(self, partition_field):
        # M0⁻¹ applied one frequency at a time (ER-15 steps 2-4).
        return apply_on_partition_lattice(self.fourier_inverse, partition_field)

## ------- Fixed Point (Strategy 1) ------- ##

@timed_online("correction_solve")
def get_fixed_point_correction(reference, residual):
    # δε = -r, or, given the FFT reference, the solution of M0 δε = -r (ER-12).
    return -residual if reference is None else reference.solve(-residual)

def fixed_point_iteration(E, P, macro_strain, partition_materials, plastic_history, P0_transformed=None):
    # Strategy 1: relaxed fixed-point iteration on the actual residual ER-4, preconditioned by the FFT reference
    # when P0_transformed is given.
    reference = None if P0_transformed is None else FFTReference(P0_transformed)
    b = (E @ macro_strain).reshape(config.partition_count, 6)
    induced_strain_history = get_induced_strain(P, plastic_history.plastic_strain)
    strain = b + induced_strain_history

    residual_history = []
    while True:
        strain, plastic_state, stress, residual = reset_elastic_partitions(
            strain, b, P, induced_strain_history, partition_materials, plastic_history)
        residual_history.append(get_relative_residual(residual, macro_strain))

        if has_converged(residual_history):
            break

        if reference is not None:
            yielding_partitions = get_yielding_partitions(plastic_state, plastic_history)
            if reference.needs_rebuild(yielding_partitions):
                reference.rebuild(yielding_partitions,
                                  get_eigenstrain_sensitivity(strain, partition_materials, plastic_history))

        strain = strain + config.tfa_relaxation_factor * get_fixed_point_correction(reference, residual)

    return StepResult(strain, stress, plastic_state, np.array(residual_history))

def get_tfa_fixed_point_solvers(P0_transformed):
    # The two fixed-point solvers for actual E/P, by the names every entry point and output file uses.
    return {"TFA Standard FP": Solver(fixed_point_iteration, get_actual_residual),
            "TFA FFT FP": Solver(functools.partial(fixed_point_iteration, P0_transformed=P0_transformed),
                                 get_actual_residual)}

## ------- Newton-Krylov (Strategy 2) ------- ##

@timed_online("correction_solve")
def get_newton_krylov_correction(P, sensitivity, residual, reference):
    # Solves J δε = -r (ER-18) by GMRES, with the matrix-free product J v = v - P (H_μ v) of ER-21. H_μ is zero
    # outside the yielding partitions, so each product reads only their columns of P. Given the FFT reference,
    # GMRES solves the right-preconditioned system J M0⁻¹ y = -r and returns δε = M0⁻¹ y: the same Newton correction
    # as ER-20, but GMRES then monitors the actual linear residual -r - J δε, as Formulation.md section 12 asks.
    # Timed as one correction solve, apart from its P products, which count as induced strain.
    yielding_partitions = np.any(sensitivity != 0, axis=(1, 2))

    def apply_jacobian(vector):
        return vector - apply_P_to_partitions(P, np.einsum('pij,pj->pi', sensitivity, vector), yielding_partitions)

    def apply_preconditioner(vector):
        return vector if reference is None else reference.solve(vector)

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
    # Strategy 2: Newton on the actual residual ER-4, each correction solved by GMRES, preconditioned by the FFT
    # reference when P0_transformed is given. The residual is evaluated exactly as in fixed_point_iteration, elastic
    # reset included, from the same starting strain, so all TFA solvers solve the same equation from the same start.
    reference = None if P0_transformed is None else FFTReference(P0_transformed)
    b = (E @ macro_strain).reshape(config.partition_count, 6)
    induced_strain_history = get_induced_strain(P, plastic_history.plastic_strain)
    strain = b + induced_strain_history

    residual_history = []
    while True:
        strain, plastic_state, stress, residual = reset_elastic_partitions(
            strain, b, P, induced_strain_history, partition_materials, plastic_history)
        residual_history.append(get_relative_residual(residual, macro_strain))

        if has_converged(residual_history, max_iterations=config.newton_max_steps):
            break

        sensitivity = get_eigenstrain_sensitivity(strain, partition_materials, plastic_history)
        if reference is not None:
            yielding_partitions = get_yielding_partitions(plastic_state, plastic_history)
            if reference.needs_rebuild(yielding_partitions):
                reference.rebuild(yielding_partitions, sensitivity)

        strain = strain + get_newton_krylov_correction(P, sensitivity, residual, reference)

    return StepResult(strain, stress, plastic_state, np.array(residual_history))

def get_tfa_newton_solvers(P0_transformed):
    # The two Newton-Krylov solvers for actual E/P, by the names every entry point and output file uses.
    return {"TFA Newton": Solver(newton_krylov_iteration, get_actual_residual),
            "TFA FFT Newton": Solver(functools.partial(newton_krylov_iteration, P0_transformed=P0_transformed),
                                     get_actual_residual)}
