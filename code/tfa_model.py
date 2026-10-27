# TFA, the baseline: the actual E/P model of notes/3_tfa_baseline.md, whose residual is r = ε - E ε̄ - P μ(ε) (ER-4),
# with E and P the exact FE influence functions of the composite (fe.py). Its regions, called partitions below as in
# the notes, are the clusters config.tfa_clusters names (fe.Clusters), each with one uniform eigenstrain and one
# material update at its average strain; the strain itself varies inside a region. Each load step is
# solved by Newton with GMRES (Strategy 2), on the matrix-free Jacobian J v = v - P (H_μ v) of ER-21.

import functools
import numpy as np

import config
from material import get_eigenstrain_sensitivity, get_plastic_eigenstrain, get_yielding_partitions
from newton import Evaluation, Solver, solve_load_step, solve_newton_correction

def get_induced_strain(P, eigenstrain):
    return (P @ eigenstrain.reshape(-1)).reshape(-1, 6)

def get_partition_runs(partitions):
    # (first, stop) of every run of consecutive partitions in the mask, in partition order.
    edges = np.diff(np.concatenate(([0], partitions.astype(int), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))

def apply_P_to_partitions(P, partition_field, partitions):
    # P times a partition field that is zero outside the given partitions, from only those partitions' columns of P.
    # P is column-major, so each run of consecutive partitions is one contiguous block of columns, used without a
    # copy. Fancy-indexing P[:, columns] instead would copy them, which costs more than the full product.
    product = np.zeros(P.shape[0])
    for first, stop in get_partition_runs(partitions):
        product += P[:, 6 * first:6 * stop] @ partition_field[first:stop].reshape(-1)
    return product.reshape(-1, 6)

def get_induced_strain_reusing_history(P, plastic_state, plastic_history, induced_strain_history):
    # Pμ = Pμₙ + PΔμ, where Δμ is zero outside the partitions yielding in this step, so only their columns of P
    # are needed. The mask is taken from Δμ itself, so every skipped column multiplies an exact zero.
    eigenstrain_increment = plastic_state.plastic_strain - plastic_history.plastic_strain
    changed_partitions = np.any(eigenstrain_increment != 0, axis=1)
    if not np.any(changed_partitions):
        return induced_strain_history
    return induced_strain_history + apply_P_to_partitions(P, eigenstrain_increment, changed_partitions)

def reset_elastic_partitions(strain, b, P, induced_strain_history, materials, plastic_history):
    # Updates the material at the given strain, solves the partitions that stay elastic exactly, and returns the
    # residual r = ε - b - Pμ(ε) of ER-4, which is then zero in every elastic partition.
    #
    # An elastic partition's eigenstrain stays at its accepted value, so setting its strain to b + Pμ zeroes its
    # residual without changing μ. Yielding partitions keep their strain, so their μ does not change either. The
    # second material update catches elastic partitions that the reset pushed past yield: only then does μ, and
    # so Pμ, change.
    plastic_state, stress = get_plastic_eigenstrain(strain, materials, plastic_history)
    induced_strain = get_induced_strain_reusing_history(P, plastic_state, plastic_history, induced_strain_history)
    yielding_partitions = get_yielding_partitions(plastic_state, plastic_history)

    strain = strain.copy()
    strain[~yielding_partitions] = b[~yielding_partitions] + induced_strain[~yielding_partitions]

    plastic_state, stress = get_plastic_eigenstrain(strain, materials, plastic_history)
    if not np.array_equal(get_yielding_partitions(plastic_state, plastic_history), yielding_partitions):
        induced_strain = get_induced_strain_reusing_history(P, plastic_state, plastic_history, induced_strain_history)

    residual = strain - b - induced_strain
    return strain, plastic_state, stress, residual

def get_newton_krylov_correction(P, sensitivity, residual):
    # Solves J δε = -r (ER-18) by GMRES, with the matrix-free product J v = v - P (H_μ v) of ER-21. H_μ is zero
    # outside the yielding partitions, so each product reads only their columns of P.
    yielding_partitions = np.any(sensitivity != 0, axis=(1, 2))

    def apply_jacobian(vector):
        return vector - apply_P_to_partitions(P, np.einsum('pij,pj->pi', sensitivity, vector), yielding_partitions)

    return solve_newton_correction(apply_jacobian, residual)

def newton_krylov_iteration(E, P, clusters, macro_strain, plastic_history):
    # One load step (Strategy 2): Newton on the actual residual ER-4, from the exact elastic predictor Eε̄ + Pμₙ.
    # The clusters have different volumes, so the residual norm is a volume average over them.
    materials = clusters.materials
    b = (E @ macro_strain).reshape(-1, 6)
    induced_strain_history = get_induced_strain(P, plastic_history.plastic_strain)

    def evaluate(strain):
        return Evaluation(*reset_elastic_partitions(strain, b, P, induced_strain_history, materials,
                                                    plastic_history))

    def get_correction(current):
        sensitivity = get_eigenstrain_sensitivity(current.strain, materials, plastic_history)
        return get_newton_krylov_correction(P, sensitivity, current.residual)

    return solve_load_step(evaluate, get_correction, b + induced_strain_history, macro_strain,
                           config.tfa_divergence_limit, clusters.average)

def get_tfa_solver(E, P, clusters):
    # TFA on the given clusters, which E and P were built on.
    return Solver(functools.partial(newton_krylov_iteration, E, P, clusters), clusters)
