# The partition-averaged Lippmann–Schwinger (LS) equation of notes/2_ls_model.md, solved by Newton–GMRES on its
# LS-20 Jacobian. Each partition carries a uniform polarization τ_A = σ_A − C0 ε_A, and partitions interact only through
# the homogeneous reference medium C0, so the whole nonlocal operator is one FFT convolution with the kernel P0 (LS-17).
# Two methods solve it: CP on the partitions, and Spectral LS on the element lattice (one partition per element
# column). The code is the same for both; the lattice is whatever P0_transformed and the clusters describe.
#
# The unknowns are the partition strains ε_A. A partition's stress σ_A comes from its clusters (fe.Clusters), one
# material state per phase: a single-phase partition's cluster takes ε_A; in a mixed partition the matrix and fiber
# clusters share it, c_m ε_m + c_f ε_f = ε_A, so that both carry the same stress, and σ_A is that stress.

import functools
import numpy as np
import scipy.fft

import config
from material import get_algorithmic_stiffness, get_plastic_eigenstrain
from newton import Evaluation, Solver, SolverDidNotConverge, solve_load_step, solve_newton_correction

def apply_on_partition_lattice(fourier_blocks, partition_field):
    # Applies a translation-invariant operator, given by its 6 × 6 block at every lattice frequency, to a
    # partition-stacked field: transform the field, multiply frequency by frequency, transform back (ER-14a).
    # The field and the operator are real, so only the non-redundant half of the frequencies is used. The lattice is
    # the operator's: its first axis runs over every frequency along one side. The FFTs use config.thread_count
    # threads, like TFA's dense products.
    lattice_shape = (fourier_blocks.shape[0],) * 2
    field_transformed = scipy.fft.rfftn(partition_field.reshape(*lattice_shape, 6), axes=(0, 1),
                                        workers=config.thread_count)
    result_transformed = np.einsum('...ij,...j->...i', fourier_blocks, field_transformed)
    return scipy.fft.irfftn(result_transformed, s=lattice_shape, axes=(0, 1),
                            workers=config.thread_count).reshape(-1, 6)

def get_P0_induced_strain(P0_transformed, partition_field):
    # P0 applied as a convolution over the partition lattice (LS-17).
    return apply_on_partition_lattice(P0_transformed, partition_field)

## ------- Partition Stress from Clusters ------- ##

def get_mixed_partition_clusters(clusters):
    # The matrix and fiber cluster of every mixed partition. Clusters are ordered by partition, matrix first, so a
    # mixed partition's fiber cluster directly follows its matrix cluster.
    matrix = np.flatnonzero((clusters.volume_fraction < 1) & (clusters.materials.material_ids == 0))
    return matrix, matrix + 1

def take_rows(materials, rows):
    return materials._make(field[rows] for field in materials)

def get_cluster_strain(strain, clusters, plastic_history):
    # Each cluster's strain: its partition's strain, except in mixed partitions, where it is shared so that the matrix
    # and fiber clusters carry the same stress. Newton on the fiber strain ε_f, with ε_m = (ε_A − c_f ε_f) / c_m and
    # residual σ_m(ε_m) − σ_f(ε_f), whose derivative is −(L_t,f + (c_f / c_m) L_t,m).
    cluster_strain = strain[clusters.partition]
    matrix, fiber = get_mixed_partition_clusters(clusters)
    if len(matrix) == 0:
        return cluster_strain
    c_m, c_f = clusters.volume_fraction[matrix, None], clusters.volume_fraction[fiber, None]
    matrix_materials, fiber_materials = take_rows(clusters.materials, matrix), take_rows(clusters.materials, fiber)
    matrix_history, fiber_history = take_rows(plastic_history, matrix), take_rows(plastic_history, fiber)
    partition_strain = cluster_strain[matrix]
    fiber_strain = partition_strain.copy()
    for _ in range(config.mixed_partition_max_iterations):
        matrix_strain = (partition_strain - c_f * fiber_strain) / c_m
        _, matrix_stress = get_plastic_eigenstrain(matrix_strain, matrix_materials, matrix_history)
        _, fiber_stress = get_plastic_eigenstrain(fiber_strain, fiber_materials, fiber_history)
        stress_difference = matrix_stress - fiber_stress
        stress_scale = max(np.abs(matrix_stress).max(), np.abs(fiber_stress).max())
        if np.abs(stress_difference).max() <= config.mixed_partition_tolerance * stress_scale:
            break
        jacobian = (get_algorithmic_stiffness(fiber_strain, fiber_materials, fiber_history)
                    + (c_f / c_m)[:, :, None] * get_algorithmic_stiffness(matrix_strain, matrix_materials,
                                                                           matrix_history))
        fiber_strain = fiber_strain + np.linalg.solve(jacobian, stress_difference[..., None])[..., 0]
    else:
        raise SolverDidNotConverge("the equal-stress split of a mixed partition did not converge")
    cluster_strain[matrix], cluster_strain[fiber] = matrix_strain, fiber_strain
    return cluster_strain

def get_partition_stress(cluster_stress, clusters):
    # σ_A = Σ c σ_cluster over the partition's clusters (a single-phase partition's own cluster stress).
    stress = np.zeros((clusters.partition[-1] + 1, 6))
    np.add.at(stress, clusters.partition, clusters.volume_fraction[:, None] * cluster_stress)
    return stress

def get_partition_tangent(cluster_strain, clusters, plastic_history):
    # dσ_A/dε_A: a single-phase partition's cluster tangent L_t; for a mixed partition, differentiating the equal-stress
    # split gives dε_f = D_f dε_A with D_f = (L_t,f + (c_f / c_m) L_t,m)⁻¹ L_t,m / c_m, dε_m = (dε_A − c_f dε_f) / c_m,
    # and the tangent c_m L_t,m D_m + c_f L_t,f D_f.
    cluster_tangent = get_algorithmic_stiffness(cluster_strain, clusters.materials, plastic_history)
    tangent = np.zeros((clusters.partition[-1] + 1, 6, 6))
    tangent[clusters.partition] = cluster_tangent
    matrix, fiber = get_mixed_partition_clusters(clusters)
    if len(matrix) > 0:
        c_m = clusters.volume_fraction[matrix, None, None]
        c_f = clusters.volume_fraction[fiber, None, None]
        L_m, L_f = cluster_tangent[matrix], cluster_tangent[fiber]
        D_f = np.linalg.solve(L_f + (c_f / c_m) * L_m, L_m / c_m)
        D_m = (np.eye(6) - c_f * D_f) / c_m
        tangent[clusters.partition[matrix]] = c_m * L_m @ D_m + c_f * L_f @ D_f
    return tangent

## ------- LS Equation ------- ##

def get_ls_state(strain, b, P0_transformed, reference_compliance, clusters, plastic_history):
    # The LS-18 residual r = ε − b − P0 μ*, where μ* = ε − C0⁻¹ σ is the equivalent eigenstrain of LS-6, with the
    # clusters' plastic state and stress.
    cluster_strain = get_cluster_strain(strain, clusters, plastic_history)
    plastic_state, cluster_stress = get_plastic_eigenstrain(cluster_strain, clusters.materials, plastic_history)
    equivalent_eigenstrain = strain - get_partition_stress(cluster_stress, clusters) @ reference_compliance.T
    residual = strain - b - get_P0_induced_strain(P0_transformed, equivalent_eigenstrain)
    return plastic_state, cluster_stress, residual

def get_ls_jacobian_blocks(strain, clusters, plastic_history, reference_compliance):
    # The block-diagonal factor I − C0⁻¹ L_t of the LS-20 Jacobian J v = v − P0 [(I − C0⁻¹ L_t) v], with L_t each
    # partition's tangent.
    cluster_strain = get_cluster_strain(strain, clusters, plastic_history)
    return np.eye(6) - reference_compliance @ get_partition_tangent(cluster_strain, clusters, plastic_history)

def get_ls_newton_correction(P0_transformed, jacobian_blocks, residual):
    # Solves the LS-20 Newton system J δε = −r by unpreconditioned GMRES.
    def apply_jacobian(vector):
        return vector - get_P0_induced_strain(P0_transformed, np.einsum('pij,pj->pi', jacobian_blocks, vector))

    return solve_newton_correction(apply_jacobian, residual)

def ls_newton_iteration(macro_strain, initial_strain, clusters, plastic_history, P0_transformed, reference_compliance):
    # One load step: Newton from initial_strain until the LS-18 residual converges.
    b = np.tile(macro_strain, (len(initial_strain), 1))

    def evaluate(strain):
        return Evaluation(strain, *get_ls_state(strain, b, P0_transformed, reference_compliance, clusters,
                                                plastic_history))

    def get_correction(current):
        jacobian_blocks = get_ls_jacobian_blocks(current.strain, clusters, plastic_history, reference_compliance)
        return get_ls_newton_correction(P0_transformed, jacobian_blocks, current.residual)

    return solve_load_step(evaluate, get_correction, initial_strain, macro_strain, config.ls_divergence_limit)

class LSSolver:
    # The Newton iteration as a load-path solver. Each step starts from the previous step's converged partition
    # strain shifted by the macrostrain increment, so the solver keeps that strain between steps.
    def __init__(self, iterate, partition_count):
        self.iterate = iterate
        self.previous_strain = np.zeros((partition_count, 6))
        self.previous_macro_strain = np.zeros(6)

    def __call__(self, macro_strain, plastic_history):
        initial_strain = self.previous_strain + (macro_strain - self.previous_macro_strain)
        step_result = self.iterate(macro_strain, initial_strain, plastic_history=plastic_history)
        self.previous_strain, self.previous_macro_strain = step_result.strain, macro_strain
        return step_result

def get_ls_solver(lattice, reference_compliance):
    # An LS solver on the given fe.Lattice: CP on the partitions, Spectral LS on the elements.
    clusters = lattice.clusters
    iterate = functools.partial(ls_newton_iteration, clusters=clusters, P0_transformed=lattice.P0_transformed,
                                reference_compliance=reference_compliance)
    return Solver(LSSolver(iterate, clusters.partition[-1] + 1), clusters)
