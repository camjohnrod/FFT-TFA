# The partition-averaged Lippmann–Schwinger (LS) equation of notes/2_ls_model.md, solved by Newton–GMRES. Partitions
# interact only through the homogeneous reference medium C0, so the whole nonlocal operator is one FFT convolution
# with the kernel P0 (LS-17). Two methods solve it, sharing the lattice code:
#   Spectral LS  on the element lattice: one partition per element, each one cluster, so a partition's stress is its
#                cluster's at the partition strain (SpectralLSSolver);
#   CP           on the partitions, whose material states are clusters (fe.Clusters) that may span several
#                partitions. Every piece of a partition carries the partition's polarization τ_A = σ − C0 ε, and every
#                piece of a cluster its eigenstrain μ_g; the pieces' strains follow from both in closed form (LS-25),
#                and each cluster's eigenstrain from the return map at its average strain (LS-26, CPSolver).
#
# The reference medium's operators are built here too: C0, and the kernel P0 averaged over partitions (LS-21); and the
# Fourier preconditioner of the Newton solvers (LS-23). They are translation invariant, so they are stored as one 6 × 6
# block per lattice frequency, rfftn's non-redundant half (shape (N_z, N, N//2 + 1, 6, 6), lattices indexed [z, y, x]
# and square in-plane), and applied by FFT (apply_on_lattice).

from typing import NamedTuple
import numpy as np
import scipy.fft
import scipy.sparse
import scipy.sparse.linalg

import config
from material import (get_algorithmic_stiffness, get_eigenstrain_sensitivity, get_plastic_eigenstrain,
                      get_unloaded_plastic_state)
from newton import Evaluation, SolverDidNotConverge, solve_load_step, solve_newton_correction

## ------- Convolution on a Lattice ------- ##

def apply_on_lattice(fourier_blocks, field):
    # A translation-invariant operator, given by its 6 × 6 block at every lattice frequency (rfftn's half spectrum),
    # applied to a field of one row of 6 per partition: transform, multiply frequency by frequency, transform back.
    # Only axes longer than one are transformed, as read from the operator's shape: a transform of length one is the
    # identity, but costs time.
    lattice_shape = (fourier_blocks.shape[0], fourier_blocks.shape[1], fourier_blocks.shape[1])
    axes = tuple(axis for axis in (0, 1) if lattice_shape[axis] > 1) + (2,)
    transformed_shape = tuple(lattice_shape[axis] for axis in axes)
    field_transformed = scipy.fft.rfftn(field.reshape(*lattice_shape, 6), axes=axes, workers=config.thread_count)
    result_transformed = np.einsum('...ij,...j->...i', fourier_blocks, field_transformed)
    return scipy.fft.irfftn(result_transformed, s=transformed_shape, axes=axes,
                            workers=config.thread_count).reshape(-1, 6)

## ------- Reference Stiffness ------- ##

def get_named_stiffness(clusters, E, matrix_L):
    # The stiffness config.cp_reference_stiffness names: volume averages over the clusters.
    if config.cp_reference_stiffness == "homogenized":
        return clusters.average(clusters.materials.L @ E.reshape(-1, 6, 6))    # <L E>
    if config.cp_reference_stiffness == "voigt":
        return clusters.average(clusters.materials.L)                          # <L>
    return matrix_L

def get_cp_reference_L(named_L):
    # CP's C0, part of its model: a fraction of the named stiffness. The same C0 sets the interaction (P0) and how a
    # partition's strain is shared among its pieces (LS-25), so it must stay below the tangent of every coupled cluster.
    return config.cp_reference_stiffness_scale * named_L

def get_spectral_reference_L(matrix_L, inclusion_L):
    # Spectral LS's reference stiffness. Its solution does not depend on it (its P0 is a projection; Schneider 2021,
    # section 2), so it is a solver setting: "midpoint" is Moulinec–Suquet's rule (1998, section 3.1), the phases'
    # Lamé constants averaged, which for two isotropic phases is (L_m + L_f) / 2.
    if config.spectral_ls_reference_stiffness == "midpoint":
        return (matrix_L + inclusion_L) / 2
    return matrix_L

## ------- Moulinec–Suquet Kernel P0 ------- ##

def get_transformed(offset_blocks, lattice_shape):
    # From one 6 × 6 block per lattice offset to one per frequency.
    return np.fft.rfftn(offset_blocks.reshape(*lattice_shape, 6, 6), axes=(0, 1, 2))

def get_offset_blocks(transformed):
    # The inverse of get_transformed, rows of 6 (offsets × 6, 6).
    lattice_shape = (transformed.shape[0], transformed.shape[1], transformed.shape[1])
    return np.fft.irfftn(transformed, s=lattice_shape, axes=(0, 1, 2)).reshape(-1, 6)

def get_direction_strain_matrix(directions):
    # B(n), shape (..., 6, 3): the engineering strain of a displacement u varying along the unit direction n, per
    # unit gradient, ε = B(n) u.
    n1, n2, n3 = np.moveaxis(directions, -1, 0)
    zero = np.zeros_like(n1)
    rows = [(n1, zero, zero), (zero, n2, zero), (zero, zero, n3), (n2, n1, zero), (zero, n3, n2), (n3, zero, n1)]
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)

def get_moulinec_suquet_kernel(lattice_shape, reference_L):
    # P̂0(ξ) = Γ̂0(ξ) C0 at every frequency of a lattice of the given shape [z, y, x] over the cubic cell (LS-21), with
    # the continuum Green operator Γ̂0 = B (Bᵀ C0 B)⁻¹ Bᵀ, B = B(ξ / |ξ|), returned as offset blocks. P̂0(0) = 0
    # (LS-9). Along an axis with an even count N, Γ̂0 is not even in a component equal to N / 2, so it has no
    # real-valued meaning there; P̂0 is then the identity, Moulinec–Suquet's Γ̂0 = C0⁻¹ (1998, section 2.4.2), which
    # makes the stress at those frequencies zero. The cell is a cube, so frequencies in units of 2π / side compare
    # across axes whatever each axis's count.
    frequencies = [np.fft.fftfreq(count, 1 / count) for count in lattice_shape[:2]]
    frequencies.append(np.fft.rfftfreq(lattice_shape[2], 1 / lattice_shape[2]))
    frequency_z, frequency_y, frequency_x = np.meshgrid(*frequencies, indexing="ij")
    wave_vectors = np.stack([frequency_x, frequency_y, frequency_z], axis=-1)
    wave_vectors[0, 0, 0] = [1, 0, 0]  # any direction at ξ = 0, whose block is set to zero below
    B = get_direction_strain_matrix(wave_vectors / np.linalg.norm(wave_vectors, axis=-1, keepdims=True))
    B_transposed = np.swapaxes(B, -1, -2)
    P0_transformed = B @ np.linalg.inv(B_transposed @ reference_L @ B) @ B_transposed @ reference_L
    P0_transformed[0, 0, 0] = 0
    highest_frequencies = np.zeros(frequency_x.shape, dtype=bool)
    for frequency, count in zip((frequency_z, frequency_y, frequency_x), lattice_shape):
        if count % 2 == 0:
            highest_frequencies |= np.abs(frequency) == count / 2
    P0_transformed[highest_frequencies] = np.eye(6)
    return get_offset_blocks(P0_transformed)

def average_over_partitions(element_P0, discretization):
    # The partitions' kernel from the elements' (same mesh), by linearity. Unit eigenstrain in a partition is unit
    # eigenstrain in each of its k × k × k_z elements, and a partition's strain is the mean over its elements, so with
    # K = (k_z, k, k) and offsets grouped by d = s − t, which (k_z − |d_z|)(k − |d_y|)(k − |d_x|) pairs share:
    #   P0[D] = Σ_d (k_z − |d_z|)(k − |d_y|)(k − |d_x|) / (k_z k²) · p0_element[K D + d]       (LS note, section 6.5)
    # Offsets are periodic. One element per partition returns the kernel unchanged. A column through the thickness
    # averages over every z offset, which keeps only the ξ_z = 0 frequencies: the in-plane kernel.
    k, k_z = discretization.elements_per_partition_side, discretization.elements_per_partition_along_z
    element_count_z, element_side, _ = discretization.element_lattice_shape
    element_P0 = element_P0.reshape(*discretization.element_lattice_shape, 6, 6)
    partition_offsets = k * np.arange(discretization.partition_number_per_side)
    partition_offsets_z = k_z * np.arange(discretization.partition_number_along_z)
    partition_P0 = np.zeros((*discretization.partition_lattice_shape, 6, 6))
    for d_k in range(1 - k_z, k_z):
        for d_j in range(1 - k, k):
            for d_i in range(1 - k, k):
                weight = (k_z - abs(d_k)) * (k - abs(d_j)) * (k - abs(d_i)) / (k_z * k**2)
                partition_P0 += weight * element_P0[np.ix_((partition_offsets_z + d_k) % element_count_z,
                                                           (partition_offsets + d_j) % element_side,
                                                           (partition_offsets + d_i) % element_side)]
    return partition_P0.reshape(-1, 6)

def get_P0_transformed(discretization, reference_L):
    # P̂0 on the discretization's partition lattice: the Moulinec–Suquet kernel on its element grid, averaged over its
    # partitions, as SCA averages its fine-grid operator over clusters (Liu et al. 2016).
    element_P0 = get_moulinec_suquet_kernel(discretization.element_lattice_shape, reference_L)
    return get_transformed(average_over_partitions(element_P0, discretization), discretization.partition_lattice_shape)

## ------- Residual in Matrix Units (LS-24) ------- ##

def get_residual_in_matrix_units(P0_transformed, reference_L, matrix_L):
    # LS-24: the LS residual r is a strain of the reference medium, so its size depends on C0. Convergence is judged
    # instead on the strain correction an all-matrix cell would need to cancel r, M_m⁻¹ r, with M_m the Newton
    # Jacobian of a homogeneous cell of matrix stiffness C_m (LS-20 with L_t = C_m everywhere):
    #   M̂_m(ξ) = I − P̂0(ξ) (I − C0⁻¹ C_m),
    # a fixed Fourier operator like P0, so the tolerance means the same whatever C0. Returns M̂_m⁻¹ per frequency.
    matrix_cell_jacobian = np.eye(6) - P0_transformed @ (np.eye(6) - np.linalg.solve(reference_L, matrix_L))
    return np.linalg.inv(matrix_cell_jacobian)

## ------- Block Helpers ------- ##

def apply_blocks(blocks, rows):
    # One 6 × 6 block times one row of 6, row by row.
    return np.einsum('pij,pj->pi', blocks, rows)

def assemble_blocks(blocks, block_rows, block_columns, shape_in_blocks):
    # A sparse matrix of 6 × 6 blocks, each block at its block row and column, blocks at the same position summed.
    rows = np.broadcast_to(6 * block_rows[:, None, None] + np.arange(6)[:, None], blocks.shape)
    columns = np.broadcast_to(6 * block_columns[:, None, None] + np.arange(6), blocks.shape)
    return scipy.sparse.coo_matrix((blocks.ravel(), (rows.ravel(), columns.ravel())),
                                   shape=(6 * shape_in_blocks[0], 6 * shape_in_blocks[1])).tocsr()

def get_block_diagonal(blocks):
    return assemble_blocks(blocks, np.arange(len(blocks)), np.arange(len(blocks)), (len(blocks), len(blocks)))

def apply_sparse(matrix, rows):
    # A sparse matrix on stacked rows of 6.
    return (matrix @ rows.reshape(-1)).reshape(-1, 6)

## ------- Newton–GMRES on a Lattice ------- ##

def get_fourier_preconditioner(P0_transformed, mean_D):
    # LS-23: the best Fourier-diagonal approximation of the Newton Jacobian J v = v − P0 (D v) (LS-20), inverted. P0
    # is diagonal in Fourier space; when D is one block per partition (D = I − C0⁻¹ L_t), its Fourier-space diagonal
    # is its lattice mean ⟨D⟩. So the Fourier-diagonal operator closest to J (Frobenius norm) is
    #   M̂(ξ) = I − P̂0(ξ) ⟨D⟩,
    # the Jacobian of a homogeneous cell with the mean current tangent ⟨L_t⟩, and its inverse is one 6 × 6 inverse per
    # frequency.
    return np.linalg.inv(np.eye(6) - P0_transformed @ mean_D)

class LatticeSolver:
    # What CP and Spectral LS share as load-path solvers on their lattice (run.Lattice): Newton on the partition
    # strains, each correction solving the LS-20 system J δε = −r by GMRES, convergence judged in matrix units (LS-24).
    # solve_step(macro_strain, plastic_history) solves one load step, from the previous step's converged strain
    # shifted by the macrostrain increment. A subclass supplies evaluate (the LS-18 residual) and get_D, which gives
    # LS-20's operator D at the current state and a function for its lattice mean ⟨D⟩.
    def __init__(self, lattice, matrix_L, use_fourier_preconditioner):
        self.clusters = lattice.clusters
        self.reference_L = lattice.reference_L                  # C0
        self.P0_transformed = lattice.P0_transformed
        self.use_fourier_preconditioner = use_fourier_preconditioner
        self.residual_in_matrix_units = get_residual_in_matrix_units(self.P0_transformed, self.reference_L, matrix_L)
        self.previous_strain = np.zeros((self.clusters.partition_count, 6))
        self.previous_macro_strain = np.zeros(6)

    def get_newton_correction(self, current, plastic_history):
        # Solves J δε = −r with J v = v − P0 (D v) (LS-20). The Fourier preconditioner M⁻¹ (LS-23) acts on the right:
        # GMRES solves J M⁻¹ y = −r and δε = M⁻¹ y, so it still measures the residual of J δε = −r, and M⁻¹ changes
        # only how many iterations that takes, not the correction.
        apply_D, get_mean_D = self.get_D(current, plastic_history)

        def apply_jacobian(vector):
            return vector - apply_on_lattice(self.P0_transformed, apply_D(vector))

        if not self.use_fourier_preconditioner:
            return solve_newton_correction(apply_jacobian, current.residual)

        preconditioner = get_fourier_preconditioner(self.P0_transformed, get_mean_D())

        def apply_preconditioner(vector):
            return apply_on_lattice(preconditioner, vector)

        y, gmres_iteration_count = solve_newton_correction(lambda vector: apply_jacobian(apply_preconditioner(vector)),
                                                           current.residual)
        return apply_preconditioner(y), gmres_iteration_count

    def solve_step(self, macro_strain, plastic_history):
        initial_strain = self.previous_strain + (macro_strain - self.previous_macro_strain)
        step_result = solve_load_step(
            evaluate=lambda strain: self.evaluate(strain, macro_strain, plastic_history),
            get_correction=lambda current: self.get_newton_correction(current, plastic_history),
            initial_strain=initial_strain, macro_strain=macro_strain, divergence_limit=config.ls_divergence_limit,
            measure_residual=lambda residual: apply_on_lattice(self.residual_in_matrix_units, residual))
        self.previous_strain, self.previous_macro_strain = step_result.strain, macro_strain
        return step_result

## ------- Spectral LS: One Cluster per Partition ------- ##

class SpectralLSSolver(LatticeSolver):
    # Spectral LS: LS-18 with one partition per element (per element column), each partition one cluster, so a
    # partition's stress is its cluster's at the partition strain.
    def __init__(self, lattice, matrix_L, use_fourier_preconditioner):
        super().__init__(lattice, matrix_L, use_fourier_preconditioner)
        if self.clusters.cluster_count != self.clusters.partition_count:
            raise ValueError("Spectral LS needs exactly one cluster per partition.")

    def evaluate(self, strain, macro_strain, plastic_history):
        # The LS-18 residual r = ε − ε̄ − P0 μ*, with μ* = ε − C0⁻¹ σ the equivalent eigenstrain (LS-6), and the
        # candidate plastic state and stress.
        plastic_state, stress = get_plastic_eigenstrain(strain, self.clusters.materials, plastic_history)
        equivalent_eigenstrain = strain - np.linalg.solve(self.reference_L, stress.T).T
        b = np.tile(macro_strain, (len(strain), 1))
        residual = strain - b - apply_on_lattice(self.P0_transformed, equivalent_eigenstrain)
        return Evaluation(strain, plastic_state, stress, residual)

    def get_D(self, current, plastic_history):
        # D = I − C0⁻¹ L_t (LS-20), one block per partition.
        tangent = get_algorithmic_stiffness(current.strain, self.clusters.materials, plastic_history)
        D_blocks = np.eye(6) - np.linalg.solve(self.reference_L, tangent)
        return (lambda vector: apply_blocks(D_blocks, vector)), (lambda: D_blocks.mean(axis=0))

def get_elastic_strain_concentration(lattice, matrix_L):
    # The strain of every partition of the lattice per unit macrostrain in elasticity, partitions × 6 × 6 (column k for
    # the k-th unit macrostrain): the strain concentration tensor that k-means clusters on (Liu–Bessa–Liu 2016,
    # section 2.2.1). Each column is one Spectral LS load step with every phase elastic.
    materials = lattice.clusters.materials
    elastic_materials = materials._replace(yield_stress=np.full(len(materials.yield_stress), np.inf))
    elastic_lattice = lattice._replace(clusters=lattice.clusters._replace(materials=elastic_materials))
    unloaded = get_unloaded_plastic_state(lattice.clusters.cluster_count)
    return np.stack([SpectralLSSolver(elastic_lattice, matrix_L, config.spectral_ls_fourier_preconditioner)
                     .solve_step(unit_strain, unloaded).strain for unit_strain in np.eye(6)], axis=-1)

## ------- CP: Pieces of Clusters (LS-25) ------- ##

class PieceLaw(NamedTuple):
    # LS-25 on a lattice, fixed along the load path. With K_g = (L_g − C0)⁻¹, N_g = K_g L_g and
    # T_A = (Σ c_p K_g)⁻¹ = C*_A − C0 over partition A's pieces p (slide 15), every piece of cluster g in partition A
    # has the strain ε_p = K_g τ_A + N_g μ_g, with τ_A = T_A (ε_A − λ*_A) and λ*_A = Σ c_p N_g μ_g. That is linear in
    # the partition strains ε and cluster eigenstrains μ, so it is kept as sparse block matrices:
    #   ε_g = E ε + F μ        each cluster's average strain over its pieces (F only between coupled clusters),
    #   μ*  = D_e (ε − Λ μ)    each partition's equivalent eigenstrain −C0⁻¹ τ_A, with Λ μ = λ*_A.
    # A coupled cluster has a piece in a partition holding several pieces; an uncoupled one's pieces each fill their
    # partition, so its strain does not depend on μ.
    coupled_clusters: np.ndarray
    strain_to_cluster_strain: scipy.sparse.csr_matrix                # E, blocks w_p K_g T_A
    coupled_eigenstrain_to_cluster_strain: scipy.sparse.csr_matrix   # F, on the coupled clusters
    eigenstrain_to_lambda_star: scipy.sparse.csr_matrix              # Λ, blocks c_p N_g
    elastic_D: scipy.sparse.csr_matrix                               # D_e, blocks −C0⁻¹ T_A

def get_partition_pairs(clusters, pieces):
    # Every pair (p, q) of a given piece p with a piece q of p's partition, q = p included. Pieces are ordered by
    # partition, so a partition's pieces are consecutive, from its first.
    pieces_per_partition = np.bincount(clusters.piece_partition)
    first_piece = np.cumsum(pieces_per_partition) - pieces_per_partition
    pair_counts = pieces_per_partition[clusters.piece_partition[pieces]]
    pair_first = np.repeat(pieces, pair_counts)
    position_in_partition = np.arange(len(pair_first)) - np.repeat(np.cumsum(pair_counts) - pair_counts, pair_counts)
    return pair_first, first_piece[clusters.piece_partition[pair_first]] + position_in_partition

def get_coupled_eigenstrain_to_cluster_strain(clusters, coupled, K, N, T):
    # F of LS-25 on the coupled clusters:
    #   ∂ε_g/∂μ_h = δ_gh N_g − Σ w_p c_q K_g T_A N_h
    # over the pairs (p, q) of pieces of one partition A, p of cluster g and q of cluster h, w_p = c_p / v_g the
    # piece's weight in its cluster's average.
    position = np.zeros(clusters.cluster_count, dtype=int)
    position[coupled] = np.arange(len(coupled))
    p, q = get_partition_pairs(clusters, np.flatnonzero(np.isin(clusters.piece_cluster, coupled)))
    g, h, A = clusters.piece_cluster[p], clusters.piece_cluster[q], clusters.piece_partition[p]
    weight = clusters.piece_volume_fraction[p] / clusters.volume[g] * clusters.piece_volume_fraction[q]
    return assemble_blocks(np.concatenate([N[coupled], -weight[:, None, None] * K[g] @ T[A] @ N[h]]),
                           np.concatenate([np.arange(len(coupled)), position[g]]),
                           np.concatenate([np.arange(len(coupled)), position[h]]), (len(coupled), len(coupled)))

def get_piece_law(clusters, reference_L):
    partition, cluster = clusters.piece_partition, clusters.piece_cluster
    share = clusters.piece_volume_fraction                   # c_p, the piece's share of its partition
    weight = share / clusters.volume[cluster]                # w_p, its weight in its cluster's average
    partition_count, cluster_count = clusters.partition_count, clusters.cluster_count
    K = np.linalg.inv(clusters.materials.L - reference_L)
    N = K @ clusters.materials.L
    partition_sum = scipy.sparse.csr_matrix((share, (partition, np.arange(len(partition)))),
                                            shape=(partition_count, len(partition)))
    T = np.linalg.inv((partition_sum @ K[cluster].reshape(-1, 36)).reshape(-1, 6, 6))
    coupled = np.unique(cluster[np.bincount(partition)[partition] > 1])
    return PieceLaw(
        coupled,
        assemble_blocks(weight[:, None, None] * K[cluster] @ T[partition], cluster, partition,
                        (cluster_count, partition_count)),
        get_coupled_eigenstrain_to_cluster_strain(clusters, coupled, K, N, T),
        assemble_blocks(share[:, None, None] * N[cluster], partition, cluster, (partition_count, cluster_count)),
        get_block_diagonal(-np.linalg.solve(reference_L, T)))

def get_cluster_strain(strain, eigenstrain, law):
    # ε_g = E ε + F μ (LS-25).
    cluster_strain = apply_sparse(law.strain_to_cluster_strain, strain)
    cluster_strain[law.coupled_clusters] += apply_sparse(law.coupled_eigenstrain_to_cluster_strain,
                                                         eigenstrain[law.coupled_clusters])
    return cluster_strain

def get_equivalent_eigenstrain(strain, eigenstrain, law):
    # μ*_A = ε_A − C0⁻¹ σ_A = −C0⁻¹ τ_A = D_e (ε − Λ μ) (LS-6 and LS-25, σ_A = C0 ε_A + τ_A).
    return apply_sparse(law.elastic_D, strain - apply_sparse(law.eigenstrain_to_lambda_star, eigenstrain))

def get_cluster_jacobian_solver(sensitivity, law):
    # Z⁻¹ (LS-27) as a function on one row per cluster, with Z = I − H F the derivative of LS-26's cluster residual
    # r_g = μ_g − μ(ε_g) at fixed partition strains, H_g = ∂μ/∂ε at each cluster's average strain. Z is the identity
    # on the uncoupled clusters, and sparse on the coupled ones, factorized here.
    coupled = law.coupled_clusters
    if len(coupled) == 0:
        return lambda rows: rows
    F = law.coupled_eigenstrain_to_cluster_strain
    Z = scipy.sparse.identity(F.shape[0], format="csr") - get_block_diagonal(sensitivity[coupled]) @ F
    factorization = scipy.sparse.linalg.splu(Z.tocsc())

    def solve(rows):
        result = rows.copy()
        result[coupled] = factorization.solve(rows[coupled].reshape(-1)).reshape(-1, 6)
        return result

    return solve

def check_reference_below_tangents(cluster_tangents, reference_L):
    # LS-26 has a solution only while C0 is below the tangent of every coupled cluster, i.e. while L_t − C0 is positive
    # definite there (Brisard–Dormieux's Assumption 1 on the soft side). Otherwise the cluster Jacobian Z is singular,
    # so stop with the cause instead.
    differences = cluster_tangents - reference_L
    lowest_eigenvalues = np.linalg.eigvalsh((differences + np.swapaxes(differences, -1, -2)) / 2)[:, 0]
    if np.any(lowest_eigenvalues <= 0):
        raise SolverDidNotConverge(
            f"C0 is not below the tangent stiffness of {np.count_nonzero(lowest_eigenvalues <= 0)} clusters sharing a "
            "partition, so their eigenstrains (LS-26) have no solution: lower config.cp_reference_stiffness_scale "
            "(a yielding phase's shear tangent is about G H / (3G + H))")

## ------- CP Solver (LS-26, LS-27) ------- ##

class CPSolver(LatticeSolver):
    # CP: LS-18 on the partitions, whose stress comes from clusters that may span several partitions (LS-25). Newton's
    # unknowns are the partition strains; every evaluation solves the clusters' eigenstrains from them (LS-26), so
    # every Newton iterate satisfies every cluster's return map.
    def __init__(self, lattice, matrix_L, use_fourier_preconditioner):
        super().__init__(lattice, matrix_L, use_fourier_preconditioner)
        self.law = get_piece_law(self.clusters, self.reference_L)

    def solve_eigenstrain(self, strain, plastic_history):
        # LS-26: the cluster eigenstrains μ_g = μ(ε_g; z_n,g) at the given partition strains, ε_g the cluster's average
        # strain (LS-25), by Newton on r_g = μ_g − μ(ε_g) with the Jacobian Z (LS-27), from the accepted history. An
        # uncoupled cluster's ε_g does not depend on μ, so one Newton step settles it. Each step goes through the outer
        # Newton's backtracking line search (line_search_*) on the residual in stress units, ‖L_g r_g‖: the return
        # map's switch between elastic and plastic clusters can make full steps cycle. Converged when every |L_g r_g|
        # is below cp_eigenstrain_tolerance times the largest cluster stress. Returns the eigenstrains, and the
        # clusters' plastic state and stress.
        materials = self.clusters.materials

        def evaluate(eigenstrain):
            # The clusters' average strains, their return map, and the residual r_g and L_g r_g.
            cluster_strain = get_cluster_strain(strain, eigenstrain, self.law)
            plastic_state, stress = get_plastic_eigenstrain(cluster_strain, materials, plastic_history)
            residual = eigenstrain - plastic_state.plastic_strain
            return cluster_strain, plastic_state, stress, residual, apply_blocks(materials.L, residual)

        eigenstrain = plastic_history.plastic_strain
        cluster_strain, plastic_state, stress, residual, stress_residual = evaluate(eigenstrain)
        for iteration in range(config.cp_eigenstrain_max_iterations + 1):
            if np.abs(stress_residual).max() <= config.cp_eigenstrain_tolerance * np.abs(stress).max():
                return eigenstrain, plastic_state, stress
            if iteration == config.cp_eigenstrain_max_iterations:
                break
            sensitivity = get_eigenstrain_sensitivity(cluster_strain, materials, plastic_history)
            correction = -get_cluster_jacobian_solver(sensitivity, self.law)(residual)
            residual_size = np.linalg.norm(stress_residual)
            step_length = 1.0
            for _ in range(config.line_search_max_reductions + 1):
                trial = evaluate(eigenstrain + step_length * correction)
                sufficient_size = (1 - config.line_search_sufficient_decrease * step_length) * residual_size
                if np.linalg.norm(trial[-1]) < sufficient_size:
                    break
                step_length *= config.line_search_step_reduction
            eigenstrain = eigenstrain + step_length * correction
            cluster_strain, plastic_state, stress, residual, stress_residual = trial
        raise SolverDidNotConverge("CP's cluster eigenstrains (LS-26) did not converge: C0 may be above a cluster's "
                                   "tangent stiffness (lower config.cp_reference_stiffness_scale)")

    def evaluate(self, strain, macro_strain, plastic_history):
        # The LS-18 residual r = ε − ε̄ − P0 μ* at the eigenstrains LS-26 gives; the stress returned is each cluster's.
        eigenstrain, plastic_state, cluster_stress = self.solve_eigenstrain(strain, plastic_history)
        residual = (strain - macro_strain
                    - apply_on_lattice(self.P0_transformed, get_equivalent_eigenstrain(strain, eigenstrain, self.law)))
        return Evaluation(strain, plastic_state, cluster_stress, residual)

    def get_D(self, current, plastic_history):
        # LS-27: D v, the equivalent eigenstrain's response to a partition strain change v with the eigenstrains
        # following through LS-26 (Z δμ = H E v):
        #   D v = D_e (v − Λ Z⁻¹ H E v).
        # When every cluster lies in one partition, D is one block per partition, the derivative of its stress;
        # otherwise the eigenstrains couple the partitions a cluster spans. Its lattice mean ⟨D⟩, for LS-23, is the
        # mean response to uniform fields: the mean block in the first case.
        law, clusters = self.law, self.clusters
        cluster_strain = get_cluster_strain(current.strain, current.plastic_state.plastic_strain, law)
        sensitivity = get_eigenstrain_sensitivity(cluster_strain, clusters.materials, plastic_history)
        cluster_tangents = clusters.materials.L @ (np.eye(6) - sensitivity)
        check_reference_below_tangents(cluster_tangents[law.coupled_clusters], self.reference_L)
        solve_cluster_jacobian = get_cluster_jacobian_solver(sensitivity, law)
        eigenstrain_load = get_block_diagonal(sensitivity) @ law.strain_to_cluster_strain     # H E

        def apply_D(vector):
            eigenstrain_change = solve_cluster_jacobian(apply_sparse(eigenstrain_load, vector))
            return apply_sparse(law.elastic_D,
                                vector - apply_sparse(law.eigenstrain_to_lambda_star, eigenstrain_change))

        def get_mean_D():
            return np.stack([apply_D(np.tile(unit, (clusters.partition_count, 1))).mean(axis=0)
                             for unit in np.eye(6)], axis=-1)

        return apply_D, get_mean_D
