# FFT-TFA (notes/2_ls_model.md section 10, equations FT-*): TFA's clusters, each with one uniform eigenstrain and one
# material update at its average strain, interacting through the Lippmann–Schwinger equation of a homogeneous medium
# of the matrix's elastic stiffness, C0 = L_m, instead of the composite's FE influence functions.
#
# With C0 = L_m, a matrix element's polarization τ = (L − C0) ε − L μ is −L_m μ_g: exact for one uniform eigenstrain per
# cluster, whatever the cluster's shape. An inclusion element's polarization depends on its strain, and is taken
# uniform on each cluster. So the unknowns are one strain per cluster (FT-1). The clusters interact through LS-5
# averaged over them (FT-2): the partition-averaged kernel on each partition's mean polarization, by FFT on the
# partitions (LS-17), plus a correction to the exact element-level interaction between the clusters of each partition
# and, optionally, of its neighbouring partitions (FT-3), and the mean strain kept at the imposed one (FT-4). Each load
# step is solved by Newton–GMRES on the cluster strains (FT-5, FT-6), preconditioned by block-Jacobi (FT-7).

from typing import NamedTuple
import numpy as np
import scipy.sparse

from ls_model import (apply_on_lattice, apply_sparse, assemble_blocks, get_moulinec_suquet_kernel, get_offset_blocks,
                      get_P0_transformed)
from material import apply_blocks, get_algorithmic_stiffness, get_plastic_eigenstrain
from newton import Evaluation, Solver, solve_by_gmres

## ------- Interaction (FT-2, FT-3) ------- ##

class Interaction(NamedTuple):
    # FFT-TFA's operators on its clusters, built once (get_interaction).
    clusters: tuple
    matrix_L: np.ndarray                         # C0
    cluster_to_partition: scipy.sparse.csr_matrix  # a partition's mean polarization from its clusters', weights c_p
    P0_transformed: np.ndarray                   # the partitions' kernel for C0 (LS-21)
    correction_blocks: scipy.sparse.bsr_matrix   # FT-3's C_pt, clusters × clusters, stored as 6 × 6 blocks
    own_strain_change: np.ndarray                # FT-7's G_pp, one 6 × 6 block per cluster

def get_neighbour_offsets(partition_lattice_shape, reach):
    # The partition offsets [z, y, x] to every partition within reach partitions along each axis, the own partition
    # first, each partition once: as residues modulo the lattice, so with two partitions along an axis the offsets +1
    # and −1 give the same partition and count once.
    residues = [np.unique(np.arange(-reach, reach + 1) % count) for count in partition_lattice_shape]
    return np.stack(np.meshgrid(*residues, indexing="ij"), axis=-1).reshape(-1, 3)

def get_correction_blocks(mesh, clusters, P0_transformed, matrix_L, reach):
    # FT-3. Between a receiving cluster s of partition A and a source cluster t of partition B = A + d, the exact
    # interaction averages the element kernel D0 = P0 C0⁻¹ over every element pair, grouped by their offset within
    # the partitions, δ = a − b, which n_st(δ) pairs share (the cross-correlation of the two clusters' indicators,
    # by FFT):
    #   D_st = (1 / n_s) Σ_δ n_st(δ) D0[δ − K d],   K = the partition's size in elements along each axis,
    # while the partition FFT applies c_t D̄0[−d], D̄0 its kernel. C_st = D_st − c_t D̄0[−d] corrects one to the other.
    # Returns C for every partition within reach, as sparse 6 × 6 blocks, clusters × clusters.
    discretization = mesh.discretization
    block_shape = np.array([discretization.elements_per_partition_along_z, discretization.elements_per_partition_side,
                            discretization.elements_per_partition_side])
    element_shape = discretization.element_lattice_shape
    partition_shape = discretization.partition_lattice_shape
    partition_count = clusters.partition_count
    matrix_compliance = np.linalg.inv(matrix_L)

    # Each cluster's rank among its partition's clusters, and every cluster's indicator on its partition's elements.
    clusters_per_partition = np.bincount(clusters.cluster_partition, minlength=partition_count)
    first_cluster = np.cumsum(clusters_per_partition) - clusters_per_partition
    cluster_rank = np.arange(clusters.cluster_count) - first_cluster[clusters.cluster_partition]
    rank_count = clusters_per_partition.max()
    present = np.zeros((partition_count, rank_count), dtype=bool)
    present[clusters.cluster_partition, cluster_rank] = True
    share = np.zeros((partition_count, rank_count))
    share[clusters.cluster_partition, cluster_rank] = clusters.volume
    element_z, element_y, element_x = np.unravel_index(np.arange(discretization.element_count), element_shape)
    indicators = np.zeros((partition_count, rank_count, *block_shape))
    element_cluster = clusters.element_cluster
    indicators[clusters.cluster_partition[element_cluster], cluster_rank[element_cluster], element_z % block_shape[0],
               element_y % block_shape[1], element_x % block_shape[2]] = 1
    element_count = np.maximum(indicators.sum(axis=(2, 3, 4)), 1)

    # Padded to twice the partition, the cross-correlation holds every offset |δ| < K without wrapping; index i holds
    # δ = i below K, i − 2K from K on.
    padded_shape = tuple(2 * block_shape)
    transformed = np.fft.rfftn(indicators, s=padded_shape, axes=(2, 3, 4))
    pair_offsets = [np.where(np.arange(2 * size) < size, np.arange(2 * size), np.arange(2 * size) - 2 * size)
                    for size in block_shape]
    element_D0 = get_moulinec_suquet_kernel(element_shape, matrix_L).reshape(*element_shape, 6, 6) @ matrix_compliance
    partition_D0 = get_offset_blocks(P0_transformed).reshape(*partition_shape, 6, 6) @ matrix_compliance
    partition_position = np.stack(np.unravel_index(np.arange(partition_count), partition_shape), axis=-1)

    correction_parts = []
    for offset in get_neighbour_offsets(partition_shape, reach):
        source = np.ravel_multi_index(tuple(((partition_position + offset) % partition_shape).T), partition_shape)
        pair_counts = np.rint(np.fft.irfftn(transformed[:, :, None] * np.conj(transformed[source][:, None]),
                                            s=padded_shape, axes=(3, 4, 5)))
        kernel = element_D0[np.ix_(*[(pair_offsets[axis] - block_shape[axis] * offset[axis]) % element_shape[axis]
                                     for axis in range(3)])]
        exact = (pair_counts.reshape(partition_count * rank_count**2, -1) @ kernel.reshape(-1, 36)).reshape(
            partition_count, rank_count, rank_count, 6, 6) / element_count[:, :, None, None, None]
        partition_kernel = partition_D0[tuple(-offset % partition_shape)]
        correction = exact - share[source][:, None, :, None, None] * partition_kernel
        pairs = present[:, :, None] & present[source][:, None, :]
        receiver, receiving_rank, source_rank = np.nonzero(pairs)
        rows = first_cluster[receiver] + receiving_rank
        columns = first_cluster[source[receiver]] + source_rank
        correction_parts.append((correction[pairs], rows, columns))
    return assemble_blocks(*[np.concatenate(parts) for parts in zip(*correction_parts)],
                           (clusters.cluster_count, clusters.cluster_count))

def get_own_strain_change(clusters, P0_transformed, matrix_L, correction_blocks):
    # FT-7: for every cluster p, the strain change of FT-2 with FT-4 at p from a unit polarization of p itself:
    #   G_pp = c_p D̄0[0] + C_pp − (1 / N) Σ_s c_s C_sp,
    # D̄0[0] the partition kernel at offset zero; the partition FFT adds nothing to the cell mean (P̂0(0) = 0).
    own_partition_kernel = get_offset_blocks(P0_transformed).reshape(-1, 6, 6)[0] @ np.linalg.inv(matrix_L)
    share = clusters.volume
    block_rows = np.repeat(np.arange(clusters.cluster_count), np.diff(correction_blocks.indptr))
    block_columns = correction_blocks.indices
    own = share[:, None, None] * own_partition_kernel
    on_diagonal = block_rows == block_columns
    np.add.at(own, block_rows[on_diagonal], correction_blocks.data[on_diagonal])
    mean = np.zeros((clusters.cluster_count, 6, 6))
    np.add.at(mean, block_columns, share[block_rows][:, None, None] * correction_blocks.data)
    return own - mean / clusters.partition_count

def get_interaction(mesh, clusters, matrix_L, reach):
    # FFT-TFA's operators on its clusters, with the exact interaction kept between partitions within reach (0: each
    # partition with itself only).
    cluster_to_partition = scipy.sparse.csr_matrix(
        (clusters.volume, (clusters.cluster_partition, np.arange(clusters.cluster_count))),
        shape=(clusters.partition_count, clusters.cluster_count))
    P0_transformed = get_P0_transformed(mesh.discretization, matrix_L)
    # The correction blocks are applied at every residual and Jacobian product; stored as dense 6 × 6 blocks (BSR),
    # a product reads one column index per block instead of one per entry.
    correction_blocks = get_correction_blocks(mesh, clusters, P0_transformed, matrix_L, reach).tobsr(blocksize=(6, 6))
    return Interaction(clusters, matrix_L, cluster_to_partition, P0_transformed, correction_blocks,
                       get_own_strain_change(clusters, P0_transformed, matrix_L, correction_blocks))

def get_strain_change(interaction, polarization):
    # FT-2 with FT-4: the strain every cluster's polarization takes from every cluster, ε_p = ε̄ − (this): the partition
    # FFT on each partition's mean polarization τ̄_A = Σ c_p τ_p, broadcast to its clusters, plus Σ_t C_pt τ_t, less its
    # volume average over the cell (FT-4).
    clusters = interaction.clusters
    partition_polarization = interaction.cluster_to_partition @ polarization
    partition_strain = apply_on_lattice(interaction.P0_transformed,
                                        np.linalg.solve(interaction.matrix_L, partition_polarization.T).T)
    strain_change = (partition_strain[clusters.cluster_partition]
                     + apply_sparse(interaction.correction_blocks, polarization))
    return strain_change - clusters.average(strain_change)

## ------- Solver (FT-5, FT-6) ------- ##

class FFTTFASolver(Solver):
    # FFT-TFA: Newton on the cluster strains. Its residual is a strain error, like TFA's, judged by its volume average
    # over the clusters.
    def __init__(self, interaction):
        super().__init__(interaction.clusters, interaction.clusters.cluster_count)
        self.interaction = interaction

    def evaluate(self, strain, macro_strain, plastic_history):
        # FT-5: r = ε − ε̄ + (FT-2's strain change), with the polarization of FT-1, τ_g = (L_g − C0) ε_g − L_g μ_g, which
        # is −L_m μ_g in a matrix cluster, and each cluster's eigenstrain from its return map at its strain.
        materials = self.clusters.materials
        plastic_state, stress = get_plastic_eigenstrain(strain, materials, plastic_history)
        polarization = (apply_blocks(materials.L - self.interaction.matrix_L, strain)
                        - apply_blocks(materials.L, plastic_state.plastic_strain))
        residual = strain - macro_strain + get_strain_change(self.interaction, polarization)
        return Evaluation(strain, plastic_state, stress, residual)

    def get_newton_correction(self, current, plastic_history):
        # FT-6: J v = v + (FT-2's strain change of the polarization change T v), with T = L_t − C0 and L_t each cluster's
        # algorithmic stiffness. GMRES solves it with J's diagonal blocks inverted (block-Jacobi, FT-7) on the right.
        interaction = self.interaction
        T = (get_algorithmic_stiffness(current.strain, self.clusters.materials, plastic_history) - interaction.matrix_L)

        def apply_jacobian(vector):
            return vector + get_strain_change(interaction, apply_blocks(T, vector))

        inverse_blocks = np.linalg.inv(np.eye(6) + interaction.own_strain_change @ T)      # FT-7: J_pp = I + G_pp T
        return solve_by_gmres(apply_jacobian, current.residual,
                              apply_preconditioner=lambda vector: apply_blocks(inverse_blocks, vector))
