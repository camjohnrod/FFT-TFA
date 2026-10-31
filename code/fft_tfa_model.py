# FFT-TFA (notes/2_ls_model.md section 10, equations FT-*): TFA's clusters, each with one uniform eigenstrain and one
# material update at its average strain, interacting through the Lippmann–Schwinger equation of a homogeneous medium
# of the matrix's elastic stiffness, C0 = L_m, instead of the composite's FE influence functions.
#
# With C0 = L_m, a matrix element's polarization τ = (L − C0) ε − L μ is −L_m μ_g: exact for one uniform eigenstrain per
# cluster, whatever the cluster's shape. An inclusion element's polarization depends on its strain, and is taken
# uniform on each inclusion cluster's piece (its part in one partition, fe.Clusters). So the polarization lives on
# regions, each matrix cluster and each inclusion piece, and the unknowns are one strain per region (FT-1). The
# regions interact through LS-5 averaged over them (FT-2): the partition-averaged kernel on each partition's mean
# polarization, by FFT on the partitions (LS-17), plus a correction to the exact element-level interaction between
# the pieces of each partition and, optionally, of its neighbouring partitions (FT-3), and the mean strain kept at the
# imposed one (FT-4). Each load step is solved by Newton–GMRES on the region strains (FT-5, FT-6), preconditioned by
# block-Jacobi (FT-7).

from typing import NamedTuple
import numpy as np
import scipy.sparse

import config
from ls_model import (apply_on_lattice, apply_sparse, assemble_blocks, get_moulinec_suquet_kernel, get_offset_blocks,
                      get_P0_transformed)
from material import get_eigenstrain_sensitivity, get_plastic_eigenstrain
from newton import Evaluation, solve_load_step, solve_newton_correction

## ------- Regions (FT-1) ------- ##

class Interaction(NamedTuple):
    # FFT-TFA's operators on its clusters, built once (get_interaction). Region volumes are in partition volumes, as
    # fe.Clusters' are. The sparse averages act on arrays of rows of 6 (@); the *_blocks matrices hold 6 × 6 blocks
    # (apply_sparse).
    clusters: tuple
    matrix_L: np.ndarray                     # C0
    region_cluster: np.ndarray               # the cluster of each region
    region_volume: np.ndarray
    piece_region: np.ndarray                 # the region of each piece
    region_to_cluster: scipy.sparse.csr_matrix   # a cluster's strain from its regions', weights v_r / v_g
    piece_to_region: scipy.sparse.csr_matrix     # a region's strain from its pieces', weights v_p / v_r
    piece_to_partition: scipy.sparse.csr_matrix  # a partition's mean polarization from its pieces', weights c_p
    P0_transformed: np.ndarray               # the partitions' kernel for C0 (LS-21)
    correction_blocks: scipy.sparse.bsr_matrix   # FT-3's C_pt, pieces × pieces, stored as 6 × 6 blocks
    own_strain_change: np.ndarray            # FT-7's G_rr, one 6 × 6 block per region
    cluster_strain_change: np.ndarray        # FT-7's S_r = Σ_{q in r's cluster} G_rq, one 6 × 6 block per region

def get_regions(clusters):
    # FT-1: a matrix cluster is one region; an inclusion cluster has one region per piece. Matrix clusters come first,
    # then the inclusion pieces in piece order. Returns each piece's region and each region's cluster.
    is_matrix = clusters.materials.material_ids[clusters.piece_cluster] == 0
    keys = np.where(is_matrix, clusters.piece_cluster, clusters.cluster_count + np.arange(len(is_matrix)))
    region_keys, piece_region = np.unique(keys, return_inverse=True)
    region_cluster = np.zeros(len(region_keys), dtype=int)
    region_cluster[piece_region] = clusters.piece_cluster
    return piece_region, region_cluster

def get_averaging(groups, members, volume, group_count):
    # The sparse average over each group of its members' rows, each member weighted by its volume.
    group_volume = np.bincount(groups, weights=volume, minlength=group_count)
    return scipy.sparse.csr_matrix((volume / group_volume[groups], (groups, members)),
                                   shape=(group_count, len(members)))

## ------- Interaction (FT-2, FT-3) ------- ##

def get_neighbour_offsets(partition_lattice_shape, reach):
    # The partition offsets [z, y, x] to every partition within reach partitions along each axis, the own partition
    # first, each partition once: as residues modulo the lattice, so with two partitions along an axis the offsets +1
    # and −1 give the same partition and count once.
    residues = [np.unique(np.arange(-reach, reach + 1) % count) for count in partition_lattice_shape]
    return np.stack(np.meshgrid(*residues, indexing="ij"), axis=-1).reshape(-1, 3)

def get_correction_blocks(mesh, clusters, P0_transformed, matrix_L, reach):
    # FT-3. Between a receiving piece s of partition A and a source piece t of partition B = A + d, the exact
    # interaction averages the element kernel D0 = P0 C0⁻¹ over every element pair, grouped by their offset within
    # the partitions, δ = a − b, which n_st(δ) pairs share (the cross-correlation of the two pieces' indicators,
    # by FFT):
    #   D_st = (1 / n_s) Σ_δ n_st(δ) D0[δ − K d],   K = the partition's size in elements along each axis,
    # while the partition FFT applies c_t D̄0[−d], D̄0 its kernel. C_st = D_st − c_t D̄0[−d] corrects one to the other.
    # Returns C for every partition within reach (fft_tfa_neighbour_blocks), as sparse 6 × 6 blocks, pieces × pieces.
    discretization = mesh.discretization
    block_shape = np.array([discretization.elements_per_partition_along_z, discretization.elements_per_partition_side,
                            discretization.elements_per_partition_side])
    element_shape = discretization.element_lattice_shape
    partition_shape = discretization.partition_lattice_shape
    partition_count = clusters.partition_count
    matrix_compliance = np.linalg.inv(matrix_L)

    # Each piece's rank among its partition's pieces, and every piece's indicator on its partition's elements.
    pieces_per_partition = np.bincount(clusters.piece_partition, minlength=partition_count)
    first_piece = np.cumsum(pieces_per_partition) - pieces_per_partition
    piece_rank = np.arange(len(clusters.piece_partition)) - first_piece[clusters.piece_partition]
    rank_count = pieces_per_partition.max()
    present = np.zeros((partition_count, rank_count), dtype=bool)
    present[clusters.piece_partition, piece_rank] = True
    share = np.zeros((partition_count, rank_count))
    share[clusters.piece_partition, piece_rank] = clusters.piece_volume_fraction
    element_z, element_y, element_x = np.unravel_index(np.arange(discretization.element_count), element_shape)
    indicators = np.zeros((partition_count, rank_count, *block_shape))
    element_piece = clusters.element_piece
    indicators[clusters.piece_partition[element_piece], piece_rank[element_piece], element_z % block_shape[0],
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
        rows = first_piece[receiver] + receiving_rank
        columns = first_piece[source[receiver]] + source_rank
        correction_parts.append((correction[pairs], rows, columns))
    piece_count = len(clusters.piece_partition)
    return assemble_blocks(*[np.concatenate(parts) for parts in zip(*correction_parts)], (piece_count, piece_count))

def get_strain_change_within_groups(mesh, clusters, P0_transformed, matrix_L, correction_blocks, piece_group):
    # FT-7: for every piece p, Σ_t G_pt over the pieces t of p's group, with G_pt the strain change of FT-2 with FT-4
    # at piece p from a unit polarization of piece t:
    #   G_pt = c_t D̄0[A_p − A_t] + C_pt − (1 / N) Σ_s c_s C_st,
    # the partition kernel taken pair by pair (the pairs within each group), C from the correction blocks within the
    # group; the partition FFT adds nothing to the cell mean (P̂0(0) = 0).
    partition_shape = mesh.discretization.partition_lattice_shape
    partition_D0 = get_offset_blocks(P0_transformed).reshape(-1, 6, 6) @ np.linalg.inv(matrix_L)
    partition_position = np.stack(np.unravel_index(clusters.piece_partition, partition_shape), axis=-1)
    share = clusters.piece_volume_fraction
    piece_count = len(piece_group)
    sums = np.zeros((piece_count, 6, 6))
    receiver, source = get_pairs_within_groups(piece_group, np.arange(piece_count))
    offset = np.ravel_multi_index(tuple(((partition_position[receiver] - partition_position[source])
                                         % partition_shape).T), partition_shape)
    np.add.at(sums, receiver, share[source][:, None, None] * partition_D0[offset])

    block_rows = np.repeat(np.arange(piece_count), np.diff(correction_blocks.indptr))
    block_columns = correction_blocks.indices
    within = piece_group[block_rows] == piece_group[block_columns]
    np.add.at(sums, block_rows[within], correction_blocks.data[within])
    group_mean = np.zeros((piece_group.max() + 1, 6, 6))
    np.add.at(group_mean, piece_group[block_columns], share[block_rows][:, None, None] * correction_blocks.data)
    return sums - group_mean[piece_group] / clusters.partition_count

def get_interaction(mesh, clusters, matrix_L, reach):
    # FFT-TFA's operators on its clusters, with the exact interaction kept between partitions within reach (0: each
    # partition with itself only).
    piece_region, region_cluster = get_regions(clusters)
    region_count = len(region_cluster)
    piece_volume = clusters.piece_volume_fraction
    region_volume = np.bincount(piece_region, weights=piece_volume)
    region_to_cluster = get_averaging(region_cluster, np.arange(region_count), region_volume, clusters.cluster_count)
    piece_to_region = get_averaging(piece_region, np.arange(len(piece_region)), piece_volume, region_count)
    piece_to_partition = scipy.sparse.csr_matrix(
        (piece_volume, (clusters.piece_partition, np.arange(len(piece_region)))),
        shape=(clusters.partition_count, len(piece_region)))
    P0_transformed = get_P0_transformed(mesh.discretization, matrix_L)
    # The correction blocks are applied at every residual and Jacobian product; stored as dense 6 × 6 blocks (BSR),
    # a product reads one column index per block instead of one per entry.
    correction_blocks = get_correction_blocks(mesh, clusters, P0_transformed, matrix_L, reach).tobsr(blocksize=(6, 6))
    own_strain_change, cluster_strain_change = [
        (piece_to_region @ get_strain_change_within_groups(mesh, clusters, P0_transformed, matrix_L, correction_blocks,
                                                           piece_group).reshape(-1, 36)).reshape(-1, 6, 6)
        for piece_group in (piece_region, clusters.piece_cluster)]
    return Interaction(clusters, matrix_L, region_cluster, region_volume, piece_region, region_to_cluster,
                       piece_to_region, piece_to_partition, P0_transformed, correction_blocks, own_strain_change,
                       cluster_strain_change)

def get_strain_change(interaction, region_polarization):
    # FT-2 with FT-4: the strain every region's polarization takes from every region, ε_r = ε̄ − (this). On the
    # pieces, m_p = D̄0 τ̄ (the partition FFT on each partition's mean polarization τ̄_A = Σ c_p τ_p, broadcast to its
    # pieces) + Σ_t C_pt τ_t, less its volume average over the cell (FT-4); then averaged over each region.
    clusters = interaction.clusters
    piece_polarization = region_polarization[interaction.piece_region]
    partition_polarization = interaction.piece_to_partition @ piece_polarization
    partition_strain = apply_on_lattice(interaction.P0_transformed,
                                        np.linalg.solve(interaction.matrix_L, partition_polarization.T).T)
    piece_strain = (partition_strain[clusters.piece_partition]
                    + apply_sparse(interaction.correction_blocks, piece_polarization))
    mean_strain = clusters.piece_volume_fraction @ piece_strain / clusters.partition_count
    return interaction.piece_to_region @ piece_strain - mean_strain

## ------- Solver (FT-5, FT-6) ------- ##

class FFTTFASolver:
    # FFT-TFA along the load path: solve_step(macro_strain, plastic_history) solves one load step by Newton on the
    # region strains, from the previous step's shifted by the macrostrain increment, as the LS methods do. Its residual
    # is a strain error, like TFA's, judged by its volume average over the regions.
    def __init__(self, interaction):
        self.interaction = interaction
        self.clusters = interaction.clusters
        materials = self.clusters.materials
        self.region_stiffness_contrast = materials.L[interaction.region_cluster] - interaction.matrix_L
        self.region_share = interaction.region_volume / self.clusters.volume[interaction.region_cluster]   # v_r / v_g
        self.previous_strain = np.zeros((len(interaction.region_cluster), 6))
        self.previous_macro_strain = np.zeros(6)

    def get_polarization(self, region_strain, eigenstrain):
        # FT-1: τ_r = (L_g − C0) ε_r − L_g μ_g, which is −L_m μ_g in a matrix region.
        materials, region_cluster = self.clusters.materials, self.interaction.region_cluster
        return (np.einsum('rij,rj->ri', self.region_stiffness_contrast, region_strain)
                - np.einsum('rij,rj->ri', materials.L[region_cluster], eigenstrain[region_cluster]))

    def evaluate(self, region_strain, macro_strain, plastic_history):
        # FT-5: r = ε − ε̄ + (FT-2's strain change), each cluster's eigenstrain from its return map at its average
        # strain ε_g = Σ (v_r / v_g) ε_r; the stress returned is each cluster's.
        cluster_strain = self.interaction.region_to_cluster @ region_strain
        plastic_state, stress = get_plastic_eigenstrain(cluster_strain, self.clusters.materials, plastic_history)
        residual = (region_strain - macro_strain
                    + get_strain_change(self.interaction, self.get_polarization(region_strain,
                                                                               plastic_state.plastic_strain)))
        return Evaluation(region_strain, plastic_state, stress, residual)

    def get_newton_correction(self, current, plastic_history):
        # FT-6: J v = v + (FT-2's strain change of the polarization change T v), with
        #   T v = (L_g − C0) v_r − L_g H_g Σ (v_q / v_g) v_q   over the regions q of r's cluster g,
        # H_g = ∂μ/∂ε at the cluster's average strain. GMRES solves it with J's diagonal blocks inverted (block-Jacobi,
        # FT-7) on the right, so it still measures the residual of J δε = −r.
        interaction, materials = self.interaction, self.clusters.materials
        cluster_strain = interaction.region_to_cluster @ current.strain
        sensitivity = get_eigenstrain_sensitivity(cluster_strain, materials, plastic_history)
        plastic_stiffness = materials.L @ sensitivity                                   # L_g H_g

        def apply_polarization_change(vector):
            return (np.einsum('rij,rj->ri', self.region_stiffness_contrast, vector)
                    - apply_cluster_blocks(plastic_stiffness, interaction.region_to_cluster @ vector,
                                           interaction.region_cluster))

        def apply_jacobian(vector):
            return vector + get_strain_change(interaction, apply_polarization_change(vector))

        # FT-7: J_rr = I + G_rr (L_g − C0) − (v_r / v_g) S_r L_g H_g.
        diagonal_blocks = (np.eye(6) + interaction.own_strain_change @ self.region_stiffness_contrast
                           - self.region_share[:, None, None] * interaction.cluster_strain_change
                           @ plastic_stiffness[interaction.region_cluster])
        inverse_blocks = np.linalg.inv(diagonal_blocks)

        def apply_preconditioner(vector):
            return np.einsum('rij,rj->ri', inverse_blocks, vector)

        y, gmres_iteration_count = solve_newton_correction(lambda vector: apply_jacobian(apply_preconditioner(vector)),
                                                           current.residual)
        return apply_preconditioner(y), gmres_iteration_count

    def solve_step(self, macro_strain, plastic_history):
        initial_strain = self.previous_strain + (macro_strain - self.previous_macro_strain)
        region_volume, partition_count = self.interaction.region_volume, self.clusters.partition_count
        step_result = solve_load_step(
            evaluate=lambda strain: self.evaluate(strain, macro_strain, plastic_history),
            get_correction=lambda current: self.get_newton_correction(current, plastic_history),
            initial_strain=initial_strain, macro_strain=macro_strain, divergence_limit=config.ls_divergence_limit,
            average=lambda values: region_volume @ values / partition_count)
        self.previous_strain, self.previous_macro_strain = step_result.strain, macro_strain
        return step_result

def apply_cluster_blocks(cluster_blocks, cluster_rows, region_cluster):
    # Every region's cluster's block times that cluster's row.
    return np.einsum('rij,rj->ri', cluster_blocks[region_cluster], cluster_rows[region_cluster])

def get_pairs_within_groups(groups, members):
    # Every pair (i, j) of a member i of the given ones with any j of i's group (j = i included).
    order = np.argsort(groups, kind="stable")
    group_size = np.bincount(groups)
    group_first = np.cumsum(group_size) - group_size
    pair_counts = group_size[groups[members]]
    first = np.repeat(members, pair_counts)
    position = np.arange(len(first)) - np.repeat(np.cumsum(pair_counts) - pair_counts, pair_counts)
    return first, order[group_first[groups[first]] + position]
