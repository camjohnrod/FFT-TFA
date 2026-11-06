"""FFT-TFA, the method under development (notes/0_fft_tfa_lecture.md, equations FT-*).

FFT-TFA keeps TFA's material model: clusters, each with one uniform eigenstrain and one material update at its average
strain. What differs is how the clusters interact. TFA uses the composite's exact influence functions, a dense matrix
P that every cluster pair fills. FFT-TFA instead uses the Lippmann–Schwinger equation of a homogeneous medium with the
matrix's elastic stiffness, C0 = L_m, which has two parts:
  - far field: one FFT on the partition lattice, acting on each partition's mean polarization (LS-17);
  - near field: an exact correction between the clusters of each partition and, if reach ≥ 1, of its neighbours (FT-3).
So the cost grows with the partitions and the clusters per partition, not with the elements.

Why C0 = L_m: a matrix element's polarization τ = (L − C0) ε − L μ is then −L_m μ_g, uniform on a cluster with one
eigenstrain, whatever the cluster's shape, so the matrix's interaction is exact. An inclusion element's polarization
depends on its strain, and is taken uniform on each cluster. The unknowns are one strain per cluster (FT-1), and each
load step is solved by Newton–GMRES on them (FT-5, FT-6), preconditioned by block-Jacobi (FT-7).
"""

from typing import NamedTuple
import numpy as np
import scipy.sparse

from cell import Clusters, get_matrix_L
from ls_model import (apply_on_lattice, apply_sparse, assemble_blocks, get_moulinec_suquet_kernel, get_P0_transformed,
                      to_offset_blocks)
from material import apply_blocks, get_algorithmic_stiffness, radial_return
from newton import Evaluation, Solver, solve_by_gmres

## ------- Interaction (FT-2, FT-3) ------- ##

class Interaction(NamedTuple):
    """FFT-TFA's operators on its clusters, built once (get_interaction)."""
    clusters: Clusters
    matrix_L: np.ndarray                           # C0
    cluster_to_partition: scipy.sparse.csr_matrix  # a partition's mean polarization from its clusters', weights c_p
    P0_transformed: np.ndarray                     # the partitions' kernel for C0 (LS-21)
    correction_blocks: scipy.sparse.bsr_matrix     # FT-3's C_pt, clusters × clusters, stored as 6 × 6 blocks
    own_strain_change: np.ndarray                  # FT-7's G_pp, one 6 × 6 block per cluster

def get_neighbour_offsets(partition_lattice_shape, reach):
    """The partition offsets [z, y, x] to every partition within reach partitions along each axis, the own partition
    first, each partition once: as residues modulo the lattice, so with two partitions along an axis the offsets +1 and
    −1 give the same partition and count once."""
    residues = [np.unique(np.arange(-reach, reach + 1) % count) for count in partition_lattice_shape]
    return np.stack(np.meshgrid(*residues, indexing="ij"), axis=-1).reshape(-1, 3)

def get_partition_shape_in_elements(discretization):
    """A partition's size in elements along [z, y, x], K in FT-3."""
    return np.array([discretization.elements_per_partition_along_z, discretization.elements_per_partition_side,
                     discretization.elements_per_partition_side])

class PartitionSlots(NamedTuple):
    """The clusters laid out partition by partition, so that every partition can be handled at once: a partition's
    clusters fill its slots 0, 1, … in order, out of slot_count, the most clusters any partition holds."""
    first_cluster: np.ndarray    # partitions: the number of the partition's first cluster
    occupied: np.ndarray         # partitions × slots: whether the slot holds a cluster
    share: np.ndarray            # partitions × slots: the cluster's share of its partition (c_t), 0 in an empty slot
    indicators: np.ndarray       # partitions × slots × the partition's elements [z, y, x]: 1 where the cluster lies
    element_count: np.ndarray    # partitions × slots: the cluster's elements (1 in an empty slot, to divide by)

def get_partition_slots(mesh, clusters):
    discretization = mesh.discretization
    block_shape = get_partition_shape_in_elements(discretization)
    partition_count = clusters.partition_count
    clusters_per_partition = np.bincount(clusters.cluster_partition, minlength=partition_count)
    first_cluster = np.cumsum(clusters_per_partition) - clusters_per_partition
    cluster_slot = np.arange(clusters.cluster_count) - first_cluster[clusters.cluster_partition]
    slot_count = clusters_per_partition.max()
    occupied = np.zeros((partition_count, slot_count), dtype=bool)
    occupied[clusters.cluster_partition, cluster_slot] = True
    share = np.zeros((partition_count, slot_count))
    share[clusters.cluster_partition, cluster_slot] = clusters.share
    element_z, element_y, element_x = np.unravel_index(np.arange(discretization.element_count),
                                                       discretization.element_lattice_shape)
    indicators = np.zeros((partition_count, slot_count, *block_shape))
    element_cluster = clusters.element_cluster
    indicators[clusters.cluster_partition[element_cluster], cluster_slot[element_cluster], element_z % block_shape[0],
               element_y % block_shape[1], element_x % block_shape[2]] = 1
    element_count = np.maximum(indicators.sum(axis=(2, 3, 4)), 1)
    return PartitionSlots(first_cluster, occupied, share, indicators, element_count)

def get_correction_blocks(mesh, clusters, P0_transformed, matrix_L, reach):
    """FT-3's correction blocks C, for every pair of clusters in partitions within reach of each other, as sparse 6 × 6
    blocks, clusters × clusters.

    Between a receiving cluster s of partition A and a source cluster t of partition B = A + d, the exact interaction
    averages the element kernel D0 = P0 C0⁻¹ over every pair of their elements. Grouped by the elements' offset within
    their partitions, δ = a − b, which n_st(δ) pairs share (the cross-correlation of the two clusters' indicators, by
    FFT), it is
      D_st = (1 / n_s) Σ_δ n_st(δ) D0[δ − K d],   K = the partition's size in elements along each axis,
    while the partition FFT applies c_t D̄0[−d], D̄0 the partition kernel. C_st = D_st − c_t D̄0[−d] corrects one to the
    other."""
    discretization = mesh.discretization
    block_shape = get_partition_shape_in_elements(discretization)
    element_shape = discretization.element_lattice_shape
    partition_shape = discretization.partition_lattice_shape
    partition_count = clusters.partition_count
    matrix_compliance = np.linalg.inv(matrix_L)
    slots = get_partition_slots(mesh, clusters)
    slot_count = slots.occupied.shape[1]

    # Step 1: every cluster's indicator, Fourier transformed. Padded to twice the partition, the cross-correlation holds
    # every offset |δ| < K without wrapping; index i holds δ = i below K, i − 2K from K on.
    padded_shape = tuple(2 * block_shape)
    transformed = np.fft.rfftn(slots.indicators, s=padded_shape, axes=(2, 3, 4))
    pair_offsets = [np.where(np.arange(2 * size) < size, np.arange(2 * size), np.arange(2 * size) - 2 * size)
                    for size in block_shape]
    # Step 2: the element kernel D0 and the partition kernel D̄0, by offset.
    element_D0 = get_moulinec_suquet_kernel(element_shape, matrix_L).reshape(*element_shape, 6, 6) @ matrix_compliance
    partition_D0 = to_offset_blocks(P0_transformed).reshape(*partition_shape, 6, 6) @ matrix_compliance
    partition_position = np.stack(np.unravel_index(np.arange(partition_count), partition_shape), axis=-1)

    # Step 3: for each partition offset d within reach, every partition at once: the pair counts n_st, the exact blocks
    # D_st, and the corrections C_st of the pairs whose slots both hold a cluster.
    correction_parts = []
    for offset in get_neighbour_offsets(partition_shape, reach):
        source = np.ravel_multi_index(tuple(((partition_position + offset) % partition_shape).T), partition_shape)
        pair_counts = np.rint(np.fft.irfftn(transformed[:, :, None] * np.conj(transformed[source][:, None]),
                                            s=padded_shape, axes=(3, 4, 5)))
        kernel = element_D0[np.ix_(*[(pair_offsets[axis] - block_shape[axis] * offset[axis]) % element_shape[axis]
                                     for axis in range(3)])]
        exact = (pair_counts.reshape(partition_count * slot_count**2, -1) @ kernel.reshape(-1, 36)).reshape(
            partition_count, slot_count, slot_count, 6, 6) / slots.element_count[:, :, None, None, None]
        partition_kernel = partition_D0[tuple(-offset % partition_shape)]
        correction = exact - slots.share[source][:, None, :, None, None] * partition_kernel
        pairs = slots.occupied[:, :, None] & slots.occupied[source][:, None, :]
        receiver, receiving_slot, source_slot = np.nonzero(pairs)
        rows = slots.first_cluster[receiver] + receiving_slot
        columns = slots.first_cluster[source[receiver]] + source_slot
        correction_parts.append((correction[pairs], rows, columns))
    return assemble_blocks(*[np.concatenate(parts) for parts in zip(*correction_parts)],
                           (clusters.cluster_count, clusters.cluster_count))

def get_own_strain_change(clusters, P0_transformed, matrix_L, correction_blocks):
    """FT-7: for every cluster p, the strain change of FT-2 with FT-4 at p from a unit polarization of p itself,
      G_pp = c_p D̄0[0] + C_pp − (1 / N) Σ_s c_s C_sp,
    D̄0[0] the partition kernel at offset zero; the partition FFT adds nothing to the cell mean (P̂0(0) = 0). These are
    the diagonal blocks of the Newton Jacobian, which the block-Jacobi preconditioner inverts."""
    own_partition_kernel = to_offset_blocks(P0_transformed).reshape(-1, 6, 6)[0] @ np.linalg.inv(matrix_L)
    share = clusters.share
    block_rows = np.repeat(np.arange(clusters.cluster_count), np.diff(correction_blocks.indptr))
    block_columns = correction_blocks.indices
    own = share[:, None, None] * own_partition_kernel
    on_diagonal = block_rows == block_columns
    np.add.at(own, block_rows[on_diagonal], correction_blocks.data[on_diagonal])
    mean = np.zeros((clusters.cluster_count, 6, 6))
    np.add.at(mean, block_columns, share[block_rows][:, None, None] * correction_blocks.data)
    return own - mean / clusters.partition_count

def resolve_reach(reach, partition_lattice_shape):
    """reach as a number of partitions: "full" is every partition, half the periodic lattice along its longest axis."""
    return max(partition_lattice_shape) // 2 if reach == "full" else reach

def get_interaction(mesh, clusters, reach):
    """FFT-TFA's operators on its clusters, with the exact interaction kept between partitions within reach (0: each
    partition with itself only; "full": every partition)."""
    matrix_L = get_matrix_L()
    reach = resolve_reach(reach, mesh.discretization.partition_lattice_shape)
    cluster_to_partition = scipy.sparse.csr_matrix(
        (clusters.share, (clusters.cluster_partition, np.arange(clusters.cluster_count))),
        shape=(clusters.partition_count, clusters.cluster_count))
    P0_transformed = get_P0_transformed(mesh.discretization, matrix_L)
    # The correction blocks are applied at every residual and Jacobian product; stored as dense 6 × 6 blocks (BSR),
    # a product reads one column index per block instead of one per entry.
    correction_blocks = get_correction_blocks(mesh, clusters, P0_transformed, matrix_L, reach).tobsr(blocksize=(6, 6))
    return Interaction(clusters, matrix_L, cluster_to_partition, P0_transformed, correction_blocks,
                       get_own_strain_change(clusters, P0_transformed, matrix_L, correction_blocks))

def get_strain_change(interaction, polarization):
    """FT-2 with FT-4: the strain every cluster takes from every cluster's polarization, ε_p = ε̄ − (this).

    Far field: the partition FFT on each partition's mean polarization τ̄_A = Σ c_p τ_p, broadcast to its clusters. Near
    field: Σ_t C_pt τ_t. Then less its volume average over the cell, so the mean strain stays the imposed one (FT-4)."""
    clusters = interaction.clusters
    partition_polarization = interaction.cluster_to_partition @ polarization
    partition_strain = apply_on_lattice(interaction.P0_transformed,
                                        np.linalg.solve(interaction.matrix_L, partition_polarization.T).T)
    strain_change = (partition_strain[clusters.cluster_partition]
                     + apply_sparse(interaction.correction_blocks, polarization))
    return strain_change - clusters.average(strain_change)

## ------- Solver (FT-5, FT-6) ------- ##

class FFTTFASolver(Solver):
    """FFT-TFA: Newton on the cluster strains. Its residual is a strain error, like TFA's, judged by its volume average
    over the clusters; with the matrix as reference medium it is in matrix units already (FT-5)."""
    def __init__(self, interaction):
        super().__init__(interaction.clusters, interaction.clusters.cluster_count)
        self.interaction = interaction

    def evaluate(self, strain, macro_strain, plastic_history):
        """FT-5: r = ε − ε̄ + (FT-2's strain change), with the polarization of FT-1, τ_g = (L_g − C0) ε_g − L_g μ_g,
        which is −L_m μ_g in a matrix cluster, and each cluster's eigenstrain from its radial return at its strain."""
        materials = self.clusters.materials
        plastic_state, stress = radial_return(strain, materials, plastic_history)
        polarization = (apply_blocks(materials.L - self.interaction.matrix_L, strain)
                        - apply_blocks(materials.L, plastic_state.plastic_strain))
        residual = strain - macro_strain + get_strain_change(self.interaction, polarization)
        return Evaluation(strain, plastic_state, stress, residual)

    def get_newton_correction(self, current, plastic_history):
        """FT-6: J v = v + (FT-2's strain change of the polarization change T v), with T = L_t − C0 and L_t each
        cluster's algorithmic stiffness. GMRES solves it with J's diagonal blocks inverted (block-Jacobi, FT-7) on the
        right."""
        interaction = self.interaction
        T = (get_algorithmic_stiffness(current.strain, self.clusters.materials, plastic_history) - interaction.matrix_L)

        def apply_jacobian(vector):
            return vector + get_strain_change(interaction, apply_blocks(T, vector))

        inverse_blocks = np.linalg.inv(np.eye(6) + interaction.own_strain_change @ T)      # FT-7: J_pp = I + G_pp T
        return solve_by_gmres(apply_jacobian, current.residual,
                              apply_preconditioner=lambda vector: apply_blocks(inverse_blocks, vector))
