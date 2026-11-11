"""FFT-TFA, the method under development (notes/fft_tfa_lecture.md, equations FT-*).

FFT-TFA keeps TFA's material model: clusters, each with one uniform eigenstrain and one material update at its average
strain. What differs is how the clusters interact. TFA uses the composite's exact influence functions, a dense matrix P
that every pair of clusters fills. FFT-TFA uses the Lippmann–Schwinger equation of a homogeneous medium with the
matrix's elastic stiffness, C0 = L_m, split by distance:
  - far field: each partition's polarization is summarized by its polynomial moments up to `degree` (equivalently, its
    best-fitting polynomial over the partition), and every partition acts on every other through these summaries, by
    one FFT on the partition lattice (FT-2);
  - near field: between the clusters of the same partition, where the interaction is sharpest, the far field's share is
    replaced by the exact interaction (the corrections, FT-3).
Degree 0 summarizes a partition by its mean polarization. Raising the degree converges to the exact clustered equation,
which degree "full" reaches: its polynomials span every function on a partition's elements. The cost grows with the
partitions and the polynomials, not with the elements.

Why C0 = L_m: a matrix element's polarization τ = (L − C0) ε − L μ is then −L_m μ, uniform on a cluster with one
eigenstrain, whatever the cluster's shape, so the matrix's interaction is exact. An inclusion element's polarization
depends on its strain, and is taken uniform on each cluster. The unknowns are one strain per cluster (FT-1), and each
load step is solved by Newton–GMRES on them (FT-5, FT-6), preconditioned by block-Jacobi (FT-7).
"""

import itertools
from typing import NamedTuple
import numpy as np
import scipy.sparse

from cell import Clusters, get_matrix_L
from ls_model import apply_on_lattice, apply_sparse, assemble_blocks, get_moulinec_suquet_kernel, to_fourier_blocks
from material import apply_blocks, get_algorithmic_stiffness, radial_return
from newton import Evaluation, Solver, solve_by_gmres

## ------- Polynomials on a Partition ------- ##

def resolve_degree(degree, discretization):
    """degree as a number: "full" is the degree whose polynomials span every function on a partition's elements, the sum
    over its axes of their element count minus one."""
    return int(np.sum(discretization.partition_shape_in_elements - 1)) if degree == "full" else degree

def get_axis_polynomials(size):
    """Orthonormal polynomials of degree 0 to size − 1 on `size` equally spaced points, the constant first: one column
    per degree. Built by the three-term (Stieltjes) recurrence, each new polynomial x times the last, orthogonalized
    against all the earlier ones twice, which stays accurate at every degree; orthogonalizing the monomials xⁿ instead
    loses the high degrees to round-off, since they are nearly parallel."""
    positions = (np.arange(size) - (size - 1) / 2) / size
    polynomials = [np.full(size, 1 / np.sqrt(size))]
    for _ in range(size - 1):
        values = positions * polynomials[-1]
        for _ in range(2):
            for polynomial in polynomials:
                values = values - (polynomial @ values) * polynomial
        polynomials.append(values / np.linalg.norm(values))
    return np.column_stack(polynomials)

def get_polynomial_basis(block_shape, degree):
    """Orthonormal polynomials of total degree at most `degree` on the element positions of a partition, the constant
    first: one column per polynomial, shape (k_z, k, k, p), in order of total degree. Each is a product of one
    polynomial per axis (get_axis_polynomials), of degrees adding up to at most `degree`: the elements lie on a grid,
    so these products are orthonormal and span the same polynomials as the monomials of that total degree. An axis of k
    elements holds degrees up to k − 1 (x² on two elements is no new function), so at full degree the basis holds one
    polynomial per element."""
    axes = [axis for axis in range(3) if block_shape[axis] > 1]
    axis_polynomials = [get_axis_polynomials(block_shape[axis]) for axis in axes]
    exponents = sorted((e for e in itertools.product(*[range(min(degree, block_shape[axis] - 1) + 1) for axis in axes])
                        if sum(e) <= degree), key=sum)
    columns = []
    for exponent in exponents:
        values = np.ones(tuple(block_shape))
        for axis, polynomials, power in zip(axes, axis_polynomials, exponent):
            shape = [1, 1, 1]
            shape[axis] = block_shape[axis]
            values = values * polynomials[:, power].reshape(shape)
        columns.append(values.ravel())
    return np.column_stack(columns).reshape(*block_shape, -1)

def get_cluster_moments(discretization, clusters, basis):
    """Every cluster's moments m, clusters × p: each polynomial summed over the cluster's elements. A cluster's
    polarization τ contributes m_i τ to its partition's moment i."""
    block = discretization.partition_shape_in_elements
    z, y, x = np.unravel_index(np.arange(discretization.element_count), discretization.element_lattice_shape)
    moments = np.zeros((clusters.cluster_count, basis.shape[-1]))
    np.add.at(moments, clusters.element_cluster, basis[z % block[0], y % block[1], x % block[2]])
    return moments

## ------- Far Field: Moment Kernels (FT-2) ------- ##

def get_moment_kernels(element_Gamma, discretization, basis):
    """How each partition's moments act on every other's, at every partition offset D = A − B, shape
    (partition lattice, p, 6, p, 6): the strain in A's polynomial j per unit polarization in B's polynomial i,
      K_ji(D) = Σ_{α in A, β in B} φ_j(α) Γ(α←β) φ_i(β) = Σ_d w_ji(d) Γ(k D + d),   w_ji(d) = Σ_r φ_j(r) φ_i(r − d),
    with d the offset between the two elements' positions inside their partitions."""
    block = discretization.partition_shape_in_elements
    element_shape = discretization.element_lattice_shape
    partition_shape = discretization.partition_lattice_shape
    p = basis.shape[-1]
    kernels = np.zeros((p, p, *partition_shape, 6, 6))
    starts = [block[axis] * np.arange(partition_shape[axis]) for axis in range(3)]
    for d in itertools.product(*[range(1 - size, size) for size in block]):
        receiver = tuple(slice(max(0, d[axis]), block[axis] + min(0, d[axis])) for axis in range(3))
        source = tuple(slice(max(0, -d[axis]), block[axis] + min(0, -d[axis])) for axis in range(3))
        weights = np.einsum('zyxj,zyxi->ji', basis[receiver], basis[source])
        Gamma = element_Gamma[np.ix_(*[(starts[axis] + d[axis]) % element_shape[axis] for axis in range(3)])]
        kernels += weights[:, :, None, None, None, None, None] * Gamma[None, None]
    return kernels.transpose(2, 3, 4, 0, 5, 1, 6)

## ------- Near Field: Corrections inside each Partition (FT-3) ------- ##

class PartitionSlots(NamedTuple):
    """The clusters laid out partition by partition, so that every partition can be handled at once: a partition's
    clusters fill its slots 0, 1, … in order, out of slot_count, the most clusters any partition holds."""
    first_cluster: np.ndarray    # partitions: the number of the partition's first cluster
    occupied: np.ndarray         # partitions × slots: whether the slot holds a cluster
    indicators: np.ndarray       # partitions × slots × the partition's elements [z, y, x]: 1 where the cluster lies
    element_count: np.ndarray    # partitions × slots: the cluster's elements (1 in an empty slot, to divide by)
    cluster_slot: np.ndarray     # clusters: each cluster's slot in its partition

def get_partition_slots(mesh, clusters):
    discretization = mesh.discretization
    block_shape = discretization.partition_shape_in_elements
    partition_count = clusters.partition_count
    clusters_per_partition = np.bincount(clusters.cluster_partition, minlength=partition_count)
    first_cluster = np.cumsum(clusters_per_partition) - clusters_per_partition
    cluster_slot = np.arange(clusters.cluster_count) - first_cluster[clusters.cluster_partition]
    slot_count = clusters_per_partition.max()
    occupied = np.zeros((partition_count, slot_count), dtype=bool)
    occupied[clusters.cluster_partition, cluster_slot] = True
    element_z, element_y, element_x = np.unravel_index(np.arange(discretization.element_count),
                                                       discretization.element_lattice_shape)
    indicators = np.zeros((partition_count, slot_count, *block_shape))
    element_cluster = clusters.element_cluster
    indicators[clusters.cluster_partition[element_cluster], cluster_slot[element_cluster], element_z % block_shape[0],
               element_y % block_shape[1], element_x % block_shape[2]] = 1
    element_count = np.maximum(indicators.sum(axis=(2, 3, 4)), 1)
    return PartitionSlots(first_cluster, occupied, indicators, element_count, cluster_slot)

def get_correction_blocks(mesh, clusters, element_Gamma, moments, self_kernels):
    """FT-3's corrections C, for every pair of clusters a, b in the same partition, as sparse 6 × 6 blocks, clusters ×
    clusters: the exact interaction Γ(a←b) minus the far field's share of it.

    Γ(a←b) averages the element kernel over every pair of their elements. Grouped by the elements' offset δ inside the
    partition, which n_ab(δ) pairs share (the cross-correlation of the two clusters' indicators, by FFT),
      Γ(a←b) = (1 / n_a) Σ_δ n_ab(δ) Γ(δ),
    while the far field gives Σ_ji (m_j(a) / n_a) K_ji(0) m_i(b), with m the clusters' moments."""
    discretization = mesh.discretization
    block_shape = discretization.partition_shape_in_elements
    element_shape = discretization.element_lattice_shape
    partition_count = clusters.partition_count
    slots = get_partition_slots(mesh, clusters)
    slot_count = slots.occupied.shape[1]

    # Step 1: the pair counts n_ab(δ) of every partition, by cross-correlation. Padded to twice the partition, it holds
    # every offset |δ| < k without wrapping; index i holds δ = i below k, i − 2k from k on.
    padded_shape = tuple(2 * block_shape)
    transformed = np.fft.rfftn(slots.indicators, s=padded_shape, axes=(2, 3, 4))
    pair_counts = np.rint(np.fft.irfftn(transformed[:, :, None] * np.conj(transformed[:, None]), s=padded_shape,
                                        axes=(3, 4, 5)))
    pair_offsets = [np.where(np.arange(2 * size) < size, np.arange(2 * size), np.arange(2 * size) - 2 * size)
                    for size in block_shape]
    # Step 2: the exact interaction of every pair of slots.
    kernel = element_Gamma[np.ix_(*[pair_offsets[axis] % element_shape[axis] for axis in range(3)])]
    exact = (pair_counts.reshape(partition_count * slot_count**2, -1) @ kernel.reshape(-1, 36)).reshape(
        partition_count, slot_count, slot_count, 6, 6) / slots.element_count[:, :, None, None, None]
    # Step 3: the far field's share of every pair, from the clusters' moments.
    slot_moments = np.zeros((partition_count, slot_count, moments.shape[1]))
    slot_moments[clusters.cluster_partition, slots.cluster_slot] = moments
    far = np.einsum('Aaj,Abi,jxiy->Aabxy', slot_moments / slots.element_count[:, :, None], slot_moments, self_kernels)
    pairs = slots.occupied[:, :, None] & slots.occupied[:, None, :]
    partition, receiving_slot, source_slot = np.nonzero(pairs)
    return assemble_blocks((exact - far)[pairs], slots.first_cluster[partition] + receiving_slot,
                           slots.first_cluster[partition] + source_slot,
                           (clusters.cluster_count, clusters.cluster_count))

## ------- Interaction (FT-2, FT-4) ------- ##

class Interaction(NamedTuple):
    """FFT-TFA's operators on its clusters, built once (get_interaction)."""
    clusters: Clusters
    matrix_L: np.ndarray                         # C0
    to_moments: scipy.sparse.csr_matrix          # every partition's polarization moments, from the clusters'
    from_moments: scipy.sparse.csr_matrix        # each cluster's average of its partition's polynomial strain field
    moment_kernels: np.ndarray                   # one 6p × 6p block per partition-lattice frequency
    correction_blocks: scipy.sparse.bsr_matrix   # FT-3's C inside each partition, stored as 6 × 6 blocks
    own_strain_change: np.ndarray                # FT-7's own blocks, one 6 × 6 block per cluster

def get_own_strain_change(clusters, moments, self_kernels, correction_blocks):
    """FT-7: for every cluster a, the strain change at a from a unit polarization of a itself: the far field's
    self-term Σ_ji (m_j(a) / n_a) K_ji(0) m_i(a), plus its own correction, less its share of FT-4's cell mean (the far
    field adds nothing to it), less what subtracting the mean polarization before the corrections takes,
    w_a (S(a) − Σ_q w_q S(q)), with S(a) the sum of a's corrections and w the clusters' shares of the cell."""
    element_counts = np.bincount(clusters.element_cluster, minlength=clusters.cluster_count)
    own = np.einsum('aj,ai,jxiy->axy', moments / element_counts[:, None], moments, self_kernels)
    weights = clusters.share / clusters.partition_count
    block_rows = np.repeat(np.arange(clusters.cluster_count), np.diff(correction_blocks.indptr))
    block_columns = correction_blocks.indices
    on_diagonal = block_rows == block_columns
    np.add.at(own, block_rows[on_diagonal], correction_blocks.data[on_diagonal])
    column_mean = np.zeros((clusters.cluster_count, 6, 6))
    np.add.at(column_mean, block_columns, weights[block_rows][:, None, None] * correction_blocks.data)
    row_sums = np.zeros((clusters.cluster_count, 6, 6))
    np.add.at(row_sums, block_rows, correction_blocks.data)
    mean_row_sum = np.einsum('q,qij->ij', weights, row_sums)
    return own - column_mean - weights[:, None, None] * (row_sums - mean_row_sum)

def get_interaction(mesh, clusters, degree):
    """FFT-TFA's operators on its clusters, with each partition's polarization summarized by its polynomial moments up
    to `degree` (a number, or "full")."""
    discretization = mesh.discretization
    matrix_L = get_matrix_L()
    element_shape = discretization.element_lattice_shape
    # Γ = P0 C0⁻¹: the element-level strain per unit polarization.
    element_Gamma = (get_moulinec_suquet_kernel(element_shape, matrix_L).reshape(*element_shape, 6, 6)
                     @ np.linalg.inv(matrix_L))
    basis = get_polynomial_basis(discretization.partition_shape_in_elements, resolve_degree(degree, discretization))
    p = basis.shape[-1]
    kernels = get_moment_kernels(element_Gamma, discretization, basis)
    self_kernels = kernels[0, 0, 0]
    moment_kernels = to_fourier_blocks(kernels.reshape(-1, 6 * p), discretization.partition_lattice_shape)

    # Moments ordered partition by partition, then polynomial by polynomial: one row of 6 per polynomial.
    moments = get_cluster_moments(discretization, clusters, basis)
    element_counts = np.bincount(clusters.element_cluster, minlength=clusters.cluster_count)
    rows = (clusters.cluster_partition[:, None] * p + np.arange(p)).ravel()
    columns = np.repeat(np.arange(clusters.cluster_count), p)
    shape = (p * clusters.partition_count, clusters.cluster_count)
    to_moments = scipy.sparse.csr_matrix((moments.ravel(), (rows, columns)), shape=shape)
    from_moments = scipy.sparse.csr_matrix(((moments / element_counts[:, None]).ravel(), (columns, rows)),
                                           shape=shape[::-1])

    # The corrections are applied at every residual and Jacobian product; stored as dense 6 × 6 blocks (BSR), a product
    # reads one column index per block instead of one per entry.
    correction_blocks = get_correction_blocks(mesh, clusters, element_Gamma, moments,
                                              self_kernels).tobsr(blocksize=(6, 6))
    return Interaction(clusters, matrix_L, to_moments, from_moments, moment_kernels, correction_blocks,
                       get_own_strain_change(clusters, moments, self_kernels, correction_blocks))

def get_strain_change(interaction, polarization):
    """FT-2 with FT-4: the strain every cluster takes from every cluster's polarization, ε_a = ε̄ − (this).

    Far field: every partition's polarization moments, by FFT through the moment kernels, read by each cluster through
    its own moments. Near field: the corrections inside each partition, applied to the polarization's deviation from
    its cell mean (a uniform polarization causes no strain, so it must not reach the corrections). Then less the cell
    mean, so the mean strain stays the imposed one (FT-4)."""
    clusters = interaction.clusters
    partition_moments = (interaction.to_moments @ polarization).reshape(clusters.partition_count, -1)
    far = interaction.from_moments @ apply_on_lattice(interaction.moment_kernels, partition_moments).reshape(-1, 6)
    near = apply_sparse(interaction.correction_blocks, polarization - clusters.average(polarization))
    change = far + near
    return change - clusters.average(change)

## ------- Solver (FT-5, FT-6) ------- ##

class FFTTFASolver(Solver):
    """FFT-TFA: Newton on the cluster strains. Its residual is a strain error, like TFA's, judged by its volume average
    over the clusters; with the matrix as reference medium it is in matrix units already (FT-5)."""
    def __init__(self, interaction):
        super().__init__(interaction.clusters, interaction.clusters.cluster_count)
        self.interaction = interaction

    def evaluate(self, strain, macro_strain, plastic_history):
        """FT-5: r = ε − ε̄ + (FT-2's strain change), with each cluster's polarization (FT-1)
        τ_a = (L_a − C0) ε_a − L_a μ_a, which is −L_m μ_a in a matrix cluster, and its eigenstrain μ_a from the radial
        return at its strain."""
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

        inverse_blocks = np.linalg.inv(np.eye(6) + interaction.own_strain_change @ T)      # FT-7: J_aa = I + G_aa T_a
        return solve_by_gmres(apply_jacobian, current.residual,
                              apply_preconditioner=lambda vector: apply_blocks(inverse_blocks, vector))
