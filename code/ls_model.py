# The Lippmann–Schwinger (LS) equation of notes/2_ls_model.md: partitions interact only through a homogeneous reference
# medium C0, so the whole nonlocal operator is one FFT convolution with the kernel P0 (LS-17). Spectral LS solves it on
# the element lattice by Newton–CG, with one partition and one cluster per element. FFT-TFA (fft_tfa_model) uses the
# kernel, its partition average, apply_on_lattice and the block helpers.
#
# These operators are translation invariant, so they are stored as one 6 × 6 block per lattice frequency, rfftn's
# non-redundant half (shape (N_z, N, N//2 + 1, 6, 6), lattices indexed [z, y, x] and square in-plane), and applied by
# FFT (apply_on_lattice).

import numpy as np
import scipy.fft
import scipy.sparse

import config
from material import apply_blocks, get_algorithmic_stiffness, get_plastic_eigenstrain, get_unloaded_plastic_state
from newton import Evaluation, Solver, SolverDidNotConverge

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

def get_spectral_C0(matrix_L, inclusion_L):
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

def get_moulinec_suquet_kernel(lattice_shape, C0):
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
    P0_transformed = B @ np.linalg.inv(B_transposed @ C0 @ B) @ B_transposed @ C0
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
    partition_offsets = k * np.arange(discretization.partitions_per_side)
    partition_offsets_z = k_z * np.arange(discretization.partitions_along_z)
    partition_P0 = np.zeros((*discretization.partition_lattice_shape, 6, 6))
    for d_k in range(1 - k_z, k_z):
        for d_j in range(1 - k, k):
            for d_i in range(1 - k, k):
                weight = (k_z - abs(d_k)) * (k - abs(d_j)) * (k - abs(d_i)) / (k_z * k**2)
                partition_P0 += weight * element_P0[np.ix_((partition_offsets_z + d_k) % element_count_z,
                                                           (partition_offsets + d_j) % element_side,
                                                           (partition_offsets + d_i) % element_side)]
    return partition_P0.reshape(-1, 6)

def get_P0_transformed(discretization, C0):
    # P̂0 on the discretization's partition lattice: the Moulinec–Suquet kernel on its element grid, averaged over its
    # partitions, as SCA averages its fine-grid operator over clusters (Liu et al. 2016).
    element_P0 = get_moulinec_suquet_kernel(discretization.element_lattice_shape, C0)
    return get_transformed(average_over_partitions(element_P0, discretization), discretization.partition_lattice_shape)

## ------- Residual in Matrix Units (LS-24) ------- ##

def get_residual_in_matrix_units(P0_transformed, C0, matrix_L):
    # LS-24: the LS residual r is a strain of the reference medium, so its size depends on C0. Convergence is judged
    # instead on the strain correction an all-matrix cell would need to cancel r, M_m⁻¹ r, with M_m the Newton
    # Jacobian of a homogeneous cell of matrix stiffness C_m (LS-20 with L_t = C_m everywhere):
    #   M̂_m(ξ) = I − P̂0(ξ) (I − C0⁻¹ C_m),
    # a fixed Fourier operator like P0, so the tolerance means the same whatever C0. Returns M̂_m⁻¹ per frequency.
    matrix_cell_jacobian = np.eye(6) - P0_transformed @ (np.eye(6) - np.linalg.solve(C0, matrix_L))
    return np.linalg.inv(matrix_cell_jacobian)

## ------- Block Helpers ------- ##

def assemble_blocks(blocks, block_rows, block_columns, shape_in_blocks):
    # A sparse matrix of 6 × 6 blocks, each block at its block row and column, blocks at the same position summed.
    rows = np.broadcast_to(6 * block_rows[:, None, None] + np.arange(6)[:, None], blocks.shape)
    columns = np.broadcast_to(6 * block_columns[:, None, None] + np.arange(6), blocks.shape)
    return scipy.sparse.coo_matrix((blocks.ravel(), (rows.ravel(), columns.ravel())),
                                   shape=(6 * shape_in_blocks[0], 6 * shape_in_blocks[1])).tocsr()

def apply_sparse(matrix, rows):
    # A sparse matrix on stacked rows of 6.
    return (matrix @ rows.reshape(-1)).reshape(-1, 6)

## ------- Spectral LS ------- ##

class SpectralLSSolver(Solver):
    # Spectral LS: LS-18 with one partition per element (per element column), each partition one cluster, so a
    # partition's stress is its cluster's at the partition strain. Newton on the partition strains, each correction by
    # conjugate gradients (LS-28), convergence judged in matrix units (LS-24).
    def __init__(self, lattice, matrix_L):
        super().__init__(lattice.clusters, lattice.clusters.partition_count)
        self.C0 = lattice.C0
        self.P0_transformed = lattice.P0_transformed
        if self.clusters.cluster_count != self.clusters.partition_count:
            raise ValueError("Spectral LS needs exactly one cluster per partition.")
        self.residual_in_matrix_units = get_residual_in_matrix_units(self.P0_transformed, self.C0, matrix_L)

    def measure_residual(self, residual):
        return apply_on_lattice(self.residual_in_matrix_units, residual)

    def evaluate(self, strain, macro_strain, plastic_history):
        # The LS-18 residual r = ε − ε̄ − P0 μ*, with μ* = ε − C0⁻¹ σ the equivalent eigenstrain (LS-6), and the
        # candidate plastic state and stress.
        plastic_state, stress = get_plastic_eigenstrain(strain, self.clusters.materials, plastic_history)
        equivalent_eigenstrain = strain - np.linalg.solve(self.C0, stress.T).T
        b = np.tile(macro_strain, (len(strain), 1))
        residual = strain - b - apply_on_lattice(self.P0_transformed, equivalent_eigenstrain)
        return Evaluation(strain, plastic_state, stress, residual)

    def get_newton_correction(self, current, plastic_history):
        # Solves J δε = −r (LS-20), J v = v − P0 (D v) with D = I − C0⁻¹ L_t, by conjugate gradients in the C0 inner
        # product ⟨u, v⟩ = Σ uᵀ C0 v, in which J is symmetric and positive definite on the compatible strains that r and
        # every correction lie in (LS-28). It stops once ‖r + J δε‖ is below krylov_tolerance ‖r‖. Returns δε and the
        # number of CG iterations.
        tangent = get_algorithmic_stiffness(current.strain, self.clusters.materials, plastic_history)
        D_blocks = np.eye(6) - np.linalg.solve(self.C0, tangent)

        def inner_product(u, v):
            return np.sum((u @ self.C0) * v)

        correction = np.zeros_like(current.residual)
        linear_residual = -current.residual                       # −r − J δε, at δε = 0
        tolerance = config.krylov_tolerance * np.linalg.norm(linear_residual)
        direction = linear_residual
        residual_product = inner_product(linear_residual, linear_residual)
        max_iterations = (10 * linear_residual.size if config.spectral_ls_cg_max_iterations is None
                          else config.spectral_ls_cg_max_iterations)
        for iteration in range(1, max_iterations + 1):
            jacobian_direction = direction - apply_on_lattice(self.P0_transformed, apply_blocks(D_blocks, direction))
            curvature = inner_product(direction, jacobian_direction)
            if curvature <= 0:
                raise SolverDidNotConverge("CG met a direction of non-positive curvature: LS-28 needs a positive "
                                           "definite tangent, so a yielding phase needs a positive hardening modulus")
            step = residual_product / curvature
            correction += step * direction
            linear_residual = linear_residual - step * jacobian_direction   # not in place: direction may be it
            if np.linalg.norm(linear_residual) <= tolerance:
                return correction, iteration
            next_residual_product = inner_product(linear_residual, linear_residual)
            direction = linear_residual + (next_residual_product / residual_product) * direction
            residual_product = next_residual_product
        raise SolverDidNotConverge(f"CG did not converge inside a Newton iteration within {max_iterations} "
                                   "iterations. Loosen krylov_tolerance or raise spectral_ls_cg_max_iterations")

def get_elastic_strain_concentration(lattice, matrix_L):
    # The strain of every partition of the lattice per unit macrostrain in elasticity, partitions × 6 × 6 (column k for
    # the k-th unit macrostrain): the strain concentration tensor that k-means clusters on (Liu–Bessa–Liu 2016,
    # section 2.2.1). Each column is one Spectral LS load step with every phase elastic.
    materials = lattice.clusters.materials
    elastic_materials = materials._replace(yield_stress=np.full(len(materials.yield_stress), np.inf))
    elastic_lattice = lattice._replace(clusters=lattice.clusters._replace(materials=elastic_materials))
    unloaded = get_unloaded_plastic_state(lattice.clusters.cluster_count)
    return np.stack([SpectralLSSolver(elastic_lattice, matrix_L).solve_step(unit_strain, unloaded).strain
                     for unit_strain in np.eye(6)], axis=-1)
