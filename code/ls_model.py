# The partition-averaged Lippmann–Schwinger (LS) equation of notes/2_ls_model.md, solved by Newton–GMRES. Partitions
# interact only through the homogeneous reference medium C0, so the whole nonlocal operator is one FFT convolution
# with the kernel P0 (LS-17). Two methods solve it with this same code: CP on the partitions, and Spectral LS on the
# element lattice (one partition per element); the lattice is whatever its kernel and clusters describe.
#
# The unknowns are the partition strains ε_A. A partition's stress σ_A comes from its clusters (fe.Clusters), one
# material state per phase: a single-phase partition's cluster takes ε_A; a mixed partition shares ε_A between its
# matrix and inclusion clusters so that both carry the same polarization σ − C0 ε (LS-22), so the polarization is
# constant in every partition.
#
# The reference medium's operators are built here too: C0, and the kernel P0 averaged over partitions (LS-21); and the
# Fourier preconditioner of the Newton solver (LS-23). Both are translation invariant, so they are stored as one 6 × 6
# block per lattice frequency, rfftn's non-redundant half (shape (N_z, N, N//2 + 1, 6, 6), lattices indexed [z, y, x]
# and square in-plane), and applied by FFT (apply_on_lattice).

import numpy as np
import scipy.fft

import config
from material import get_algorithmic_stiffness, get_plastic_eigenstrain
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
    # The stiffness config.reference_stiffness names: volume averages over the clusters.
    if config.reference_stiffness == "homogenized":
        return clusters.average(clusters.materials.L @ E.reshape(-1, 6, 6))    # <L E>
    if config.reference_stiffness == "voigt":
        return clusters.average(clusters.materials.L)                          # <L>
    return matrix_L

def get_cp_reference_L(named_L):
    # CP's C0, part of its model: a fraction of the named stiffness. The same C0 sets the interaction (P0) and how a
    # mixed partition shares its strain (LS-22), so it must stay below every phase's tangent stiffness.
    return config.reference_stiffness_scale * named_L

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

## ------- Fourier Preconditioner (LS-23) ------- ##

def get_fourier_preconditioner(P0_transformed, jacobian_blocks):
    # The best Fourier-diagonal approximation of the Newton Jacobian J v = v − P0 (D v) (LS-20), inverted. P0 is
    # diagonal in Fourier space; D = I − C0⁻¹ L_t is diagonal in real space, and in Fourier space its diagonal is its
    # mean ⟨D⟩ over the lattice. So the Fourier-diagonal operator closest to J (Frobenius norm) is
    #   M̂(ξ) = I − P̂0(ξ) ⟨D⟩,
    # the Jacobian of a homogeneous cell with the mean tangent ⟨L_t⟩, and its inverse is one 6 × 6 inverse per
    # frequency. Built from each method's own kernel and tangents, so the same rule serves CP and Spectral LS.
    return np.linalg.inv(np.eye(6) - P0_transformed @ jacobian_blocks.mean(axis=0))

## ------- Partition Stress from Clusters (LS-22) ------- ##

def get_mixed_partition_clusters(clusters):
    # The matrix and inclusion cluster of every mixed partition (an inclusion cluster directly follows its matrix
    # cluster).
    matrix = np.flatnonzero((clusters.volume_fraction < 1) & (clusters.materials.material_ids == 0))
    return matrix, matrix + 1

def take_rows(rows_tuple, rows):
    return rows_tuple._make(field[rows] for field in rows_tuple)

def get_cluster_strain(strain, clusters, plastic_history, reference_L):
    # Each cluster's strain: its partition's, except in a mixed partition, where c_m ε_m + c_f ε_f = ε_A is shared so
    # that both clusters carry the same polarization σ − C0 ε (LS-22). Newton on ε_f, with ε_m = (ε_A − c_f ε_f) / c_m:
    #   residual  (σ_m − C0 ε_m) − (σ_f − C0 ε_f),   derivative  −[(L_t,f − C0) + (c_f / c_m)(L_t,m − C0)].
    # It has a solution only while C0 is below both clusters' tangents.
    cluster_strain = strain[clusters.partition]
    matrix, inclusion = get_mixed_partition_clusters(clusters)
    if len(matrix) == 0:
        return cluster_strain
    c_m, c_f = clusters.volume_fraction[matrix, None], clusters.volume_fraction[inclusion, None]
    matrix_materials = take_rows(clusters.materials, matrix)
    inclusion_materials = take_rows(clusters.materials, inclusion)
    matrix_history = take_rows(plastic_history, matrix)
    inclusion_history = take_rows(plastic_history, inclusion)
    partition_strain = cluster_strain[matrix]
    inclusion_strain = partition_strain.copy()
    for _ in range(config.mixed_partition_max_iterations):
        matrix_strain = (partition_strain - c_f * inclusion_strain) / c_m
        _, matrix_stress = get_plastic_eigenstrain(matrix_strain, matrix_materials, matrix_history)
        _, inclusion_stress = get_plastic_eigenstrain(inclusion_strain, inclusion_materials, inclusion_history)
        polarization_difference = ((matrix_stress - matrix_strain @ reference_L.T)
                                   - (inclusion_stress - inclusion_strain @ reference_L.T))
        stress_scale = max(np.abs(matrix_stress).max(), np.abs(inclusion_stress).max())
        if np.abs(polarization_difference).max() <= config.mixed_partition_tolerance * stress_scale:
            break
        jacobian = ((get_algorithmic_stiffness(inclusion_strain, inclusion_materials, inclusion_history) - reference_L)
                    + (c_f / c_m)[:, :, None] * (get_algorithmic_stiffness(matrix_strain, matrix_materials,
                                                                            matrix_history) - reference_L))
        inclusion_strain = inclusion_strain + np.linalg.solve(jacobian, polarization_difference[..., None])[..., 0]
    else:
        raise SolverDidNotConverge("the equal-polarization split of a mixed partition did not converge: C0 may be "
                                   "above a cluster's tangent stiffness (lower config.reference_stiffness_scale)")
    cluster_strain[matrix], cluster_strain[inclusion] = matrix_strain, inclusion_strain
    return cluster_strain

def get_partition_stress(cluster_stress, clusters):
    # σ_A = Σ c σ over the partition's clusters.
    stress = np.zeros((clusters.partition_count, 6))
    np.add.at(stress, clusters.partition, clusters.volume_fraction[:, None] * cluster_stress)
    return stress

def check_reference_below_tangents(mixed_cluster_tangents, reference_L):
    # LS-22 has a solution only while C0 is below the tangent of both clusters of every mixed partition, i.e. while
    # L_t − C0 is positive definite there. Otherwise its tangent below is singular and GMRES fails, so stop with the
    # cause instead.
    differences = mixed_cluster_tangents - reference_L
    lowest_eigenvalues = np.linalg.eigvalsh((differences + np.swapaxes(differences, -1, -2)) / 2)[:, 0]
    if np.any(lowest_eigenvalues <= 0):
        raise SolverDidNotConverge(
            f"C0 is not below the tangent stiffness of {np.count_nonzero(lowest_eigenvalues <= 0)} clusters in mixed "
            "partitions, so their equal-polarization split (LS-22) has no solution: lower "
            "config.reference_stiffness_scale (a yielding phase's shear tangent is about G H / (3G + H))")

def get_partition_tangent(cluster_strain, clusters, plastic_history, reference_L):
    # dσ_A/dε_A: a single-phase partition's cluster tangent L_t. For a mixed partition, differentiating LS-22 gives
    #   dε_f = D_f dε_A,  D_f = [(L_t,f − C0) + (c_f / c_m)(L_t,m − C0)]⁻¹ (L_t,m − C0) / c_m,
    #   dε_m = D_m dε_A,  D_m = (I − c_f D_f) / c_m,     and the tangent c_m L_t,m D_m + c_f L_t,f D_f.
    cluster_tangent = get_algorithmic_stiffness(cluster_strain, clusters.materials, plastic_history)
    tangent = np.zeros((clusters.partition_count, 6, 6))
    tangent[clusters.partition] = cluster_tangent
    matrix, inclusion = get_mixed_partition_clusters(clusters)
    if len(matrix) > 0:
        c_m = clusters.volume_fraction[matrix, None, None]
        c_f = clusters.volume_fraction[inclusion, None, None]
        L_m, L_f = cluster_tangent[matrix], cluster_tangent[inclusion]
        check_reference_below_tangents(np.concatenate([L_m, L_f]), reference_L)
        D_f = np.linalg.solve((L_f - reference_L) + (c_f / c_m) * (L_m - reference_L), (L_m - reference_L) / c_m)
        D_m = (np.eye(6) - c_f * D_f) / c_m
        tangent[clusters.partition[matrix]] = c_m * L_m @ D_m + c_f * L_f @ D_f
    return tangent

## ------- Newton–GMRES Solver ------- ##

class LSSolver:
    # One LS method on its lattice (run.Lattice), as a load-path solver: solve_step(macro_strain, plastic_history)
    # solves one load step. Each step starts from the previous step's converged strain shifted by the macrostrain
    # increment. With use_fourier_preconditioner, GMRES is preconditioned by LS-23.
    def __init__(self, lattice, matrix_L, use_fourier_preconditioner):
        self.clusters = lattice.clusters
        self.reference_L = lattice.reference_L                  # C0
        self.P0_transformed = lattice.P0_transformed
        self.use_fourier_preconditioner = use_fourier_preconditioner
        # LS-24: the residual r is a strain of the reference medium, so its size depends on C0. Convergence is judged
        # instead on the strain correction an all-matrix cell would need to cancel r, M_m⁻¹ r, with M_m the Newton
        # Jacobian of a homogeneous cell of matrix stiffness C_m (LS-20 with L_t = C_m everywhere):
        #   M̂_m(ξ) = I − P̂0(ξ) (I − C0⁻¹ C_m),
        # a fixed Fourier operator like P0, so the tolerance means the same whatever C0.
        matrix_cell_jacobian = np.eye(6) - self.P0_transformed @ (np.eye(6) - np.linalg.solve(self.reference_L,
                                                                                                matrix_L))
        self.residual_in_matrix_units = np.linalg.inv(matrix_cell_jacobian)
        self.previous_strain = np.zeros((self.clusters.partition_count, 6))
        self.previous_macro_strain = np.zeros(6)

    def evaluate(self, strain, macro_strain, plastic_history):
        # The LS-18 residual r = ε − ε̄ − P0 μ*, with μ* = ε − C0⁻¹ σ the equivalent eigenstrain (LS-6), and the
        # clusters' candidate plastic state and stress.
        cluster_strain = get_cluster_strain(strain, self.clusters, plastic_history, self.reference_L)
        plastic_state, cluster_stress = get_plastic_eigenstrain(cluster_strain, self.clusters.materials,
                                                                plastic_history)
        partition_stress = get_partition_stress(cluster_stress, self.clusters)
        equivalent_eigenstrain = strain - np.linalg.solve(self.reference_L, partition_stress.T).T
        b = np.tile(macro_strain, (len(strain), 1))
        residual = strain - b - apply_on_lattice(self.P0_transformed, equivalent_eigenstrain)
        return Evaluation(strain, plastic_state, cluster_stress, residual)

    def get_newton_correction(self, current, plastic_history):
        # Solves the LS-20 system J δε = −r, J v = v − P0 (D v) with D = I − C0⁻¹ L_t and L_t each partition's
        # tangent, by GMRES. The Fourier preconditioner M⁻¹ acts on the right: GMRES solves J M⁻¹ y = −r and
        # δε = M⁻¹ y, so it still measures the residual of J δε = −r, and M⁻¹ changes only how many iterations that
        # takes, not the correction.
        cluster_strain = get_cluster_strain(current.strain, self.clusters, plastic_history, self.reference_L)
        partition_tangent = get_partition_tangent(cluster_strain, self.clusters, plastic_history, self.reference_L)
        jacobian_blocks = np.eye(6) - np.linalg.solve(self.reference_L, partition_tangent)

        def apply_jacobian(vector):
            return vector - apply_on_lattice(self.P0_transformed, np.einsum('pij,pj->pi', jacobian_blocks, vector))

        if not self.use_fourier_preconditioner:
            return solve_newton_correction(apply_jacobian, current.residual)

        preconditioner = get_fourier_preconditioner(self.P0_transformed, jacobian_blocks)

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
