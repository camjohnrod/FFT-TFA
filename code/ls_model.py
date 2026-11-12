"""The Lippmann–Schwinger (LS) equation (notes/fft_tfa_lecture.md section 3 and appendix B, equations LS-*), and Spectral LS, which solves it.

The LS equation replaces the composite by a homogeneous reference medium of stiffness C0, loaded by a polarization
where the material differs from it. The strain the polarization causes is a convolution with a kernel P0, so on a
regular lattice it is one FFT, a 6 × 6 product at every frequency and an inverse FFT (apply_on_lattice). Spectral LS
(Moulinec–Suquet) solves the equation on the elements by Newton, each correction by conjugate gradients. FFT-TFA
(fft_tfa_model) uses this file's kernel and apply_on_lattice.

Translation-invariant operators are stored as one block per lattice frequency, rfftn's non-redundant half: shape
(N_z, N, N//2 + 1, b, b), lattices indexed [z, y, x] and square in-plane; b = 6 here, 6 per polynomial in FFT-TFA.
"""

import numpy as np
import scipy.fft

import config
from cell import cluster_by_phase, get_clusters, get_inclusion_L, get_matrix_L, get_mesh
from material import get_algorithmic_stiffness, get_unloaded_plastic_state, radial_return
from newton import Evaluation, Solver, SolverDidNotConverge, apply_blocks

## ------- Convolution on a Lattice ------- ##

def apply_on_lattice(fourier_blocks, field):
    """A translation-invariant operator, given by its b × b block at every lattice frequency (rfftn's half spectrum),
    applied to a field of b values per lattice point (rows of b, lattice point by lattice point): transform, multiply
    frequency by frequency, transform back. Only axes longer than one are transformed, as read from the operator's
    shape: a transform of length one is the identity, but costs time."""
    lattice_shape = (fourier_blocks.shape[0], fourier_blocks.shape[1], fourier_blocks.shape[1])
    block = fourier_blocks.shape[-1]
    axes = tuple(axis for axis in (0, 1) if lattice_shape[axis] > 1) + (2,)
    transformed_shape = tuple(lattice_shape[axis] for axis in axes)
    field_transformed = scipy.fft.rfftn(field.reshape(*lattice_shape, block), axes=axes, workers=config.thread_count)
    result_transformed = np.einsum('...ij,...j->...i', fourier_blocks, field_transformed)
    return scipy.fft.irfftn(result_transformed, s=transformed_shape, axes=axes,
                            workers=config.thread_count).reshape(-1, block)

## ------- Reference Stiffness ------- ##

def get_spectral_C0(name):
    """Spectral LS's reference stiffness by name (config.py). Its solution does not depend on it (its P0 is a
    projection; Schneider 2021, section 2), so it is a solver setting: "midpoint" is Moulinec–Suquet's rule (1998,
    section 3.1), the phases' Lamé constants averaged, which for two isotropic phases is (L_m + L_f) / 2."""
    if name == "midpoint":
        return (get_matrix_L() + get_inclusion_L()) / 2
    return get_matrix_L()

## ------- Moulinec–Suquet Kernel P0 ------- ##

def to_fourier_blocks(offset_blocks, lattice_shape):
    """An operator's b × b blocks by lattice offset (stacked as offsets × b rows of b) to its blocks by frequency
    (rfftn's half spectrum)."""
    block = offset_blocks.shape[-1]
    return np.fft.rfftn(offset_blocks.reshape(*lattice_shape, block, block), axes=(0, 1, 2))

def to_offset_blocks(transformed):
    """The inverse of to_fourier_blocks: an operator's blocks by frequency to its blocks by lattice offset, stacked as
    rows of 6 (offsets × 6, 6)."""
    lattice_shape = (transformed.shape[0], transformed.shape[1], transformed.shape[1])
    return np.fft.irfftn(transformed, s=lattice_shape, axes=(0, 1, 2)).reshape(-1, 6)

def get_direction_strain_matrix(directions):
    """B(n), shape (..., 6, 3): the engineering strain of a displacement u varying along the unit direction n, per unit
    gradient, ε = B(n) u."""
    n1, n2, n3 = np.moveaxis(directions, -1, 0)
    zero = np.zeros_like(n1)
    rows = [(n1, zero, zero), (zero, n2, zero), (zero, zero, n3), (n2, n1, zero), (zero, n3, n2), (n3, zero, n1)]
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)

def get_moulinec_suquet_kernel(lattice_shape, C0):
    """The Moulinec–Suquet kernel on a lattice of the given shape [z, y, x] over the cubic cell, as offset blocks: at
    every frequency ξ, P̂0(ξ) = Γ̂0(ξ) C0 (LS-21), with the continuum Green operator Γ̂0 = B (Bᵀ C0 B)⁻¹ Bᵀ and
    B = B(ξ / |ξ|). P̂0(0) = 0 (LS-9).

    Along an axis with an even count N, Γ̂0 is not even in a component equal to N / 2, so it has no real-valued meaning
    there; P̂0 is then the identity, Moulinec–Suquet's Γ̂0 = C0⁻¹ (1998, section 2.4.2), which makes the stress at those
    frequencies zero. The cell is a cube, so frequencies in units of 2π / side compare across axes whatever each axis's
    count."""
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
    return to_offset_blocks(P0_transformed)

def get_P0_transformed(lattice_shape, C0):
    """P̂0 at every frequency of a lattice of the given shape: the Moulinec–Suquet kernel, as Spectral LS applies it."""
    return to_fourier_blocks(get_moulinec_suquet_kernel(lattice_shape, C0), lattice_shape)

## ------- Residual in Matrix Units (LS-24) ------- ##

def get_matrix_units_operator(P0_transformed, C0, matrix_L):
    """M̂_m⁻¹ per frequency, which turns Spectral LS's residual into matrix units (LS-24, lecture appendix B.3).

    The LS residual r is a strain of the reference medium, so its size depends on C0. Convergence is judged instead on
    the strain correction an all-matrix cell would need to cancel r, M_m⁻¹ r, with M_m the Newton Jacobian of a
    homogeneous cell of matrix stiffness C_m (LS-20 with L_t = C_m everywhere):
      M̂_m(ξ) = I − P̂0(ξ) (I − C0⁻¹ C_m),
    a fixed Fourier operator like P0, so the tolerance means the same whatever C0."""
    matrix_cell_jacobian = np.eye(6) - P0_transformed @ (np.eye(6) - np.linalg.solve(C0, matrix_L))
    return np.linalg.inv(matrix_cell_jacobian)

## ------- Spectral LS ------- ##

def get_element_clusters(elements):
    """Clusters of one element each, on a mesh of `elements` per side: Spectral LS's material states."""
    mesh = get_mesh(elements, elements)
    return get_clusters(mesh, cluster_by_phase(mesh))

class SpectralLSSolver(Solver):
    """Spectral LS: the LS equation LS-18 on the elements, each element its own partition and cluster. Newton on the
    element strains, each correction by conjugate gradients (LS-28), convergence judged in matrix units (LS-24).
    clusters: one per element (get_element_clusters); C0: the reference stiffness's name, "midpoint" or "matrix"."""
    def __init__(self, clusters, C0):
        super().__init__(clusters, clusters.cluster_count)
        if clusters.cluster_count != clusters.mesh.element_count:
            raise ValueError("Spectral LS needs exactly one cluster per element.")
        self.C0 = get_spectral_C0(C0)
        self.P0_transformed = get_P0_transformed(clusters.mesh.element_shape, self.C0)
        self.to_matrix_units = get_matrix_units_operator(self.P0_transformed, self.C0, get_matrix_L())

    def measure_residual(self, residual):
        """The residual in matrix units, M_m⁻¹ r (LS-24), which convergence and the line search judge."""
        return apply_on_lattice(self.to_matrix_units, residual)

    def evaluate(self, strain, macro_strain, plastic_history):
        """The LS-18 residual r = ε − ε̄ − P0 μ*, with μ* = ε − C0⁻¹ σ the equivalent eigenstrain (LS-6), and the
        candidate plastic state and stress."""
        plastic_state, stress = radial_return(strain, self.clusters.materials, plastic_history)
        equivalent_eigenstrain = strain - np.linalg.solve(self.C0, stress.T).T
        residual = strain - macro_strain - apply_on_lattice(self.P0_transformed, equivalent_eigenstrain)
        return Evaluation(strain, plastic_state, stress, residual)

    def get_newton_correction(self, current, plastic_history):
        """Solves J δε = −r (LS-20), J v = v − P0 (D v) with D = I − C0⁻¹ L_t, by conjugate gradients in the C0 inner
        product ⟨u, v⟩ = Σ uᵀ C0 v, in which J is symmetric and positive definite on the compatible strains that r and
        every correction lie in (LS-28). It stops once ‖r + J δε‖ is below krylov_tolerance ‖r‖. Returns δε and the
        number of CG iterations."""
        tangent = get_algorithmic_stiffness(current.strain, self.clusters.materials, plastic_history)
        D_blocks = np.eye(6) - np.linalg.solve(self.C0, tangent)

        def inner_product(u, v):
            return np.sum((u @ self.C0) * v)

        correction = np.zeros_like(current.residual)
        linear_residual = -current.residual                       # −r − J δε, at δε = 0
        tolerance = config.krylov_tolerance * np.linalg.norm(linear_residual)
        direction = linear_residual
        residual_product = inner_product(linear_residual, linear_residual)
        max_iterations = (10 * linear_residual.size if config.cg_max_iterations is None
                          else config.cg_max_iterations)
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
                                   "iterations. Loosen krylov_tolerance or raise cg_max_iterations")

def get_elastic_strain_concentration(clusters, C0):
    """The strain of every element per unit macro strain in elasticity, elements × 6 × 6 (column k for the k-th unit
    macro strain): the strain concentration tensor that k-means clusters on (Liu–Bessa–Liu 2016, section 2.2.1). Each
    column is one Spectral LS load step from zero, with every phase elastic."""
    materials = clusters.materials
    elastic_materials = materials._replace(yield_stress=np.full(len(materials.yield_stress), np.inf))
    solver = SpectralLSSolver(clusters._replace(materials=elastic_materials), C0)
    unloaded = get_unloaded_plastic_state(clusters.cluster_count)
    columns = []
    for unit_strain in np.eye(6):
        solver.start_over()
        columns.append(solver.solve_step(unit_strain, unloaded).strain)
    return np.stack(columns, axis=-1)
