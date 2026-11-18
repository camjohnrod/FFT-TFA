"""SCA, self-consistent clustering analysis (Liu, Bessa & Liu 2016; notes/fft_tfa_lecture.md appendix E, equations
SC-*).

SCA solves the clustered equation of the lecture's section 5, the one FFT-TFA approximates: every cluster's strain from
every cluster's polarization through the averaged kernel Γ̂(a←b), with nothing summarized. Its clusters are k-means
over the whole cell (one partition), so a cluster's elements may lie anywhere. Their interaction D is a dense
(6 × clusters)² matrix, built once by FFT on the elements (Liu–Bessa–Liu, eq. 27). Each load step solves the
equation in increments from the last accepted step, as Liu–Bessa–Liu write it, r = Δε − Δε̄ + D (Δσ − C0 Δε) = 0
(SC-1), by Newton–GMRES, as the other methods do.

The reference stiffness C0 changes SCA's answer, unlike Spectral LS's (Schneider 2019, §2.2), so it is a setting:
  "matrix"           C0 = L_m, fixed: the clustered equation itself, FFT-TFA's on one partition;
  "self-consistent"  C0 isotropic, its two Lamé constants refitted to the macroscopic stress and strain increments at
                     every load step, and the step solved again until they settle (Liu–Bessa–Liu, §2.3.2 and Box II).
                     Γ0 of an isotropic medium is a sum of fixed kernels times functions of the two constants (their
                     eq. 25), so D is built once per kernel and assembled at each refit.
"""

from typing import NamedTuple
import numpy as np
import scipy.fft
import scipy.sparse

import config
from cell import get_matrix_L
from ls_model import get_element_kernel
from material import get_algorithmic_stiffness, radial_return
from newton import (Evaluation, Solver, SolverDidNotConverge, apply_blocks, get_free_compliance, get_free_strain,
                    solve_by_gmres, stress_norm_weights)

## ------- Isotropic Reference Media ------- ##

def get_isotropic_stiffness(lame_lambda, mu):
    """The stiffness of an isotropic medium with Lamé constants λ and μ, in Voigt form with engineering shear."""
    trace = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    return lame_lambda * np.outer(trace, trace) + np.diag([2 * mu, 2 * mu, 2 * mu, mu, mu, mu])

def get_matrix_lame_constants():
    """The matrix's λ and μ, read off its stiffness."""
    L = get_matrix_L()
    return L[0, 1], L[3, 3]

def get_kernel_coefficients(lame_lambda, mu):
    """How Γ0 of an isotropic medium depends on its Lamé constants: Γ̂0 = Σ_k coefficient_k(λ, μ) Φ̂_k, with the kernels
    Φ̂_k fixed (get_kernel_parts). The first two are Liu–Bessa–Liu's eq. 25; the third covers the highest frequencies of
    Moulinec–Suquet's kernel, where P̂0 = I and Γ̂0 = C0⁻¹ = K / 2μ + J / (3λ + 2μ) (K, J the deviatoric and spherical
    projectors), whose spherical part is not of the first two's form. It is zero for Willot's kernel."""
    return np.array([1 / mu, (lame_lambda + mu) / (mu * (lame_lambda + 2 * mu)), 1 / (3 * lame_lambda + 2 * mu)])

def get_kernel_parts(lattice_shape):
    """The fixed kernels Φ̂_k of get_kernel_coefficients, per frequency: Γ̂0 is computed for three isotropic media and
    the three kernels solved for, then checked on a fourth medium. Parts that are zero everywhere are left out."""
    samples = [(0.0, 1.0), (1.0, 1.0), (1.0, 2.0)]
    coefficients = np.array([get_kernel_coefficients(*sample) for sample in samples])
    kernels = np.array([get_element_kernel(lattice_shape, C0) @ np.linalg.inv(C0)
                        for C0 in map(lambda sample: get_isotropic_stiffness(*sample), samples)])
    parts = np.einsum('ks,s...->k...', np.linalg.inv(coefficients), kernels)
    check_lambda, check_mu = 2.0, 0.5
    C0 = get_isotropic_stiffness(check_lambda, check_mu)
    reconstructed = np.einsum('k,k...->...', get_kernel_coefficients(check_lambda, check_mu), parts)
    exact = get_element_kernel(lattice_shape, C0) @ np.linalg.inv(C0)
    if np.abs(reconstructed - exact).max() > config.sca_kernel_parts_tolerance * np.abs(exact).max():
        raise ValueError("the element kernel is not a combination of the three isotropic kernels SCA assumes")
    return [(k, part) for k, part in enumerate(parts) if np.abs(part).max() > 0]

def get_stored_matrix_count(C0):
    """How many dense (6 × clusters)² matrices SCA stores: D alone for a fixed C0; for a self-consistent one also D's
    parts, one per kernel of get_kernel_parts, two for Willot's kernel and three for Moulinec–Suquet's."""
    if C0 != "self-consistent":
        return 1
    return 1 + (3 if config.element_kernel == "Moulinec-Suquet" else 2)

## ------- Interaction (SC-2) ------- ##

def get_interaction(clusters, Gamma_transformed):
    """SC-2, Liu–Bessa–Liu's eq. 27: D, the strain averaged over every cluster a per unit polarization on every
    cluster b (Γ̂(a←b) of the lecture's section 5), as a dense (6 × clusters)² matrix, a's six rows by b's six
    columns, for the kernel Γ̂0 given per frequency (real 6 × 6 blocks). One source cluster and one polarization
    component j at a time: column j of Γ0 applied to the cluster's indicator by FFT on the elements, which gives a
    strain field (6 numbers per element) per unit polarization along j, then averaged over every cluster.
    Γ̂0 is zero at frequency zero, so each column of D averages to zero over the cell: the mean strain stays the imposed
    one."""
    mesh, cluster_count = clusters.mesh, clusters.cluster_count
    shape = mesh.element_shape
    axes = tuple(axis for axis in range(3) if shape[axis] > 1)       # a transform of length one is the identity
    element_counts = clusters.element_counts
    to_cluster_averages = scipy.sparse.csr_matrix(
        (1 / element_counts[clusters.element_cluster], (clusters.element_cluster, np.arange(mesh.element_count))),
        shape=(cluster_count, mesh.element_count))
    D = np.empty((cluster_count, 6, cluster_count, 6))
    for source in range(cluster_count):
        indicator = (clusters.element_cluster == source).reshape(shape).astype(float)
        indicator_transformed = scipy.fft.rfftn(indicator, axes=axes, workers=config.thread_count)
        for column in range(6):
            strain = scipy.fft.irfftn(Gamma_transformed[..., :, column] * indicator_transformed[..., None],
                                      s=[shape[axis] for axis in axes], axes=axes, workers=config.thread_count)
            D[:, :, source, column] = to_cluster_averages @ strain.reshape(-1, 6)
    return D.reshape(6 * cluster_count, 6 * cluster_count)

def get_strain_change(D, polarization):
    """The strain every cluster takes from every cluster's polarization, ε_a = ε̄ − (this)."""
    return (D @ polarization.reshape(-1)).reshape(-1, 6)

def fit_reference(macro_strain_increment, macro_stress_increment, previous_lambda):
    """SC-3: the isotropic C0 closest to the load step's macroscopic response, Δσ̄ ≈ C0(λ, μ) Δε̄ in least squares in
    the tensor norm (Liu–Bessa–Liu, eqs. 41–44). C0 Δε̄ = λ tr(Δε̄) 1 + 2μ Δε̄ is linear in (λ, μ): two normal equations.
    Only the volume change tr(Δε̄) informs λ. Under pure shear it is zero, λ drops out and keeps its previous value, as
    in their §2.3.2. When the fit comes out indefinite, the volume change did not determine λ either (a random cell,
    not exactly isotropic, contracts slightly under shear, and that contraction is no bulk response): λ then keeps its
    previous value too, and μ is fitted alone."""
    trace_term = macro_strain_increment[:3].sum() * np.array([1, 1, 1, 0, 0, 0], dtype=float)
    shear_term = macro_strain_increment * np.array([2, 2, 2, 1, 1, 1])
    weighted = lambda u, v: np.sum(stress_norm_weights * u * v)
    is_pure_shear = (abs(macro_strain_increment[:3].sum())
                     < config.sca_pure_shear_tolerance * np.sqrt(weighted(shear_term, shear_term)))
    if is_pure_shear:
        return previous_lambda, weighted(shear_term, macro_stress_increment) / weighted(shear_term, shear_term)
    normal_matrix = np.array([[weighted(trace_term, trace_term), weighted(trace_term, shear_term)],
                              [weighted(trace_term, shear_term), weighted(shear_term, shear_term)]])
    right_hand_side = np.array([weighted(trace_term, macro_stress_increment),
                                weighted(shear_term, macro_stress_increment)])
    lame_lambda, mu = np.linalg.solve(normal_matrix, right_hand_side)
    if mu > 0 and 3 * lame_lambda + 2 * mu > 0:
        return lame_lambda, mu
    stress_less_volumetric = macro_stress_increment - previous_lambda * trace_term
    return previous_lambda, weighted(shear_term, stress_less_volumetric) / weighted(shear_term, shear_term)

## ------- Solver (SC-1, SC-4) ------- ##

class StepStart(NamedTuple):
    """The last accepted load step, which a step's increments start from (subscript n in SC-1)."""
    strain: np.ndarray
    stress: np.ndarray
    macro_strain: np.ndarray

class SCASolver(Solver):
    """SCA: Newton on the cluster strains, each correction by GMRES. clusters: k-means over the whole cell; C0: the
    reference stiffness's name, "matrix" or "self-consistent"."""
    def __init__(self, clusters, C0):
        super().__init__(clusters, clusters.cluster_count)
        self.is_self_consistent = C0 == "self-consistent"
        if self.is_self_consistent:
            self.parts = [(k, get_interaction(clusters, part))
                          for k, part in get_kernel_parts(clusters.mesh.element_shape)]
            self.D = np.empty_like(self.parts[0][1])
            self.set_reference(*get_matrix_lame_constants())
        else:
            self.C0 = get_matrix_L()
            self.free_compliance = get_free_compliance(self.C0, self.stress_free)
            self.D = get_interaction(clusters, get_element_kernel(clusters.mesh.element_shape, self.C0)
                                     @ np.linalg.inv(self.C0))
        self.previous_stress = np.zeros((clusters.cluster_count, 6))
        self.step_start = None                        # set by solve_step

    def set_reference(self, lame_lambda, mu):
        """C0 = the isotropic medium (λ, μ), and D assembled from its parts, sca_assembly_rows rows at a time so that no
        full-size temporary is made."""
        self.lame_lambda, self.mu = lame_lambda, mu
        self.C0 = get_isotropic_stiffness(lame_lambda, mu)
        self.free_compliance = get_free_compliance(self.C0, self.stress_free)
        coefficients = get_kernel_coefficients(lame_lambda, mu)
        for rows in np.array_split(np.arange(len(self.D)), max(1, len(self.D) // config.sca_assembly_rows)):
            self.D[rows] = sum(coefficients[k] * part[rows] for k, part in self.parts)

    def start_over(self):
        super().start_over()
        self.previous_stress = np.zeros_like(self.previous_stress)
        if self.is_self_consistent:
            self.set_reference(*get_matrix_lame_constants())

    def evaluate(self, strain, macro_strain, plastic_history):
        """SC-1, in increments from the last accepted step n, as Liu–Bessa–Liu write it (their eq. 28):
        r = Δε − Δε̄ + D (Δσ − C0 Δε), Δε = ε − εₙ, with each cluster's stress from the radial return at its strain. For
        a fixed C0 this is the clustered equation r = ε − ε̄ + D (σ − C0 ε) less its residual at step n; a changing C0
        changes only the step's increment. With stress-free components, Δε̄ is joined by the mean strain increment the
        cell takes there (SF-2, in increments: the mean stress there was zero at step n)."""
        start = self.step_start
        plastic_state, stress = radial_return(strain, self.clusters.materials, plastic_history)
        strain_increment = strain - start.strain
        polarization_increment = (stress - start.stress) - strain_increment @ self.C0
        macro_strain_increment = macro_strain - start.macro_strain
        free_strain_increment = get_free_strain(self.free_compliance, self.C0,
                                                self.clusters.average(polarization_increment), macro_strain_increment)
        residual = (strain_increment - (macro_strain_increment + free_strain_increment)
                    + get_strain_change(self.D, polarization_increment))
        return Evaluation(strain, plastic_state, stress, residual)

    def get_newton_correction(self, current, plastic_history):
        """SC-4: J v = v + D (T v), with T = L_t − C0 and L_t each cluster's algorithmic stiffness; with stress-free
        components, less the mean strain T v frees (SF-2). GMRES solves it with J's diagonal blocks I + D_aa T_a
        inverted (block-Jacobi) on the right, as FFT-TFA's FT-6 and FT-7."""
        clusters, free_compliance = self.clusters, self.free_compliance
        T = get_algorithmic_stiffness(current.strain, clusters.materials, plastic_history) - self.C0

        def apply_jacobian(vector):
            polarization_change = apply_blocks(T, vector)
            return (vector + free_compliance @ clusters.average(polarization_change)
                    + get_strain_change(self.D, polarization_change))

        # D_aa, joined by the cluster's share of the cell times the free compliance (SF-2).
        numbers = np.arange(clusters.cluster_count)
        cell_share = (clusters.share / clusters.partition_count)[:, None, None]
        own_blocks = self.D.reshape(len(numbers), 6, len(numbers), 6)[numbers, :, numbers, :]
        inverse_blocks = np.linalg.inv(np.eye(6) + (own_blocks + cell_share * free_compliance) @ T)
        return solve_by_gmres(apply_jacobian, current.residual,
                              apply_preconditioner=lambda vector: apply_blocks(inverse_blocks, vector))

    def solve_step(self, macro_strain, plastic_history):
        """One load step from the last accepted one. With C0 "self-consistent", Liu–Bessa–Liu's Box II: solve, refit
        (λ, μ) to the step's macroscopic increments (SC-3), and solve again until both change by less than
        sca_reference_tolerance. Each solve starts from the last one's solution, and every solve's iterations count."""
        self.step_start = StepStart(self.previous_strain, self.previous_stress, self.previous_macro_strain)
        newton_iteration_count = krylov_iteration_count = 0
        for _ in range(config.sca_max_reference_updates):
            result = super().solve_step(macro_strain, plastic_history)
            newton_iteration_count += result.newton_iteration_count
            krylov_iteration_count += result.krylov_iteration_count
            if not self.is_self_consistent:
                break
            mean_strain_increment = self.get_mean_strain(result.strain - self.step_start.strain,
                                                         macro_strain - self.step_start.macro_strain)
            lame_lambda, mu = fit_reference(mean_strain_increment,
                                            self.clusters.average(result.stress - self.step_start.stress),
                                            self.lame_lambda)
            if not (mu > 0 and 3 * lame_lambda + 2 * mu > 0):
                raise SolverDidNotConverge(f"the self-consistent reference stiffness came out indefinite "
                                           f"(λ = {lame_lambda:.3e}, μ = {mu:.3e})")
            has_settled = (abs(lame_lambda - self.lame_lambda) <= config.sca_reference_tolerance * abs(self.lame_lambda)
                           and abs(mu - self.mu) <= config.sca_reference_tolerance * self.mu)
            self.set_reference(lame_lambda, mu)
            if has_settled:
                break
        else:
            raise SolverDidNotConverge(f"the self-consistent reference stiffness did not settle within "
                                       f"{config.sca_max_reference_updates} solves of the load step")
        self.previous_stress = result.stress
        return result._replace(newton_iteration_count=newton_iteration_count,
                               krylov_iteration_count=krylov_iteration_count)
