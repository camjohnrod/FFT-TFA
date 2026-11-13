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
  "self-consistent"  C0 = c I, with c refitted to the macroscopic stress and strain increments at every load step, and
                     the step solved again until c settles (Liu–Bessa–Liu, Box II, with Schneider's C0 proportional to
                     the identity, §5.1). Γ0 for c I is Γ0 for I divided by c, so D is built once and rescaled.
"""

from typing import NamedTuple
import numpy as np
import scipy.fft
import scipy.sparse

import config
from cell import get_matrix_L
from ls_model import get_P0_transformed
from material import get_algorithmic_stiffness, radial_return
from newton import Evaluation, Solver, SolverDidNotConverge, apply_blocks, solve_by_gmres, tensor_norm_weights

# The identity tensor I as a stiffness in Voigt form: with engineering shear, I ε has shear components γ / 2.
identity_stiffness = np.diag(tensor_norm_weights)

## ------- Interaction (SC-2) ------- ##

def get_interaction(clusters, C0):
    """SC-2, Liu–Bessa–Liu's eq. 27: D, the strain averaged over every cluster a per unit polarization on every
    cluster b (Γ̂(a←b) of the lecture's section 5), as a dense (6 × clusters)² matrix, a's six rows by b's six
    columns. One source cluster at a time: Γ0 applied to its indicator by FFT on the elements, which gives a 6 × 6
    field (column j: a unit polarization along j), then averaged over every cluster. Γ̂0 is zero at frequency zero, so
    each column of D averages to zero over the cell: the mean strain stays the imposed one."""
    mesh, cluster_count = clusters.mesh, clusters.cluster_count
    shape = mesh.element_shape
    axes = tuple(axis for axis in range(3) if shape[axis] > 1)       # a transform of length one is the identity
    Gamma_transformed = get_P0_transformed(shape, C0) @ np.linalg.inv(C0)          # Γ̂0 = P̂0 C0⁻¹, per frequency
    element_counts = np.bincount(clusters.element_cluster)
    to_cluster_averages = scipy.sparse.csr_matrix(
        (1 / element_counts[clusters.element_cluster], (clusters.element_cluster, np.arange(mesh.element_count))),
        shape=(cluster_count, mesh.element_count))
    D = np.empty((cluster_count, 6, cluster_count, 6))
    for source in range(cluster_count):
        indicator = (clusters.element_cluster == source).reshape(shape).astype(float)
        indicator_transformed = scipy.fft.rfftn(indicator, axes=axes, workers=config.thread_count)
        strain = scipy.fft.irfftn(Gamma_transformed * indicator_transformed[..., None, None],
                                  s=[shape[axis] for axis in axes], axes=axes, workers=config.thread_count)
        D[:, :, source, :] = (to_cluster_averages @ strain.reshape(-1, 36)).reshape(cluster_count, 6, 6)
    return D.reshape(6 * cluster_count, 6 * cluster_count)

def get_strain_change(D, polarization):
    """The strain every cluster takes from every cluster's polarization, ε_a = ε̄ − (this)."""
    return (D @ polarization.reshape(-1)).reshape(-1, 6)

def fit_reference_scale(macro_strain_increment, macro_stress_increment):
    """SC-3: the c for which c I is closest to the load step's macroscopic response, Δσ̄ ≈ c Δε̄ in least squares in the
    tensor norm (Liu–Bessa–Liu, eqs. 41–44, for Schneider's C0 = c I): c = Δσ̄ : Δε̄ / Δε̄ : Δε̄."""
    return (macro_stress_increment @ macro_strain_increment
            / (macro_strain_increment @ identity_stiffness @ macro_strain_increment))

def get_first_reference_scale():
    """The self-consistent c before the first load step: twice the matrix's shear modulus, so that C0 = c I has the
    matrix's shear stiffness."""
    return 2 * get_matrix_L()[3, 3]

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
        self.C0 = identity_stiffness if self.is_self_consistent else get_matrix_L()
        self.D, self.reference_scale = get_interaction(clusters, self.C0), 1.0
        self.previous_stress = np.zeros((clusters.cluster_count, 6))
        self.step_start = None                        # set by solve_step
        if self.is_self_consistent:
            self.set_reference_scale(get_first_reference_scale())

    def set_reference_scale(self, scale):
        """C0 = c I, and D rescaled to match: Γ0 for c I is Γ0 for I divided by c."""
        self.D *= self.reference_scale / scale
        self.reference_scale, self.C0 = scale, scale * identity_stiffness

    def start_over(self):
        super().start_over()
        self.previous_stress = np.zeros_like(self.previous_stress)
        if self.is_self_consistent:
            self.set_reference_scale(get_first_reference_scale())

    def evaluate(self, strain, macro_strain, plastic_history):
        """SC-1, in increments from the last accepted step n, as Liu–Bessa–Liu write it (their eq. 28):
        r = Δε − Δε̄ + D (Δσ − C0 Δε), Δε = ε − εₙ, with each cluster's stress from the radial return at its strain. For
        a fixed C0 this is the clustered equation r = ε − ε̄ + D (σ − C0 ε) less its residual at step n; a changing C0
        changes only the step's increment."""
        start = self.step_start
        plastic_state, stress = radial_return(strain, self.clusters.materials, plastic_history)
        strain_increment = strain - start.strain
        polarization_increment = (stress - start.stress) - strain_increment @ self.C0
        residual = (strain_increment - (macro_strain - start.macro_strain)
                    + get_strain_change(self.D, polarization_increment))
        return Evaluation(strain, plastic_state, stress, residual)

    def get_newton_correction(self, current, plastic_history):
        """SC-4: J v = v + D (T v), with T = L_t − C0 and L_t each cluster's algorithmic stiffness. GMRES solves it with
        J's diagonal blocks I + D_aa T_a inverted (block-Jacobi) on the right, as FFT-TFA's FT-6 and FT-7."""
        T = get_algorithmic_stiffness(current.strain, self.clusters.materials, plastic_history) - self.C0

        def apply_jacobian(vector):
            return vector + get_strain_change(self.D, apply_blocks(T, vector))

        clusters = np.arange(self.clusters.cluster_count)
        own_blocks = self.D.reshape(len(clusters), 6, len(clusters), 6)[clusters, :, clusters, :]       # D_aa
        inverse_blocks = np.linalg.inv(np.eye(6) + own_blocks @ T)
        return solve_by_gmres(apply_jacobian, current.residual,
                              apply_preconditioner=lambda vector: apply_blocks(inverse_blocks, vector))

    def solve_step(self, macro_strain, plastic_history):
        """One load step from the last accepted one. With C0 "self-consistent", Liu–Bessa–Liu's Box II: solve, refit c
        to the step's macroscopic increments (SC-3), and solve again until c changes by less than
        sca_reference_tolerance. Each solve starts from the last one's solution, and every solve's iterations count."""
        self.step_start = StepStart(self.previous_strain, self.previous_stress, self.previous_macro_strain)
        newton_iteration_count = krylov_iteration_count = 0
        for _ in range(config.sca_max_reference_updates):
            result = super().solve_step(macro_strain, plastic_history)
            newton_iteration_count += result.newton_iteration_count
            krylov_iteration_count += result.krylov_iteration_count
            if not self.is_self_consistent:
                break
            scale = fit_reference_scale(macro_strain - self.step_start.macro_strain,
                                        self.clusters.average(result.stress - self.step_start.stress))
            if not scale > 0:
                raise SolverDidNotConverge(f"the self-consistent reference stiffness came out as {scale:.3e} I")
            has_settled = abs(scale - self.reference_scale) <= config.sca_reference_tolerance * self.reference_scale
            self.set_reference_scale(scale)
            if has_settled:
                break
        else:
            raise SolverDidNotConverge(f"the self-consistent reference stiffness did not settle within "
                                       f"{config.sca_max_reference_updates} solves of the load step")
        self.previous_stress = result.stress
        return result._replace(newton_iteration_count=newton_iteration_count,
                               krylov_iteration_count=krylov_iteration_count)
