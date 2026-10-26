# The partition-averaged Lippmann–Schwinger (LS) equation of notes/2_ls_model.md, solved by Newton–GMRES on its
# LS-20 Jacobian. Each partition carries a uniform polarization, and partitions interact only through the homogeneous
# reference medium C0, so the whole nonlocal operator is one FFT convolution with the kernel P0 (LS-17). Two methods
# solve it: CP on the partitions, and Spectral LS on the element lattice (one partition per element column). The
# code is the same for both; the lattice is whatever P0_transformed and partition_materials describe.

import functools
import numpy as np
import scipy.fft

import config
from material import get_eigenstrain_sensitivity, get_plastic_eigenstrain
from newton import Solver, StepResult, get_relative_residual, has_converged, solve_newton_correction

def apply_on_partition_lattice(fourier_blocks, partition_field):
    # Applies a translation-invariant operator, given by its 6 × 6 block at every lattice frequency, to a
    # partition-stacked field: transform the field, multiply frequency by frequency, transform back (ER-14a).
    # The field and the operator are real, so only the non-redundant half of the frequencies is used. The lattice is
    # the operator's: its first axis runs over every frequency along one side. The FFTs use config.thread_count
    # threads, like TFA's dense products.
    lattice_shape = (fourier_blocks.shape[0],) * 2
    field_transformed = scipy.fft.rfftn(partition_field.reshape(*lattice_shape, 6), axes=(0, 1),
                                        workers=config.thread_count)
    result_transformed = np.einsum('...ij,...j->...i', fourier_blocks, field_transformed)
    return scipy.fft.irfftn(result_transformed, s=lattice_shape, axes=(0, 1),
                            workers=config.thread_count).reshape(-1, 6)

def get_P0_induced_strain(P0_transformed, partition_field):
    # P0 applied as a convolution over the partition lattice (LS-17).
    return apply_on_partition_lattice(P0_transformed, partition_field)

def get_ls_state(strain, b, P0_transformed, reference_compliance, partition_materials, plastic_history):
    # The LS-18 residual r = ε - b - P0 μ*, where μ* = ε - C0⁻¹ σ is the equivalent eigenstrain of LS-6.
    plastic_state, stress = get_plastic_eigenstrain(strain, partition_materials, plastic_history)
    equivalent_eigenstrain = strain - stress @ reference_compliance.T
    residual = strain - b - get_P0_induced_strain(P0_transformed, equivalent_eigenstrain)
    return plastic_state, stress, residual

def get_ls_jacobian_blocks(strain, partition_materials, plastic_history, reference_compliance):
    # The block-diagonal factor I - C0⁻¹ L_t of the LS-20 Jacobian J v = v - P0 [(I - C0⁻¹ L_t) v], with the
    # algorithmic stress tangent L_t = L (I - H_μ).
    sensitivity = get_eigenstrain_sensitivity(strain, partition_materials, plastic_history)
    algorithmic_stiffness = partition_materials.L @ (np.eye(6) - sensitivity)
    return np.eye(6) - reference_compliance @ algorithmic_stiffness

def get_ls_newton_correction(P0_transformed, jacobian_blocks, residual):
    # Solves the LS-20 Newton system J δε = -r by unpreconditioned GMRES.
    def apply_jacobian(vector):
        return vector - get_P0_induced_strain(P0_transformed, np.einsum('pij,pj->pi', jacobian_blocks, vector))

    return solve_newton_correction(apply_jacobian, residual)

def ls_newton_iteration(macro_strain, initial_strain, partition_materials, plastic_history, P0_transformed,
                        reference_compliance):
    # One load step: Newton from initial_strain until the LS-18 residual converges.
    b = np.tile(macro_strain, (len(initial_strain), 1))
    strain = initial_strain
    gmres_iteration_count = 0

    residual_history = []
    while True:
        plastic_state, stress, residual = get_ls_state(strain, b, P0_transformed, reference_compliance,
                                                       partition_materials, plastic_history)
        residual_history.append(get_relative_residual(residual, macro_strain))

        if has_converged(residual_history, config.ls_divergence_limit):
            break

        jacobian_blocks = get_ls_jacobian_blocks(strain, partition_materials, plastic_history, reference_compliance)
        correction, iteration_count = get_ls_newton_correction(P0_transformed, jacobian_blocks, residual)
        strain = strain + correction
        gmres_iteration_count += iteration_count

    return StepResult(strain, stress, plastic_state, np.array(residual_history), gmres_iteration_count)

class LSSolver:
    # The Newton iteration as a load-path solver. Each step starts from the previous step's converged strain
    # shifted by the macrostrain increment, so the solver keeps that strain between steps.
    def __init__(self, iterate, partition_count):
        self.iterate = iterate
        self.previous_strain = np.zeros((partition_count, 6))
        self.previous_macro_strain = np.zeros(6)

    def __call__(self, macro_strain, partition_materials, plastic_history):
        initial_strain = self.previous_strain + (macro_strain - self.previous_macro_strain)
        step_result = self.iterate(macro_strain, initial_strain, partition_materials, plastic_history)
        self.previous_strain, self.previous_macro_strain = step_result.strain, macro_strain
        return step_result

def get_ls_solver(lattice, reference_compliance):
    # An LS solver on the given fe.Lattice: CP on the partitions, Spectral LS on the elements.
    iterate = functools.partial(ls_newton_iteration, P0_transformed=lattice.P0_transformed,
                                reference_compliance=reference_compliance)
    return Solver(LSSolver(iterate, len(lattice.partition_materials.L)), lattice.partition_materials)
