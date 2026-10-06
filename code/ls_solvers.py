# The partition-averaged Lippmann-Schwinger (LS) model of notes/Partition_Lippmann_Schwinger_New_9_28_2026: a
# partitionwise-constant polarization interacting through the homogeneous reference medium, so the whole nonlocal
# operator is the P0 convolution (LS-17). It is a different model from actual E/P, so it matches the TFA solvers only
# under matched phase stiffness. It is solved either by damped fixed-point iteration (LS-19) or by matrix-free
# Newton-GMRES on the LS-20 Jacobian.

import functools
import numpy as np
import scipy.sparse.linalg

import config
from convergence import SolverDidNotConverge, StepResult, get_relative_residual, has_converged
from lattice_fft import apply_on_partition_lattice
from load_path import Solver
from material import get_eigenstrain_sensitivity, get_plastic_eigenstrain
from online_timing import timed_online

## ------- LS Model ------- ##

@timed_online("induced_strain")
def get_P0_induced_strain(P0_transformed, partition_field):
    # P0 applied as a convolution over the partition lattice (LS-17): the reference medium's counterpart of the TFA
    # get_induced_strain.
    return apply_on_partition_lattice(P0_transformed, partition_field)

def get_ls_state(strain, b, P0_transformed, reference_compliance, partition_materials, plastic_history):
    # The LS-18 residual r = ε - b - P0 μ*, where μ* = ε - C0⁻¹ σ is the equivalent eigenstrain of LS-6.
    plastic_state, stress = get_plastic_eigenstrain(strain, partition_materials, plastic_history)
    equivalent_eigenstrain = strain - stress @ reference_compliance.T
    residual = strain - b - get_P0_induced_strain(P0_transformed, equivalent_eigenstrain)
    return plastic_state, stress, residual

def get_ls_residual(macro_strain, strain, partition_materials, plastic_history, P0_transformed, reference_compliance):
    # LS-18 at a converged strain, for the independent check in run_strain_path. The LS solvers keep no bookkeeping
    # between iterations, so evaluating the residual afresh is the whole check.
    b = np.tile(macro_strain, (len(strain), 1))
    _, _, residual = get_ls_state(strain, b, P0_transformed, reference_compliance, partition_materials,
                                  plastic_history)
    return residual

## ------- Fixed Point ------- ##

def get_ls_relaxation_factor(partition_materials, reference_L):
    # eig(P0) in [0, 1] makes the Jacobian I - P0(I - C0⁻¹ L_t) similar to a symmetric matrix with spectrum inside
    # [min(1, λ_min), max(1, λ_max)], λ the eigenvalues of C0⁻¹ L; each reference stiffness lies between the phase
    # stiffnesses, which puts 1 inside [λ_min, λ_max].
    # The optimal damping for that interval is 2 / (λ_min + λ_max). The elastic L is used because the plastic tangent
    # is softer.
    if config.ls_relaxation_factor is not None:
        return config.ls_relaxation_factor
    stiffness_ratio = np.linalg.eigvals(np.linalg.inv(reference_L) @ partition_materials.L).real
    return config.ls_stability_safety * 2 / (np.min(stiffness_ratio) + np.max(stiffness_ratio))

def ls_fixed_point_iteration(macro_strain, initial_strain, partition_materials, plastic_history, P0_transformed,
                                    reference_compliance, relaxation):
    b = np.tile(macro_strain, (len(initial_strain), 1))
    strain = initial_strain

    residual_history = []
    while True:
        plastic_state, stress, residual = get_ls_state(strain, b, P0_transformed, reference_compliance,
                                                              partition_materials, plastic_history)
        residual_history.append(get_relative_residual(residual, macro_strain))

        if has_converged(residual_history, config.ls_divergence_limit, config.ls_max_iterations):
            break

        # LS-19, damped.
        strain = strain - relaxation * residual

    return StepResult(strain, stress, plastic_state, np.array(residual_history))

## ------- Newton ------- ##

@timed_online("sensitivity")
def get_ls_jacobian_blocks(strain, partition_materials, plastic_history, reference_compliance):
    # The block-diagonal factor I - C0⁻¹ L_t of the LS-20 Jacobian J v = v - P0 [(I - C0⁻¹ L_t) v], with the
    # algorithmic stress tangent L_t = L (I - H_μ).
    sensitivity = get_eigenstrain_sensitivity(strain, partition_materials, plastic_history)
    algorithmic_stiffness = partition_materials.L @ (np.eye(6) - sensitivity)
    return np.eye(6) - reference_compliance @ algorithmic_stiffness

@timed_online("correction_solve")
def get_ls_newton_correction(P0_transformed, jacobian_blocks, residual):
    # Solves the LS-20 Newton system J δε = -r by unpreconditioned GMRES. Timed as one correction solve, apart from
    # its P0 applications, which count as induced strain.
    def apply_jacobian(flat_vector):
        vector = flat_vector.reshape(-1, 6)
        coupled_strain = get_P0_induced_strain(P0_transformed, np.einsum('pij,pj->pi', jacobian_blocks, vector))
        return (vector - coupled_strain).reshape(-1)

    system_size = residual.size
    jacobian = scipy.sparse.linalg.LinearOperator((system_size, system_size), matvec=apply_jacobian)
    correction, info = scipy.sparse.linalg.gmres(jacobian, -residual.reshape(-1), rtol=config.newton_krylov_tolerance,
                                                 restart=config.newton_krylov_restart)
    if info != 0:
        raise SolverDidNotConverge(f"GMRES did not converge inside the LS Newton step (info {info}). Loosen "
                                   "newton_krylov_tolerance, raise newton_krylov_restart, or use MAIN_FP.py")
    return correction.reshape(-1, 6)

def ls_newton_iteration(macro_strain, initial_strain, partition_materials, plastic_history, P0_transformed,
                               reference_compliance):
    # A Newton step costs a whole Krylov solve, so it gets a far smaller cap than the fixed point: without one, a
    # stagnating solve would grind through hundreds of thousands of convolutions instead of failing.
    b = np.tile(macro_strain, (len(initial_strain), 1))
    strain = initial_strain

    residual_history = []
    while True:
        plastic_state, stress, residual = get_ls_state(strain, b, P0_transformed, reference_compliance,
                                                              partition_materials, plastic_history)
        residual_history.append(get_relative_residual(residual, macro_strain))

        if has_converged(residual_history, config.ls_divergence_limit, config.newton_max_steps):
            break

        jacobian_blocks = get_ls_jacobian_blocks(strain, partition_materials, plastic_history,
                                                        reference_compliance)
        strain = strain + get_ls_newton_correction(P0_transformed, jacobian_blocks, residual)

    return StepResult(strain, stress, plastic_state, np.array(residual_history))

## ------- Load-Path Solvers ------- ##

class LSSolver:
    # An LS iteration as a load-path solver. Each step starts from the previous step's converged strain shifted by
    # the macrostrain increment, so the solver keeps that strain between steps; the get_ls_*_solver functions build a
    # fresh one for every load path. iterate(macro_strain, initial_strain, partition_materials, plastic_history)
    # solves one step.
    def __init__(self, iterate, partition_count):
        self.iterate = iterate
        self.previous_strain = np.zeros((partition_count, 6))
        self.previous_macro_strain = np.zeros(6)

    def __call__(self, macro_strain, partition_materials, plastic_history):
        initial_strain = self.previous_strain + (macro_strain - self.previous_macro_strain)
        step_result = self.iterate(macro_strain, initial_strain, partition_materials, plastic_history)
        self.previous_strain, self.previous_macro_strain = step_result.strain, macro_strain
        return step_result

def get_ls_residual_function(P0_transformed, reference_compliance):
    return functools.partial(get_ls_residual, P0_transformed=P0_transformed,
                             reference_compliance=reference_compliance)

def get_ls_fixed_point_solver(P0_transformed, reference_compliance, relaxation, partition_materials):
    # An LS solver on the lattice that P0_transformed and partition_materials describe.
    iterate = functools.partial(ls_fixed_point_iteration, P0_transformed=P0_transformed,
                                reference_compliance=reference_compliance, relaxation=relaxation)
    return Solver(LSSolver(iterate, len(partition_materials.L)),
                  get_ls_residual_function(P0_transformed, reference_compliance), partition_materials)

def get_ls_newton_solver(P0_transformed, reference_compliance, partition_materials):
    # An LS solver on the lattice that P0_transformed and partition_materials describe.
    iterate = functools.partial(ls_newton_iteration, P0_transformed=P0_transformed,
                                reference_compliance=reference_compliance)
    return Solver(LSSolver(iterate, len(partition_materials.L)),
                  get_ls_residual_function(P0_transformed, reference_compliance), partition_materials)
