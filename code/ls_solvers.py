# The partition-averaged Lippmann-Schwinger (LS) model of notes/Partition_Lippmann_Schwinger_New_9_28_2026: a
# partitionwise-constant polarization interacting through the homogeneous reference medium, so the whole nonlocal
# operator is the P0 convolution (LS-17). It is a different model from actual E/P, so it matches the TFA solvers only
# under matched phase stiffness. Here it is solved by damped fixed-point iteration (LS-19).

import functools
import numpy as np

import config
from convergence import StepResult, get_relative_residual, has_converged
from lattice_fft import apply_on_partition_lattice
from load_path import Solver
from material import get_plastic_eigenstrain
from online_timing import timed_online
from tfa_solvers import get_standard_correction

@timed_online("induced_strain")
def get_reference_induced_strain(P0_transformed, partition_field):
    # P0 applied as a convolution over the partition lattice (LS-17), the reference counterpart of
    # get_induced_strain.
    return apply_on_partition_lattice(P0_transformed, partition_field)

def get_reference_state(strain, b, P0_transformed, reference_compliance, partition_materials, plastic_history):
    # The LS-18 residual r = ε - b - P0 μ*, where μ* = ε - C0⁻¹ σ is the equivalent eigenstrain of LS-6.
    plastic_state, stress = get_plastic_eigenstrain(strain, partition_materials, plastic_history)
    equivalent_eigenstrain = strain - stress @ reference_compliance.T
    residual = strain - b - get_reference_induced_strain(P0_transformed, equivalent_eigenstrain)
    return plastic_state, stress, residual

def get_reference_model_residual(E, P, macro_strain, strain, partition_materials, plastic_history, P0_transformed,
                                 reference_compliance):
    # LS-18 at a converged strain, for the independent check in run_strain_path. The fixed point keeps no
    # bookkeeping between iterations, so evaluating the residual afresh is the whole check. E and P belong to the
    # actual model and go unused.
    b = np.tile(macro_strain, (config.partition_count, 1))
    _, _, residual = get_reference_state(strain, b, P0_transformed, reference_compliance, partition_materials,
                                         plastic_history)
    return residual

def get_reference_stiffness_ratio_bounds(partition_materials, reference_L):
    # eig(P0) in [0, 1] makes the Jacobian I - P0(I - C0^-1 L_t) similar to a symmetric matrix with spectrum
    # inside [lambda_min, lambda_max] of C0^-1 L. The elastic L is used because the plastic tangent is softer.
    stiffness_ratio = np.linalg.eigvals(np.linalg.inv(reference_L) @ partition_materials.L).real
    return np.min(stiffness_ratio), np.max(stiffness_ratio)

def get_reference_stability_limit(partition_materials, reference_L):
    return 2 / get_reference_stiffness_ratio_bounds(partition_materials, reference_L)[1]

def get_reference_relaxation_factor(partition_materials, reference_L):
    if config.ls_relaxation_factor is not None:
        return config.ls_relaxation_factor
    smallest_ratio, largest_ratio = get_reference_stiffness_ratio_bounds(partition_materials, reference_L)
    return config.ls_stability_safety * 2 / (smallest_ratio + largest_ratio)

def reference_fixed_point_iteration(macro_strain, initial_strain, partition_materials, plastic_history, P0_transformed,
                                    reference_compliance, relaxation):
    b = np.tile(macro_strain, (config.partition_count, 1))
    strain = initial_strain

    residual_history = []
    while True:
        plastic_state, stress, residual = get_reference_state(strain, b, P0_transformed, reference_compliance,
                                                              partition_materials, plastic_history)
        residual_history.append(get_relative_residual(residual, macro_strain))

        if has_converged(residual_history[-1], len(residual_history), "reference_fixed_point",
                         config.ls_divergence_limit, config.ls_max_iterations):
            break

        # LS-19, damped.
        strain = strain + relaxation * get_standard_correction(residual)

    return StepResult(strain, stress, plastic_state, np.array(residual_history))

class ReferenceFixedPointSolver:
    # The LS fixed point as a load-path solver. Each step starts from the previous step's converged strain shifted
    # by the macrostrain increment, so the solver keeps that strain between steps; get_ls_fixed_point_solver builds a
    # fresh one for every load path.
    def __init__(self, P0_transformed, reference_compliance, relaxation):
        self.P0_transformed = P0_transformed
        self.reference_compliance = reference_compliance
        self.relaxation = relaxation
        self.previous_strain = np.zeros((config.partition_count, 6))
        self.previous_macro_strain = np.zeros(6)

    def __call__(self, E, P, macro_strain, partition_materials, plastic_history):
        # E and P describe the actual heterogeneous model, which the reference model replaces, so they go unused.
        initial_strain = self.previous_strain + (macro_strain - self.previous_macro_strain)
        step_result = reference_fixed_point_iteration(macro_strain, initial_strain, partition_materials,
                                                      plastic_history, self.P0_transformed,
                                                      self.reference_compliance, self.relaxation)
        self.previous_strain, self.previous_macro_strain = step_result.strain, macro_strain
        return step_result

def get_ls_fixed_point_solver(P0_transformed, reference_compliance, relaxation):
    return Solver(ReferenceFixedPointSolver(P0_transformed, reference_compliance, relaxation),
                  functools.partial(get_reference_model_residual, P0_transformed=P0_transformed,
                                    reference_compliance=reference_compliance))
