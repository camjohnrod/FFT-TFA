# J2 plasticity with linear isotropic hardening: closed-form radial return for all partitions at once, and its
# eigenstrain sensitivity H_μ.

from typing import NamedTuple
import numpy as np

import config
from online_timing import timed_online

class PlasticState(NamedTuple):
    # Partition-stacked plastic history: either the accepted history z_n of ER-3 or a candidate for the current step.
    plastic_strain: np.ndarray
    accumulated_plastic_strain: np.ndarray

def get_unloaded_plastic_state():
    # The plastic history before the first load step: no plastic strain anywhere.
    return PlasticState(np.zeros((config.partition_count, 6)), np.zeros(config.partition_count))

def get_yielding_partitions(plastic_state, plastic_history):
    # The partitions that flow plastically in a candidate state: those that accumulate plastic strain beyond the
    # accepted history.
    return plastic_state.accumulated_plastic_strain > plastic_history.accumulated_plastic_strain

class TrialState(NamedTuple):
    stress: np.ndarray
    mean_stress: np.ndarray
    deviatoric_stress: np.ndarray
    equivalent_stress: np.ndarray
    yield_function: np.ndarray
    plastic_multiplier: np.ndarray

volumetric_direction = np.array([1, 1, 1, 0, 0, 0])

deviatoric_projector = np.zeros((6, 6))
deviatoric_projector[:3, :3] = np.eye(3) - np.ones((3, 3)) / 3
deviatoric_projector[3:, 3:] = np.eye(3)

def get_deviatoric_stress(stress):
    return stress @ deviatoric_projector.T

def get_equivalent_stress(deviatoric_stress):
    return np.sqrt(1.5 * np.sum(deviatoric_stress[:, :3]**2, axis=1) + 3 * np.sum(deviatoric_stress[:, 3:]**2, axis=1))

def get_flow_stress(partition_materials, accumulated_plastic_strain):
    return partition_materials.yield_stress + partition_materials.hardening_modulus * accumulated_plastic_strain

def get_trial_state(strain, partition_materials, plastic_history):
    stress = np.einsum('pij,pj->pi', partition_materials.L, strain - plastic_history.plastic_strain)
    mean_stress = np.sum(stress[:, :3], axis=1) / 3
    deviatoric_stress = stress - np.outer(mean_stress, volumetric_direction)
    equivalent_stress = get_equivalent_stress(deviatoric_stress)
    yield_function = equivalent_stress - get_flow_stress(partition_materials,
                                                         plastic_history.accumulated_plastic_strain)
    shear_modulus = partition_materials.L[:, 3, 3]
    plastic_multiplier = np.maximum(yield_function, 0) / (3 * shear_modulus + partition_materials.hardening_modulus)
    return TrialState(stress, mean_stress, deviatoric_stress, equivalent_stress, yield_function, plastic_multiplier)

def get_flow_direction(deviatoric_stress, equivalent_stress):
    flow_direction = np.zeros_like(deviatoric_stress)
    flow_direction[:, :3] = 1.5 * deviatoric_stress[:, :3] / equivalent_stress[:, None]
    flow_direction[:, 3:] = 3 * deviatoric_stress[:, 3:] / equivalent_stress[:, None]
    return flow_direction

@timed_online("material_update")
def get_plastic_eigenstrain(strain, partition_materials, plastic_history):
    # ER-3 by radial return from the accepted history. The eigenstrain μ is the plastic strain.
    trial = get_trial_state(strain, partition_materials, plastic_history)

    plastic_strain = plastic_history.plastic_strain.copy()
    accumulated_plastic_strain = plastic_history.accumulated_plastic_strain.copy()
    stress = trial.stress.copy()

    yielding = trial.yield_function > 0
    yielding_deviatoric_stress = trial.deviatoric_stress[yielding]
    yielding_equivalent_stress = trial.equivalent_stress[yielding]
    shear_modulus = partition_materials.L[yielding, 3, 3]
    plastic_multiplier = trial.plastic_multiplier[yielding]

    flow_direction = get_flow_direction(yielding_deviatoric_stress, yielding_equivalent_stress)
    plastic_strain[yielding] += plastic_multiplier[:, None] * flow_direction
    accumulated_plastic_strain[yielding] += plastic_multiplier

    deviatoric_scaling = 1 - 3 * shear_modulus * plastic_multiplier / yielding_equivalent_stress
    corrected_deviatoric_stress = deviatoric_scaling[:, None] * yielding_deviatoric_stress
    stress[yielding] = np.outer(trial.mean_stress[yielding], volumetric_direction) + corrected_deviatoric_stress

    return PlasticState(plastic_strain, accumulated_plastic_strain), stress

@timed_online("sensitivity")
def get_eigenstrain_sensitivity(strain, partition_materials, plastic_history):
    # H_μ = ∂μ/∂ε at the current strain (ER-8), zero in partitions that stay elastic.
    trial = get_trial_state(strain, partition_materials, plastic_history)

    sensitivity = np.zeros((config.partition_count, 6, 6))

    yielding = trial.yield_function > 0
    yielding_deviatoric_stress = trial.deviatoric_stress[yielding]
    yielding_equivalent_stress = trial.equivalent_stress[yielding]
    shear_modulus = partition_materials.L[yielding, 3, 3]
    hardening_modulus = partition_materials.hardening_modulus[yielding]
    plastic_multiplier = trial.plastic_multiplier[yielding]

    flow_direction = get_flow_direction(yielding_deviatoric_stress, yielding_equivalent_stress)
    normalized_deviatoric_stress = yielding_deviatoric_stress / yielding_equivalent_stress[:, None]
    flow_projector = np.einsum('pi,pj->pij', flow_direction, normalized_deviatoric_stress)

    radial_sensitivity = 3 * shear_modulus / (3 * shear_modulus + hardening_modulus)
    rotational_sensitivity = 3 * shear_modulus * plastic_multiplier / yielding_equivalent_stress

    sensitivity[yielding] = (radial_sensitivity[:, None, None] * flow_projector
                             + rotational_sensitivity[:, None, None] * (deviatoric_projector - flow_projector))
    return sensitivity
