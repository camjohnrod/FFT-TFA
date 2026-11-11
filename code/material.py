"""The material law: J2 (von Mises) plasticity with linear isotropic hardening, applied to every cluster at once.

Each cluster (cell.Clusters) has one strain and one plastic history. From them, radial_return gives the cluster's
stress and its new plastic strain, which is its eigenstrain μ: the strain the material no longer resists elastically,
σ = L (ε − μ). Newton's method also needs how μ changes with ε, the sensitivity H_μ = ∂μ/∂ε, which gives the
algorithmic stiffness L_t = dσ/dε = L (I − H_μ). Equations ER-* are in notes/fft_tfa_lecture.md, appendix A.

Strains and stresses are rows of 6 Voigt components (11, 22, 33, 12, 23, 13), shears as engineering shear γ = 2ε (ER-2),
one row per cluster.
"""

from typing import NamedTuple
import numpy as np

## ------- Plastic History ------- ##

class PlasticState(NamedTuple):
    """Every cluster's plastic history: the history accepted at the end of the last load step (z_n of ER-3), or a
    candidate for the current one."""
    plastic_strain: np.ndarray                # clusters × 6: the eigenstrain μ
    accumulated_plastic_strain: np.ndarray    # clusters: the equivalent plastic strain, which sets the hardening

def get_unloaded_plastic_state(cluster_count):
    """The plastic history before the first load step: no plastic strain anywhere."""
    return PlasticState(np.zeros((cluster_count, 6)), np.zeros(cluster_count))

def get_yielding_clusters(plastic_state, plastic_history):
    """Which clusters flow plastically in a candidate state: those whose accumulated plastic strain grew beyond the
    accepted history."""
    return plastic_state.accumulated_plastic_strain > plastic_history.accumulated_plastic_strain

## ------- Stress Measures ------- ##

volumetric_direction = np.array([1, 1, 1, 0, 0, 0])

# Takes a stress to its deviatoric part (the stress minus its mean).
deviatoric_projector = np.zeros((6, 6))
deviatoric_projector[:3, :3] = np.eye(3) - np.ones((3, 3)) / 3
deviatoric_projector[3:, 3:] = np.eye(3)

def get_deviatoric_stress(stress):
    return stress @ deviatoric_projector.T

def get_equivalent_stress(deviatoric_stress):
    """The von Mises stress √(3/2 s:s) of each deviatoric stress s, its 6 components along the last axis (shear
    components count twice in s:s)."""
    return np.sqrt(1.5 * np.sum(deviatoric_stress[..., :3]**2, axis=-1)
                   + 3 * np.sum(deviatoric_stress[..., 3:]**2, axis=-1))

def get_von_mises_stress(stress):
    return get_equivalent_stress(get_deviatoric_stress(stress))

def get_flow_stress(materials, accumulated_plastic_strain):
    """The yield stress after hardening: σ_y + H p, with p the accumulated plastic strain."""
    return materials.yield_stress + materials.hardening_modulus * accumulated_plastic_strain

## ------- Radial Return ------- ##

class TrialState(NamedTuple):
    """The elastic trial state: the stress if the step were purely elastic, and how far it lies beyond yield."""
    stress: np.ndarray
    mean_stress: np.ndarray
    deviatoric_stress: np.ndarray
    equivalent_stress: np.ndarray
    yield_function: np.ndarray        # equivalent stress minus flow stress: positive where the cluster yields
    plastic_multiplier: np.ndarray    # the plastic strain increment Δp that brings it back to the yield surface

def get_trial_state(strain, materials, plastic_history):
    """Step 1 of the radial return: freeze the plastic strain, compute the elastic stress L (ε − μ_n), and compare its
    von Mises stress with the flow stress. Where it lies beyond, the closed-form multiplier Δp = f / (3G + H) brings it
    back (G the shear modulus, H the hardening modulus)."""
    stress = np.einsum('pij,pj->pi', materials.L, strain - plastic_history.plastic_strain)
    mean_stress = np.sum(stress[:, :3], axis=1) / 3
    deviatoric_stress = stress - np.outer(mean_stress, volumetric_direction)
    equivalent_stress = get_equivalent_stress(deviatoric_stress)
    yield_function = equivalent_stress - get_flow_stress(materials,
                                                         plastic_history.accumulated_plastic_strain)
    shear_modulus = materials.L[:, 3, 3]
    plastic_multiplier = np.maximum(yield_function, 0) / (3 * shear_modulus + materials.hardening_modulus)
    return TrialState(stress, mean_stress, deviatoric_stress, equivalent_stress, yield_function, plastic_multiplier)

def get_flow_direction(deviatoric_stress, equivalent_stress):
    """The plastic flow direction n = 3/2 s / σ_eq, in engineering Voigt form (shear components doubled)."""
    flow_direction = np.zeros_like(deviatoric_stress)
    flow_direction[:, :3] = 1.5 * deviatoric_stress[:, :3] / equivalent_stress[:, None]
    flow_direction[:, 3:] = 3 * deviatoric_stress[:, 3:] / equivalent_stress[:, None]
    return flow_direction

def radial_return(strain, materials, plastic_history):
    """Every cluster's new plastic state and stress at the given strains, from the accepted history (ER-3).

    Step 1 (get_trial_state): the elastic trial stress. Clusters below yield keep it and their plastic history. Step 2,
    in the yielding clusters: the plastic strain grows by Δp n along the flow direction, and the deviatoric stress
    shrinks radially (its direction unchanged) back onto the grown yield surface; the mean stress is unchanged."""
    trial = get_trial_state(strain, materials, plastic_history)

    plastic_strain = plastic_history.plastic_strain.copy()
    accumulated_plastic_strain = plastic_history.accumulated_plastic_strain.copy()
    stress = trial.stress.copy()

    yielding = trial.yield_function > 0
    yielding_deviatoric_stress = trial.deviatoric_stress[yielding]
    yielding_equivalent_stress = trial.equivalent_stress[yielding]
    shear_modulus = materials.L[yielding, 3, 3]
    plastic_multiplier = trial.plastic_multiplier[yielding]

    flow_direction = get_flow_direction(yielding_deviatoric_stress, yielding_equivalent_stress)
    plastic_strain[yielding] += plastic_multiplier[:, None] * flow_direction
    accumulated_plastic_strain[yielding] += plastic_multiplier

    deviatoric_scaling = 1 - 3 * shear_modulus * plastic_multiplier / yielding_equivalent_stress
    corrected_deviatoric_stress = deviatoric_scaling[:, None] * yielding_deviatoric_stress
    stress[yielding] = np.outer(trial.mean_stress[yielding], volumetric_direction) + corrected_deviatoric_stress

    return PlasticState(plastic_strain, accumulated_plastic_strain), stress

## ------- Derivatives for Newton ------- ##

def get_eigenstrain_sensitivity(strain, materials, plastic_history):
    """H_μ = ∂μ/∂ε at the given strains (ER-8), one 6 × 6 block per cluster, zero in clusters that stay elastic.

    It has two parts: a change of strain along the flow direction changes how far the trial stress lies beyond yield
    (radial part), and one across it turns the flow direction (rotational part)."""
    trial = get_trial_state(strain, materials, plastic_history)

    sensitivity = np.zeros((len(strain), 6, 6))

    yielding = trial.yield_function > 0
    yielding_deviatoric_stress = trial.deviatoric_stress[yielding]
    yielding_equivalent_stress = trial.equivalent_stress[yielding]
    shear_modulus = materials.L[yielding, 3, 3]
    hardening_modulus = materials.hardening_modulus[yielding]
    plastic_multiplier = trial.plastic_multiplier[yielding]

    flow_direction = get_flow_direction(yielding_deviatoric_stress, yielding_equivalent_stress)
    normalized_deviatoric_stress = yielding_deviatoric_stress / yielding_equivalent_stress[:, None]
    flow_projector = np.einsum('pi,pj->pij', flow_direction, normalized_deviatoric_stress)

    radial_sensitivity = 3 * shear_modulus / (3 * shear_modulus + hardening_modulus)
    rotational_sensitivity = 3 * shear_modulus * plastic_multiplier / yielding_equivalent_stress

    sensitivity[yielding] = (radial_sensitivity[:, None, None] * flow_projector
                             + rotational_sensitivity[:, None, None] * (deviatoric_projector - flow_projector))
    return sensitivity

def get_algorithmic_stiffness(strain, materials, plastic_history):
    """L_t = dσ/dε = L (I − H_μ), the stiffness consistent with the radial return, one 6 × 6 block per cluster."""
    return materials.L @ (np.eye(6) - get_eigenstrain_sensitivity(strain, materials, plastic_history))
