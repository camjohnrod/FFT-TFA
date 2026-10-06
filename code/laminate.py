# The exact solution of the laminate (config.inclusion_shape = "laminate"): an inclusion layer between matrix layers,
# with interfaces normal to x. Under a uniform macrostrain each phase has one uniform state, even in plasticity, so the
# whole cell reduces to two states, one per phase:
#   - compatibility keeps the strains along the interface, ε22, ε33 and γ23, equal to the macrostrain in both phases;
#   - equilibrium makes the tractions on the interface, σ11, σ12 and σ13, equal in both phases;
#   - the phase average of the strain is the macrostrain.
# That leaves the matrix phase's ε11, γ12 and γ13 as the only unknowns of a load step, found here by Newton. Every
# solver must reproduce this solution on a laminate (report.check_laminate_solution), and the FE homogenization must
# reproduce its elastic stiffness (verification.run_verification).

import numpy as np

import config
from material import get_eigenstrain_sensitivity, get_plastic_eigenstrain, get_unloaded_plastic_state

# Voigt positions 0, 3, 5: the strain components ε11, γ12, γ13 may jump across an interface normal to x, while the
# stress components at the same positions, the tractions σ11, σ12, σ13, may not.
interface_normal_components = [0, 3, 5]

def get_phase_materials(partition_materials):
    # One row per phase, matrix first, copied from the first partition of each phase.
    rows = [np.flatnonzero(partition_materials.material_ids == material_id)[0] for material_id in (0, 1)]
    return partition_materials._make(field[rows] for field in partition_materials)

def get_phase_strains(macro_strain, matrix_normal_strain, inclusion_fraction):
    # The strain of each phase, matrix first, given the matrix's interface-normal components.
    phase_strains = np.tile(macro_strain, (2, 1))
    phase_strains[0, interface_normal_components] = matrix_normal_strain
    phase_strains[1, interface_normal_components] = ((macro_strain[interface_normal_components]
                                                      - (1 - inclusion_fraction) * matrix_normal_strain)
                                                     / inclusion_fraction)
    return phase_strains

def solve_laminate_step(macro_strain, matrix_normal_strain, phase_materials, phase_history, inclusion_fraction):
    # Newton on the interface tractions' jump, starting from the given matrix_normal_strain. The inclusion's
    # interface-normal strain moves by -(1 - f)/f times the matrix's, which gives the Jacobian below. Returns the
    # converged matrix_normal_strain and the phases' plastic state and stress.
    for _ in range(50):
        phase_strains = get_phase_strains(macro_strain, matrix_normal_strain, inclusion_fraction)
        phase_state, phase_stress = get_plastic_eigenstrain(phase_strains, phase_materials, phase_history)
        traction_jump = (phase_stress[0] - phase_stress[1])[interface_normal_components]
        if np.abs(traction_jump).max() <= 1e-11 * np.abs(phase_stress).max():
            return matrix_normal_strain, phase_state, phase_stress
        sensitivity = get_eigenstrain_sensitivity(phase_strains, phase_materials, phase_history)
        algorithmic_stiffness = phase_materials.L @ (np.eye(6) - sensitivity)
        normal_block = np.ix_(interface_normal_components, interface_normal_components)
        jacobian = (algorithmic_stiffness[0][normal_block]
                    + (1 - inclusion_fraction) / inclusion_fraction * algorithmic_stiffness[1][normal_block])
        matrix_normal_strain = matrix_normal_strain - np.linalg.solve(jacobian, traction_jump)
    raise RuntimeError(f"the laminate's exact solution did not converge at macrostrain {macro_strain}.")

def get_laminate_phase_stress(partition_materials):
    # The exact stress of each phase at every step of config's load path, shape (load steps, 2, 6), matrix first.
    phase_materials = get_phase_materials(partition_materials)
    inclusion_fraction = partition_materials.material_ids.mean()
    phase_history = get_unloaded_plastic_state(2)
    matrix_normal_strain = np.zeros(len(interface_normal_components))
    previous_macro_strain = np.zeros(6)
    phase_stress = np.zeros((config.load_step_count, 2, 6))
    for step, macro_strain in enumerate(config.applied_macro_strain):
        # Start from the previous step's matrix strain, moved by the macrostrain increment.
        matrix_normal_strain, phase_history, phase_stress[step] = solve_laminate_step(
            macro_strain, matrix_normal_strain + (macro_strain - previous_macro_strain)[interface_normal_components],
            phase_materials, phase_history, inclusion_fraction)
        previous_macro_strain = macro_strain
    return phase_stress

def get_laminate_elastic_stiffness(partition_materials):
    # The exact homogenized elastic stiffness: the phase-averaged stress for each unit macrostrain, with yield removed.
    phase_materials = get_phase_materials(partition_materials)._replace(yield_stress=np.full(2, np.inf))
    inclusion_fraction = partition_materials.material_ids.mean()
    no_plastic_history = get_unloaded_plastic_state(2)
    stiffness = np.zeros((6, 6))
    for component, unit_macro_strain in enumerate(np.eye(6)):
        _, _, phase_stress = solve_laminate_step(unit_macro_strain, unit_macro_strain[interface_normal_components],
                                                 phase_materials, no_plastic_history, inclusion_fraction)
        stiffness[:, component] = (1 - inclusion_fraction) * phase_stress[0] + inclusion_fraction * phase_stress[1]
    return stiffness
