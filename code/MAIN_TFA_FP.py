import functools
import time
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import config
import plots
from load_path import run_interleaved_repeats
from material import get_von_mises_stress
from offline import (get_B, get_F_eigenstrain, get_F_macrostrain, get_K, get_L, get_element_dofs, get_free_dofs,
                     get_integrated_B, get_offline_operators, get_partition_material_ids, get_partition_materials,
                     get_periodic_mesh, get_value_per_material)
from report import (check_solver_agreement, get_completed_step_count, get_most_complete_solver_name,
                    print_comparison, save_load_path_plots)
from tfa_solvers import fft_preconditioned_richardson_iteration, standard_richardson_iteration

## ------- Load Path ------- ##

def get_solvers(P0_transformed):
    return {"Standard": standard_richardson_iteration,
            "FFT-preconditioned": functools.partial(fft_preconditioned_richardson_iteration,
                                                    P0_transformed=P0_transformed)}

## ------- Post-processing ------- ##

def get_F_state(B, L_per_element, mesh, macro_strain, eigenstrain):
    return (get_F_macrostrain(B, L_per_element, mesh.element_nodes) @ macro_strain
            + get_F_eigenstrain(B, L_per_element, mesh.element_nodes, mesh.element_partition_ids)
            @ eigenstrain.ravel())

def solve_state_displacements(K, F):
    free_dofs = get_free_dofs()
    K_free = K[free_dofs][:, free_dofs]
    jacobi_preconditioner = scipy.sparse.diags(1 / K_free.diagonal())
    conjugate_gradient_tolerance = 1e-10
    free_displacements, info = scipy.sparse.linalg.cg(K_free, F[free_dofs], rtol=conjugate_gradient_tolerance,
                                                      M=jacobi_preconditioner)
    if info != 0:
        raise RuntimeError(f"conjugate gradient did not converge (info {info}).")
    displacements = np.zeros(config.dof_count)
    displacements[free_dofs] = free_displacements
    return displacements

def get_element_average_fluctuation_strain(displacements, B, element_nodes):
    element_dofs = get_element_dofs(element_nodes)
    return displacements[element_dofs] @ get_integrated_B(B).T / config.element_volume

def get_element_stress(mesh, B, L_per_element, macro_strain, eigenstrain):
    start_time = time.perf_counter()
    K = get_K(B, L_per_element, mesh.element_nodes)
    F = get_F_state(B, L_per_element, mesh, macro_strain, eigenstrain)
    displacements = solve_state_displacements(K, F)

    element_strain = macro_strain + get_element_average_fluctuation_strain(displacements, B, mesh.element_nodes)
    element_eigenstrain = eigenstrain[mesh.element_partition_ids]
    element_stress = np.einsum('eij,ej->ei', L_per_element, element_strain - element_eigenstrain)
    print(f"element stress post-processing: {time.perf_counter() - start_time:.2f} seconds.")
    return element_stress

## ------- Main ------- ##

def save_von_mises_plots(result, mesh, B, L_per_element):
    partition_von_mises_stress_MPa = get_von_mises_stress(result.final_stress) / 1e6
    plots.plot_von_mises_cross_section(
        partition_von_mises_stress_MPa.reshape(config.partition_number_per_side, config.partition_number_per_side),
        config.element_number_per_side, config.partition_number_per_side, config.von_mises_plot_min_stress / 1e6,
        config.von_mises_plot_max_stress / 1e6, config.output_folder / "von_mises_cross_section.png")

    if config.post_process_element_stress:
        # The last step the solver actually solved, which is not the last requested one if it stopped early.
        final_macro_strain = result.applied_macro_strain[result.completed_step_count - 1]
        element_stress = get_element_stress(mesh, B, L_per_element, final_macro_strain, result.final_plastic_strain)
        bottom_layer_element_stress = element_stress[:config.element_number_per_side**2]
        bottom_layer_von_mises_stress_MPa = get_von_mises_stress(bottom_layer_element_stress) / 1e6
        plots.plot_von_mises_cross_section(
            bottom_layer_von_mises_stress_MPa.reshape(config.element_number_per_side, config.element_number_per_side),
            config.element_number_per_side, config.partition_number_per_side, config.von_mises_plot_min_stress / 1e6,
            config.von_mises_plot_max_stress / 1e6, config.output_folder / "von_mises_element_cross_section.png")

def main():
    config.output_folder.mkdir(exist_ok=True)
    partition_material_ids = get_partition_material_ids()
    print(f"inclusion volume fraction on the partition grid: {partition_material_ids.mean():.4f}")
    plots.plot_cross_section(
        partition_material_ids.reshape(config.partition_number_per_side, config.partition_number_per_side),
        config.element_number_per_side, config.output_folder / "cross_section.png")

    L_matrix = get_L(config.elastic_modulus_matrix, config.poisson_ratio_matrix)
    L_inclusion = get_L(config.elastic_modulus_inclusion, config.poisson_ratio_inclusion)
    partition_materials = get_partition_materials(L_matrix, L_inclusion, partition_material_ids)
    mesh = get_periodic_mesh(partition_material_ids)
    B = get_B()
    L_per_element = get_value_per_material(L_matrix, L_inclusion, mesh.element_material_ids)
    E, P, P0_transformed = get_offline_operators(mesh, B, L_per_element, L_matrix, L_inclusion, partition_materials.L)

    results = run_interleaved_repeats(functools.partial(get_solvers, P0_transformed), E, P, partition_materials)
    # Timings and stresses are compared only over the steps every solver completed.
    comparable_step_count = min(get_completed_step_count(solver_results) for solver_results in results.values())
    print_comparison(results, comparable_step_count)
    plotted_solver_name = get_most_complete_solver_name(results)
    save_load_path_plots(results, comparable_step_count, plotted_solver_name)
    save_von_mises_plots(results[plotted_solver_name][-1], mesh, B, L_per_element)
    check_solver_agreement(results, comparable_step_count)

if __name__ == "__main__":
    main()
