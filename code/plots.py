# Every figure the entry points save: the macroscopic stress-strain curves and the solver performance (iterations
# and time per iteration per step), the time breakdown per timing group, the partition cross-section and the von
# Mises stress maps on it. Every solver is drawn and named by its role, the same in every figure (solver_role_styles,
# solver_role_labels). No solver logic.

from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.colors

## ------- Shared Style ------- ##

# Every figure's text in Computer Modern, the math included, so labels, ticks and symbols match. Tick labels go through
# mathtext, since cmr10 has no minus sign of its own.
figure_style = {'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
                'axes.formatter.use_mathtext': True, 'axes.formatter.useoffset': False, 'savefig.dpi': 200}

# One look per solver role, keyed like report.SolverRoles and in legend order: told apart by colour (Okabe–Ito, whose
# pairs stay distinct under simulated colour blindness) and by dash pattern. CP, the method being tested,
# is solid, the fine lattice dotted and the coarse one dash-dot. TFA, the baseline, is a thinner black dash drawn on
# top, so it stays visible where the curves agree.
solver_role_styles = {
    "ls_fine":      {'color': '#0072B2', 'linestyle': (0, (1, 1.6)), 'dash_capstyle': 'round', 'zorder': 3},
    "cp":           {'color': '#D55E00', 'linestyle': '-', 'zorder': 2},
    "ls_coarse":    {'color': '#009E73', 'linestyle': (0, (7, 2, 1.5, 2)), 'zorder': 3},
    "tfa_baseline": {'color': 'black', 'linestyle': (0, (5, 3)), 'linewidth': 1.4, 'zorder': 4},
    "tfa_fft":      {'color': '#CC79A7', 'linestyle': (0, (3, 1.5, 1, 1.5, 1, 1.5)), 'zorder': 4},
}
# Short names: the strategy (fixed point or Newton) is in the file name.
solver_role_labels = {"ls_fine": "LS fine", "cp": "CP", "ls_coarse": "LS coarse", "tfa_baseline": "TFA",
                      "tfa_fft": "TFA FFT"}

load_path_style = {**figure_style, 'font.size': 12, 'axes.labelsize': 13, 'xtick.labelsize': 11,
                   'ytick.labelsize': 11, 'legend.fontsize': 12, 'lines.linewidth': 1.8, 'axes.spines.top': False,
                   'axes.spines.right': False}
legend_style = {'loc': 'outside upper center', 'frameon': False, 'handlelength': 3.2}

def get_roles_in_legend_order(role_per_solver, solver_names):
    # The named solvers, as (role, name), in the order of solver_role_styles.
    solver_per_role = {role_per_solver[solver_name]: solver_name for solver_name in solver_names}
    return [(role, solver_per_role[role]) for role in solver_role_styles if role in solver_per_role]

def mark_solver_failure(axis, x_values, values, completed_step_count, color):
    # An x at the step the solver gave up on, drawn level with the last value it produced so it reads as the
    # curve stopping rather than as a data point.
    if completed_step_count >= len(x_values) or completed_step_count == 0:
        return
    axis.plot(x_values[completed_step_count], values[completed_step_count - 1], marker='x', markersize=9,
              markeredgewidth=2, color=color, linestyle='none', clip_on=False, zorder=5)

## ------- Stress-Strain Curves ------- ##

macro_strain_component_labels = [r"$\bar{\varepsilon}_{11}$", r"$\bar{\varepsilon}_{22}$", r"$\bar{\varepsilon}_{33}$",
                                 r"$\bar{\gamma}_{12}$", r"$\bar{\gamma}_{23}$", r"$\bar{\gamma}_{13}$"]
# Normal stresses on the top row, shears on the bottom, each as (component in the engineering order 11, 22, 33, 12,
# 23, 13; its subscript).
stress_panel_rows = [[(0, "11"), (1, "22"), (2, "33")], [(4, "23"), (5, "13"), (3, "12")]]
# A component whose stress stays below this fraction of the largest, for every solver, is zero up to solver
# tolerance and round-off.
zero_stress_fraction = 1e-4
stress_strain_figure_width = 11
stress_strain_row_height = 2.7

def get_load_axis(applied_macro_strain, max_macro_strain):
    # The stress-strain curves' x axis, from the unloaded state, and its label: the applied strain in % when one
    # component is loaded, otherwise the load factor t of ε̄ = t ε̄max, labelled with ε̄max.
    loaded_components = np.flatnonzero(max_macro_strain)
    applied_macro_strain = np.vstack([np.zeros(6), applied_macro_strain])
    if len(loaded_components) == 1:
        component = loaded_components[0]
        return 100 * applied_macro_strain[:, component], f"{macro_strain_component_labels[component]} (%)"
    largest_component = np.abs(max_macro_strain).argmax()
    max_strain_description = ",  ".join(f"{macro_strain_component_labels[component]} = "
                                        f"{100 * max_macro_strain[component]:g} %" for component in loaded_components)
    return (applied_macro_strain[:, largest_component] / max_macro_strain[largest_component],
            r"Load factor $t$,   $\bar{\varepsilon} = t\,\bar{\varepsilon}_{\max}$,   $\bar{\varepsilon}_{\max}$:  "
            + max_strain_description)

def plot_stress_strain(stress_MPa_per_solver, role_per_solver, completed_step_count_per_solver, applied_macro_strain,
                       max_macro_strain, plot_path):
    # Each solver's macroscopic stress (MPa, one row per step) against the applied strain, one panel per component,
    # from the unloaded state to the last step it solved. role_per_solver maps each solver's name to its role. A row
    # whose components are zero for every solver is left out (uniaxial loading leaves every shear at zero), and a zero
    # panel in a kept row shows only "0".
    load_axis, load_axis_label = get_load_axis(applied_macro_strain, max_macro_strain)
    solved_stress_MPa = {solver_name: np.vstack([np.zeros(6), stress[:completed_step_count_per_solver[solver_name]]])
                         for solver_name, stress in stress_MPa_per_solver.items()}
    largest_stress = max(np.abs(stress).max() for stress in solved_stress_MPa.values())

    def is_zero(components):
        return all(np.abs(stress[:, components]).max() < zero_stress_fraction * largest_stress
                   for stress in solved_stress_MPa.values())

    panel_rows = [row for row in stress_panel_rows if not is_zero([component for component, _ in row])]
    with plt.rc_context(load_path_style):
        figure_height = 0.6 + stress_strain_row_height * len(panel_rows)
        figure, axes = plt.subplots(len(panel_rows), 3, squeeze=False, sharex=True, layout='constrained',
                                    figsize=(stress_strain_figure_width, figure_height))
        figure.get_layout_engine().set(w_pad=0.12, h_pad=0.1)
        for axis, (component, subscript) in zip(axes.flat, [panel for row in panel_rows for panel in row]):
            axis.set_ylabel(rf"$\bar{{\sigma}}_{{{subscript}}}$ (MPa)")
            axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
            axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            if is_zero([component]):
                axis.text(0.5, 0.5, "0", transform=axis.transAxes, ha='center', va='center', fontsize=20, color='0.6')
                axis.set_yticks([])
                axis.spines['left'].set_color('0.8')
                continue
            axis.grid(True, color='0.9', linewidth=0.8)
            axis.set_axisbelow(True)
            for role, solver_name in get_roles_in_legend_order(role_per_solver, solved_stress_MPa):
                stress = solved_stress_MPa[solver_name][:, component]
                axis.plot(load_axis[:len(stress)], stress, label=solver_role_labels[role], **solver_role_styles[role])
                mark_solver_failure(axis, load_axis[1:], stress[1:], len(stress) - 1,
                                    solver_role_styles[role]['color'])
        figure.supxlabel(load_axis_label, fontsize=13)
        figure.align_ylabels(axes)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, ncols=len(handles), columnspacing=2.4, **legend_style)
        figure.savefig(plot_path)
    plt.close(figure)
    print(f"stress-strain plot saved to {plot_path}")

## ------- Solver Performance ------- ##

solver_performance_figure_size = (10.5, 5.6)
iterations_log_axis_ratio = 20

class SolverSteps(NamedTuple):
    iterations_per_step: np.ndarray
    solve_time_per_step: np.ndarray
    completed_step_count: int

def shade_elastic_steps(axis, load_steps, elastic_steps):
    for load_step in load_steps[elastic_steps]:
        axis.axvspan(load_step - 0.5, load_step + 0.5, color='0.93', linewidth=0, zorder=0)

def get_time_per_iteration_ms(solver_steps):
    # Abandoned steps have zero iterations and zero time, so they stay at zero instead of being divided.
    time_per_iteration_ms = np.zeros(len(solver_steps.iterations_per_step))
    solved = solver_steps.iterations_per_step > 0
    time_per_iteration_ms[solved] = (1000 * solver_steps.solve_time_per_step[solved]
                                     / solver_steps.iterations_per_step[solved])
    return time_per_iteration_ms

def plot_solver_performance(steps_per_solver, role_per_solver, elastic_steps, plot_path):
    # Iterations and time per iteration at every step, for every solver in steps_per_solver (a dict from solver name
    # to SolverSteps), each up to the last step it solved, with its total solve time in the legend. elastic_steps
    # marks the steps where no partition yielded, which are shaded.
    load_step_count = len(elastic_steps)
    load_steps = np.arange(1, load_step_count + 1)
    with plt.rc_context(load_path_style):
        figure, (iterations_axis, time_per_iteration_axis) = plt.subplots(
            2, 1, sharex=True, figsize=solver_performance_figure_size, layout='constrained')
        for axis in (iterations_axis, time_per_iteration_axis):
            shade_elastic_steps(axis, load_steps, elastic_steps)

        for role, solver_name in get_roles_in_legend_order(role_per_solver, steps_per_solver):
            solver_steps = steps_per_solver[solver_name]
            style = solver_role_styles[role]
            solved_steps = load_steps[:solver_steps.completed_step_count]
            label = f"{solver_role_labels[role]}  {solver_steps.solve_time_per_step.sum():.2f} s"
            if solver_steps.completed_step_count < load_step_count:
                label += f", stopped at step {solver_steps.completed_step_count + 1}"
            time_per_iteration_ms = get_time_per_iteration_ms(solver_steps)
            for axis, values in ((iterations_axis, solver_steps.iterations_per_step),
                                 (time_per_iteration_axis, time_per_iteration_ms)):
                axis.step(solved_steps, values[:len(solved_steps)], where='mid', label=label, **style)
                mark_solver_failure(axis, load_steps, values, solver_steps.completed_step_count, style['color'])

        iterations_axis.set_ylabel("Iterations")
        # Counts more than an order of magnitude apart, such as the CP fixed point against TFA, share a log axis so
        # the smaller ones stay readable. Otherwise the axis is linear from zero.
        largest_counts = [solver_steps.iterations_per_step.max() for solver_steps in steps_per_solver.values()]
        if max(largest_counts) > iterations_log_axis_ratio * max(min(largest_counts), 1):
            iterations_axis.set_yscale('log')
            iterations_axis.set_ylim(bottom=0.8)
        else:
            iterations_axis.set_ylim(bottom=0)
            iterations_axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        time_per_iteration_axis.set_ylabel("ms per iteration")
        time_per_iteration_axis.set_ylim(bottom=0)
        time_per_iteration_axis.set_xlabel("Load step")
        time_per_iteration_axis.set_xlim(0.5, load_step_count + 0.5)
        figure.align_ylabels()
        handles, labels = iterations_axis.get_legend_handles_labels()
        figure.legend(handles, labels, ncols=len(handles), columnspacing=1.6, fontsize=11, **legend_style)
        figure.savefig(plot_path)
    plt.close(figure)
    print(f"solver performance plot saved to {plot_path}")

## ------- Time per Iteration Breakdown ------- ##

# One label per online time group (online_timing.online_time_groups), in the notes' symbols, then "Other". The
# sensitivity is H_μ for the Newton Jacobians and its partition average H_μ,0 for the FFT reference.
online_time_group_labels = [r"Material update $\mu(\varepsilon)$",
                            r"Induced strain $\mathbf{P}\mu$ or $\mathbf{P}_0\mu^*$",
                            r"Sensitivity $\mathbf{H}_\mu$", r"Build $\mathbf{M}_0^{-1}$",
                            r"Solve for $\delta\varepsilon$", "Other"]
online_time_group_colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#4a3aa7', '#d3d2cc']
time_breakdown_style = {**figure_style, 'font.size': 13, 'axes.labelsize': 14, 'xtick.labelsize': 12,
                        'ytick.labelsize': 14, 'legend.fontsize': 13}
time_breakdown_figure_size = (10, 3.6)
time_breakdown_legend_row_count = 2

def plot_time_per_iteration_breakdown(time_per_iteration_per_group_ms_per_solver, plot_path):
    # One bar per solver, top to bottom in the order given, split into the online time groups.
    solver_names = list(time_per_iteration_per_group_ms_per_solver)
    time_per_iteration_ms = np.array(list(time_per_iteration_per_group_ms_per_solver.values()))
    bar_positions = np.arange(len(solver_names))[::-1]

    segment_ends = np.cumsum(time_per_iteration_ms, axis=1)
    total_time_per_iteration_ms = segment_ends[:, -1]

    with plt.rc_context(time_breakdown_style):
        figure, axis = plt.subplots(figsize=time_breakdown_figure_size, layout='constrained')
        for group_time_ms, group_end, label, color in zip(time_per_iteration_ms.T, segment_ends.T,
                                                          online_time_group_labels, online_time_group_colors):
            axis.barh(bar_positions, group_time_ms, left=group_end - group_time_ms, height=0.62, color=color,
                      label=label, edgecolor='white', linewidth=1.2)
        for bar_position, total_time_ms in zip(bar_positions, total_time_per_iteration_ms):
            axis.annotate(f"{total_time_ms:.2f} ms", xy=(total_time_ms, bar_position), xytext=(8, 0),
                          textcoords='offset points', va='center', color='0.2')

        axis.set_yticks(bar_positions, solver_names)
        axis.tick_params(axis='y', length=0)
        axis.set_xlabel("Time per iteration (ms)")
        axis.set_xlim(0, 1.12 * total_time_per_iteration_ms.max())
        axis.xaxis.grid(True, color='0.9')
        axis.set_axisbelow(True)
        axis.spines[['top', 'right', 'left']].set_visible(False)
        handles, labels = axis.get_legend_handles_labels()
        row_major_order = np.arange(len(handles)).reshape(time_breakdown_legend_row_count, -1).T.ravel()
        axis.legend([handles[i] for i in row_major_order], [labels[i] for i in row_major_order], loc='lower center',
                    bbox_to_anchor=(0.5, 1.02), ncols=len(handles) // time_breakdown_legend_row_count, frameon=False,
                    handlelength=1.0, handletextpad=0.5, columnspacing=2.0)

        figure.savefig(plot_path)
    plt.close(figure)
    print(f"time per iteration breakdown plot saved to {plot_path}")

## ------- Cross Sections ------- ##

cross_section_figure_size = (5, 5)

def draw_element_and_partition_edges(axis, element_number_per_side, partition_number_per_side):
    # Element edges thin and light, partition edges darker. With one partition per element only the light grid is
    # drawn, which would otherwise bury a fine map under dark lines.
    element_edges = np.linspace(0, 1, element_number_per_side + 1)
    axis.vlines(element_edges, 0, 1, color='0.75', linewidth=0.5)
    axis.hlines(element_edges, 0, 1, color='0.75', linewidth=0.5)
    if partition_number_per_side < element_number_per_side:
        partition_edges = np.linspace(0, 1, partition_number_per_side + 1)
        axis.vlines(partition_edges, 0, 1, color='0.25', linewidth=1.2, clip_on=False)
        axis.hlines(partition_edges, 0, 1, color='0.25', linewidth=1.2, clip_on=False)

def plot_cross_section(cross_section_material_ids, element_number_per_side, plot_path):
    partition_number_per_side = cross_section_material_ids.shape[0]

    with plt.rc_context(figure_style):
        figure, axis = plt.subplots(figsize=cross_section_figure_size, layout='constrained')
        axis.imshow(cross_section_material_ids, cmap=matplotlib.colors.ListedColormap(['0.92', '0.55']), vmin=0,
                    vmax=1, origin='lower', extent=(0, 1, 0, 1))
        draw_element_and_partition_edges(axis, element_number_per_side, partition_number_per_side)
        axis.set_axis_off()
        figure.savefig(plot_path)
    plt.close(figure)
    print(f"cross section plot saved to {plot_path}")

## ------- von Mises Stress Maps ------- ##

von_mises_style = {**figure_style, 'font.size': 13, 'axes.titlesize': 15, 'figure.titlesize': 14}
von_mises_panel_size = 4.2

def draw_inclusion_outline(axis, cross_section_material_ids):
    # The partition edges between inclusion and matrix, in white, which stands out from the stress map underneath.
    partition_number_per_side = cross_section_material_ids.shape[0]
    edge_length = 1 / partition_number_per_side
    rows, columns = np.nonzero(cross_section_material_ids != np.roll(cross_section_material_ids, 1, axis=0))
    axis.hlines(rows * edge_length, columns * edge_length, (columns + 1) * edge_length, color='white', linewidth=2.5)
    rows, columns = np.nonzero(cross_section_material_ids != np.roll(cross_section_material_ids, 1, axis=1))
    axis.vlines(columns * edge_length, rows * edge_length, (rows + 1) * edge_length, color='white', linewidth=2.5)

def draw_stress_map(axis, stress_map_MPa, cross_section_material_ids, element_number_per_side, title, **imshow_kwargs):
    image = axis.imshow(stress_map_MPa, origin='lower', extent=(0, 1, 0, 1), **imshow_kwargs)
    draw_element_and_partition_edges(axis, element_number_per_side, cross_section_material_ids.shape[0])
    draw_inclusion_outline(axis, cross_section_material_ids)
    axis.set_title(title)
    axis.set_axis_off()
    return image

class StressMap(NamedTuple):
    # One solver's map, on its own lattice: von Mises stress per partition (MPa) and phase per partition, both
    # (side × side), and the side of its element grid.
    title: str
    von_mises_MPa: np.ndarray
    material_ids: np.ndarray
    element_number_per_side: int

def plot_von_mises_stress(stress_maps, load_step, load_step_count, stress_min_MPa, stress_max_MPa, plot_path):
    # One map per solver, side by side on one colour scale, between the given limits or, where a limit is None, the
    # maps' own. Maps may sit on different lattices, since each is drawn over the whole cross-section.
    with plt.rc_context(von_mises_style):
        figure, axes = plt.subplots(1, len(stress_maps), squeeze=False, layout='constrained',
                                    figsize=(von_mises_panel_size * len(stress_maps) + 1.2, von_mises_panel_size))
        all_maps_MPa = [stress_map.von_mises_MPa for stress_map in stress_maps]
        stress_limits = {'vmin': min(map(np.min, all_maps_MPa)) if stress_min_MPa is None else stress_min_MPa,
                         'vmax': max(map(np.max, all_maps_MPa)) if stress_max_MPa is None else stress_max_MPa}
        for axis, stress_map in zip(axes[0], stress_maps):
            image = draw_stress_map(axis, stress_map.von_mises_MPa, stress_map.material_ids,
                                    stress_map.element_number_per_side, stress_map.title, cmap='jet', **stress_limits)
        figure.colorbar(image, ax=axes[0], label="von Mises stress (MPa)", shrink=0.85)
        figure.suptitle(f"Load step {load_step} of {load_step_count}", color='0.3')
        figure.savefig(plot_path)
    plt.close(figure)
    print(f"von Mises stress plot saved to {plot_path}")
