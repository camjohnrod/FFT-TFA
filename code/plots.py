from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.colors
import matplotlib.legend
import matplotlib.lines

## ------- Load Path Summary ------- ##

macro_strain_component_labels = [r"$\bar{\varepsilon}_{11}$", r"$\bar{\varepsilon}_{22}$", r"$\bar{\varepsilon}_{33}$",
                                 r"$\bar{\gamma}_{12}$", r"$\bar{\gamma}_{23}$", r"$\bar{\gamma}_{13}$"]

load_path_summary_style = {'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
                           'legend.fontsize': 15, 'axes.titlesize': 22, 'lines.linewidth': 2.5}
load_path_summary_figure_size = (10, 10)
deviatoric_stress_component_styles = [('red', 'o', r"$\bar{\mathbf{S}}_{11}$"),
                                      ('blue', 's', r"$\bar{\mathbf{S}}_{22}$"),
                                      ('green', '^', r"$\bar{\mathbf{S}}_{33}$")]
# One colour per solver, in the order the solvers are given. Their legends are stacked beside the iterations panel.
solver_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

class SolverSteps(NamedTuple):
    iterations_per_step: np.ndarray
    solve_time_per_step: np.ndarray
    completed_step_count: int

def get_applied_strain_description(max_macro_strain, strain_increment_count):
    nonzero_components = [f"{label} = {value:g}" for label, value in zip(macro_strain_component_labels, max_macro_strain)
                          if value != 0]
    component_lines = [", ".join(nonzero_components[start:start + 3]) for start in range(0, len(nonzero_components), 3)]
    return "\n".join([f"Applied Strain ({strain_increment_count} steps)"] + component_lines)

def shade_elastic_steps(axis, load_steps, elastic_steps):
    for load_step in load_steps[elastic_steps]:
        axis.axvspan(load_step - 0.5, load_step + 0.5, color='0.9', linewidth=0, zorder=0)

def label_elastic_steps(axis, load_steps, elastic_steps, max_font_size=13, min_font_size=8, band_fill_fraction=0.85):
    elastic_load_steps = load_steps[elastic_steps]
    if len(elastic_load_steps) == 0:
        return
    band_left, band_right = elastic_load_steps[0] - 0.5, elastic_load_steps[-1] + 0.5
    label = axis.annotate("Elastic\nregime", xy=((band_left + band_right) / 2, 1), xycoords=axis.get_xaxis_transform(),
                          xytext=(0, -8), textcoords='offset points', ha='center', va='top', color='0.4',
                          fontsize=max_font_size, linespacing=1.1)
    figure = axis.get_figure()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    band_width_pixels = abs(np.diff(axis.transData.transform([(band_left, 0), (band_right, 0)])[:, 0])[0])
    label_width_pixels = label.get_window_extent(renderer).width
    fitted_font_size = max_font_size * min(1.0, band_fill_fraction * band_width_pixels / label_width_pixels)
    rotated_font_size = min(max_font_size, band_fill_fraction * band_width_pixels * 72 / figure.dpi / 1.2)
    if fitted_font_size >= min_font_size:
        label.set_fontsize(fitted_font_size)
    elif rotated_font_size >= min_font_size:
        label.set_text("Elastic")
        label.set_rotation(90)
        label.set_fontsize(rotated_font_size)
    else:
        label.set_text("◂ Elastic")
        label.xy = (band_right, 1)
        label.set_horizontalalignment('left')
        label.set_position((4, -8))
        label.set_fontsize(min_font_size + 2)

def add_solver_legend(axis, line, solver_name, solver_steps, vertical_anchor):
    no_line = matplotlib.lines.Line2D([], [], linestyle='none')
    totals_label = (f"  {solver_steps.iterations_per_step.sum()} iterations,\n"
                    f"  {solver_steps.solve_time_per_step.sum():.2f} s total time")
    if solver_steps.completed_step_count < len(solver_steps.iterations_per_step):
        totals_label += f"\n  (stopped at step {solver_steps.completed_step_count + 1})"
    solver_legend = matplotlib.legend.Legend(axis, [line, no_line], [f"{solver_name}:", totals_label],
                                             labelspacing=0.4, handlelength=1.5, frameon=False,
                                             loc='center left', bbox_to_anchor=(1.01, vertical_anchor))
    solver_name_text, totals_text = solver_legend.get_texts()
    solver_name_text.set_fontweight('bold')
    solver_name_text.set_fontsize(15)
    totals_text.set_fontsize(13)
    axis.add_artist(solver_legend)
    solver_legend.set_clip_on(False)

def mark_solver_failure(axis, load_steps, values, completed_step_count, color):
    # An x at the step the solver gave up on, drawn level with the last value it produced so it reads as the
    # curve stopping rather than as a data point.
    if completed_step_count >= len(load_steps) or completed_step_count == 0:
        return
    axis.plot(load_steps[completed_step_count], values[completed_step_count - 1], marker='x', markersize=11,
              markeredgewidth=2.5, color=color, linestyle='none', clip_on=False, zorder=5)

def get_time_per_iteration_ms(solver_steps):
    # Abandoned steps have zero iterations and zero time, so they stay at zero instead of being divided.
    time_per_iteration_ms = np.zeros(len(solver_steps.iterations_per_step))
    solved = solver_steps.iterations_per_step > 0
    time_per_iteration_ms[solved] = (1000 * solver_steps.solve_time_per_step[solved]
                                     / solver_steps.iterations_per_step[solved])
    return time_per_iteration_ms

def plot_load_path_summary(macroscopic_deviatoric_stress_MPa, stress_solver_name, steps_per_solver, elastic_steps,
                           title, plot_path):
    # The stress panel shows the solver named stress_solver_name; the iteration and timing panels show every solver
    # in steps_per_solver, a dict from solver name to SolverSteps. elastic_steps marks the steps where no partition
    # yielded, which are shaded.
    stress_solver_steps = steps_per_solver[stress_solver_name]
    strain_increment_count = len(stress_solver_steps.iterations_per_step)
    load_steps = np.arange(1, strain_increment_count + 1)

    with plt.rc_context(load_path_summary_style):
        figure, (stress_axis, iterations_axis, time_per_iteration_axis) = plt.subplots(
            3, 1, sharex=True, figsize=load_path_summary_figure_size, layout='constrained')
        title_heading, title_details = title.split("\n", 1)
        stress_axis.set_title(title_details, pad=16, linespacing=1.4, fontsize=17, color='0.2')
        stress_axis.annotate(title_heading, xy=(0.5, 1), xycoords=stress_axis.title, xytext=(0, 6),
                             textcoords='offset points', ha='center', va='bottom',
                             fontsize=load_path_summary_style['axes.titlesize'])
        for axis in (stress_axis, iterations_axis, time_per_iteration_axis):
            shade_elastic_steps(axis, load_steps, elastic_steps)

        stress_step_count = stress_solver_steps.completed_step_count
        solved_stress = macroscopic_deviatoric_stress_MPa[:stress_step_count]
        for component, (color, marker, label) in enumerate(deviatoric_stress_component_styles):
            stress_axis.plot(load_steps[:stress_step_count], solved_stress[:, component], color=color, marker=marker,
                             markersize=4, linewidth=2, label=label)
        stress_axis.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
        stress_axis.set_ylabel("Deviatoric stress (MPa)")

        for solver_index, (solver_name, solver_steps) in enumerate(steps_per_solver.items()):
            color = solver_colors[solver_index]
            legend_anchor = 1 - (solver_index + 0.5) / len(steps_per_solver)
            solved_steps = load_steps[:solver_steps.completed_step_count]
            time_per_iteration_ms = get_time_per_iteration_ms(solver_steps)

            line, = iterations_axis.step(solved_steps, solver_steps.iterations_per_step[:len(solved_steps)],
                                         where='mid', color=color)
            add_solver_legend(iterations_axis, line, solver_name, solver_steps, legend_anchor)
            mark_solver_failure(iterations_axis, load_steps, solver_steps.iterations_per_step,
                                solver_steps.completed_step_count, color)

            time_per_iteration_axis.step(solved_steps, time_per_iteration_ms[:len(solved_steps)], where='mid',
                                         color=color)
            mark_solver_failure(time_per_iteration_axis, load_steps, time_per_iteration_ms,
                                solver_steps.completed_step_count, color)

        iterations_axis.set_ylabel("Iterations")
        iterations_axis.set_ylim(bottom=0)
        iterations_axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        time_per_iteration_axis.set_ylabel("Time per iteration (ms)")
        time_per_iteration_axis.set_ylim(bottom=0)
        time_per_iteration_axis.set_xlabel("Load step", labelpad=12)
        time_per_iteration_axis.set_xlim(0.5, strain_increment_count + 0.5)
        figure.align_ylabels()
        label_elastic_steps(stress_axis, load_steps, elastic_steps)

        figure.savefig(plot_path)
    plt.close(figure)
    print(f"load path summary plot saved to {plot_path}")

## ------- Time per Iteration Breakdown ------- ##

online_time_group_labels = [r"Material update $\mu(\varepsilon)$", r"Induced strain $\mathbf{P}\mu$",
                            r"Sensitivity $\tilde{\mathbf{M}}_{\mu,0}$",
                            r"Build $(\mathbf{I}-\mathbf{P}_0\tilde{\mathbf{M}}_{\mu,0})^{-1}$",
                            r"Solve for $\delta\varepsilon$", "Other"]
online_time_group_colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#4a3aa7', '#d3d2cc']
time_breakdown_style = {'font.size': 13, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 14,
                        'legend.fontsize': 13}
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
    element_edges = np.linspace(0, 1, element_number_per_side + 1)
    partition_edges = np.linspace(0, 1, partition_number_per_side + 1)
    axis.vlines(element_edges, 0, 1, color='0.75', linewidth=0.5)
    axis.hlines(element_edges, 0, 1, color='0.75', linewidth=0.5)
    axis.vlines(partition_edges, 0, 1, color='0.25', linewidth=1.2, clip_on=False)
    axis.hlines(partition_edges, 0, 1, color='0.25', linewidth=1.2, clip_on=False)

def plot_cross_section(cross_section_material_ids, element_number_per_side, plot_path):
    partition_number_per_side = cross_section_material_ids.shape[0]

    figure, axis = plt.subplots(figsize=cross_section_figure_size, layout='constrained')
    axis.imshow(cross_section_material_ids, cmap=matplotlib.colors.ListedColormap(['0.92', '0.55']), vmin=0, vmax=1,
                origin='lower', extent=(0, 1, 0, 1))
    draw_element_and_partition_edges(axis, element_number_per_side, partition_number_per_side)
    axis.set_axis_off()

    figure.savefig(plot_path)
    plt.close(figure)
    print(f"cross section plot saved to {plot_path}")
