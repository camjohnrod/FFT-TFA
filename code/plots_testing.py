from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.colors
import matplotlib.legend
import matplotlib.lines

## ------- Load Path Summary ------- ##

class SolverSeries(NamedTuple):
    name: str
    color: str
    iterations_per_step: np.ndarray
    solve_time_per_step: np.ndarray

macro_strain_component_labels = [r"$\bar{\varepsilon}_{11}$", r"$\bar{\varepsilon}_{22}$", r"$\bar{\varepsilon}_{33}$",
                                 r"$\bar{\gamma}_{12}$", r"$\bar{\gamma}_{23}$", r"$\bar{\gamma}_{13}$"]

load_path_summary_style = {'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
                           'legend.fontsize': 15, 'axes.titlesize': 22, 'lines.linewidth': 2.5}
load_path_summary_figure_size = (10, 12.5)

def get_applied_strain_description(max_macro_strain, strain_increment_count):
    nonzero_components = [f"{label} = {value:g}" for label, value in zip(macro_strain_component_labels, max_macro_strain)
                          if value != 0]
    component_lines = [", ".join(nonzero_components[start:start + 3]) for start in range(0, len(nonzero_components), 3)]
    return "\n".join([f"Applied Strain ({strain_increment_count} steps)"] + component_lines)

def shade_elastic_steps(axis, load_steps, elastic_steps_mask):
    for load_step in load_steps[elastic_steps_mask]:
        axis.axvspan(load_step - 0.5, load_step + 0.5, color='0.9', linewidth=0, zorder=0)

def label_elastic_steps(axis, load_steps, elastic_steps_mask, max_font_size=13, min_font_size=8,
                        band_fill_fraction=0.85):
    elastic_steps = load_steps[elastic_steps_mask]
    if len(elastic_steps) == 0:
        return
    band_left, band_right = elastic_steps[0] - 0.5, elastic_steps[-1] + 0.5
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

def add_solver_legend(axis, line, solver_name, iterations_per_step, solve_time_per_step, vertical_anchor):
    no_line = matplotlib.lines.Line2D([], [], linestyle='none')
    totals_label = f"  {iterations_per_step.sum()} iterations,\n  {solve_time_per_step.sum():.2f} s total time"
    solver_legend = matplotlib.legend.Legend(axis, [line, no_line], [f"{solver_name}:", totals_label],
                                             labelspacing=0.4, handlelength=1.5, frameon=False,
                                             loc='center left', bbox_to_anchor=(1.01, vertical_anchor))
    solver_name_text, totals_text = solver_legend.get_texts()
    solver_name_text.set_fontweight('bold')
    solver_name_text.set_fontsize(15)
    totals_text.set_fontsize(13)
    axis.add_artist(solver_legend)
    solver_legend.set_clip_on(False)

def plot_load_path_summary(macroscopic_deviatoric_stress_MPa, solver_series, elastic_steps_mask,
                           model_difference_percent, model_difference_label, title, plot_path):
    strain_increment_count = len(elastic_steps_mask)
    load_steps = np.arange(1, strain_increment_count + 1)

    with plt.rc_context(load_path_summary_style):
        figure, (stress_axis, difference_axis, iterations_axis, time_axis) = plt.subplots(
            4, 1, sharex=True, figsize=load_path_summary_figure_size, layout='constrained')
        title_heading, title_details = title.split("\n", 1)
        stress_axis.set_title(title_details, pad=16, linespacing=1.4, fontsize=17, color='0.2')
        stress_axis.annotate(title_heading, xy=(0.5, 1), xycoords=stress_axis.title, xytext=(0, 6),
                             textcoords='offset points', ha='center', va='bottom',
                             fontsize=load_path_summary_style['axes.titlesize'])
        for axis in (stress_axis, difference_axis, iterations_axis, time_axis):
            shade_elastic_steps(axis, load_steps, elastic_steps_mask)

        stress_axis.plot(load_steps, macroscopic_deviatoric_stress_MPa[:, 0], color='red', marker='o', markersize=4,
                         linewidth=2, label=r"$\bar{\mathbf{S}}_{11}$")
        stress_axis.plot(load_steps, macroscopic_deviatoric_stress_MPa[:, 1], color='blue', marker='s', markersize=4,
                         linewidth=2, label=r"$\bar{\mathbf{S}}_{22}$")
        stress_axis.plot(load_steps, macroscopic_deviatoric_stress_MPa[:, 2], color='green', marker='^', markersize=4,
                         linewidth=2, label=r"$\bar{\mathbf{S}}_{33}$")
        stress_axis.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
        stress_axis.set_ylabel("Deviatoric stress (MPa)")

        difference_axis.plot(load_steps, model_difference_percent, color='tab:green', marker='o', markersize=3,
                             linewidth=2)
        difference_axis.set_ylabel("Model difference (%)")
        difference_axis.set_ylim(bottom=0)
        difference_axis.annotate(model_difference_label, xy=(0, 1), xycoords='axes fraction', xytext=(8, -8),
                                 textcoords='offset points', ha='left', va='top', fontsize=12, color='0.35')

        for index, series in enumerate(solver_series):
            line, = iterations_axis.step(load_steps, series.iterations_per_step, where='mid', color=series.color)
            add_solver_legend(iterations_axis, line, series.name, series.iterations_per_step,
                              series.solve_time_per_step, 1 - (index + 0.5) / len(solver_series))
            time_axis.step(load_steps, 1000 * series.solve_time_per_step, where='mid', color=series.color)
        iterations_axis.set_ylabel("Iterations")
        # Headroom so the caveat below sits clear of the curves rather than on top of them.
        iterations_axis.set_ylim(bottom=0, top=1.45 * max(series.iterations_per_step.max()
                                                          for series in solver_series))
        iterations_axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        iterations_axis.annotate("one iteration is not the same unit for every solver:\n"
                                 "a Newton step contains a whole Krylov solve", xy=(0, 1), xycoords='axes fraction',
                                 xytext=(8, -8), textcoords='offset points', ha='left', va='top', fontsize=12,
                                 color='0.35', linespacing=1.3)

        time_axis.set_ylabel("Solve time per step (ms)")
        time_axis.set_ylim(bottom=0)
        time_axis.set_xlabel("Load step", labelpad=12)
        time_axis.set_xlim(0.5, strain_increment_count + 0.5)
        figure.align_ylabels()
        label_elastic_steps(stress_axis, load_steps, elastic_steps_mask)

        figure.savefig(plot_path)
    plt.close(figure)
    print(f"load path summary plot saved to {plot_path}")

## ------- Cost Breakdown ------- ##

online_time_group_labels = [r"Material update $\mu(\varepsilon)$",
                            r"Operator applies $\mathbf{P}\mu$ or $\mathbf{P}_0\mu^*$",
                            r"Sensitivity / tangent",
                            r"Build $(\mathbf{I}-\mathbf{P}_0\tilde{\mathbf{M}}_{\mu,0})^{-1}$",
                            r"Correction solve, excluding its operator applies", "Other"]
online_time_group_colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#4a3aa7', '#d3d2cc']
time_breakdown_style = {'font.size': 13, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 13,
                        'legend.fontsize': 13}
time_breakdown_figure_size = (10, 5.2)
time_breakdown_legend_row_count = 3

def plot_cost_breakdown(total_time_per_group_s, solver_names, iteration_counts, application_counts, plot_path):
    total_time_per_group_s = np.array(total_time_per_group_s)
    bar_positions = np.arange(len(solver_names))[::-1]

    segment_ends = np.cumsum(total_time_per_group_s, axis=1)
    total_time_s = segment_ends[:, -1]

    with plt.rc_context(time_breakdown_style):
        figure, axis = plt.subplots(figsize=time_breakdown_figure_size, layout='constrained')
        for group_time_s, group_end, label, color in zip(total_time_per_group_s.T, segment_ends.T,
                                                         online_time_group_labels, online_time_group_colors):
            axis.barh(bar_positions, group_time_s, left=group_end - group_time_s, height=0.62, color=color,
                      label=label, edgecolor='white', linewidth=1.2)
        for bar_position, total, iterations, applications in zip(bar_positions, total_time_s, iteration_counts,
                                                                application_counts):
            axis.annotate(f"{total:.2f} s", xy=(total, bar_position), xytext=(8, 5), textcoords='offset points',
                          va='center', color='0.2')
            axis.annotate(f"{iterations} iterations, {applications} operator applications",
                          xy=(total, bar_position), xytext=(8, -8), textcoords='offset points', va='center',
                          color='0.45', fontsize=10)

        axis.set_yticks(bar_positions, solver_names)
        axis.tick_params(axis='y', length=0)
        axis.set_xlabel("Total online solve time over the load path (s)")
        # Room for the two annotation lines to the right of the longest bar.
        axis.set_xlim(0, 1.62 * total_time_s.max())
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
    print(f"cost breakdown plot saved to {plot_path}")

## ------- Cross Sections ------- ##

cross_section_figure_size = (5, 5)
von_mises_cross_section_figure_size = (6, 5)

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

def plot_von_mises_difference_cross_section(cross_section_difference_MPa, element_number_per_side,
                                            partition_number_per_side, plot_path):
    difference_limit = max(np.abs(cross_section_difference_MPa).max(), np.finfo(float).tiny)

    figure, axis = plt.subplots(figsize=von_mises_cross_section_figure_size, layout='constrained')
    image = axis.imshow(cross_section_difference_MPa, cmap='RdBu_r', vmin=-difference_limit, vmax=difference_limit,
                        origin='lower', extent=(0, 1, 0, 1))
    figure.colorbar(image, ax=axis, label="Reference model $-$ actual E/P (MPa)")
    draw_element_and_partition_edges(axis, element_number_per_side, partition_number_per_side)
    axis.set_axis_off()

    figure.savefig(plot_path)
    plt.close(figure)
    print(f"von Mises model difference plot saved to {plot_path}")

def plot_von_mises_cross_section(cross_section_von_mises_stress_MPa, element_number_per_side, partition_number_per_side,
                                 min_stress_MPa, max_stress_MPa, plot_path):
    figure, axis = plt.subplots(figsize=von_mises_cross_section_figure_size, layout='constrained')
    image = axis.imshow(cross_section_von_mises_stress_MPa, cmap='jet', vmin=min_stress_MPa, vmax=max_stress_MPa,
                        origin='lower', extent=(0, 1, 0, 1))
    figure.colorbar(image, ax=axis, extend='both', label="von Mises stress (MPa)")
    draw_element_and_partition_edges(axis, element_number_per_side, partition_number_per_side)
    axis.set_axis_off()

    figure.savefig(plot_path)
    plt.close(figure)
    print(f"von Mises cross section plot saved to {plot_path}")
