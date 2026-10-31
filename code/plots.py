# Every figure run.py saves: the macroscopic stress-strain curves, each method's discretization of the cross-section,
# and maps of the von Mises stress and the accumulated plastic strain.
# Every method is drawn the same way in every figure (method_styles). Plot appearance is set here; no solver logic.

from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.colors
import matplotlib.lines

## ------- Shared Style ------- ##

# Every figure's text in Computer Modern, the math included, so labels, ticks and symbols match. Tick labels go through
# mathtext, since cmr10 has no minus sign of its own.
figure_style = {'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
                'axes.formatter.use_mathtext': True, 'axes.formatter.useoffset': False, 'savefig.dpi': 200}

# One look per line, in legend order: told apart by colour (Okabe–Ito, whose pairs stay distinct under simulated
# colour blindness) and by dash pattern. The reference, when it runs, is solid black, drawn first so every method shows
# on top of it. FFT-TFA, the method being tested, is dash-dotted, Spectral LS dotted. TFA, the baseline, is a thinner
# dash drawn on top, so it stays visible where the curves agree. CP, the earlier method, keeps its colour,
# dash-dot-dotted.
method_styles = {
    "Reference":   {'color': 'black', 'linestyle': '-', 'zorder': 0},
    "FFT-TFA":     {'color': '#009E73', 'linestyle': (0, (6, 2, 1.5, 2)), 'zorder': 2},
    "Spectral LS": {'color': '#0072B2', 'linestyle': (0, (1, 1.6)), 'dash_capstyle': 'round', 'zorder': 3},
    "TFA":         {'color': '#CC79A7', 'linestyle': (0, (5, 3)), 'linewidth': 1.4, 'zorder': 4},
    "CP":          {'color': '#D55E00', 'linestyle': (0, (6, 2, 1.5, 2, 1.5, 2)), 'zorder': 1},
}

load_path_style = {**figure_style, 'font.size': 12, 'axes.labelsize': 13, 'xtick.labelsize': 11,
                   'ytick.labelsize': 11, 'legend.fontsize': 12, 'lines.linewidth': 1.8, 'axes.spines.top': False,
                   'axes.spines.right': False}
legend_style = {'loc': 'outside upper center', 'frameon': False, 'handlelength': 3.2}

def mark_method_failure(axis, x_values, values, completed_step_count, color):
    # An x at the step the method gave up on, drawn level with the last value it produced so it reads as the
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
# A component whose stress stays below this fraction of the largest, for every method, is zero up to solver
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

def plot_stress_strain(stress_MPa_per_method, completed_step_count_per_method, applied_macro_strain,
                       max_macro_strain, plot_path):
    # Each method's macroscopic stress (MPa, one row per step) against the applied strain, one panel per component,
    # from the unloaded state to the last step it solved. A row whose components are zero for every method is left
    # out (uniaxial loading leaves every shear at zero), and a zero panel in a kept row shows only "0".
    load_axis, load_axis_label = get_load_axis(applied_macro_strain, max_macro_strain)
    solved_stress_MPa = {method: np.vstack([np.zeros(6), stress[:completed_step_count_per_method[method]]])
                         for method, stress in stress_MPa_per_method.items()}
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
            for method, method_stress in solved_stress_MPa.items():
                stress = method_stress[:, component]
                axis.plot(load_axis[:len(stress)], stress, label=method, **method_styles[method])
                mark_method_failure(axis, load_axis[1:], stress[1:], len(stress) - 1, method_styles[method]['color'])
        figure.supxlabel(load_axis_label, fontsize=13)
        figure.align_ylabels(axes)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, ncols=len(handles), columnspacing=2.4, **legend_style)
        figure.savefig(plot_path)
    plt.close(figure)

## ------- Macroscopic Stress Errors ------- ##

# Each stress component's subscript, in the engineering order 11, 22, 33, 12, 23, 13.
stress_component_subscripts = ["11", "22", "33", "12", "23", "13"]
error_zero_line_style = {'color': '0.6', 'linewidth': 0.8, 'zorder': 1}

def plot_macro_stress_errors(error_percent_per_method, components, step_count, plot_path):
    # Each method's signed macroscopic stress error (%, one row per step it solved, one column per component in
    # components) against the load step, one panel per component, three to a row as in the stress-strain figure,
    # from the unloaded state (step 0, no error) to the last step the method solved. The load step, not the applied
    # strain, so that a cyclic path unrolls instead of folding back on itself.
    load_steps = np.arange(step_count + 1)
    row_count = -(-len(components) // 3)
    with plt.rc_context(load_path_style):
        figure, axes = plt.subplots(row_count, 3, squeeze=False, sharex=True, layout='constrained',
                                    figsize=(stress_strain_figure_width, 0.6 + stress_strain_row_height * row_count))
        figure.get_layout_engine().set(w_pad=0.12, h_pad=0.1)
        for axis in axes.flat[len(components):]:
            axis.set_visible(False)
        for column, (axis, component) in enumerate(zip(axes.flat, components)):
            subscript = stress_component_subscripts[component]
            axis.set_ylabel(rf"$\Delta\bar{{\sigma}}_{{{subscript}}}\,/\,\max|\bar{{\sigma}}_{{{subscript}}}^{{\,"
                            rf"\mathrm{{ref}}}}|$ (%)")
            axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5, integer=True))
            axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            axis.grid(True, color='0.9', linewidth=0.8)
            axis.set_axisbelow(True)
            axis.axhline(0, **error_zero_line_style)
            for method, error_percent in error_percent_per_method.items():
                error = np.concatenate([[0], error_percent[:, column]])
                axis.plot(load_steps[:len(error)], error, label=method, **method_styles[method])
                mark_method_failure(axis, load_steps[1:], error[1:], len(error) - 1, method_styles[method]['color'])
        figure.supxlabel("Load step", fontsize=13)
        figure.align_ylabels(axes)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, ncols=len(handles), columnspacing=2.4, **legend_style)
        figure.savefig(plot_path)
    plt.close(figure)

## ------- Cross Sections ------- ##

cross_section_style = {**figure_style, 'font.size': 12, 'axes.titlesize': 13}
cross_section_panel_size = 4.2
cross_section_legend_row_height = 0.3    # inches of figure height per row of the legend

# A method's clusters, when it colours them in, fill their elements: each matrix cluster in its own colour, lightened so
# the boundaries stay visible, and the inclusion's clusters in greys, so the fiber still reads as the fiber. Elements
# of one cluster share its colour wherever they lie, and methods with the same clusters colour them alike. The matrix
# colours are tab20's dark colours, then its light ones (so consecutive clusters differ), then tab20b's and tab20c's,
# without any grey (no colour whose channels lie within grey_tolerance of each other), so no matrix cluster looks like
# the inclusion; they repeat past 53 matrix clusters, where the cluster boundaries still tell clusters apart.
grey_tolerance = 0.1
matrix_cluster_colors = np.array([color for color in plt.cm.tab20.colors[::2] + plt.cm.tab20.colors[1::2]
                                  + plt.cm.tab20b.colors + plt.cm.tab20c.colors
                                  if max(color) - min(color) > grey_tolerance])
cluster_color_lightening = 0.25
inclusion_cluster_greys = (0.62, 0.48, 0.76)

class CellMap(NamedTuple):
    # One method's discretization on its element grid, with a title and a line of its settings. boundaries maps each
    # kind of line it draws ("voxel", "cluster", "partition", boundary_styles) to a map of ids on that grid, the line
    # running along every element edge whose two sides have different ids. fill_cluster_ids colours each element by its
    # cluster (k-means clusters), or is None to show only the phases. element_material_ids are the phases on the
    # method's own elements (Spectral LS's), or None for the mesh's.
    title: str
    settings: str
    boundaries: dict
    fill_cluster_ids: np.ndarray = None
    element_material_ids: np.ndarray = None

# The kinds of line, drawn in this order, each named in the legend by what it bounds: hairline grey for Spectral LS's
# voxels, so its fine grid shows without competing, lighter than the clusters' lines so the two read apart; thin dark
# grey for the clusters (one material state each), visible on the light phases and the cluster colours without
# drawing the eye; thick black for the partitions of FFT-TFA and CP, the cells of their FFT, drawn on top where the
# two coincide. Every panel is framed alike.
boundary_styles = {
    "voxel":     {'label': "voxel", 'color': '0.62', 'linewidth': 0.3},
    "cluster":   {'label': "cluster: uniform eigenstrain", 'color': '0.2', 'linewidth': 0.5},
    "partition": {'label': "partition: FFT cell", 'color': 'black', 'linewidth': 1.2},
}
frame_style = {'color': '0.1', 'linewidth': 0.8}

def get_runs(rows, columns):
    # Element edges, by the row they lie on and their column (sorted by row, then column), merged into straight runs:
    # each run's row, first column and stop column.
    breaks = np.flatnonzero((np.diff(rows) != 0) | (np.diff(columns) != 1)) + 1
    firsts = np.concatenate(([0], breaks))
    lasts = np.concatenate((breaks, [len(rows)])) - 1
    return rows[firsts], columns[firsts], columns[lasts] + 1

def draw_boundaries(axis, ids, style):
    # A line along every element edge inside the cross-section whose two sides have different ids, each straight run
    # of edges drawn as one line. The edges on the cell's periodic boundary are left to the panel's frame.
    edge_length = 1 / ids.shape[0]
    line_style = {key: value for key, value in style.items() if key != 'label'}
    rows, firsts, stops = get_runs(*np.nonzero(ids[1:] != ids[:-1]))
    axis.hlines((rows + 1) * edge_length, firsts * edge_length, stops * edge_length, **line_style)
    columns, firsts, stops = get_runs(*np.nonzero((ids[:, 1:] != ids[:, :-1]).T))
    axis.vlines((columns + 1) * edge_length, firsts * edge_length, stops * edge_length, **line_style)

def get_cluster_image(cluster_ids, element_material_ids):
    # An RGB image of the cross-section with every element in its cluster's colour.
    colors = np.zeros((cluster_ids.max() + 1, 3))
    matrix_clusters = np.unique(cluster_ids[element_material_ids == 0])
    inclusion_clusters = np.unique(cluster_ids[element_material_ids == 1])
    matrix_colors = matrix_cluster_colors[np.arange(len(matrix_clusters)) % len(matrix_cluster_colors)]
    colors[matrix_clusters] = matrix_colors + cluster_color_lightening * (1 - matrix_colors)
    colors[inclusion_clusters] = np.array(inclusion_cluster_greys)[np.arange(len(inclusion_clusters))
                                                                   % len(inclusion_cluster_greys), None]
    return colors[cluster_ids]

def plot_cross_section(element_material_ids, cell_maps, plot_path):
    # One panel per method: the phase of every element (fiber dark), or its cluster's colour when the method's clusters
    # are coloured in, with the method's boundaries on top; one legend below for the kinds of line the panels use.
    used_kinds = [kind for kind in boundary_styles if any(kind in cell_map.boundaries for cell_map in cell_maps)]
    # The legend lies side by side under several panels, stacked under one, which is too narrow for a row.
    legend_columns = len(used_kinds) if len(cell_maps) > 1 else 1
    legend_rows = -(-len(used_kinds) // legend_columns)
    figure_height = cross_section_panel_size + 0.6 + cross_section_legend_row_height * legend_rows
    with plt.rc_context(cross_section_style):
        figure, axes = plt.subplots(1, len(cell_maps), squeeze=False, layout='constrained',
                                    figsize=(cross_section_panel_size * len(cell_maps), figure_height))
        for axis, cell_map in zip(axes[0], cell_maps):
            material_ids = (element_material_ids if cell_map.element_material_ids is None
                            else cell_map.element_material_ids)
            if cell_map.fill_cluster_ids is None:
                axis.imshow(material_ids, cmap=matplotlib.colors.ListedColormap(['0.92', '0.62']), vmin=0,
                            vmax=1, origin='lower', extent=(0, 1, 0, 1))
            else:
                axis.imshow(get_cluster_image(cell_map.fill_cluster_ids, material_ids), origin='lower',
                            extent=(0, 1, 0, 1))
            for kind in boundary_styles:
                if kind in cell_map.boundaries:
                    draw_boundaries(axis, cell_map.boundaries[kind], boundary_styles[kind])
            axis.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], **frame_style)
            axis.set_title(f"{cell_map.title}\n{cell_map.settings}")
            axis.set_axis_off()
            # Square panels sit at the bottom of their slots, so the slots' spare height goes to the titles.
            axis.set_anchor('S')
        handles = [matplotlib.lines.Line2D([], [], **boundary_styles[kind]) for kind in used_kinds]
        figure.legend(handles=handles, loc='outside lower center', ncols=legend_columns, frameon=False,
                      handlelength=3)
        figure.savefig(plot_path)
    plt.close(figure)

## ------- Field Maps (von Mises Stress, Plastic Strain) ------- ##

field_map_style = {**figure_style, 'font.size': 13, 'axes.titlesize': 15, 'figure.titlesize': 14}
field_map_panel_size = 4.2
# A colormap per field, so a stress map is never read as a strain map: jet for the von Mises stress, magma for the
# accumulated plastic strain (dark where the matrix stays elastic, bright in the plastic bands).
von_mises_colormap = 'jet'
plastic_strain_colormap = 'magma'
field_map_hidden_color = '0.8'    # elements a map leaves out (NaN), such as the inclusion in a matrix-only map

class FieldMap(NamedTuple):
    # One method's map on its element grid (each element showing its cluster's value); maps may differ in resolution.
    title: str
    values: np.ndarray

def plot_field_maps(field_maps, label, colormap_name, load_step, load_step_count, value_min, value_max, plot_path):
    # One map per method, side by side on one colour scale between the given limits, in the named colormap (one of
    # the field colormaps above); label names the field and its unit on the colour bar, whose ends point outwards
    # where a map goes beyond them (those values take the end colour). Elements without a value (NaN) are drawn in
    # field_map_hidden_color.
    with plt.rc_context(field_map_style):
        figure, axes = plt.subplots(1, len(field_maps), squeeze=False, layout='constrained',
                                    figsize=(field_map_panel_size * len(field_maps) + 1.2, field_map_panel_size))
        limits = {'vmin': value_min, 'vmax': value_max}
        below = any(np.nanmin(field_map.values) < value_min for field_map in field_maps)
        above = any(np.nanmax(field_map.values) > value_max for field_map in field_maps)
        extend = {(False, False): 'neither', (True, False): 'min', (False, True): 'max', (True, True): 'both'}
        colormap = plt.get_cmap(colormap_name).with_extremes(bad=field_map_hidden_color)
        for axis, field_map in zip(axes[0], field_maps):
            image = axis.imshow(field_map.values, origin='lower', extent=(0, 1, 0, 1), cmap=colormap, **limits)
            axis.set_title(field_map.title)
            axis.set_axis_off()
        figure.colorbar(image, ax=axes[0], label=label, shrink=0.85, extend=extend[below, above])
        figure.suptitle(f"Load step {load_step} of {load_step_count}", color='0.3')
        figure.savefig(plot_path)
    plt.close(figure)

## ------- Field Profiles ------- ##

profile_inclusion_color = '0.93'    # the inclusion's stretches along the profile, where an elastic phase has no flow
profile_figure_size = (11, 3.6)

def plot_field_profiles(field_maps, profile_height, inclusion_map, label, load_step, load_step_count, side_length_mm,
                        plot_path):
    # Each map's values along the horizontal line at profile_height (a fraction of the side), left to right as on the
    # maps, as steps (one value per element or voxel), in the style of the stress-strain figure. The inclusion's
    # stretches along the line, read from inclusion_map (True inside, the finest available grid), are shaded.
    def get_row(values):
        return values[int(profile_height * len(values))]

    def get_edges(values):
        return np.linspace(0, side_length_mm, len(values) + 1)

    with plt.rc_context(load_path_style):
        figure, axis = plt.subplots(layout='constrained', figsize=profile_figure_size)
        inclusion = get_row(inclusion_map)
        edges = get_edges(inclusion)
        changes = np.flatnonzero(np.diff(np.concatenate([[0], inclusion.astype(int), [0]])))
        for first, stop in zip(changes[::2], changes[1::2]):
            axis.axvspan(edges[first], edges[stop], color=profile_inclusion_color, linewidth=0, zorder=0)
        for field_map in field_maps:
            values = get_row(field_map.values)
            axis.plot(get_edges(values), np.append(values, values[-1]), drawstyle='steps-post', label=field_map.title,
                      **method_styles[field_map.title])
        axis.set_xlim(0, side_length_mm)
        axis.set_ylim(bottom=0)
        axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        axis.grid(True, color='0.9', linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_ylabel(label)
        axis.set_xlabel(f"x (mm), along the middle element row at mid-thickness; load step {load_step} of "
                        f"{load_step_count}")
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(handles, labels, ncols=len(handles), columnspacing=2.4, **legend_style)
        figure.savefig(plot_path)
    plt.close(figure)
