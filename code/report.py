"""Saving a run and drawing it: what a run leaves in its folder, and the tables and figures drawn from it.

run.py solves; this file saves what it solved (save_run) and draws a saved run (draw_run), so a run can be redrawn
with other figure settings without solving it again (python code/run.py replot). A run's folder holds:
  config.py                    the settings it ran with
  run.json, results.npz        what each entry did and cost, and its results (read them with get_saved_run): the
                               macroscopic stress and strain at every step, and maps of the element layer at
                               mid-thickness (each element's cluster, partition and phase; its stress and accumulated
                               plastic strain at every turning point)
  summary.txt                  each entry's clusters, times, iterations and peak memory, and with the reference on,
                               its relative macroscopic stress error (|Σ − Σ_ref| / |Σ_ref|) at the first step and
                               the turning points, and its error per loaded component
  stress_strain.png            macroscopic stress against applied strain, every entry and the reference
  macro_stress_error.png       with the reference on: each entry's macroscopic stress error in every loaded component
  error_against_time.png       with the reference on and several entries of a method: the relative error against
                               online time, one panel per turning point away from zero strain
  cross_section.png            each entry's elements, clusters and partitions
  von_mises_stress.png         the von Mises stress at config.map_turning_point, one map per entry
  plastic_strain.png           the accumulated plastic strain there, when there is any
  *_profile.png                each of the two along the middle element row, every entry on one plot
code/output/latest points at the last run. Every figure is drawn by plots.py, from the plain arrays this file hands it.
"""

import datetime
import json
import re
import shutil
from typing import Callable, NamedTuple

import numpy as np

import config
import plots
from material import get_von_mises_stress
from newton import stress_norm_weights

## ------- Saving a Run ------- ##

def get_mid_layer_clusters(clusters):
    """The cluster of every element on the element layer at mid-thickness, which the maps show. A finer mesh's layer
    there (the reference's) lies in a coarser one's, since its elements subdivide them."""
    shape = clusters.mesh.element_shape
    return clusters.element_cluster.reshape(shape)[shape[0] // 2]

def get_cluster_maps(label, clusters):
    """Each element's cluster, partition and phase on the mid-thickness layer: what the cross-section shows."""
    layer = get_mid_layer_clusters(clusters)
    return {f"{label}/element_cluster": layer, f"{label}/element_partition": clusters.cluster_partition[layer],
            f"{label}/element_phase": clusters.materials.phase[layer]}

def save_run(load_path, results, entries, reference):
    """A new folder in config.output_folder with the settings, run.json (the run, each entry's line from solve_all and
    the reference's from get_reference) and results.npz (every result, the reference's under "Reference"), and
    code/output/latest pointing at it."""
    date = datetime.datetime.now()
    folder = config.output_folder / f"{date:%Y-%m-%d_%H%M%S}_{re.sub(r'[^A-Za-z0-9.-]+', '-', config.results_name)}"
    folder.mkdir(parents=True)
    shutil.copy(config.__file__, folder / "config.py")
    info = {"results_name": config.results_name, "date": date.isoformat(timespec="seconds"),
            "geometry": config.geometry, "materials": config.materials, "load_path": config.load_path,
            "unloaded_directions": config.unloaded_directions,
            "clustered_mesh": config.clustered_mesh, "element_kernel": config.element_kernel,
            "side_length_mm": 1e3 * config.cell_side,
            "reference": reference, "entries": entries}
    (folder / "run.json").write_text(json.dumps(info, indent=1))

    arrays = {"applied_macro_strain": load_path.applied_macro_strain, "turning_steps": load_path.turning_steps}
    for label, result in results.items():
        if result is None:
            continue
        layer = get_mid_layer_clusters(result.clusters)
        arrays |= get_cluster_maps(label, result.clusters)
        # The fields at the turning points in single precision, since they are only drawn.
        arrays |= {f"{label}/macro_stress": result.macro_stress, f"{label}/macro_strain": result.macro_strain,
                   f"{label}/stress": result.cluster_stress[:, layer].astype(np.float32),
                   f"{label}/accumulated_plastic_strain": result.accumulated_plastic_strain[:, layer].astype(np.float32)}
    np.savez_compressed(folder / "results.npz", **arrays)

    latest = config.output_folder / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(folder.name)
    return folder

def get_saved_run(folder):
    """A saved run's run.json, and its results.npz as a dict of arrays."""
    with np.load(folder / "results.npz") as saved:
        return json.loads((folder / "run.json").read_text()), {name: saved[name] for name in saved.files}

## ------- Tables ------- ##

def format_table(rows):
    """Rows of cells as aligned text lines: the first column left-aligned, the others right-aligned."""
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return ["   ".join(cell.ljust(width) if column == 0 else cell.rjust(width)
                       for column, (cell, width) in enumerate(zip(row, widths))) for row in rows]

def get_timing_table(info, step_count):
    """Offline: building the entry's solver. Online: solving the load steps it completed, with its Newton and Krylov
    (GMRES, or CG for Spectral LS) iterations over them. Peaks: the largest memory it held in each. Then why an entry
    stopped early."""
    rows = [("Entry", "Clusters", "Offline (s)", "Online (s)", "Newton its", "Krylov its", "Steps", "Offline peak (MB)",
             "Online peak (MB)")]
    for entry in info["entries"]:
        rows.append((entry["label"], str(entry["cluster_count"]), f"{entry['offline_seconds']:.2f}",
                     f"{entry['online_seconds']:.2f}", str(entry["newton_iterations"]), str(entry["krylov_iterations"]),
                     f"{entry['completed_step_count']}/{step_count}", f"{entry['peak_offline_MB']:.0f}",
                     f"{entry['peak_online_MB']:.0f}"))
    lines = format_table(rows)
    lines += [f"{entry['label']} stopped at step {entry['completed_step_count'] + 1}: {entry['failure_reason']}"
              for entry in info["entries"] if entry["completed_step_count"] < step_count]
    if info["reference"] is not None:
        lines.append(f"Reference: Spectral LS on {info['reference']['voxels']} voxels, built in "
                     f"{info['reference']['build_seconds']:.0f} s (not timed against the entries)")
    return "\n".join(lines)

def get_relative_stress_errors(info, arrays):
    """Each entry's relative macroscopic stress error |Σ − Σ_ref| / |Σ_ref| (%, tensor norms) at every step it solved:
    Schneider 2019's eq. (5.1), there for Σ11, on the whole tensor, so it depends on no choice of axes."""
    reference_stress = arrays["Reference/macro_stress"]
    errors = {}
    for entry in info["entries"]:
        solved = entry["completed_step_count"]
        if solved > 0:
            difference = arrays[f"{entry['label']}/macro_stress"][:solved] - reference_stress[:solved]
            errors[entry["label"]] = 100 * np.sqrt(
                (stress_norm_weights * difference**2).sum(axis=1)
                / (stress_norm_weights * reference_stress[:solved]**2).sum(axis=1))
    return errors

def get_error_turning_points(arrays):
    """The turning points the relative error is reported at, as (turning point index, step): all but those at zero
    strain, where the reference stress is only the residual stress and the ratio means nothing."""
    return [(index, step) for index, step in enumerate(arrays["turning_steps"])
            if np.any(arrays["applied_macro_strain"][step] != 0)]

def get_relative_error_table(info, relative_errors, error_turning_points, step_count):
    """The relative errors of get_relative_stress_errors at the first step (elastic, or nearly) and at the reported
    turning points; – where an entry stopped before."""
    columns = [("Step 1", 0), *((f"TP {index + 1} (step {step + 1})", step) for index, step in error_turning_points)]
    rows = [("Entry", *(name for name, _ in columns), "Steps")]
    for label, error in relative_errors.items():
        rows.append((label, *(f"{error[step]:.2f}" if step < len(error) else "–" for _, step in columns),
                     f"{len(error)}/{step_count}"))
    return "\n".join([f"Relative macroscopic stress error |Σ − Σ_ref| / |Σ_ref| (%, tensor norms; Schneider 2019, eq. "
                      f"5.1) against the reference (Spectral LS on {info['reference']['voxels']} voxels), at the first "
                      "step and the turning points (TP):", *format_table(rows)])

def get_loaded_components(reference_stress):
    """The stress components the load path produces: those whose reference value reaches config.loaded_stress_fraction
    of the largest reference component somewhere along the path."""
    largest_per_component = np.abs(reference_stress).max(axis=0)
    return np.flatnonzero(largest_per_component >= config.loaded_stress_fraction * largest_per_component.max())

def get_macro_stress_errors(info, arrays, components):
    """Each entry's signed macroscopic stress error in the given components at every step it solved, in % of each
    component's largest reference value along the path, so a smaller component is read on its own scale."""
    reference_stress = arrays["Reference/macro_stress"][:, components]
    scale = np.abs(reference_stress).max(axis=0)
    errors = {}
    for entry in info["entries"]:
        solved = entry["completed_step_count"]
        if solved > 0:
            stress = arrays[f"{entry['label']}/macro_stress"][:solved, components]
            errors[entry["label"]] = 100 * (stress - reference_stress[:solved]) / scale
    return errors

def get_error_table(info, macro_stress_errors, components, step_count):
    """The errors of get_macro_stress_errors summarized per component, to see where an error comes from: the largest
    magnitude over the steps an entry solved, and their root mean square."""
    rows = [("Entry", *(f"σ{plots.stress_component_subscripts[component]} max / rms (%)" for component in components),
             "Steps")]
    for label, error in macro_stress_errors.items():
        rows.append((label, *(f"{np.abs(error[:, column]).max():.2f} / {np.sqrt(np.mean(error[:, column]**2)):.2f}"
                              for column in range(len(components))), f"{len(error)}/{step_count}"))
    return "\n".join([f"Macroscopic stress error against the reference (Spectral LS on {info['reference']['voxels']} "
                      "voxels), per loaded component, in % of its largest reference value along the path:",
                      *format_table(rows)])

## ------- Figures ------- ##

class Field(NamedTuple):
    """A field drawn at config.map_turning_point: one map per entry (name.png) and a profile (name_profile.png)."""
    name: str                   # its figures' file names
    label: str                  # its colour bar's and axis's label
    get_values: Callable        # its value at every element, from the maps of stress (Pa) and accumulated plastic strain
    range_setting: str          # the config.py setting that holds its colour range

# The fields drawn, each a line: its name, label, values from the saved maps, and its colour range's setting in config.py.
# A field that is zero everywhere (the plastic strain of an elastic run) is not drawn.
fields = [Field("von_mises_stress", "von Mises stress (MPa)", lambda stress, _: get_von_mises_stress(stress) / 1e6,
                "von_mises_range_MPa"),
          Field("plastic_strain", "Accumulated plastic strain (%)", lambda _, plastic_strain: 100 * plastic_strain,
                "plastic_strain_range_percent")]

def get_scheme_indices(info):
    """Every drawn label's place among its method's schemes, which sets its colour (the reference's is 0)."""
    return {"Reference": 0, **{entry["label"]: entry.get("scheme_index", 0) for entry in info["entries"]}}

def get_finest_entries(entries):
    """Each scheme's last entry that solved a load step: the one the line figures draw."""
    finest = {}
    for entry in entries:
        if entry["completed_step_count"] > 0:
            finest[entry["method"], entry.get("scheme", entry["label"])] = entry
    return list(finest.values())

def get_scheme_points(entries, relative_errors, error_turning_points):
    """The error-against-time plot's schemes, from the entries that completed the load path: one dict per scheme, its
    points in the order listed, each with its relative error at every reported turning point."""
    schemes = {}
    for entry in entries:
        key = entry["method"], entry.get("scheme", entry["method"])
        scheme = schemes.setdefault(key, {"method": entry["method"], "index": entry.get("scheme_index", 0),
                                          "name": key[1], "points": []})
        scheme["points"].append((entry.get("point", entry["label"]), entry["online_seconds"],
                                 [relative_errors[entry["label"]][step] for _, step in error_turning_points]))
    return list(schemes.values())

def get_cell_map(entry, arrays):
    """An entry's discretization on its element grid at mid-thickness, with its settings: k-means clusters coloured in.
    """
    label, settings = entry["label"], entry["settings"]
    cluster_ids, element_phase = arrays[f"{label}/element_cluster"], arrays[f"{label}/element_phase"]
    is_k_means = max(settings.get("matrix_clusters", 1), settings.get("inclusion_clusters", 1)) > 1
    fill = cluster_ids if is_k_means else None
    clusters_line = f"{entry['cluster_count']} {'k-means' if is_k_means else 'partition phase'} clusters"
    if entry["method"] == "Spectral LS":
        return plots.CellMap(label, f"{settings['elements']} voxels per side, C0: {settings['C0']}",
                             {"voxel": cluster_ids}, element_phase)
    if entry["method"] == "SCA":
        return plots.CellMap(label, f"over the whole cell, C0: {settings['C0']}\n{clusters_line}",
                             {"cluster": cluster_ids}, element_phase, fill)
    if entry["method"] == "TFA":
        return plots.CellMap(label, f"{settings['partitions']} partitions per side\n{clusters_line}",
                             {"cluster": cluster_ids}, element_phase, fill)
    return plots.CellMap(label, f"{settings['partitions']} partitions per side, degree {settings['degree']}\n"
                                f"{clusters_line}",
                         {"cluster": cluster_ids, "partition": arrays[f"{label}/element_partition"]},
                         element_phase, fill)

def draw_cross_section(entries, arrays, plot_path):
    """Every entry's elements, clusters and partitions on the mid-thickness layer, side by side (get_cluster_maps)."""
    plots.plot_cross_section([get_cell_map(entry, arrays) for entry in entries], plot_path)

def get_color_limits(field_maps, value_min, value_max):
    """A map figure's colour limits: the given ones, and those left at None from the finest field's map, the reference's
    or else Spectral LS's when no map is finer, so the others are read against it (values beyond it are clipped);
    otherwise from every map. Elements without a value (NaN, the inclusion in a matrix-only map) are left out."""
    finest_side = max(len(field_map.values) for field_map in field_maps)
    scale_maps = ([field_map.values for field_map in field_maps if field_map.title == "Reference"]
                  or [field_map.values for field_map in field_maps
                      if field_map.title.startswith("Spectral LS") and len(field_map.values) == finest_side]
                  or [field_map.values for field_map in field_maps])
    return (min(map(np.nanmin, scale_maps)) if value_min is None else value_min,
            max(map(np.nanmax, scale_maps)) if value_max is None else value_max)

def get_field_maps(info, arrays, turning_point, field):
    """A field's map for the reference and every entry that reached the turning point, each on its own elements, and
    each one's map of its inclusion, by label."""
    turning_step = arrays["turning_steps"][turning_point]
    labels = (["Reference"] if info["reference"] is not None else []) + [
        entry["label"] for entry in info["entries"] if entry["completed_step_count"] > turning_step]
    field_maps = [plots.FieldMap(label, field.get_values(arrays[f"{label}/stress"][turning_point],
                                                         arrays[f"{label}/accumulated_plastic_strain"][turning_point]))
                  for label in labels]
    return field_maps, {label: arrays[f"{label}/element_phase"] == 1 for label in labels}

def get_matrix_only(field_maps, inclusion_maps):
    """The maps with their inclusion elements hidden (NaN) when config.field_maps_matrix_only is on."""
    if not config.field_maps_matrix_only:
        return field_maps
    return [plots.FieldMap(field_map.title, np.where(inclusion_maps[field_map.title], np.nan, field_map.values))
            for field_map in field_maps]

def draw_load_path_figures(folder, info, arrays, drawn):
    """The figures along the load path, each scheme's finest entry in the stress-strain and stress-error figures, every
    entry in the error-against-time plot; and with the reference on, the error tables of every entry."""
    applied_macro_strain = arrays["applied_macro_strain"]
    step_count = len(applied_macro_strain)
    finest = get_finest_entries(drawn)
    scheme_indices = get_scheme_indices(info)
    stress_MPa = {entry["label"]: arrays[f"{entry['label']}/macro_stress"] / 1e6 for entry in finest}
    completed_step_counts = {entry["label"]: entry["completed_step_count"] for entry in finest}
    error_table = ""
    if info["reference"] is not None:
        reference_stress = arrays["Reference/macro_stress"]
        stress_MPa = {"Reference": reference_stress / 1e6, **stress_MPa}
        completed_step_counts["Reference"] = step_count
        relative_errors = get_relative_stress_errors(info, arrays)
        error_turning_points = get_error_turning_points(arrays)
        components = get_loaded_components(reference_stress)
        errors = get_macro_stress_errors(info, arrays, components)
        error_table = "\n\n".join([get_relative_error_table(info, relative_errors, error_turning_points, step_count),
                                   get_error_table(info, errors, components, step_count)])
        plots.plot_macro_stress_errors({entry["label"]: errors[entry["label"]] for entry in finest}, components,
                                       step_count, scheme_indices, folder / "macro_stress_error.png")
        schemes = get_scheme_points([entry for entry in drawn if entry["completed_step_count"] == step_count],
                                    relative_errors, error_turning_points)
        if any(len(scheme["points"]) > 1 for scheme in schemes):
            plots.plot_error_against_time(schemes, [f"Turning point {index + 1} (step {step + 1})"
                                                    for index, step in error_turning_points],
                                          folder / "error_against_time.png")
    plots.plot_stress_strain(stress_MPa, completed_step_counts, applied_macro_strain, scheme_indices,
                             folder / "stress_strain.png")
    return error_table

def draw_field_figures(folder, info, arrays):
    """Every field's maps and profile at config.map_turning_point. A saved run may have fewer turning points than
    config.py's load path."""
    turning_steps = arrays["turning_steps"]
    turning_point = (len(turning_steps) if config.map_turning_point == "last"
                     else min(config.map_turning_point, len(turning_steps))) - 1
    title = (f"Turning point {turning_point + 1} of {len(turning_steps)}: load step {turning_steps[turning_point] + 1} "
             f"of {len(arrays['applied_macro_strain'])}")
    # The profile follows the middle row of the mesh's elements, which the reference's subdivide; a map on other
    # elements (Spectral LS's) shows its row at that height. The inclusion's stretches come from the finest map.
    profile_height = (info["clustered_mesh"] // 2 + 0.5) / info["clustered_mesh"]
    reached = [entry for entry in info["entries"] if entry["completed_step_count"] > turning_steps[turning_point]]
    profile_labels = {"Reference", *(entry["label"] for entry in get_finest_entries(reached))}
    for field in fields:
        field_maps, inclusion_maps = get_field_maps(info, arrays, turning_point, field)
        if not field_maps:
            print(f"No entry reached turning point {turning_point + 1}, so no maps: see config.map_turning_point.")
            return
        if all(np.nanmax(np.abs(field_map.values)) == 0 for field_map in field_maps):
            continue
        shown_maps = get_matrix_only(field_maps, inclusion_maps)
        plots.plot_field_maps(shown_maps, field.label, title,
                              *get_color_limits(shown_maps, *getattr(config, field.range_setting)),
                              folder / f"{field.name}.png")
        plots.plot_field_profiles([field_map for field_map in field_maps if field_map.title in profile_labels],
                                  profile_height, max(inclusion_maps.values(), key=len), field.label, title,
                                  info["side_length_mm"], get_scheme_indices(info), folder / f"{field.name}_profile.png")

def draw_run(folder):
    """Every table and figure of a saved run, from what it saved, with config.py's figure settings: the tables printed
    and written to summary.txt."""
    info, arrays = get_saved_run(folder)
    # A run.json without these settings ran strain-controlled, on Moulinec–Suquet's kernel.
    problem = (f"{info['geometry']}, {info['materials']}, {info['load_path']} (unloaded directions "
               f"{info.get('unloaded_directions', 'fixed')}); element kernel "
               f"{info.get('element_kernel', 'Moulinec-Suquet')}")
    tables = [problem, get_timing_table(info, len(arrays["applied_macro_strain"]))]
    drawn = [entry for entry in info["entries"] if entry["completed_step_count"] > 0]
    if drawn:
        tables.append(draw_load_path_figures(folder, info, arrays, drawn))
        draw_cross_section(drawn, arrays, folder / "cross_section.png")
        draw_field_figures(folder, info, arrays)
    summary = "\n\n".join(table for table in tables if table)
    print(f"\n{summary}\n")
    (folder / "summary.txt").write_text(summary + "\n")
