# The figures of notes/0_fft_tfa_lecture.md, computed with the code itself: python notes/figures/make_figures.py
# (a minute or so). Inputs are set here in memory, as check.py does; config.py, the cache and code/output are not
# touched.

import pathlib
import sys
import tempfile

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent.parent / "code"))

import config  # first: it sets the thread count before numpy loads
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches
import scipy.sparse

import cell
import fft_tfa_model
import ls_model
import plots
import run
from config import strain
from material import get_unloaded_plastic_state

config.cache_folder = pathlib.Path(tempfile.mkdtemp())
config.geometry, config.materials = "one fiber", "glass/PA"
style = {**plots.figure_style, 'font.size': 11, 'axes.titlesize': 12}
phase_colors = matplotlib.colors.ListedColormap(['0.92', '0.62'])

def get_clusters(clustered_mesh, partitions, matrix_clusters):
    config.clustered_mesh = clustered_mesh
    return run.get_mesh_and_clusters(dict(partitions=partitions, matrix_clusters=matrix_clusters,
                                          inclusion_clusters=1))

def draw_phases(axis, image, title):
    axis.imshow(image, origin='lower', extent=(0, 1, 0, 1), **({} if image.ndim == 3 else
                                                              dict(cmap=phase_colors, vmin=0, vmax=1)))
    axis.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], **plots.frame_style)
    axis.set_title(title)
    axis.set_axis_off()

## ------- Figure 1: the three grids ------- ##

def draw_grids():
    mesh, clusters = get_clusters(32, 4, 3)
    shape = mesh.discretization.element_lattice_shape[1:]
    phase = mesh.element_phase.reshape(shape)
    element_ids = np.arange(phase.size).reshape(shape)
    partition = mesh.element_partition.reshape(shape)
    element_cluster = clusters.element_cluster.reshape(shape)
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 3, figsize=(10, 3.9), layout='constrained')
        draw_phases(axes[0], phase, "Elements\nresolve the geometry")
        plots.draw_boundaries(axes[0], element_ids, plots.boundary_styles["voxel"])
        draw_phases(axes[1], phase, "Partitions\na regular lattice of blocks")
        plots.draw_boundaries(axes[1], element_ids, plots.boundary_styles["voxel"])
        plots.draw_boundaries(axes[1], partition, plots.boundary_styles["partition"])
        draw_phases(axes[2], plots.get_cluster_image(element_cluster, phase),
                    "Clusters\none phase, one partition, one state")
        plots.draw_boundaries(axes[2], element_cluster, plots.boundary_styles["cluster"])
        plots.draw_boundaries(axes[2], partition, plots.boundary_styles["partition"])
        figure.savefig(folder / "grids.png")
    plt.close(figure)

## ------- Figure 2: the polarization with the matrix as reference medium ------- ##

def get_tensor_norm(stress):
    # The norm of a stress-like Voigt row (shear components count twice in the double contraction).
    return np.sqrt(np.sum(stress[:, :3]**2, axis=1) + 2 * np.sum(stress[:, 3:]**2, axis=1))

def draw_polarization():
    elements = 64
    operators = ls_model.get_spectral_ls_operators(elements, "midpoint")
    solver = ls_model.SpectralLSSolver(operators)
    matrix_L = cell.get_matrix_L()
    history = get_unloaded_plastic_state(operators.clusters.cluster_count)
    path = np.linspace(0, 1, 21)[1:, None] * np.array(strain(e11=0.02, g12=0.02))
    maps = {}
    for step, macro_strain in enumerate(path):
        result = solver.solve_step(macro_strain, history)
        history = result.plastic_state
        if step in (0, len(path) - 1):
            polarization = result.stress - result.strain @ matrix_L.T
            maps[step] = get_tensor_norm(polarization).reshape(elements, elements) / 1e6
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), layout='constrained')
        titles = {0: r"Elastic: $\bar\varepsilon_{11}=\bar\gamma_{12}=0.1$ %",
                  len(path) - 1: r"Plastic: $\bar\varepsilon_{11}=\bar\gamma_{12}=2$ %"}
        for axis, (step, values) in zip(axes, maps.items()):
            image = axis.imshow(values, origin='lower', extent=(0, 1, 0, 1), cmap=plots.field_map_colormap)
            axis.set_title(titles[step])
            axis.set_axis_off()
            figure.colorbar(image, ax=axis, label=r"$|\boldsymbol{\tau}|$ (MPa)", shrink=0.85)
        figure.savefig(folder / "polarization.png")
    plt.close(figure)

## ------- Figure 3: what the corrections fix, and how far ------- ##

def get_partition_distance(clusters, partition_shape):
    # Every cluster pair's distance in partitions along the farthest axis, periodic.
    position = np.stack(np.unravel_index(clusters.cluster_partition, partition_shape), axis=-1)
    offsets = np.abs(position[:, None, :] - position[None, :, :])
    return np.max(np.minimum(offsets, np.array(partition_shape) - offsets), axis=-1)

def get_interaction_strain(interaction, polarization):
    # FT-2 before FT-4's mean subtraction: the partition FFT on the partitions' mean polarization plus the corrections.
    # (FT-4 spreads the truncated interaction's small cell mean over every cluster, which would hide the decay.)
    clusters = interaction.clusters
    partition_polarization = interaction.cluster_to_partition @ polarization
    partition_strain = ls_model.apply_on_lattice(interaction.P0_transformed,
                                                 np.linalg.solve(interaction.matrix_L, partition_polarization.T).T)
    return (partition_strain[clusters.cluster_partition]
            + ls_model.apply_sparse(interaction.correction_blocks, polarization))

def draw_reach():
    mesh, clusters = get_clusters(64, 16, 2)
    partition_shape = mesh.discretization.partition_lattice_shape
    exact = fft_tfa_model.get_interaction(mesh, clusters, "full")
    variants = {"partition FFT alone": exact._replace(correction_blocks=scipy.sparse.bsr_matrix(
                    exact.correction_blocks.shape, blocksize=(6, 6))),
                "reach 0": fft_tfa_model.get_interaction(mesh, clusters, 0),
                "reach 1": fft_tfa_model.get_interaction(mesh, clusters, 1)}
    distance = get_partition_distance(clusters, partition_shape)
    largest_distance = distance.max()
    # Per distance, the volume-weighted root mean square over every pair of clusters at that distance (a unit
    # polarization in the source), as a share of the same measure of the exact interaction within a partition.
    squared_errors = {label: np.zeros(largest_distance + 1) for label in variants}
    squared_own = 0.0
    unit = np.zeros((clusters.cluster_count, 6))
    for source in range(clusters.cluster_count):
        unit[:] = 0
        unit[source, 0] = 1
        exact_change = get_interaction_strain(exact, unit)
        weights = clusters.share * clusters.share[source]
        squared_own += np.sum((weights * np.sum(exact_change**2, axis=1))[distance[:, source] == 0])
        for label, interaction in variants.items():
            squared_error = weights * np.sum((get_interaction_strain(interaction, unit) - exact_change)**2, axis=1)
            np.add.at(squared_errors[label], distance[:, source], squared_error)
    pair_weights = np.zeros(largest_distance + 1)
    for source in range(clusters.cluster_count):
        np.add.at(pair_weights, distance[:, source], clusters.share * clusters.share[source])
    errors = {label: np.sqrt(squared / pair_weights / (squared_own / pair_weights[0]))
              for label, squared in squared_errors.items()}
    colors = {"partition FFT alone": '#E69F00', "reach 0": '#56B4E9', "reach 1": '#D55E00'}
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained', width_ratios=(1, 1.6))
        axis = axes[0]
        for i in range(-3, 4):
            for j in range(-3, 4):
                ring = max(abs(i), abs(j))
                color = ('#56B4E9' if ring == 0 else '#D55E00' if ring == 1 else '0.93')
                axis.add_patch(matplotlib.patches.Rectangle((i - 0.5, j - 0.5), 1, 1, facecolor=color,
                                                            edgecolor='white', linewidth=2))
        axis.text(0, 0, "A", ha='center', va='center', fontsize=14, color='white')
        axis.set_xlim(-3.5, 3.5)
        axis.set_ylim(-3.5, 3.5)
        axis.set_aspect('equal')
        axis.set_axis_off()
        axis.set_title("Exact corrections around partition A:\nreach 0 (A itself), reach 1 (and its neighbours)")
        axis = axes[1]
        for label, error in errors.items():
            shown = error > 1e-14
            axis.plot(np.arange(largest_distance + 1)[shown], 100 * error[shown], marker='o', color=colors[label],
                      label=label)
        axis.set_yscale('log')
        axis.set_xlabel("Distance between the two clusters' partitions")
        axis.set_ylabel("Interaction error, rms (%)")
        axis.grid(True, which='major', color='0.9', linewidth=0.8)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(frameon=False)
        axis.set_title("Error of the strain a cluster's polarization causes in\nanother, against the exact "
                       "interaction (full reach)")
        figure.savefig(folder / "reach.png")
    plt.close(figure)
    return errors

## ------- Figure 4: the patch test ------- ##

def draw_patch_test():
    mesh, clusters = get_clusters(64, 16, 2)
    matrix_L = cell.get_matrix_L()
    uniform_strain = np.array(strain(e11=1.0))
    polarization = np.tile(matrix_L @ uniform_strain, (clusters.cluster_count, 1))
    shape = mesh.discretization.element_lattice_shape[1:]
    maps = {}
    for reach in (0, 1, "full"):
        change = fft_tfa_model.get_strain_change(fft_tfa_model.get_interaction(mesh, clusters, reach), polarization)
        maps[reach] = 100 * np.abs(change[:, 0])[clusters.element_cluster].reshape(shape)
    largest = max(values.max() for values in maps.values())
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 3, figsize=(11, 3.9), layout='constrained')
        for axis, (reach, values) in zip(axes, maps.items()):
            image = axis.imshow(values, origin='lower', extent=(0, 1, 0, 1), cmap=plots.field_map_colormap, vmin=0,
                                vmax=largest)
            axis.set_title(f"{'full reach' if reach == 'full' else f'reach {reach}'}: largest {values.max():.2g} %")
            axis.set_axis_off()
        figure.colorbar(image, ax=axes, label=r"spurious strain $|\Delta\varepsilon_{11}|$ (% of $\mathbf{L}_M^{-1}\boldsymbol{\tau}$)",
                        shrink=0.85)
        figure.savefig(folder / "patch_test.png")
    plt.close(figure)
    return {reach: values.max() for reach, values in maps.items()}

if __name__ == "__main__":
    draw_grids()
    draw_polarization()
    errors = draw_reach()
    print("interaction error by distance (%):", {label: np.round(100 * error, 3) for label, error in errors.items()})
    print("patch test, largest spurious strain (%):", draw_patch_test())
