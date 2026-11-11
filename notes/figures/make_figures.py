# The figures of notes/0_fft_tfa_lecture.md, computed with the code itself: python notes/figures/make_figures.py
# (a few minutes). Inputs are pinned here in memory, as check.py does: glass/PA, a centred fiber, check.py's solver
# settings; config.py, the cache and code/output are not touched.

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

import cell
import check
import fft_tfa_model
import ls_model
import plots
import run
from config import strain
from material import get_unloaded_plastic_state

glass_PA = config.all_materials["glass/PA"]
for name, value in check.pinned_inputs.items():
    setattr(config, name, value)
config.all_materials, config.materials, config.geometry = {"glass/PA": glass_PA}, "glass/PA", "fiber"
config.cache_folder = pathlib.Path(tempfile.mkdtemp())
style = {**plots.figure_style, 'font.size': 11, 'axes.titlesize': 12}
phase_colors = matplotlib.colors.ListedColormap(['0.92', '0.62'])
# The load of figures 2 and 3: tension and shear together, out to 2 % in 20 steps.
load_path = np.linspace(0, 1, 21)[1:, None] * np.array(strain(e11=0.02, g12=0.02))

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
    maps = {}
    for step, macro_strain in enumerate(load_path):
        result = solver.solve_step(macro_strain, history)
        history = result.plastic_state
        if step in (0, len(load_path) - 1):
            polarization = result.stress - result.strain @ matrix_L.T
            maps[step] = get_tensor_norm(polarization).reshape(elements, elements) / 1e6
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), layout='constrained')
        titles = {0: r"Elastic: $\bar\varepsilon_{11}=\bar\gamma_{12}=0.1$ %",
                  len(load_path) - 1: r"Plastic: $\bar\varepsilon_{11}=\bar\gamma_{12}=2$ %"}
        for axis, (step, values) in zip(axes, maps.items()):
            image = axis.imshow(values, origin='lower', extent=(0, 1, 0, 1), cmap=plots.field_map_colormap)
            axis.set_title(titles[step])
            axis.set_axis_off()
            figure.colorbar(image, ax=axis, label=r"$|\boldsymbol{\tau}|$ (MPa)", shrink=0.85)
        figure.savefig(folder / "polarization.png")
    plt.close(figure)

## ------- Figure 3: a partition's polarization summarized by polynomials ------- ##

def draw_summary():
    """FFT-TFA's cluster polarizations at the end of figure 2's load, and each partition's best-fitting polynomial of
    degree 0, 1 and 3: τ(β) ≈ Σ_ψ ψ(β) τ(B, ψ)."""
    mesh, clusters = get_clusters(64, 4, 2)
    discretization = mesh.discretization
    interaction = fft_tfa_model.get_interaction(mesh, clusters, 1)
    solver = fft_tfa_model.FFTTFASolver(interaction)
    history = get_unloaded_plastic_state(clusters.cluster_count)
    for macro_strain in load_path:
        result = solver.solve_step(macro_strain, history)
        history = result.plastic_state
    matrix_L = cell.get_matrix_L()
    polarization = result.stress - result.strain @ matrix_L.T
    component, shape = 3, discretization.element_lattice_shape
    maps = {"Polarization: uniform\non each cluster": polarization[clusters.element_cluster, component]}
    block = discretization.partition_shape_in_elements
    z, y, x = np.unravel_index(np.arange(discretization.element_count), shape)
    for degree in (0, 1, 3):
        basis = fft_tfa_model.get_polynomial_basis(block, degree)
        element_basis = basis[z % block[0], y % block[1], x % block[2]]
        moments = fft_tfa_model.get_cluster_moments(discretization, clusters, basis)
        partition_moments = np.zeros((clusters.partition_count, basis.shape[-1]))
        np.add.at(partition_moments, clusters.cluster_partition,
                  moments * polarization[:, component, None])
        maps[f"Its summary,\ndegree {degree}"] = np.sum(element_basis * partition_moments[mesh.element_partition],
                                                         axis=1)
    largest = max(np.abs(values).max() for values in maps.values()) / 1e6
    partition = mesh.element_partition.reshape(shape[1:])
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 4, figsize=(12, 3.6), layout='constrained')
        for axis, (title, values) in zip(axes, maps.items()):
            image = axis.imshow(values.reshape(shape[1:]) / 1e6, origin='lower', extent=(0, 1, 0, 1),
                                cmap='RdBu_r', vmin=-largest, vmax=largest)
            plots.draw_boundaries(axis, partition, plots.boundary_styles["partition"])
            axis.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], **plots.frame_style)
            axis.set_title(title)
            axis.set_axis_off()
        figure.colorbar(image, ax=axes, label=r"$\tau_{12}$ (MPa)", shrink=0.85)
        figure.savefig(folder / "summary.png")
    plt.close(figure)

## ------- Figure 4: the orthonormal polynomials, and a cluster's indicator projected onto them ------- ##

def draw_basis():
    """The six orthonormal polynomials of degree at most 2 on a partition of 8 × 8 elements, and one cluster's indicator
    χ_b with its projection Σ_ψ m(b, ψ) ψ at increasing degree, with the share of χ_b the projection misses."""
    mesh, clusters = get_clusters(64, 8, 2)
    discretization = mesh.discretization
    block = discretization.partition_shape_in_elements
    basis = fft_tfa_model.get_polynomial_basis(block, 2)[0]
    # A partition the fiber's edge crosses, and its larger matrix cluster.
    inclusion_share = np.bincount(mesh.element_partition, weights=mesh.element_phase) / np.prod(block)
    partition = int(np.argmin(np.abs(inclusion_share - 0.4)))
    in_partition = np.flatnonzero(clusters.cluster_partition == partition)
    matrix_clusters = in_partition[clusters.materials.phase[in_partition] == 0]
    cluster = matrix_clusters[np.argmax(np.bincount(clusters.element_cluster)[matrix_clusters])]
    elements = np.flatnonzero(mesh.element_partition == partition)
    _, y, x = np.unravel_index(elements, discretization.element_lattice_shape)
    indicator = np.zeros(block[1:])
    indicator[y % block[1], x % block[2]] = clusters.element_cluster[elements] == cluster
    titles = ["$1$", "$x$", "$y$", "$x^2$", "$xy$", "$y^2$"]
    with plt.rc_context(style):
        figure, axes = plt.subplots(2, 6, figsize=(12, 4.6), layout='constrained')
        largest = np.abs(basis).max()
        for axis, values, title in zip(axes[0], np.moveaxis(basis, -1, 0), titles):
            image = axis.imshow(values, origin='lower', cmap='RdBu_r', vmin=-largest, vmax=largest)
            axis.set_title(f"from {title}")
            axis.set_axis_off()
        figure.colorbar(image, ax=axes[0], label="value", shrink=0.85)
        axes[1, 0].imshow(indicator, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, 0].set_title(r"$\chi_b$, one cluster")
        for axis, degree in zip(axes[1, 1:], (0, 1, 2, 4, 6)):
            polynomials = fft_tfa_model.get_polynomial_basis(block, degree)[0]
            moments = np.einsum('yxp,yx->p', polynomials, indicator)
            projection = polynomials @ moments
            missed = np.linalg.norm(indicator - projection) / np.linalg.norm(indicator)
            image = axis.imshow(projection, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
            axis.set_title(f"degree {degree}: misses {100 * missed:.0f} %")
        for axis in axes[1]:
            axis.set_axis_off()
        figure.colorbar(image, ax=axes[1], label="value", shrink=0.85)
        figure.savefig(folder / "basis.png")
    plt.close(figure)

## ------- Figure 5: where the far field is accurate enough ------- ##

def get_partition_distance(clusters, partition_shape):
    # Every cluster pair's distance in partitions along the farthest axis, periodic.
    position = np.stack(np.unravel_index(clusters.cluster_partition, partition_shape), axis=-1)
    offsets = np.abs(position[:, None, :] - position[None, :, :])
    return np.max(np.minimum(offsets, np.array(partition_shape) - offsets), axis=-1)

def get_exact_pairs(mesh, clusters):
    """Γ(a←b) for every pair of clusters, 6 × 6 blocks: a unit polarization on b's elements, its strain by the element
    kernel, averaged over a's elements."""
    discretization, matrix_L = mesh.discretization, cell.get_matrix_L()
    shape = discretization.element_lattice_shape
    kernel = ls_model.to_fourier_blocks(ls_model.get_moulinec_suquet_kernel(shape, matrix_L), shape)
    counts = np.bincount(clusters.element_cluster)
    pairs = np.zeros((clusters.cluster_count, clusters.cluster_count, 6, 6))
    for source in range(clusters.cluster_count):
        for column in range(6):
            polarization = np.zeros((discretization.element_count, 6))
            polarization[clusters.element_cluster == source, column] = 1
            element_strain = ls_model.apply_on_lattice(kernel, np.linalg.solve(matrix_L, polarization.T).T)
            average = np.zeros((clusters.cluster_count, 6))
            np.add.at(average, clusters.element_cluster, element_strain)
            pairs[:, source, :, column] = average / counts[:, None]
    return pairs

def get_far_pairs(interaction):
    """Γ_far(a←b) for every pair: the far field alone (moments, FFT, read back by the clusters' moments)."""
    clusters = interaction.clusters
    pairs = np.zeros((clusters.cluster_count, clusters.cluster_count, 6, 6))
    for source in range(clusters.cluster_count):
        for column in range(6):
            polarization = np.zeros((clusters.cluster_count, 6))
            polarization[source, column] = 1
            moments = (interaction.to_moments @ polarization).reshape(clusters.partition_count, -1)
            pairs[:, source, :, column] = interaction.from_moments @ ls_model.apply_on_lattice(
                interaction.moment_kernels, moments).reshape(-1, 6)
    return pairs

def draw_far_field():
    mesh, clusters = get_clusters(64, 8, 2)
    distance = get_partition_distance(clusters, mesh.discretization.partition_lattice_shape)
    exact = get_exact_pairs(mesh, clusters)
    # Per distance, the volume-weighted rms over every pair at that distance of the 6 × 6 block's norm, as a share of the
    # same measure of the exact interaction within a partition.
    weights = clusters.share[:, None] * clusters.share[None, :]
    def get_rms(blocks, at_distance):
        squared = weights * np.sum(blocks**2, axis=(-1, -2))
        return np.sqrt(squared[at_distance].sum() / weights[at_distance].sum())
    own_scale = get_rms(exact, distance == 0)
    distances = np.arange(distance.max() + 1)
    degrees = (0, 1, 2, 3, 4)
    errors = {}
    for degree in degrees:
        far = get_far_pairs(fft_tfa_model.get_interaction(mesh, clusters, degree))
        errors[degree] = np.array([get_rms(far - exact, distance == r) for r in distances]) / own_scale
    green = np.array(matplotlib.colors.to_rgb(plots.method_styles["FFT-TFA"]['color']))
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained', width_ratios=(1, 1.6))
        axis = axes[0]
        for i in range(-3, 4):
            for j in range(-3, 4):
                own = i == 0 and j == 0
                axis.add_patch(matplotlib.patches.Rectangle((i - 0.5, j - 0.5), 1, 1,
                                                            facecolor='#56B4E9' if own else '0.88',
                                                            edgecolor='white', linewidth=2))
        axis.text(0, 0, "A", ha='center', va='center', fontsize=14, color='white')
        axis.set_xlim(-3.5, 3.5)
        axis.set_ylim(-3.5, 3.5)
        axis.set_aspect('equal')
        axis.set_axis_off()
        axis.set_title("Seen from partition A: its own clusters exactly,\nevery other partition through its moments")
        axis = axes[1]
        axis.axvspan(-0.3, 0.3, color='#56B4E9', alpha=0.25, linewidth=0)
        axis.text(0.35, 0.03, "replaced by the\nexact interaction", transform=axis.get_xaxis_transform(), fontsize=10,
                  va='bottom')
        for index, degree in enumerate(degrees):
            color = green + plots.first_entry_lightening * (1 - index / (len(degrees) - 1)) * (1 - green)
            axis.plot(distances, 100 * errors[degree], marker='o', color=tuple(color), label=f"degree {degree}")
        axis.set_yscale('log')
        axis.set_xticks(distances)
        axis.set_xlabel("Distance between the two clusters' partitions")
        axis.set_ylabel("Far field's error, rms (%)")
        axis.grid(True, which='major', color='0.9', linewidth=0.8)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(frameon=False)
        axis.set_title("The far field alone against the exact interaction")
        figure.savefig(folder / "far_field.png")
    plt.close(figure)
    return errors

if __name__ == "__main__":
    draw_grids()
    draw_polarization()
    draw_summary()
    draw_basis()
    errors = draw_far_field()
    print("far field's error by distance (%):", {degree: np.round(100 * error, 3) for degree, error in errors.items()})
