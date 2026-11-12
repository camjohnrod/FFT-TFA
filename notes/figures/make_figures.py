# The figures of notes/fft_tfa_lecture.md, computed with the code itself: python notes/figures/make_figures.py
# (under a minute). Inputs are pinned here in memory, as check.py does: glass/PA, a centred fiber, check.py's solver
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
# The load of figures 2 and 4: tension and shear together, out to 2 % in 20 steps.
load_path = np.linspace(0, 1, 21)[1:, None] * np.array(strain(e11=0.02, g12=0.02))

def get_clusters(clustered_mesh, partitions, matrix_clusters):
    config.clustered_mesh = clustered_mesh
    clusters = run.get_partition_clusters(dict(partitions=partitions, matrix_clusters=matrix_clusters,
                                               inclusion_clusters=1))
    return clusters.mesh, clusters

def draw_phases(axis, image, title):
    axis.imshow(image, origin='lower', extent=(0, 1, 0, 1), **({} if image.ndim == 3 else
                                                              dict(cmap=phase_colors, vmin=0, vmax=1)))
    axis.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], **plots.frame_style)
    axis.set_title(title)
    axis.set_axis_off()

## ------- Figure 1: the three grids ------- ##

def draw_grids():
    mesh, clusters = get_clusters(32, 4, 3)
    shape = mesh.element_shape[1:]
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

## ------- Figures 2 and 4: FFT-TFA's polarization, and its polynomial fits ------- ##

def get_shear_polarization():
    """FFT-TFA's polarization τ12 on every cluster at the end of the load: 64² elements, 4² partitions, two matrix
    clusters per partition's matrix part, solved at degree 1."""
    mesh, clusters = get_clusters(64, 4, 2)
    solver = fft_tfa_model.FFTTFASolver(clusters, 1)
    history = get_unloaded_plastic_state(clusters.cluster_count)
    for macro_strain in load_path:
        result = solver.solve_step(macro_strain, history)
        history = result.plastic_state
    polarization = result.stress - result.strain @ cell.get_matrix_L().T
    return mesh, clusters, polarization[:, 3]

def get_polynomial_fit(mesh, clusters, values, degree):
    """At every element, its partition's best-fitting polynomial of the given degree to a field uniform on each
    cluster: Σ_i τ_i(B) ψ_i, with the moments τ_i(B) = Σ_b m_i(b) τ(b) (section 7.1)."""
    block = mesh.partition_size
    z, y, x = np.unravel_index(np.arange(mesh.element_count), mesh.element_shape)
    basis = fft_tfa_model.get_polynomial_basis(block, degree)
    moments = fft_tfa_model.get_cluster_moments(clusters, basis)
    partition_moments = np.zeros((clusters.partition_count, basis.shape[-1]))
    np.add.at(partition_moments, clusters.cluster_partition, moments * values[:, None])
    return np.sum(basis[z % block[0], y % block[1], x % block[2]] * partition_moments[mesh.element_partition], axis=1)

def draw_polarization_maps(maps, mesh, file_name, figure_width, outlined_partition=None):
    """One map per entry, on one shared colour scale, with the partitions drawn, and optionally one partition
    outlined."""
    shape = mesh.element_shape[1:]
    largest = max(np.abs(values).max() for values in maps.values()) / 1e6
    partition = mesh.element_partition.reshape(shape)
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, len(maps), figsize=(figure_width, 3.6), layout='constrained')
        for axis, (title, values) in zip(axes, maps.items()):
            image = axis.imshow(values.reshape(shape) / 1e6, origin='lower', extent=(0, 1, 0, 1),
                                cmap='RdBu_r', vmin=-largest, vmax=largest)
            plots.draw_boundaries(axis, partition, plots.boundary_styles["partition"])
            axis.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], **plots.frame_style)
            if outlined_partition is not None:
                rows, columns = np.nonzero(partition == outlined_partition)
                x0, y0 = columns.min() / shape[1], rows.min() / shape[0]
                width, height = (columns.max() + 1) / shape[1] - x0, (rows.max() + 1) / shape[0] - y0
                axis.add_patch(matplotlib.patches.Rectangle((x0, y0), width, height, fill=False, edgecolor='0.1',
                                                            linewidth=2.2))
                axis.text(x0 + width / 2, y0 + height / 2, "A", ha='center', va='center', fontsize=14)
            axis.set_title(title)
            axis.set_axis_off()
        figure.colorbar(image, ax=axes, label=r"$\tau_{12}$ (MPa)", shrink=0.85)
        figure.savefig(folder / file_name)
    plt.close(figure)

## ------- Figure 2: what one partition sees ------- ##

def draw_split():
    """FFT-TFA's polarization, and the same as seen from one partition A the fiber's edge crosses: A's own clusters
    exactly, every other partition through its mean (degree 0)."""
    mesh, clusters, polarization = get_shear_polarization()
    element_values = polarization[clusters.element_cluster]
    inclusion_share = (np.bincount(mesh.element_partition, weights=mesh.element_phase)
                       / mesh.elements_per_partition)
    partition = int(np.argmin(np.abs(inclusion_share - 0.5)))
    seen = np.where(mesh.element_partition == partition, element_values,
                    get_polynomial_fit(mesh, clusters, polarization, 0))
    draw_polarization_maps({"The polarization,\nuniform on each cluster": element_values,
                            "As partition A sees it:\nits own exactly, the others by their means": seen},
                           mesh, "split.png", 7.6, outlined_partition=partition)

## ------- Figure 3: where the far field is accurate enough ------- ##

def get_partition_distance(clusters, partition_shape):
    # Every cluster pair's distance in partitions along the farthest axis, periodic.
    position = np.stack(np.unravel_index(clusters.cluster_partition, partition_shape), axis=-1)
    offsets = np.abs(position[:, None, :] - position[None, :, :])
    return np.max(np.minimum(offsets, np.array(partition_shape) - offsets), axis=-1)

def get_exact_pairs(mesh, clusters):
    """Γ(a←b) for every pair of clusters, 6 × 6 blocks: a unit polarization on b's elements, its strain by the element
    kernel, averaged over a's elements."""
    matrix_L, shape = cell.get_matrix_L(), mesh.element_shape
    kernel = ls_model.get_P0_transformed(shape, matrix_L)
    counts = np.bincount(clusters.element_cluster)
    pairs = np.zeros((clusters.cluster_count, clusters.cluster_count, 6, 6))
    for source in range(clusters.cluster_count):
        for column in range(6):
            polarization = np.zeros((mesh.element_count, 6))
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
    distance = get_partition_distance(clusters, mesh.partition_shape)
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
        far = get_far_pairs(fft_tfa_model.get_interaction(clusters, degree))
        errors[degree] = np.array([get_rms(far - exact, distance == r) for r in distances]) / own_scale
    green = np.array(matplotlib.colors.to_rgb(plots.method_styles["FFT-TFA"]['color']))
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(6.4, 3.8), layout='constrained')
        axis.axvspan(-0.3, 0.3, color='#56B4E9', alpha=0.25, linewidth=0)
        axis.text(0.35, 0.03, "replaced by the\nexact interaction", transform=axis.get_xaxis_transform(), fontsize=10,
                  va='bottom')
        for index, degree in enumerate(degrees):
            color = green + 0.55 * (1 - index / (len(degrees) - 1)) * (1 - green)     # lighter for lower degrees
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

## ------- Figure 4: a partition's polarization summarized by polynomials ------- ##

def draw_summary():
    """FFT-TFA's cluster polarizations, and each partition's best-fitting polynomial of degree 0, 1 and 3:
    τ ≈ Σ_i τ_i(B) ψ_i."""
    mesh, clusters, polarization = get_shear_polarization()
    maps = {"Polarization: uniform\non each cluster": polarization[clusters.element_cluster]}
    for degree in (0, 1, 3):
        maps[f"Its summary,\ndegree {degree}"] = get_polynomial_fit(mesh, clusters, polarization, degree)
    draw_polarization_maps(maps, mesh, "summary.png", 12)

if __name__ == "__main__":
    draw_grids()
    draw_split()
    errors = draw_far_field()
    draw_summary()
    print("far field's error by distance (%):", {degree: np.round(100 * error, 3) for degree, error in errors.items()})
