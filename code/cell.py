# The cell and its grids:
#   elements    the periodic hexahedral mesh, which resolves the geometry;
#   partitions  a uniform lattice of blocks of k × k elements (k × k × k for a geometry that varies along z; columns
#               through the thickness otherwise);
#   clusters    regions of one phase and one material state each, inside one partition.
# Every method keeps one material state per cluster. The same code builds the partitions (FFT-TFA's, TFA's) and
# Spectral LS's lattice, which is the same mesh with one partition per element (per element column). The cache of
# expensive results is here too.

import hashlib
import warnings
from typing import NamedTuple
import numpy as np
import scipy.cluster.vq

import config

## ------- Shapes ------- ##

# The inclusion shapes whose geometry does not vary along z. They have one element layer, which is exact, and so one
# partition layer (columns through the thickness); every other shape is meshed and partitioned alike along all three
# axes.
shapes_constant_along_z = ("fiber", "laminate", "random fibers")

def is_shape_constant_along_z():
    return config.inclusion_shape in shapes_constant_along_z

def get_fiber_radius():
    # Of each of config.fiber_count equal fibers filling config.fiber_volume_fraction of the cell's cross-section.
    return config.domain_side_length * np.sqrt(config.fiber_volume_fraction / (np.pi * config.fiber_count))

def get_periodic_offsets(points, centre):
    # Each (x, y) point's offset from centre, to the nearest of the centre's periodic images.
    offsets = points - centre
    return offsets - config.domain_side_length * np.round(offsets / config.domain_side_length)

def get_fiber_centres():
    # The (x, y) centres of the "random fibers", one row each (config.py): by random sequential addition, and if that
    # jams, the rest at random and every overlap pushed apart.
    minimum_distance = 2 * get_fiber_radius() + config.fiber_gap * config.domain_side_length
    random_generator = np.random.default_rng(config.fiber_seed)
    fiber_centres = np.empty((0, 2))
    for _ in range(config.fiber_placement_attempts):
        candidate = random_generator.uniform(0, config.domain_side_length, 2)
        if np.all(np.sum(get_periodic_offsets(fiber_centres, candidate)**2, axis=1) >= minimum_distance**2):
            fiber_centres = np.vstack([fiber_centres, candidate])
            if len(fiber_centres) == config.fiber_count:
                return fiber_centres
    fiber_centres = np.vstack([fiber_centres, random_generator.uniform(0, config.domain_side_length,
                                                                       (config.fiber_count - len(fiber_centres), 2))])
    # Each pass moves both fibers of every pair closer than minimum_distance apart by half the shortfall from
    # (1 + fiber_push_margin) × minimum_distance, all pairs at once: pushing to minimum_distance itself would only
    # approach it.
    for _ in range(config.fiber_push_passes):
        offsets = get_periodic_offsets(fiber_centres[None, :, :], fiber_centres[:, None, :])
        distances = np.sqrt(np.sum(offsets**2, axis=2))
        np.fill_diagonal(distances, np.inf)
        if np.all(distances >= minimum_distance):
            return fiber_centres
        shortfalls = np.maximum((1 + config.fiber_push_margin) * minimum_distance - distances, 0)
        fiber_centres = ((fiber_centres - np.sum((shortfalls / (2 * distances))[:, :, None] * offsets, axis=1))
                         % config.domain_side_length)
    padded_fraction = config.fiber_count * np.pi * (minimum_distance / 2)**2 / config.domain_side_length**2
    raise ValueError(f"Pushing the fibers apart left some overlapping after {config.fiber_push_passes} passes. The "
                     f"fibers with half the gap around each would fill {padded_fraction:.3g} of the cell (at most "
                     "0.907, hexagonal packing): lower fiber_volume_fraction or fiber_gap, or raise "
                     "fiber_push_passes.")

def is_inside_inclusion(centres):
    # Whether each point, one (x, y, z) row each, lies inside the inclusion.
    if config.inclusion_shape == "random fibers":
        radius = get_fiber_radius()
        return np.any([np.sum(get_periodic_offsets(centres[:, :2], fiber_centre)**2, axis=1) < radius**2
                       for fiber_centre in get_fiber_centres()], axis=0)
    offsets_from_centre = centres - config.domain_side_length / 2
    if config.inclusion_shape == "laminate":
        return np.abs(offsets_from_centre[:, 0]) < config.layer_thickness / 2
    if config.inclusion_shape == "laminate_z":
        return np.abs(offsets_from_centre[:, 2]) < config.layer_thickness / 2
    if config.inclusion_shape == "sphere":
        return np.sum(offsets_from_centre**2, axis=1) < config.inclusion_radius**2
    return np.sum(offsets_from_centre[:, :2]**2, axis=1) < config.inclusion_radius**2

## ------- Discretization ------- ##

class Discretization(NamedTuple):
    # A mesh of elements_per_side² × elements_along_z hexahedra over the periodic cubic cell, and its
    # lattice of partitions_per_side² × partitions_along_z partitions. Lattices are indexed [z, y, x], as
    # get_grid_id orders them (x fastest).
    elements_per_side: int
    elements_along_z: int
    partitions_per_side: int
    partitions_along_z: int

    @property
    def elements_per_partition_side(self):
        return self.elements_per_side // self.partitions_per_side

    @property
    def elements_per_partition_along_z(self):
        return self.elements_along_z // self.partitions_along_z

    @property
    def elements_per_partition(self):
        return self.elements_per_partition_side**2 * self.elements_per_partition_along_z

    @property
    def element_lattice_shape(self):
        return (self.elements_along_z, self.elements_per_side, self.elements_per_side)

    @property
    def partition_lattice_shape(self):
        return (self.partitions_along_z, self.partitions_per_side, self.partitions_per_side)

    @property
    def element_count(self):
        return self.elements_per_side**2 * self.elements_along_z

    @property
    def dof_count(self):
        return 3 * self.element_count

    @property
    def element_side_length(self):
        return config.domain_side_length / self.elements_per_side

    @property
    def element_length_along_z(self):
        return config.domain_side_length / self.elements_along_z

    @property
    def element_volume(self):
        return self.element_side_length**2 * self.element_length_along_z

def get_discretization(elements_per_side, partitions_per_side):
    if is_shape_constant_along_z():
        return Discretization(elements_per_side, 1, partitions_per_side, 1)
    return Discretization(elements_per_side, elements_per_side, partitions_per_side, partitions_per_side)

def get_partition_discretization():
    # The mesh and partitions config.py describes, shared by FFT-TFA and TFA.
    return get_discretization(config.elements_per_side, config.partitions_per_side)

def get_spectral_discretization(elements_per_side):
    # A lattice of one partition per element, elements_per_side per side.
    return get_discretization(elements_per_side, elements_per_side)

def get_spectral_ls_discretization():
    # Spectral LS's lattice: on its own elements (config.spectral_ls_elements_per_side), or the mesh's when None.
    elements = config.spectral_ls_elements_per_side
    return get_spectral_discretization(config.elements_per_side if elements is None else elements)

## ------- Periodic Mesh ------- ##

class Mesh(NamedTuple):
    discretization: Discretization
    element_partition_ids: np.ndarray
    element_material_ids: np.ndarray     # 0 matrix, 1 inclusion

def get_grid_id(i, j, k, count_per_side):
    return i + count_per_side * j + count_per_side**2 * k

def get_element_material_ids(discretization):
    # The geometry, resolved on the elements: 1 for an element whose centre lies inside the inclusion, else 0.
    centres_xy = (np.arange(discretization.elements_per_side) + 0.5) * discretization.element_side_length
    centres_z = (np.arange(discretization.elements_along_z) + 0.5) * discretization.element_length_along_z
    z, y, x = np.meshgrid(centres_z, centres_xy, centres_xy, indexing="ij")
    return is_inside_inclusion(np.column_stack([x.ravel(), y.ravel(), z.ravel()])).astype(int)

def get_periodic_mesh(discretization):
    k, j, i = np.meshgrid(*map(np.arange, discretization.element_lattice_shape), indexing="ij")
    element_partition_ids = get_grid_id(i // discretization.elements_per_partition_side,
                                        j // discretization.elements_per_partition_side,
                                        k // discretization.elements_per_partition_along_z,
                                        discretization.partitions_per_side).reshape(-1)
    return Mesh(discretization, element_partition_ids, get_element_material_ids(discretization))

## ------- Materials ------- ##

class Materials(NamedTuple):
    # One row per material state (cluster): the phase (0 matrix, 1 inclusion), the elastic stiffness and the yield
    # properties.
    material_ids: np.ndarray
    L: np.ndarray
    yield_stress: np.ndarray
    hardening_modulus: np.ndarray

def get_L(elastic_modulus, poisson_ratio):
    # Isotropic elastic stiffness in engineering Voigt notation (ε11, ε22, ε33, γ12, γ23, γ13).
    lame_lambda   = elastic_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    shear_modulus = elastic_modulus / (2 * (1 + poisson_ratio))
    L = np.zeros((6, 6))
    L[:3, :3] = lame_lambda * np.ones((3, 3)) + 2 * shear_modulus * np.eye(3)
    L[3:, 3:] = shear_modulus * np.eye(3)
    return L

def get_matrix_L():
    return get_L(config.matrix_elastic_modulus, config.matrix_poisson_ratio)

def get_inclusion_L():
    return get_L(config.inclusion_elastic_modulus, config.inclusion_poisson_ratio)

def get_value_per_material(matrix_value, inclusion_value, material_ids):
    return np.array([matrix_value, inclusion_value])[material_ids]

def get_materials(material_ids):
    return Materials(
        material_ids=material_ids,
        L=get_value_per_material(get_matrix_L(), get_inclusion_L(), material_ids),
        yield_stress=get_value_per_material(config.matrix_yield_stress, config.inclusion_yield_stress, material_ids),
        hardening_modulus=get_value_per_material(config.matrix_hardening_modulus, config.inclusion_hardening_modulus,
                                                 material_ids))

## ------- Clusters ------- ##

class Clusters(NamedTuple):
    # The material states of a lattice and where they lie. A cluster is a set of elements of one phase in one partition,
    # with one material state (one uniform eigenstrain). Clusters are ordered by partition.
    materials: Materials               # one row per cluster
    volume: np.ndarray                 # each cluster's volume, as a share of its partition's
    element_cluster: np.ndarray        # the cluster of every element
    cluster_partition: np.ndarray      # the partition of each cluster

    @property
    def cluster_count(self):
        return len(self.volume)

    @property
    def partition_count(self):
        return self.cluster_partition[-1] + 1

    def average(self, values):
        # The volume average over the cell of one value per cluster (rows of values). The cell's volume is
        # partition_count partition volumes.
        weights = self.volume.reshape(-1, *[1] * (np.ndim(values) - 1))
        return np.sum(weights * values, axis=0) / self.partition_count

def label_partition_phases(mesh):
    # Every element's cluster when the clusters are the partitions' single-phase parts: one cluster for a single-phase
    # partition, two for a mixed one. A cluster's key is 2 × partition + phase, so the clusters are ordered by
    # partition, matrix first.
    _, element_cluster = np.unique(2 * mesh.element_partition_ids + mesh.element_material_ids, return_inverse=True)
    return element_cluster

def label_partition_k_means(mesh, element_features):
    # Every element's cluster by k-means inside each partition, in each phase apart, on its features (one row per
    # element; the elastic strain concentration, Liu–Bessa–Liu 2016, section 2.2.1): up to
    # k_means_matrix_clusters_per_partition clusters in a partition's matrix part and
    # k_means_inclusion_clusters_per_partition in its inclusion part, fewer when a part has fewer distinct features,
    # and clusters k-means leaves empty are dropped. A cluster's key is
    # (2 × partition + phase) × the larger count + its label in its part, so the clusters are ordered by partition,
    # matrix first, and with one cluster per part this is label_partition_phases.
    rng = np.random.default_rng(config.k_means_seed)
    cluster_counts = (config.k_means_matrix_clusters_per_partition, config.k_means_inclusion_clusters_per_partition)
    part_keys = 2 * mesh.element_partition_ids + mesh.element_material_ids
    part_order = np.argsort(part_keys, kind="stable")
    part_firsts = np.flatnonzero(np.diff(part_keys[part_order], prepend=-1))
    labels = np.zeros(len(element_features), dtype=int)
    for elements in np.split(part_order, part_firsts[1:]):
        features = element_features[elements]
        cluster_count = min(cluster_counts[mesh.element_material_ids[elements[0]]],
                            len(np.unique(features, axis=0)))
        if cluster_count == 1:
            continue
        with warnings.catch_warnings():
            # An empty cluster is dropped below, and run.py prints how many clusters there are.
            warnings.filterwarnings("ignore", "One of the clusters is empty")
            _, part_labels = scipy.cluster.vq.kmeans2(features, cluster_count, iter=config.k_means_iterations,
                                                      minit="++", rng=rng)
        labels[elements] = part_labels
    _, element_cluster = np.unique(part_keys * max(cluster_counts) + labels, return_inverse=True)
    return element_cluster

def get_clusters(mesh, element_cluster):
    # The clusters, given every element's cluster.
    if mesh.element_material_ids.min() == mesh.element_material_ids.max():
        raise ValueError("Every element is the same phase, so the cell is homogeneous: the inclusion is too small or "
                         "too large for this element grid.")
    cluster_count = element_cluster.max() + 1
    material_ids, cluster_partition = np.zeros(cluster_count, dtype=int), np.zeros(cluster_count, dtype=int)
    material_ids[element_cluster] = mesh.element_material_ids
    cluster_partition[element_cluster] = mesh.element_partition_ids
    if np.any(material_ids[element_cluster] != mesh.element_material_ids):
        raise ValueError("A cluster holds both phases; every cluster must lie in one phase.")
    if np.any(cluster_partition[element_cluster] != mesh.element_partition_ids):
        raise ValueError("A cluster spans partitions; every cluster must lie in one partition.")
    return Clusters(get_materials(material_ids), np.bincount(element_cluster) / mesh.discretization.elements_per_partition,
                    element_cluster, cluster_partition)

## ------- Cache ------- ##

def get_cache_path(kind, parameters):
    # One file per parameter set, so switching between configurations reuses each one's operators instead of
    # overwriting a single file. The name is a hash of every parameter, so a file that exists matches them.
    digest = hashlib.sha256()
    for name in sorted(parameters):
        value = np.asarray(parameters[name])
        digest.update(f"{name}:{value.dtype.str}:{value.shape}:".encode())
        digest.update(value.tobytes())
    return config.cache_folder / f"{kind}_{digest.hexdigest()[:12]}.npz"

def is_cached(cache_path):
    # A cache file counts only if it records how long its array took to build, which the offline times report.
    if not cache_path.exists():
        return False
    with np.load(cache_path) as cached:
        return "build_seconds" in cached.files

def get_cached(names, parameters, compute):
    # The named arrays and the seconds each took to build, as two dicts by name, each from its own cache file for these
    # parameters when they all exist, otherwise computed together by compute(), which returns both dicts, and each
    # saved to its own file.
    cache_paths = {name: get_cache_path(name, parameters) for name in names}
    if all(is_cached(cache_path) for cache_path in cache_paths.values()):
        arrays, build_seconds = {}, {}
        for name, cache_path in cache_paths.items():
            print(f"{cache_path.name}: loaded from cache.")
            with np.load(cache_path) as cached:
                arrays[name], build_seconds[name] = cached[name], float(cached["build_seconds"])
        return arrays, build_seconds
    print(f"{', '.join(cache_path.name for cache_path in cache_paths.values())}: no cache found, computing.")
    arrays, build_seconds = compute()
    config.cache_folder.mkdir(exist_ok=True)
    for name, cache_path in cache_paths.items():
        np.savez(cache_path, **{name: arrays[name], "build_seconds": build_seconds[name]})
    return arrays, build_seconds

def get_geometry_parameters():
    # Every input that sets the geometry, for a cache key: the cell, the shape and the shape's own inputs.
    parameters = {"domain_side_length": config.domain_side_length, "inclusion_shape": config.inclusion_shape}
    if config.inclusion_shape in ("laminate", "laminate_z"):
        parameters["layer_thickness"] = config.layer_thickness
    elif config.inclusion_shape == "random fibers":
        # The centres stand for the count, volume fraction, gap and seed that placed them.
        parameters["fiber_radius"] = get_fiber_radius()
        parameters["fiber_centres"] = get_fiber_centres()
    else:
        parameters["inclusion_radius"] = config.inclusion_radius
    return parameters
