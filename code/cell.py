"""The periodic cell and its three grids, from fine to coarse:

  elements    a regular mesh of hexahedra over the cubic cell, which resolves the geometry (each element is either
              matrix or inclusion: its phase);
  partitions  a regular lattice of blocks of k × k × k elements (columns of k × k through the thickness when the
              geometry does not vary along z): the cells of FFT-TFA's FFT;
  clusters    groups of elements of one phase inside one partition, each with one material state: one strain, stress
              and plastic strain. Every method keeps one material state per cluster.

Spectral LS uses the same code with one partition and one cluster per element. Also here: the materials, k-means
clustering, peak memory and the cache of expensive results (E and P, the reference).
"""

import hashlib
import re
import warnings
from typing import NamedTuple
import numpy as np
import scipy.cluster.vq

import config

## ------- Shapes ------- ##

# The inclusion shapes whose geometry does not vary along z. They get one element layer, which is exact, and so one
# partition layer (columns through the thickness); every other shape is meshed and partitioned alike along x, y and z.
shapes_constant_along_z = ("fiber", "laminate", "random fibers")

def get_geometry():
    """The geometry config.py names: its shape and the shape's own settings."""
    return config.all_geometries[config.geometry]

def is_shape_constant_along_z():
    return get_geometry()["shape"] in shapes_constant_along_z

def get_fiber_radius():
    """The radius of each of the "random fibers": equal fibers that together fill volume_fraction of the section."""
    geometry = get_geometry()
    return config.cell_side * np.sqrt(geometry["volume_fraction"] / (np.pi * geometry["count"]))

def get_periodic_offsets(points, centre):
    """Each (x, y) point's offset from centre, to the nearest of the centre's periodic images."""
    offsets = points - centre
    return offsets - config.cell_side * np.round(offsets / config.cell_side)

def get_fiber_centres():
    """The (x, y) centres of the "random fibers", one row each (config.py): by random sequential addition, and if that
    jams, the rest at random and every overlap pushed apart."""
    geometry = get_geometry()
    fiber_count = geometry["count"]
    minimum_distance = 2 * get_fiber_radius() + geometry["gap"] * config.cell_side
    random_generator = np.random.default_rng(geometry["seed"])
    fiber_centres = np.empty((0, 2))
    for _ in range(config.fiber_placement_attempts):
        candidate = random_generator.uniform(0, config.cell_side, 2)
        if np.all(np.sum(get_periodic_offsets(fiber_centres, candidate)**2, axis=1) >= minimum_distance**2):
            fiber_centres = np.vstack([fiber_centres, candidate])
            if len(fiber_centres) == fiber_count:
                return fiber_centres
    fiber_centres = np.vstack([fiber_centres, random_generator.uniform(0, config.cell_side,
                                                                       (fiber_count - len(fiber_centres), 2))])
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
                         % config.cell_side)
    padded_fraction = fiber_count * np.pi * (minimum_distance / 2)**2 / config.cell_side**2
    raise ValueError(f"Pushing the fibers apart left some overlapping after {config.fiber_push_passes} passes. The "
                     f"fibers with half the gap around each would fill {padded_fraction:.3g} of the cell (at most "
                     "0.907, hexagonal packing): lower volume_fraction or gap, or raise "
                     "fiber_push_passes.")

def is_inside_inclusion(centres):
    """Whether each point, one (x, y, z) row each, lies inside the inclusion. A new shape goes here, in
    config.check_geometry and in get_geometry_parameters (and in shapes_constant_along_z if it is constant along z)."""
    geometry = get_geometry()
    if geometry["shape"] == "random fibers":
        radius = get_fiber_radius()
        return np.any([np.sum(get_periodic_offsets(centres[:, :2], fiber_centre)**2, axis=1) < radius**2
                       for fiber_centre in get_fiber_centres()], axis=0)
    offsets_from_centre = centres - config.cell_side / 2
    if geometry["shape"] == "laminate":
        return np.abs(offsets_from_centre[:, 0]) < geometry["thickness"] / 2
    if geometry["shape"] == "laminate_z":
        return np.abs(offsets_from_centre[:, 2]) < geometry["thickness"] / 2
    if geometry["shape"] == "sphere":
        return np.sum(offsets_from_centre**2, axis=1) < geometry["radius"]**2
    if geometry["shape"] == "fiber":
        return np.sum(offsets_from_centre[:, :2]**2, axis=1) < geometry["radius"]**2
    raise ValueError(f"Unknown shape {geometry['shape']!r}")

## ------- Discretization ------- ##

class Discretization(NamedTuple):
    """A mesh of elements_per_side² × elements_along_z hexahedra over the periodic cubic cell, and its lattice of
    partitions_per_side² × partitions_along_z partitions. Lattices are indexed [z, y, x], as get_flat_index orders them
    (x fastest)."""
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
    def partition_shape_in_elements(self):
        """A partition's size in elements along [z, y, x]."""
        return np.array([self.elements_per_partition_along_z, self.elements_per_partition_side,
                         self.elements_per_partition_side])

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
        return config.cell_side / self.elements_per_side

    @property
    def element_length_along_z(self):
        return config.cell_side / self.elements_along_z

    @property
    def element_volume(self):
        return self.element_side_length**2 * self.element_length_along_z

def get_discretization(elements_per_side, partitions_per_side):
    if is_shape_constant_along_z():
        return Discretization(elements_per_side, 1, partitions_per_side, 1)
    return Discretization(elements_per_side, elements_per_side, partitions_per_side, partitions_per_side)

## ------- Periodic Mesh ------- ##

class Mesh(NamedTuple):
    """The elements, one entry per element in lattice order [z, y, x] (x fastest)."""
    discretization: Discretization
    element_partition: np.ndarray    # the partition of each element
    element_phase: np.ndarray        # the phase of each element: 0 matrix, 1 inclusion

def get_flat_index(i, j, k, count_per_side):
    """The position of lattice point (x index i, y index j, z index k) when the lattice is flattened, x fastest."""
    return i + count_per_side * j + count_per_side**2 * k

def get_element_phase(discretization):
    """The geometry, resolved on the elements: 1 for an element whose centre lies inside the inclusion, else 0."""
    centres_xy = (np.arange(discretization.elements_per_side) + 0.5) * discretization.element_side_length
    centres_z = (np.arange(discretization.elements_along_z) + 0.5) * discretization.element_length_along_z
    z, y, x = np.meshgrid(centres_z, centres_xy, centres_xy, indexing="ij")
    return is_inside_inclusion(np.column_stack([x.ravel(), y.ravel(), z.ravel()])).astype(int)

def get_periodic_mesh(discretization):
    k, j, i = np.meshgrid(*map(np.arange, discretization.element_lattice_shape), indexing="ij")
    element_partition = get_flat_index(i // discretization.elements_per_partition_side,
                                        j // discretization.elements_per_partition_side,
                                        k // discretization.elements_per_partition_along_z,
                                        discretization.partitions_per_side).reshape(-1)
    return Mesh(discretization, element_partition, get_element_phase(discretization))

## ------- Materials ------- ##

class Materials(NamedTuple):
    """The material properties, one row per material state (cluster)."""
    phase: np.ndarray                # 0 matrix, 1 inclusion
    L: np.ndarray                    # the elastic stiffness, 6 × 6 in engineering Voigt notation
    yield_stress: np.ndarray         # infinite for a phase that stays elastic
    hardening_modulus: np.ndarray

def get_L(elastic_modulus, poisson_ratio):
    """Isotropic elastic stiffness in engineering Voigt notation (ε11, ε22, ε33, γ12, γ23, γ13)."""
    lame_lambda   = elastic_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    shear_modulus = elastic_modulus / (2 * (1 + poisson_ratio))
    L = np.zeros((6, 6))
    L[:3, :3] = lame_lambda * np.ones((3, 3)) + 2 * shear_modulus * np.eye(3)
    L[3:, 3:] = shear_modulus * np.eye(3)
    return L

def get_phase_properties(name):
    """A phase's properties ("matrix" or "inclusion") in the material set config.py names."""
    return config.all_materials[config.materials][name]

def get_matrix_L():
    return get_L(get_phase_properties("matrix")["E"], get_phase_properties("matrix")["poisson"])

def get_inclusion_L():
    return get_L(get_phase_properties("inclusion")["E"], get_phase_properties("inclusion")["poisson"])

def get_value_per_phase(matrix_value, inclusion_value, phase):
    """Each entry of phase replaced by the matrix's or the inclusion's value."""
    return np.array([matrix_value, inclusion_value])[phase]

def get_materials(phase):
    """The material properties of every cluster, given each cluster's phase."""
    matrix, inclusion = get_phase_properties("matrix"), get_phase_properties("inclusion")
    return Materials(
        phase=phase,
        L=get_value_per_phase(get_matrix_L(), get_inclusion_L(), phase),
        yield_stress=get_value_per_phase(matrix["yield_stress"], inclusion["yield_stress"], phase),
        hardening_modulus=get_value_per_phase(matrix["hardening"], inclusion["hardening"], phase))

## ------- Clusters ------- ##

class Clusters(NamedTuple):
    """A method's clusters and where they lie. A cluster is a set of elements of one phase in one partition, with one
    material state (one strain, stress and eigenstrain). Clusters are numbered partition by partition, so a partition's
    clusters are consecutive. Clusters have unequal volumes, so every cell average goes through average."""
    materials: Materials               # one row per cluster
    share: np.ndarray                  # each cluster's share of its partition's volume (c_p)
    element_cluster: np.ndarray        # the cluster of every element
    cluster_partition: np.ndarray      # the partition of each cluster

    @property
    def cluster_count(self):
        return len(self.share)

    @property
    def partition_count(self):
        return self.cluster_partition[-1] + 1

    def average(self, values):
        """The volume average over the cell of one value per cluster (rows of values). The cell's volume is
        partition_count partition volumes."""
        weights = self.share.reshape(-1, *[1] * (np.ndim(values) - 1))
        return np.sum(weights * values, axis=0) / self.partition_count

def cluster_by_phase(mesh):
    """The simplest clusters: each partition's part in each phase, one cluster for a single-phase partition, two for a
    mixed one. Returns every element's cluster. A cluster's key is 2 × partition + phase, so the clusters are numbered
    partition by partition, matrix first."""
    _, element_cluster = np.unique(2 * mesh.element_partition + mesh.element_phase, return_inverse=True)
    return element_cluster

def cluster_by_k_means(mesh, element_features, matrix_clusters, inclusion_clusters):
    """Clusters by k-means: each partition's part in each phase is split by k-means on its elements' features (one row
    per element: the elastic strain concentration, Liu–Bessa–Liu 2016, section 2.2.1). Returns every element's cluster.

    A partition's matrix part gets up to matrix_clusters clusters and its inclusion part up to inclusion_clusters, fewer
    when a part has fewer distinct features; clusters k-means leaves empty are dropped. A cluster's key is
    (2 × partition + phase) × the larger count + its label in its part, so the clusters are numbered partition by
    partition, matrix first, and with one cluster per part this is cluster_by_phase."""
    rng = np.random.default_rng(config.k_means_seed)
    cluster_counts = (matrix_clusters, inclusion_clusters)
    part_keys = 2 * mesh.element_partition + mesh.element_phase
    part_order = np.argsort(part_keys, kind="stable")
    part_firsts = np.flatnonzero(np.diff(part_keys[part_order], prepend=-1))
    labels = np.zeros(len(element_features), dtype=int)
    for elements in np.split(part_order, part_firsts[1:]):
        features = element_features[elements]
        cluster_count = min(cluster_counts[mesh.element_phase[elements[0]]],
                            len(np.unique(features, axis=0)))
        if cluster_count == 1:
            continue
        with warnings.catch_warnings():
            # An empty cluster is dropped below, and run.py prints how many clusters there are.
            warnings.filterwarnings("ignore", "One of the clusters is empty")
            _, part_labels = scipy.cluster.vq.kmeans2(features, cluster_count, iter=config.k_means_iterations,
                                                      minit=config.k_means_start, rng=rng)
        labels[elements] = part_labels
    _, element_cluster = np.unique(part_keys * max(cluster_counts) + labels, return_inverse=True)
    return element_cluster

def get_clusters(mesh, element_cluster):
    """The Clusters, given every element's cluster (from cluster_by_phase or cluster_by_k_means)."""
    if mesh.element_phase.min() == mesh.element_phase.max():
        raise ValueError("Every element is the same phase, so the cell is homogeneous: the inclusion is too small or "
                         "too large for this element grid.")
    cluster_count = element_cluster.max() + 1
    phase, cluster_partition = np.zeros(cluster_count, dtype=int), np.zeros(cluster_count, dtype=int)
    phase[element_cluster] = mesh.element_phase
    cluster_partition[element_cluster] = mesh.element_partition
    if np.any(phase[element_cluster] != mesh.element_phase):
        raise ValueError("A cluster holds both phases; every cluster must lie in one phase.")
    if np.any(cluster_partition[element_cluster] != mesh.element_partition):
        raise ValueError("A cluster spans partitions; every cluster must lie in one partition.")
    share = np.bincount(element_cluster) / mesh.discretization.elements_per_partition
    return Clusters(get_materials(phase), share, element_cluster, cluster_partition)

## ------- Memory ------- ##

def get_memory_MB(field):
    """A line of the kernel's account of this process: VmRSS (now) or VmHWM (the peak since run.reset_peak_memory)."""
    with open("/proc/self/status") as status:
        return int(re.search(rf"{field}:\s+(\d+) kB", status.read()).group(1)) / 1024

## ------- Cache ------- ##

def get_cache_path(kind, parameters):
    """One file per parameter set, so switching between configurations reuses each one's results instead of overwriting
    a single file. The name is a hash of every parameter, so a file that exists matches them."""
    digest = hashlib.sha256()
    for name in sorted(parameters):
        value = np.asarray(parameters[name])
        digest.update(f"{name}:{value.dtype.str}:{value.shape}:".encode())
        digest.update(value.tobytes())
    return config.cache_folder / f"{kind}_{digest.hexdigest()[:12]}.npz"

def get_cached(kind, parameters, description, compute):
    """A dict of arrays, from its cache file for these parameters when it exists, otherwise from compute(), and saved
    there with its description, which `run.py cache` lists."""
    cache_path = get_cache_path(kind, parameters)
    if cache_path.exists():
        print(f"{cache_path.name}: loaded from cache.")
        with np.load(cache_path) as cached:
            return {name: cached[name] for name in cached.files}
    print(f"{cache_path.name}: no cache found, computing.")
    arrays = compute()
    config.cache_folder.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **arrays, description=description)
    return arrays

def get_cache_entries():
    """Every cache file and its description."""
    entries = []
    for cache_path in sorted(config.cache_folder.glob("*.npz")):
        with np.load(cache_path) as cached:
            entries.append((cache_path, str(cached["description"]) if "description" in cached.files else ""))
    return entries

def get_geometry_parameters():
    """Every input that sets the geometry, for a cache key: the cell, the shape and the shape's own inputs. The keys'
    names are hashed into the cache files' names, so renaming one makes every existing cache file unreachable."""
    geometry = get_geometry()
    parameters = {"domain_side_length": config.cell_side, "inclusion_shape": geometry["shape"]}
    if geometry["shape"] in ("laminate", "laminate_z"):
        parameters["layer_thickness"] = geometry["thickness"]
    elif geometry["shape"] == "random fibers":
        # The centres stand for the count, volume fraction, gap and seed that placed them.
        parameters["fiber_radius"] = get_fiber_radius()
        parameters["fiber_centres"] = get_fiber_centres()
    else:
        parameters["inclusion_radius"] = geometry["radius"]
    return parameters
