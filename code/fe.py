# The cell, its grids and its finite elements:
#   elements    the periodic hexahedral mesh, which resolves the geometry;
#   partitions  a uniform lattice of blocks of k × k elements (k × k × k for a geometry that varies along z; columns
#               through the thickness otherwise);
#   clusters    regions of one phase and one material state each, possibly spanning several partitions; a cluster's
#               part in one partition is a piece.
# Every method keeps one material state per cluster. The same code builds CP's partitions and Spectral LS's lattice,
# which is the same mesh with one partition per element (per element column).
#
# From the 8-node element matrices come TFA's influence functions on the clusters (notes/3_tfa_baseline.md):
#   E  the cluster-average strain per unit macrostrain (ER-1),
#   P  the cluster-average strain per unit eigenstrain in each cluster (ER-24), a dense (6 × clusters)² matrix.
# Both come from one factorization of the stiffness K and are cached in code/cache/, since P takes minutes.

import hashlib
import sys
import time
import warnings
from typing import NamedTuple
import numpy as np
import scipy.cluster.vq
import scipy.sparse
import scipy.sparse.linalg
import tqdm

import config

## ------- Shapes ------- ##

# The inclusion shapes whose geometry does not vary along z. They are meshed with config.element_number_along_z
# element layers (one is exact) and partitioned in config.partition_number_along_z layers (one: columns through the
# thickness); every other shape is meshed and partitioned alike along all three axes.
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
    # The (x, y) centres of the "random fibers", one row each, by random sequential addition (config.py).
    element_side_length = config.domain_side_length / config.element_number_per_side
    minimum_distance = 2 * get_fiber_radius() + config.fiber_gap_elements * element_side_length
    random_generator = np.random.default_rng(config.fiber_seed)
    fiber_centres = np.empty((0, 2))
    for _ in range(config.fiber_placement_attempts):
        candidate = random_generator.uniform(0, config.domain_side_length, 2)
        if np.all(np.sum(get_periodic_offsets(fiber_centres, candidate)**2, axis=1) >= minimum_distance**2):
            fiber_centres = np.vstack([fiber_centres, candidate])
            if len(fiber_centres) == config.fiber_count:
                return fiber_centres
    # Random sequential addition of discs jams near 0.55 of the area, here discs of the fiber plus half the gap.
    padded_fraction = config.fiber_count * np.pi * (minimum_distance / 2)**2 / config.domain_side_length**2
    raise ValueError(f"Random sequential addition placed only {len(fiber_centres)} of {config.fiber_count} fibers in "
                     f"{config.fiber_placement_attempts} attempts. The fibers with half the gap around each would "
                     f"fill {padded_fraction:.3g} of the cell, and this placement jams near 0.55: lower "
                     "fiber_volume_fraction or fiber_gap_elements, or raise fiber_placement_attempts.")

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
    # A mesh of element_number_per_side² × element_number_along_z hexahedra over the periodic cubic cell, and its
    # lattice of partition_number_per_side² × partition_number_along_z partitions. Lattices are indexed [z, y, x], as
    # get_grid_id orders them (x fastest).
    element_number_per_side: int
    element_number_along_z: int
    partition_number_per_side: int
    partition_number_along_z: int

    @property
    def elements_per_partition_side(self):
        return self.element_number_per_side // self.partition_number_per_side

    @property
    def elements_per_partition_along_z(self):
        return self.element_number_along_z // self.partition_number_along_z

    @property
    def elements_per_partition(self):
        return self.elements_per_partition_side**2 * self.elements_per_partition_along_z

    @property
    def element_lattice_shape(self):
        return (self.element_number_along_z, self.element_number_per_side, self.element_number_per_side)

    @property
    def partition_lattice_shape(self):
        return (self.partition_number_along_z, self.partition_number_per_side, self.partition_number_per_side)

    @property
    def element_count(self):
        return self.element_number_per_side**2 * self.element_number_along_z

    @property
    def dof_count(self):
        return 3 * self.element_count

    @property
    def element_side_length(self):
        return config.domain_side_length / self.element_number_per_side

    @property
    def element_length_along_z(self):
        return config.domain_side_length / self.element_number_along_z

    @property
    def element_volume(self):
        return self.element_side_length**2 * self.element_length_along_z

    @property
    def partition_volume(self):
        return self.elements_per_partition * self.element_volume

def get_partition_discretization():
    # CP's partitions, which are also TFA's: the mesh and lattice config.py describes.
    elements, partitions = config.element_number_per_side, config.partition_number_per_side
    if is_shape_constant_along_z():
        return Discretization(elements, config.element_number_along_z, partitions, config.partition_number_along_z)
    return Discretization(elements, elements, partitions, partitions)

def get_spectral_discretization():
    # Spectral LS's lattice: the same mesh with one partition per element (per element column for a shape constant
    # along z, whatever CP's partition_number_along_z).
    discretization = get_partition_discretization()
    return discretization._replace(
        partition_number_per_side=discretization.element_number_per_side,
        partition_number_along_z=1 if is_shape_constant_along_z() else discretization.element_number_along_z)

## ------- Periodic Mesh ------- ##

class Mesh(NamedTuple):
    discretization: Discretization
    element_nodes: np.ndarray            # the 8 nodes of each element, in corner_natural_coordinates order
    element_partition_ids: np.ndarray
    element_material_ids: np.ndarray     # 0 matrix, 1 inclusion

def get_grid_id(i, j, k, count_per_side):
    return i + count_per_side * j + count_per_side**2 * k

def get_element_material_ids(discretization):
    # The geometry, resolved on the elements: 1 for an element whose centre lies inside the inclusion, else 0.
    centres_xy = (np.arange(discretization.element_number_per_side) + 0.5) * discretization.element_side_length
    centres_z = (np.arange(discretization.element_number_along_z) + 0.5) * discretization.element_length_along_z
    z, y, x = np.meshgrid(centres_z, centres_xy, centres_xy, indexing="ij")
    material_ids = is_inside_inclusion(np.column_stack([x.ravel(), y.ravel(), z.ravel()])).astype(int)
    # A shape partitioned in columns must indeed be the same in every layer.
    layers = material_ids.reshape(discretization.element_number_along_z, -1)
    if is_shape_constant_along_z() and not np.all(layers == layers[0]):
        raise ValueError(f"{config.inclusion_shape!r} is in fe.shapes_constant_along_z, but its geometry varies "
                         "along z.")
    return material_ids

def get_periodic_mesh(discretization):
    element_number_per_side = discretization.element_number_per_side
    element_nodes = np.zeros((discretization.element_count, 8), dtype=int)
    element_partition_ids = np.zeros(discretization.element_count, dtype=int)

    for k in range(discretization.element_number_along_z):
        for j in range(element_number_per_side):
            for i in range(element_number_per_side):
                element_id = get_grid_id(i, j, k, element_number_per_side)

                for offset_k in range(2):
                    for offset_j in range(2):
                        for offset_i in range(2):
                            corner = get_grid_id(offset_i, offset_j, offset_k, 2)
                            element_nodes[element_id, corner] = get_grid_id(
                                (i + offset_i) % element_number_per_side,
                                (j + offset_j) % element_number_per_side,
                                (k + offset_k) % discretization.element_number_along_z,
                                element_number_per_side)

                element_partition_ids[element_id] = get_grid_id(i // discretization.elements_per_partition_side,
                                                                j // discretization.elements_per_partition_side,
                                                                k // discretization.elements_per_partition_along_z,
                                                                discretization.partition_number_per_side)

    return Mesh(discretization, element_nodes, element_partition_ids, get_element_material_ids(discretization))

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
    return get_L(config.elastic_modulus_matrix, config.poisson_ratio_matrix)

def get_inclusion_L():
    return get_L(config.elastic_modulus_inclusion, config.poisson_ratio_inclusion)

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
    # The material states of a lattice and where they lie. A cluster is a set of elements of one phase with one material
    # state (one uniform eigenstrain); it may span several partitions. A piece is the part of a cluster inside one
    # partition, so a partition holds one piece per cluster it meets. Pieces are ordered by partition.
    materials: Materials               # one row per cluster
    volume: np.ndarray                 # each cluster's volume, in units of one partition's volume
    element_piece: np.ndarray          # the piece of every element
    piece_partition: np.ndarray        # the partition of each piece
    piece_cluster: np.ndarray          # the cluster of each piece
    piece_volume_fraction: np.ndarray  # each piece's share of its partition's volume

    @property
    def cluster_count(self):
        return len(self.volume)

    @property
    def partition_count(self):
        return self.piece_partition[-1] + 1

    @property
    def element_cluster(self):
        return self.piece_cluster[self.element_piece]

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

def label_k_means(mesh, element_features):
    # Every element's cluster by k-means on its features (one row per element; the elastic strain concentration,
    # Liu–Bessa–Liu 2016, section 2.2.1), in each phase apart, so every cluster lies in one phase: up to
    # k_means_matrix_clusters in the matrix, then up to k_means_inclusion_clusters in the inclusion. A phase gets no
    # more clusters than it has distinct features, and clusters k-means leaves empty are dropped.
    rng = np.random.default_rng(config.k_means_seed)
    element_cluster = np.zeros(len(element_features), dtype=int)
    first_cluster = 0
    for phase, cluster_count in ((0, config.k_means_matrix_clusters), (1, config.k_means_inclusion_clusters)):
        elements = np.flatnonzero(mesh.element_material_ids == phase)
        features = element_features[elements]
        cluster_count = min(cluster_count, len(np.unique(features, axis=0)))
        with warnings.catch_warnings():
            # An empty cluster is dropped below, and run.py prints how many clusters each phase got.
            warnings.filterwarnings("ignore", "One of the clusters is empty")
            _, labels = scipy.cluster.vq.kmeans2(features, cluster_count, iter=config.k_means_iterations,
                                                 minit="++", rng=rng)
        _, labels = np.unique(labels, return_inverse=True)
        element_cluster[elements] = first_cluster + labels
        first_cluster += labels.max() + 1
    return element_cluster

def get_clusters(mesh, element_cluster):
    # The clusters, given every element's cluster, and their pieces. A piece's key is partition × cluster count +
    # cluster, so sorting the keys orders the pieces by partition.
    if mesh.element_material_ids.min() == mesh.element_material_ids.max():
        raise ValueError("Every element is the same phase, so the cell is homogeneous: the inclusion is too small or "
                         "too large for this element grid.")
    cluster_count = element_cluster.max() + 1
    material_ids = np.zeros(cluster_count, dtype=int)
    material_ids[element_cluster] = mesh.element_material_ids
    if np.any(material_ids[element_cluster] != mesh.element_material_ids):
        raise ValueError("A cluster holds both phases; every cluster must lie in one phase.")
    piece_keys, element_piece = np.unique(cluster_count * mesh.element_partition_ids + element_cluster,
                                          return_inverse=True)
    elements_per_partition = mesh.discretization.elements_per_partition
    return Clusters(get_materials(material_ids), np.bincount(element_cluster) / elements_per_partition, element_piece,
                    piece_keys // cluster_count, piece_keys % cluster_count,
                    np.bincount(element_piece) / elements_per_partition)

## ------- Element Matrices ------- ##

# The element's corners in natural coordinates, in the order get_grid_id gives them (x fastest).
corner_natural_coordinates = np.array([[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
                                       [-1, -1,  1], [1, -1,  1], [-1, 1,  1], [1, 1,  1]])

def get_shape_function_derivatives(xi, eta, zeta):
    # The trilinear shape functions' derivatives with respect to the natural coordinates, one entry per corner.
    corner_xi, corner_eta, corner_zeta = corner_natural_coordinates.T
    xi_bracket   = 1 + corner_xi   * xi
    eta_bracket  = 1 + corner_eta  * eta
    zeta_bracket = 1 + corner_zeta * zeta
    derivative_wrt_xi   = corner_xi   * eta_bracket * zeta_bracket / 8
    derivative_wrt_eta  = corner_eta  * xi_bracket  * zeta_bracket / 8
    derivative_wrt_zeta = corner_zeta * xi_bracket  * eta_bracket  / 8
    return derivative_wrt_xi, derivative_wrt_eta, derivative_wrt_zeta

def get_B(discretization):
    # The strain-displacement matrix ε = B u (engineering shear) at each of the 2 × 2 × 2 Gauss points, the same for
    # every element of the regular mesh: shape (8 Gauss points, 6, 24).
    gauss_point_coordinates = corner_natural_coordinates / np.sqrt(3)
    B = np.zeros((8, 6, 24))

    for gauss_index, (xi, eta, zeta) in enumerate(gauss_point_coordinates):
        derivatives_wrt_xi, derivatives_wrt_eta, derivatives_wrt_zeta = get_shape_function_derivatives(xi, eta, zeta)
        derivatives_wrt_x = derivatives_wrt_xi   * 2 / discretization.element_side_length
        derivatives_wrt_y = derivatives_wrt_eta  * 2 / discretization.element_side_length
        derivatives_wrt_z = derivatives_wrt_zeta * 2 / discretization.element_length_along_z

        for corner in range(8):
            column = 3 * corner
            B[gauss_index, 0, column]     = derivatives_wrt_x[corner]
            B[gauss_index, 1, column + 1] = derivatives_wrt_y[corner]
            B[gauss_index, 2, column + 2] = derivatives_wrt_z[corner]
            B[gauss_index, 3, column]     = derivatives_wrt_y[corner]
            B[gauss_index, 3, column + 1] = derivatives_wrt_x[corner]
            B[gauss_index, 4, column + 1] = derivatives_wrt_z[corner]
            B[gauss_index, 4, column + 2] = derivatives_wrt_y[corner]
            B[gauss_index, 5, column]     = derivatives_wrt_z[corner]
            B[gauss_index, 5, column + 2] = derivatives_wrt_x[corner]

    return B

def get_K_element(B, gauss_point_volume, L):
    # ∫ Bᵀ L B dV by 2 × 2 × 2 Gauss quadrature.
    K_element = np.zeros((24, 24))
    for gauss_index in range(8):
        K_element += gauss_point_volume * B[gauss_index].T @ L @ B[gauss_index]
    return K_element

## ------- Assembly ------- ##

def get_element_dofs(mesh):
    node_dofs = 3 * mesh.element_nodes[:, :, None] + np.arange(3)
    return node_dofs.reshape(-1, 24)

def get_cluster_components(element_cluster):
    # The six entries of each element's cluster in a cluster-stacked vector such as ε, μ or b.
    return 6 * element_cluster[:, None] + np.arange(6)

def assemble_element_blocks(element_blocks, element_rows, element_columns, shape):
    # Adds each element's block into a sparse global matrix at the given global rows and columns. Blocks that share
    # an entry are summed, as in finite-element assembly.
    rows = np.broadcast_to(element_rows[:, :, None], element_blocks.shape)
    columns = np.broadcast_to(element_columns[:, None, :], element_blocks.shape)
    return scipy.sparse.coo_matrix((element_blocks.ravel(), (rows.ravel(), columns.ravel())), shape=shape).tocsr()

class FEModel:
    # The periodic FE model of the composite on a mesh: its element matrices and the global operators of ER-24.
    def __init__(self, mesh):
        self.mesh = mesh
        self.discretization = mesh.discretization
        self.B = get_B(self.discretization)
        self.gauss_point_volume = self.discretization.element_volume / 8
        self.integrated_B = self.gauss_point_volume * np.sum(self.B, axis=0)    # ∫ B dV over one element
        self.L_per_element = get_value_per_material(get_matrix_L(), get_inclusion_L(), mesh.element_material_ids)
        self.element_dofs = get_element_dofs(mesh)

    def get_K(self):
        K_elements = np.array([get_K_element(self.B, self.gauss_point_volume, L) for L in self.L_per_element])
        dof_count = self.discretization.dof_count
        return assemble_element_blocks(K_elements, self.element_dofs, self.element_dofs, (dof_count, dof_count))

    def get_F_elements(self):
        # Each element's nodal load per unit eigenstrain, ∫ Bᵀ L dV (24 × 6).
        return self.integrated_B.T @ self.L_per_element

    def get_F_macrostrain(self):
        # Nodal loads per unit macrostrain. Moving the applied strain to the right-hand side gives −∫ Bᵀ L ε̄ dV.
        macrostrain_columns = np.tile(np.arange(6), (self.discretization.element_count, 1))
        return -assemble_element_blocks(self.get_F_elements(), self.element_dofs, macrostrain_columns,
                                        (self.discretization.dof_count, 6))

    def get_F_eigenstrain(self, clusters):
        # F_μ in ER-24: nodal loads per unit cluster eigenstrain, one block of six columns per cluster.
        return assemble_element_blocks(self.get_F_elements(), self.element_dofs,
                                       get_cluster_components(clusters.element_cluster),
                                       (self.discretization.dof_count, 6 * clusters.cluster_count))

    def get_cluster_averaging_operator(self, clusters):
        # A_ε in ER-24: maps nodal displacements to cluster-average strain.
        cluster_volume = clusters.volume * self.discretization.partition_volume
        element_blocks = self.integrated_B / cluster_volume[clusters.element_cluster, None, None]
        return assemble_element_blocks(element_blocks, get_cluster_components(clusters.element_cluster),
                                       self.element_dofs, (6 * clusters.cluster_count, self.discretization.dof_count))

## ------- Influence Functions E and P ------- ##

def get_influence_function_solver(model, clusters):
    # Factorizes K once and returns solve(F), the cluster-average strain response A_ε K⁻¹ F to every load column of
    # F, so that several sets of load columns share one factorization. Pinning one node's three displacements removes
    # the rigid translation of the periodic cell.
    dof_count = model.discretization.dof_count
    free_dofs = np.arange(3, dof_count)
    K_free = model.get_K()[free_dofs][:, free_dofs].tocsc()
    print(f"  {f'factor K ({K_free.shape[0]} DOFs)':<26}: ", end="", flush=True)
    start_time = time.perf_counter()
    K_free_factorization = scipy.sparse.linalg.splu(K_free, permc_spec='MMD_AT_PLUS_A')
    print(f"{time.perf_counter() - start_time:.1f} s")
    A = model.get_cluster_averaging_operator(clusters)

    def solve(F):
        F_free = F.tocsr()[free_dofs].tocsc()
        columns_per_solve = config.fe_columns_per_solve
        average_strain = np.zeros((A.shape[0], F.shape[1]))
        column_batches = tqdm.tqdm(range(0, F.shape[1], columns_per_solve),
                                   desc=f"  {f'solve {F.shape[1]} load columns':<26}",
                                   bar_format="{desc}: {percentage:3.0f}% |{bar:25}| {n_fmt}/{total_fmt} batches "
                                              "[{elapsed} elapsed, {remaining} left]", file=sys.stdout)
        for first_column in column_batches:
            columns = slice(first_column, first_column + columns_per_solve)
            batch_F_free = F_free[:, columns].toarray()
            batch_displacements = np.zeros((dof_count, batch_F_free.shape[1]))
            batch_displacements[free_dofs] = K_free_factorization.solve(batch_F_free)
            average_strain[:, columns] = A @ batch_displacements
        return average_strain

    return solve

def compute_E_and_P(mesh, clusters, include_P):
    # E, and P when include_P (None otherwise), from one factorization of K, each with the seconds it took: E's include
    # assembling and factorizing K, P's only its own solves. E's six load columns are solved on their own, so E is the
    # same to the last bit whether or not P is built.
    start_time = time.perf_counter()
    model = FEModel(mesh)
    solve = get_influence_function_solver(model, clusters)
    # ER-1: E is the applied macrostrain plus the fluctuation it causes; P is A_ε K⁻¹ F_μ (ER-24).
    E = np.tile(np.eye(6), (clusters.cluster_count, 1)) + solve(model.get_F_macrostrain())
    E_seconds = time.perf_counter() - start_time
    P = solve(model.get_F_eigenstrain(clusters)) if include_P else None
    P_seconds = time.perf_counter() - start_time - E_seconds
    return {"E": E, "P": P}, {"E": E_seconds, "P": P_seconds}

## ------- Cache ------- ##

# Increase whenever a code change alters E or P, so caches made by the old code are recomputed.
# 2: the geometry is resolved on the elements, not the partitions. 3: E and P are built on clusters.
cache_version = 3

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
    # saved to its own file. One file per array lets a run load only the arrays it needs (E without P when TFA is off).
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

def get_E_and_P(mesh, clusters, include_P):
    # E, and P only when include_P (None otherwise), on the clusters, and the seconds they took to build, by name
    # ("E", "P"), measured when they were computed, even if they now come from the cache.
    discretization = mesh.discretization
    parameters = {"cache_version": cache_version,
                  "domain_side_length": config.domain_side_length,
                  "inclusion_shape": config.inclusion_shape,
                  "element_number_per_side": discretization.element_number_per_side,
                  "element_number_along_z": discretization.element_number_along_z,
                  "partition_number_per_side": discretization.partition_number_per_side,
                  "L_matrix": get_matrix_L(),
                  "L_inclusion": get_inclusion_L()}
    if config.inclusion_shape in ("laminate", "laminate_z"):
        parameters["layer_thickness"] = config.layer_thickness
    elif config.inclusion_shape == "random fibers":
        # The centres stand for the count, volume fraction, gap and seed that placed them.
        parameters["fiber_radius"] = get_fiber_radius()
        parameters["fiber_centres"] = get_fiber_centres()
    else:
        parameters["inclusion_radius"] = config.inclusion_radius

    # The clusters themselves, so any change of clustering (k-means settings included) finds its own cache.
    parameters["element_cluster"] = clusters.element_cluster
    E_P, build_seconds = get_cached(("E", "P") if include_P else ("E",), parameters,
                                    lambda: compute_E_and_P(mesh, clusters, include_P))
    if not include_P:
        return E_P["E"], None, build_seconds
    # Column-major, so the columns of a run of consecutive clusters are contiguous and P products can read just the
    # yielding clusters' columns without copying them (tfa_model.apply_P_to_partitions).
    return E_P["E"], np.asfortranarray(E_P["P"]), build_seconds
