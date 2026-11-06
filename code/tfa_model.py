"""TFA, the baseline (transformation field analysis, notes/3_tfa_baseline.md, equations ER-*).

Each cluster has one uniform eigenstrain μ and one material update at its average strain ε. Offline, a finite-element
model of the whole composite gives two influence functions on the clusters:
  E  the cluster-average strain per unit macro strain (ER-1), clusters × 6 rows by 6 columns;
  P  the cluster-average strain per unit eigenstrain in each cluster (ER-24), a dense (6 × clusters)² matrix.
Online, each load step solves r = ε − E ε̄ − P μ(ε) = 0 (ER-4) by Newton, each correction by GMRES on the matrix-free
Jacobian J v = v − P (H_μ v) of ER-21. E and P are exact for the composite in elasticity, but P is dense: its memory and
its products grow with the square of the clusters. Both come from one factorization of the FE stiffness K and are
cached in code/cache/, since P can take minutes.
"""

import sys
import time
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import tqdm

import config
from cell import (get_cached, get_geometry_parameters, get_flat_index, get_inclusion_L, get_matrix_L, get_memory_MB,
                  get_value_per_phase)
from material import apply_blocks, get_eigenstrain_sensitivity, radial_return, get_yielding_clusters
from newton import Evaluation, Solver, solve_by_gmres

## ------- Element Matrices ------- ##

# The element's corners in natural coordinates, in the order get_flat_index gives them (x fastest).
corner_natural_coordinates = np.array([[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
                                       [-1, -1,  1], [1, -1,  1], [-1, 1,  1], [1, 1,  1]])

def get_shape_function_derivatives(xi, eta, zeta):
    """The trilinear shape functions' derivatives with respect to the natural coordinates, one entry per corner."""
    corner_xi, corner_eta, corner_zeta = corner_natural_coordinates.T
    xi_bracket   = 1 + corner_xi   * xi
    eta_bracket  = 1 + corner_eta  * eta
    zeta_bracket = 1 + corner_zeta * zeta
    derivative_wrt_xi   = corner_xi   * eta_bracket * zeta_bracket / 8
    derivative_wrt_eta  = corner_eta  * xi_bracket  * zeta_bracket / 8
    derivative_wrt_zeta = corner_zeta * xi_bracket  * eta_bracket  / 8
    return derivative_wrt_xi, derivative_wrt_eta, derivative_wrt_zeta

def get_B(discretization):
    """The strain-displacement matrix ε = B u (engineering shear) at each of the 2 × 2 × 2 Gauss points, the same for
    every element of the regular mesh: shape (8 Gauss points, 6, 24)."""
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
    """∫ Bᵀ L B dV by 2 × 2 × 2 Gauss quadrature."""
    K_element = np.zeros((24, 24))
    for gauss_index in range(8):
        K_element += gauss_point_volume * B[gauss_index].T @ L @ B[gauss_index]
    return K_element

## ------- Assembly ------- ##

def get_element_dofs(discretization):
    """The 24 degrees of freedom of each element, its 8 nodes in corner_natural_coordinates order, nodes numbered as the
    elements are, periodically."""
    k, j, i = [axis.reshape(-1, 1) for axis in np.meshgrid(*map(np.arange, discretization.element_lattice_shape),
                                                           indexing="ij")]
    offset_i, offset_j, offset_k = (corner_natural_coordinates.T + 1) // 2
    element_nodes = get_flat_index((i + offset_i) % discretization.elements_per_side,
                                (j + offset_j) % discretization.elements_per_side,
                                (k + offset_k) % discretization.elements_along_z, discretization.elements_per_side)
    return (3 * element_nodes[:, :, None] + np.arange(3)).reshape(-1, 24)

def get_cluster_components(element_cluster):
    """The six entries of each element's cluster in a cluster-stacked vector such as ε, μ or b."""
    return 6 * element_cluster[:, None] + np.arange(6)

def assemble_element_blocks(element_blocks, element_rows, element_columns, shape):
    """Adds each element's block into a sparse global matrix at the given global rows and columns. Blocks that share an
    entry are summed, as in finite-element assembly."""
    rows = np.broadcast_to(element_rows[:, :, None], element_blocks.shape)
    columns = np.broadcast_to(element_columns[:, None, :], element_blocks.shape)
    return scipy.sparse.coo_matrix((element_blocks.ravel(), (rows.ravel(), columns.ravel())), shape=shape).tocsr()

class FEModel:
    """The periodic FE model of the composite on a mesh: its element matrices and the global operators of ER-24."""
    def __init__(self, mesh):
        self.mesh = mesh
        self.discretization = mesh.discretization
        self.B = get_B(self.discretization)
        self.gauss_point_volume = self.discretization.element_volume / 8
        self.integrated_B = self.gauss_point_volume * np.sum(self.B, axis=0)    # ∫ B dV over one element
        self.L_per_element = get_value_per_phase(get_matrix_L(), get_inclusion_L(), mesh.element_phase)
        self.element_dofs = get_element_dofs(self.discretization)

    def get_K(self):
        K_elements = np.array([get_K_element(self.B, self.gauss_point_volume, L) for L in self.L_per_element])
        dof_count = self.discretization.dof_count
        return assemble_element_blocks(K_elements, self.element_dofs, self.element_dofs, (dof_count, dof_count))

    def get_F_elements(self):
        """Each element's nodal load per unit eigenstrain, ∫ Bᵀ L dV (24 × 6)."""
        return self.integrated_B.T @ self.L_per_element

    def get_F_macro_strain(self):
        """Nodal loads per unit macro strain. Moving the applied strain to the right-hand side gives −∫ Bᵀ L ε̄ dV."""
        macro_strain_columns = np.tile(np.arange(6), (self.discretization.element_count, 1))
        return -assemble_element_blocks(self.get_F_elements(), self.element_dofs, macro_strain_columns,
                                        (self.discretization.dof_count, 6))

    def get_F_eigenstrain(self, clusters):
        """F_μ in ER-24: nodal loads per unit cluster eigenstrain, one block of six columns per cluster."""
        return assemble_element_blocks(self.get_F_elements(), self.element_dofs,
                                       get_cluster_components(clusters.element_cluster),
                                       (self.discretization.dof_count, 6 * clusters.cluster_count))

    def get_cluster_averaging_operator(self, clusters):
        """A_ε in ER-24: maps nodal displacements to cluster-average strain."""
        partition_volume = self.discretization.elements_per_partition * self.discretization.element_volume
        cluster_volume = clusters.share * partition_volume
        element_blocks = self.integrated_B / cluster_volume[clusters.element_cluster, None, None]
        return assemble_element_blocks(element_blocks, get_cluster_components(clusters.element_cluster),
                                       self.element_dofs, (6 * clusters.cluster_count, self.discretization.dof_count))

## ------- Influence Functions E and P ------- ##

def get_influence_function_solver(model, clusters):
    """Factorizes K once and returns solve(F), the cluster-average strain response A_ε K⁻¹ F to every load column of F,
    so that several sets of load columns share one factorization. Pinning one node's three displacements removes the
    rigid translation of the periodic cell."""
    dof_count = model.discretization.dof_count
    free_dofs = np.arange(3, dof_count)
    K_free = model.get_K()[free_dofs][:, free_dofs].tocsc()
    print(f"  {f'factor K ({K_free.shape[0]} DOFs)':<26}: ", end="", flush=True)
    start_time = time.perf_counter()
    K_free_factorization = scipy.sparse.linalg.splu(K_free, permc_spec=config.tfa_factorization_ordering)
    print(f"{time.perf_counter() - start_time:.1f} s")
    A = model.get_cluster_averaging_operator(clusters)

    def solve(F):
        F_free = F.tocsr()[free_dofs].tocsc()
        columns_per_solve = config.tfa_columns_per_solve
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

def compute_E_and_P(mesh, clusters):
    """E and P from one factorization of K, the seconds they took and the memory they needed."""
    start_time, start_memory = time.perf_counter(), get_memory_MB("VmRSS")
    model = FEModel(mesh)
    solve = get_influence_function_solver(model, clusters)
    # ER-1: E is the applied macro strain plus the fluctuation it causes; P is A_ε K⁻¹ F_μ (ER-24).
    E = np.tile(np.eye(6), (clusters.cluster_count, 1)) + solve(model.get_F_macro_strain())
    P = solve(model.get_F_eigenstrain(clusters))
    return {"E": E, "P": P, "build_seconds": time.perf_counter() - start_time,
            "build_peak_MB": get_memory_MB("VmHWM") - start_memory}

## ------- Cache of E and P ------- ##

# Increase whenever a code change alters E or P, so caches made by the old code are recomputed.
cache_version = 4

def get_E_and_P(mesh, clusters):
    """E and P on the clusters, and the seconds and memory they took to build, measured when they were computed, even if
    they now come from the cache."""
    discretization = mesh.discretization
    parameters = {"cache_version": cache_version,
                  **get_geometry_parameters(),
                  "elements_per_side": discretization.elements_per_side,
                  "elements_along_z": discretization.elements_along_z,
                  "partitions_per_side": discretization.partitions_per_side,
                  "L_matrix": get_matrix_L(),
                  "L_inclusion": get_inclusion_L(),
                  # The clusters themselves, so any change of clustering (k-means settings included) finds its own
                  # cache.
                  "element_cluster": clusters.element_cluster}
    description = (f"E and P on {clusters.cluster_count} clusters, {discretization.elements_per_side} elements per "
                   f"side, {discretization.partitions_per_side} partitions per side, {config.geometry}, "
                   f"{config.materials}")
    arrays = get_cached("E_and_P", parameters, description, lambda: compute_E_and_P(mesh, clusters))
    # Column-major, so the columns of a run of consecutive clusters are contiguous and P products can read just the
    # yielding clusters' columns without copying them (apply_P_to_clusters).
    return arrays["E"], np.asfortranarray(arrays["P"]), float(arrays["build_seconds"]), float(arrays["build_peak_MB"])

## ------- Solver ------- ##

def get_cluster_runs(clusters):
    """(first, stop) of every run of consecutive clusters in the mask, in cluster order."""
    edges = np.diff(np.concatenate(([0], clusters.astype(int), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))

def apply_P_to_clusters(P, cluster_field, clusters):
    """P times a cluster field that is zero outside the given clusters, from only those clusters' columns of P. P is
    column-major, so each run of consecutive clusters is one contiguous block of columns, used without a copy.
    Fancy-indexing P[:, columns] instead would copy them, which costs more than the full product."""
    product = np.zeros(P.shape[0])
    for first, stop in get_cluster_runs(clusters):
        product += P[:, 6 * first:6 * stop] @ cluster_field[first:stop].reshape(-1)
    return product.reshape(-1, 6)

class TFASolver(Solver):
    """TFA on the given clusters, which E and P were built on. Each step starts from the exact elastic predictor
    E ε̄ + P μₙ, and every evaluation solves the clusters that stay elastic exactly (evaluate)."""
    def __init__(self, E, P, clusters):
        super().__init__(clusters, clusters.cluster_count)
        self.E = E
        self.P = P
        # Set at the start of each step (start_step): b = E ε̄, the strain the macro strain alone causes (ER-4), and
        # P μₙ, the strain the accepted eigenstrains cause.
        self.b = None
        self.induced_strain_history = None

    def divergence_limit(self):
        return config.tfa_divergence_limit

    def start_step(self, macro_strain, plastic_history):
        """The exact elastic predictor: the strain if no cluster yields further in this step."""
        self.b = (self.E @ macro_strain).reshape(-1, 6)
        self.induced_strain_history = (self.P @ plastic_history.plastic_strain.reshape(-1)).reshape(-1, 6)
        return self.b + self.induced_strain_history

    def get_induced_strain(self, plastic_state, plastic_history):
        """Pμ = Pμₙ + PΔμ, where Δμ is zero outside the clusters yielding in this step, so only their columns of P are
        needed. The mask is taken from Δμ itself, so every skipped column multiplies an exact zero."""
        eigenstrain_increment = plastic_state.plastic_strain - plastic_history.plastic_strain
        changed_clusters = np.any(eigenstrain_increment != 0, axis=1)
        if not np.any(changed_clusters):
            return self.induced_strain_history
        return self.induced_strain_history + apply_P_to_clusters(self.P, eigenstrain_increment, changed_clusters)

    def evaluate(self, strain, macro_strain, plastic_history):
        """Updates the material at the given strain, solves the clusters that stay elastic exactly, and returns the
        residual r = ε - b - Pμ(ε) of ER-4, which is then zero in every elastic cluster.

        An elastic cluster's eigenstrain stays at its accepted value, so setting its strain to b + Pμ zeroes its
        residual without changing μ. Yielding clusters keep their strain, so their μ does not change either. The second
        material update catches elastic clusters that the reset pushed past yield: only then does μ, and so Pμ, change.
        """
        materials = self.clusters.materials
        plastic_state, stress = radial_return(strain, materials, plastic_history)
        induced_strain = self.get_induced_strain(plastic_state, plastic_history)
        yielding_clusters = get_yielding_clusters(plastic_state, plastic_history)

        strain = strain.copy()
        strain[~yielding_clusters] = self.b[~yielding_clusters] + induced_strain[~yielding_clusters]

        plastic_state, stress = radial_return(strain, materials, plastic_history)
        if not np.array_equal(get_yielding_clusters(plastic_state, plastic_history), yielding_clusters):
            induced_strain = self.get_induced_strain(plastic_state, plastic_history)

        return Evaluation(strain, plastic_state, stress, strain - self.b - induced_strain)

    def get_newton_correction(self, current, plastic_history):
        """Solves J δε = -r (ER-18) by GMRES, with the matrix-free product J v = v - P (H_μ v) of ER-21. H_μ is zero
        outside the yielding clusters, so each product reads only their columns of P."""
        sensitivity = get_eigenstrain_sensitivity(current.strain, self.clusters.materials, plastic_history)
        yielding_clusters = np.any(sensitivity != 0, axis=(1, 2))

        def apply_jacobian(vector):
            return vector - apply_P_to_clusters(self.P, apply_blocks(sensitivity, vector), yielding_clusters)

        return solve_by_gmres(apply_jacobian, current.residual)
