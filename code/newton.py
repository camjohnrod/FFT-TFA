"""What every method shares: solving one load step by Newton's method.

Every method writes the load step as a residual equation r(ε) = 0 in its unknown strains ε (one row of 6 per cluster,
or per element for Spectral LS) and solves it by Newton: at the current strain, solve the linearized equation
J δε = −r for a correction δε by a Krylov method (GMRES here, or conjugate gradients for Spectral LS, in ls_model), step
along it with a backtracking line search, and repeat until the residual is small. Solver runs that loop; a method only
says how to evaluate its residual (evaluate) and how to compute a Newton correction (get_newton_correction).
Also here: helpers for fields of 6 × 6 blocks, which every method uses.
"""

from typing import NamedTuple
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import config
from material import PlasticState

class StepResult(NamedTuple):
    """A solved load step: every cluster's strain, stress and plastic state, and the iterations it took."""
    strain: np.ndarray
    stress: np.ndarray
    plastic_state: PlasticState
    newton_iteration_count: int
    krylov_iteration_count: int    # GMRES or CG iterations, summed over the step's Newton corrections

class Evaluation(NamedTuple):
    """A method's state at one strain: the strain itself (a method may adjust it while evaluating, as TFA's elastic
    reset does), the candidate plastic state and stress, and the residual."""
    strain: np.ndarray
    plastic_state: PlasticState
    stress: np.ndarray
    residual: np.ndarray

class SolverDidNotConverge(RuntimeError):
    """A load step the method cannot solve. The method stops there; the others carry on."""

## ------- Fields of 6 × 6 Blocks ------- ##

def apply_blocks(blocks, rows):
    """One 6 × 6 block times one row of 6, row by row: blocks[p] @ rows[p] for every p."""
    return np.einsum('pij,pj->pi', blocks, rows)

def assemble_blocks(blocks, rows, columns, shape):
    """A sparse matrix of the given shape holding the dense blocks, block p at the global rows[p] and columns[p];
    entries that land on the same place are summed, as in finite-element assembly."""
    rows = np.broadcast_to(rows[:, :, None], blocks.shape)
    columns = np.broadcast_to(columns[:, None, :], blocks.shape)
    return scipy.sparse.coo_matrix((blocks.ravel(), (rows.ravel(), columns.ravel())), shape=shape).tocsr()

def get_block_indices(block_numbers):
    """The six rows (or columns) of each numbered 6 × 6 block: 6 × number + 0, ..., 5."""
    return 6 * block_numbers[:, None] + np.arange(6)

def apply_sparse(matrix, rows):
    """A sparse matrix on stacked rows of 6."""
    return (matrix @ rows.reshape(-1)).reshape(-1, 6)

## ------- Residual Size ------- ##

# Engineering shear (ER-2) is twice the tensor shear, so its squares are weighted by ½ to give the tensor norm.
tensor_norm_weights = np.array([1, 1, 1, 0.5, 0.5, 0.5])

def get_strain_norm(strain, average=np.mean):
    """The root mean square of each row's tensor norm, the mean taken by average (a volume average for clusters of
    unequal volume)."""
    squared_tensor_norms = np.sum(tensor_norm_weights * np.atleast_2d(strain)**2, axis=1)
    return np.sqrt(average(squared_tensor_norms))

def get_relative_residual(residual, macro_strain, average):
    """The size of a strain residual relative to the imposed strain, never divided by less than
    residual_strain_scale_floor, so that it stays meaningful where the imposed strain passes through zero."""
    strain_scale = max(get_strain_norm(macro_strain), config.residual_strain_scale_floor)
    return get_strain_norm(residual, average) / strain_scale

## ------- Newton Correction by GMRES ------- ##

def solve_by_gmres(apply_jacobian, residual, apply_preconditioner=None):
    """Solves J δε = −r by GMRES, with J given only through its product apply_jacobian on fields of rows of 6, until
    ‖r + J δε‖ is below krylov_tolerance ‖r‖, restarting every gmres_restart iterations. Returns δε and the number of
    GMRES iterations.

    A preconditioner M⁻¹ acts on the right: GMRES solves J M⁻¹ y = −r and δε = M⁻¹ y, so it still measures the residual
    of J δε = −r, and M⁻¹ changes only how many iterations that takes, not the correction."""
    def apply_flat_jacobian(flat_vector):
        vector = flat_vector.reshape(-1, 6)
        return apply_jacobian(vector if apply_preconditioner is None else apply_preconditioner(vector)).reshape(-1)

    system_size = residual.size
    jacobian = scipy.sparse.linalg.LinearOperator((system_size, system_size), matvec=apply_flat_jacobian)
    iteration_count = 0

    def count_iteration(_):
        nonlocal iteration_count
        iteration_count += 1

    solution, info = scipy.sparse.linalg.gmres(jacobian, -residual.reshape(-1), rtol=config.krylov_tolerance,
                                               restart=config.gmres_restart, maxiter=config.gmres_max_restarts,
                                               callback=count_iteration, callback_type='pr_norm')
    if info != 0:
        raise SolverDidNotConverge(f"GMRES did not converge inside a Newton iteration (info {info}). Loosen "
                                   "krylov_tolerance or raise gmres_restart")
    solution = solution.reshape(-1, 6)
    return (solution if apply_preconditioner is None else apply_preconditioner(solution)), iteration_count

## ------- The Newton Loop ------- ##

class Solver:
    """One method along the load path. solve_step(macro_strain, plastic_history) solves one load step; its stress and
    plastic state have one row per cluster (cell.Clusters).

    A method (a subclass) supplies
      evaluate(strain, macro_strain, plastic_history)      an Evaluation of its residual at the given strain,
      get_newton_correction(current, plastic_history)      the Newton correction there and its Krylov iteration count,
    and may change where a step starts (start_step), how the residual's size is judged (measure_residual) or when a step
    counts as diverging (divergence_limit).

    offline_from_cache is the (seconds, peak MB) of offline work the method loaded from the cache instead of doing it:
    run.py counts it as if it were done now."""
    offline_from_cache = (0.0, 0.0)

    def __init__(self, clusters, unknown_count):
        self.clusters = clusters
        self.previous_strain = np.zeros((unknown_count, 6))
        self.previous_macro_strain = np.zeros(6)

    def start_over(self):
        """Forget the steps solved so far: the next starts from zero strain, as a new solver would."""
        self.previous_strain = np.zeros_like(self.previous_strain)
        self.previous_macro_strain = np.zeros(6)

    def start_step(self, macro_strain, plastic_history):
        """The step's first strain: the previous step's, shifted by the macro strain increment."""
        return self.previous_strain + (macro_strain - self.previous_macro_strain)

    def measure_residual(self, residual):
        """The residual as convergence judges it: the residual itself, unless a method says otherwise."""
        return residual

    def divergence_limit(self):
        return config.divergence_limit

    def get_relative_residual(self, evaluation, macro_strain):
        return get_relative_residual(self.measure_residual(evaluation.residual), macro_strain,
                                     self.clusters.average)

    def take_step(self, current, correction, relative_residual, macro_strain, plastic_history):
        """Backtracking line search: the full Newton step if it lowers the relative residual enough (by the fraction
        line_search_sufficient_decrease of the step length), otherwise the step shortened by line_search_step_reduction,
        at most line_search_max_reductions times; the shortest step is taken if none does. Where the full step works,
        this is plain Newton."""
        step_length = 1.0
        for _ in range(config.line_search_max_reductions + 1):
            trial = self.evaluate(current.strain + step_length * correction, macro_strain, plastic_history)
            trial_relative_residual = self.get_relative_residual(trial, macro_strain)
            if trial_relative_residual < (1 - config.line_search_sufficient_decrease * step_length) * relative_residual:
                break
            step_length *= config.line_search_step_reduction
        return trial, trial_relative_residual

    def solve_step(self, macro_strain, plastic_history):
        """Newton until the relative residual is below convergence_tolerance.

        Diverging means exceeding the divergence limit times the larger of 1 and the step's first relative residual: at
        a zero crossing of a cyclic load path, the first residual legitimately exceeds the imposed strain. Convergence
        and the line search judge the measured residual; the Newton correction always solves for the residual itself."""
        current = self.evaluate(self.start_step(macro_strain, plastic_history), macro_strain, plastic_history)
        relative_residual = first_relative_residual = self.get_relative_residual(current, macro_strain)
        newton_iteration_count = krylov_iteration_count = 0
        while True:
            if (not np.isfinite(relative_residual)
                    or relative_residual > self.divergence_limit() * max(1.0, first_relative_residual)):
                raise SolverDidNotConverge(f"diverged after {newton_iteration_count} Newton iterations (relative "
                                           f"residual {relative_residual:.3e})")
            if relative_residual < config.convergence_tolerance:
                break
            if newton_iteration_count >= config.newton_max_iterations:
                raise SolverDidNotConverge(f"did not converge within {config.newton_max_iterations} Newton iterations "
                                           f"(relative residual {relative_residual:.3e})")
            correction, iteration_count = self.get_newton_correction(current, plastic_history)
            krylov_iteration_count += iteration_count
            current, relative_residual = self.take_step(current, correction, relative_residual, macro_strain,
                                                        plastic_history)
            newton_iteration_count += 1
        self.previous_strain, self.previous_macro_strain = current.strain, macro_strain
        return StepResult(current.strain, current.stress, current.plastic_state, newton_iteration_count,
                          krylov_iteration_count)
