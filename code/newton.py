# What every method shares to solve a load step by Newton–GMRES: the Newton loop with its backtracking line search, the
# relative residual it converges on, the convergence test, the GMRES solve of a Newton correction, the result of a
# step, and the errors that end a method's load path early. A method supplies only how to evaluate its residual at a
# strain and how to compute a Newton correction there.

from typing import Callable, NamedTuple
import numpy as np
import scipy.sparse.linalg

import config
from material import PlasticState

class StepResult(NamedTuple):
    strain: np.ndarray
    stress: np.ndarray
    plastic_state: PlasticState
    # The relative residual at every residual evaluation, one per Newton iteration plus the first.
    residual_history: np.ndarray
    gmres_iteration_count: int

class Evaluation(NamedTuple):
    # A method's state at one strain: the strain itself (a method may adjust it while evaluating, as TFA's elastic reset
    # does), the candidate plastic state and stress, and the residual.
    strain: np.ndarray
    plastic_state: PlasticState
    stress: np.ndarray
    residual: np.ndarray

class Solver(NamedTuple):
    # A method ready to run along the load path: solve_step(macro_strain, plastic_history) returns a StepResult whose
    # stress and plastic state have one row per cluster of clusters (fe.Clusters). TFA uses this; the other methods'
    # solvers (ls_model, fft_tfa_model) are classes with the same two attributes.
    solve_step: Callable
    clusters: tuple

class LoadPathAbandoned(RuntimeError):
    # A load step the method cannot solve. The method stops there; the others carry on.
    pass

class SolverDidNotConverge(LoadPathAbandoned):
    pass

class MaterialFullySoftened(LoadPathAbandoned):
    # A flow stress reached zero, past which the return map would flip the stress's sign.
    pass

# Engineering shear (ER-2) is twice the tensor shear, so its squares are weighted by ½ to give the tensor norm.
tensor_norm_weights = np.array([1, 1, 1, 0.5, 0.5, 0.5])

def get_strain_norm(strain, average=np.mean):
    # The root mean square of each row's tensor norm, the mean taken by average: a plain mean for rows of equal
    # volume (partitions), a volume average (Clusters.average) for TFA's clusters.
    squared_tensor_norms = np.sum(tensor_norm_weights * np.atleast_2d(strain)**2, axis=1)
    return np.sqrt(average(squared_tensor_norms))

def get_relative_residual(residual, macro_strain, average=np.mean):
    # The strain residual relative to the imposed strain, never divided by less than residual_strain_scale_floor
    # (3_tfa_baseline.md section 12).
    strain_scale = max(get_strain_norm(macro_strain), config.residual_strain_scale_floor)
    return get_strain_norm(residual, average) / strain_scale

def has_converged(residual_history, divergence_limit):
    # Whether the latest relative residual is below convergence_tolerance. Diverging means exceeding divergence_limit
    # times the larger of 1 and the step's first relative residual: at a zero crossing of the cyclic load path, the
    # first residual legitimately exceeds the imposed strain.
    relative_residual, newton_iteration_count = residual_history[-1], len(residual_history) - 1
    if not np.isfinite(relative_residual) or relative_residual > divergence_limit * max(1.0, residual_history[0]):
        raise SolverDidNotConverge(f"diverged after {newton_iteration_count} Newton iterations (relative residual "
                                   f"{relative_residual:.3e})")
    if relative_residual < config.convergence_tolerance:
        return True
    if newton_iteration_count >= config.newton_max_iterations:
        raise SolverDidNotConverge(f"did not converge within {config.newton_max_iterations} Newton iterations "
                                   f"(relative residual {relative_residual:.3e})")
    return False

def solve_load_step(evaluate, get_correction, initial_strain, macro_strain, divergence_limit, average=np.mean,
                    measure_residual=None):
    # Newton from initial_strain: evaluate(strain) returns an Evaluation, get_correction(evaluation) returns the Newton
    # correction there and its GMRES iteration count. Each correction is applied through take_step's line search.
    # average is how the residual norm averages over the unknowns' rows (get_strain_norm). Convergence and the line
    # search judge measure_residual(r) when it is given (the LS methods use it to put r in units that do not depend
    # on their reference medium, LS-24), r itself otherwise; the Newton correction always solves for r.
    def get_measured_residual(evaluation):
        residual = evaluation.residual if measure_residual is None else measure_residual(evaluation.residual)
        return get_relative_residual(residual, macro_strain, average)

    current = evaluate(initial_strain)
    residual_history = [get_measured_residual(current)]
    gmres_iteration_count = 0
    while not has_converged(residual_history, divergence_limit):
        correction, iteration_count = get_correction(current)
        gmres_iteration_count += iteration_count
        current, relative_residual = take_step(evaluate, current, correction, residual_history[-1],
                                               get_measured_residual)
        residual_history.append(relative_residual)
    return StepResult(current.strain, current.stress, current.plastic_state, np.array(residual_history),
                      gmres_iteration_count)

def take_step(evaluate, current, correction, relative_residual, get_measured_residual):
    # Backtracking line search: the full Newton step if it lowers the relative residual enough (by the fraction
    # line_search_sufficient_decrease of the step length), otherwise the step shortened by line_search_step_reduction,
    # at most line_search_max_reductions times; the shortest step is taken if none does. Where the full step
    # works, this is plain Newton.
    step_length = 1.0
    for _ in range(config.line_search_max_reductions + 1):
        trial = evaluate(current.strain + step_length * correction)
        trial_relative_residual = get_measured_residual(trial)
        if trial_relative_residual < (1 - config.line_search_sufficient_decrease * step_length) * relative_residual:
            break
        step_length *= config.line_search_step_reduction
    return trial, trial_relative_residual

def solve_newton_correction(apply_jacobian, residual, restart=None):
    # Solves J δε = -r by GMRES, with J given as its product apply_jacobian on partition fields (partitions × 6),
    # restarting every restart iterations (None: config.newton_krylov_restart). Returns δε and the number of GMRES
    # iterations it took.
    def apply_flat_jacobian(flat_vector):
        return apply_jacobian(flat_vector.reshape(-1, 6)).reshape(-1)

    system_size = residual.size
    jacobian = scipy.sparse.linalg.LinearOperator((system_size, system_size), matvec=apply_flat_jacobian)
    iteration_count = 0

    def count_iteration(_):
        nonlocal iteration_count
        iteration_count += 1

    correction, info = scipy.sparse.linalg.gmres(jacobian, -residual.reshape(-1), rtol=config.newton_krylov_tolerance,
                                                 restart=config.newton_krylov_restart if restart is None else restart,
                                                 maxiter=config.newton_krylov_max_restarts, callback=count_iteration,
                                                 callback_type='pr_norm')
    if info != 0:
        raise SolverDidNotConverge(f"GMRES did not converge inside a Newton iteration (info {info}). Loosen "
                                   "newton_krylov_tolerance or raise newton_krylov_restart")
    return correction.reshape(-1, 6), iteration_count
