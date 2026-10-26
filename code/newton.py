# What every method shares to solve a load step by Newton–GMRES: the relative residual it converges on, the
# convergence test, the GMRES solve of a Newton correction, the result of a step, and the errors that end a method's
# load path early.

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

class Solver(NamedTuple):
    # A method ready to run along the load path: solve_step(macro_strain, partition_materials, plastic_history)
    # returns a StepResult, on the lattice partition_materials describes.
    solve_step: Callable
    partition_materials: tuple

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

def get_strain_norm(strain):
    # The root mean square over the partitions of each partition's tensor norm.
    squared_tensor_norms = np.sum(tensor_norm_weights * np.atleast_2d(strain)**2, axis=1)
    return np.sqrt(np.mean(squared_tensor_norms))

def get_relative_residual(residual, macro_strain):
    # The strain residual relative to the imposed strain, never divided by less than residual_strain_scale_floor
    # (3_tfa_baseline.md section 12).
    strain_scale = max(get_strain_norm(macro_strain), config.residual_strain_scale_floor)
    return get_strain_norm(residual) / strain_scale

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

def solve_newton_correction(apply_jacobian, residual):
    # Solves J δε = -r by GMRES, with J given as its product apply_jacobian on partition fields (partitions × 6).
    # Returns δε and the number of GMRES iterations it took.
    def apply_flat_jacobian(flat_vector):
        return apply_jacobian(flat_vector.reshape(-1, 6)).reshape(-1)

    system_size = residual.size
    jacobian = scipy.sparse.linalg.LinearOperator((system_size, system_size), matvec=apply_flat_jacobian)
    iteration_count = 0

    def count_iteration(_):
        nonlocal iteration_count
        iteration_count += 1

    correction, info = scipy.sparse.linalg.gmres(jacobian, -residual.reshape(-1), rtol=config.newton_krylov_tolerance,
                                                 restart=config.newton_krylov_restart,
                                                 maxiter=config.newton_krylov_max_restarts, callback=count_iteration,
                                                 callback_type='pr_norm')
    if info != 0:
        raise SolverDidNotConverge(f"GMRES did not converge inside a Newton iteration (info {info}). Loosen "
                                   "newton_krylov_tolerance or raise newton_krylov_restart")
    return correction.reshape(-1, 6), iteration_count
