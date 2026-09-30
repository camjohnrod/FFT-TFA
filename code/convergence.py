# The residual norm and convergence test every solver uses, and the ways a load step can fail.

import numpy as np

import config

class LoadPathAbandoned(RuntimeError):
    # A load step that cannot be solved. run_strain_path catches only this type and keeps the steps before it, so
    # genuine bugs, such as the online timer's nesting guard or a failed conjugate gradient in post-processing,
    # still stop the run.
    pass

class SolverDidNotConverge(LoadPathAbandoned):
    pass

class MaterialFullySoftened(LoadPathAbandoned):
    pass

# Weights that turn the engineering shear convention of ER-2 into the tensor norm ε:ε, since γ = 2ε for shear.
tensor_norm_weights = np.array([1, 1, 1, 0.5, 0.5, 0.5])

def get_strain_norm(strain):
    # Tensor norm of one strain, or the root-mean-square over partitions of a partition-stacked strain. Every
    # partition has the same volume, so the mean is also the volume average.
    squared_tensor_norms = np.sum(tensor_norm_weights * np.atleast_2d(strain)**2, axis=1)
    return np.sqrt(np.mean(squared_tensor_norms))

def get_relative_residual(residual, macro_strain):
    # Scaled by the imposed strain rather than the current iterate, and never by less than a fixed floor, so the
    # criterion stays meaningful when a load path passes through zero strain (Formulation.md section 12).
    strain_scale = max(get_strain_norm(macro_strain), config.residual_strain_scale_floor)
    return get_strain_norm(residual) / strain_scale

def has_converged(relative_residual, iteration_count, solver_name, divergence_limit=config.divergence_residual_limit,
                  max_iterations=config.fixed_point_max_iterations):
    if not np.isfinite(relative_residual) or relative_residual > divergence_limit:
        raise SolverDidNotConverge(f"{solver_name} diverged after {iteration_count} iterations "
                                   f"(relative residual {relative_residual})")
    if relative_residual < config.fixed_point_tolerance:
        return True
    if iteration_count >= max_iterations:
        raise SolverDidNotConverge(f"{solver_name} did not converge within {max_iterations} iterations "
                                   f"(relative residual {relative_residual})")
    return False
