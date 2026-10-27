# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase comparing three reduced-order methods for elastoplastic composites; the README explains the
problem, the three grids (elements, partitions, clusters) and the methods. Use the method names exactly, and in this
order everywhere (table, figures, code): **CP**, **Spectral LS**, **TFA**. "fine LS", "coarse LS", "LS Newton" and
"MAIN_*" are retired names. CP and TFA are different models: their difference is modelling error, not a bug.

Read the relevant note before changing solver math. Code names map onto the notes' symbols and numbered equations
(LS-* in `notes/2_ls_model.md`, ER-* in `notes/3_tfa_baseline.md`), which code comments cite. Comments cite equations
and papers, never the history of a decision.

## The user's standing decisions

- **All settings live in `code/config.py`, and only there**: a new knob goes there with a comment, never as a literal
  elsewhere. Exceptions: plot appearance in `plots.py`, check problems in `check.py`. config.py is often left at a
  local study configuration; "default" means `git show HEAD:code/config.py`.
- **One C0 for everything in CP, constant polarization in every partition.** The same C0 builds P0 and splits mixed
  partitions (LS-22, `ls_model.get_cluster_strain`). C0 must stay below every cluster's tangent or the split has no
  solution (`check_reference_below_tangents` stops with that message). Don't reintroduce a separate C0 for the split.
- **Spectral LS's reference stiffness is its own** (`spectral_ls_reference_stiffness`): its answer doesn't depend on
  it, so it is a solver setting, not CP's C0.
- **Fairness between CP and Spectral LS:** each gets its best setting from the same principles, justified from the
  LS literature only (no FE papers), and both stop at the same accuracy (LS-24).
- **Small codebase:** don't add modules, and don't bring back what the 2026-10-26 cut-back removed (git tag
  `pre-cleanup`: fixed-point solvers, FFT-preconditioned TFA, the FE kernel for P0, the coarse LS lattice, extra
  checks, saved results) without the user asking.
- **The user has said they don't really understand LS-24.** When it comes up, explain it from the LS note's
  section 7.5 before building on it.

## Running the code

```bash
python code/run.py           # solve config.py's problem with every enabled method; table + figures to code/output/
python code/check.py         # the test: five small problems, a few seconds
python code/check.py save    # the same, then save the results as the new reference
```

There is no pytest suite and no way to run one check problem alone. `check.py` solves `"laminate"`,
`"cut laminate"`, `"circle"`, `"z laminate"` and `"sphere"` with every method. On the laminates every method must match the
exact solution to `laminate_tolerance`, except CP on the cut laminate (`inexact_methods`: printed as INFO). CP's
mixed partitions must match LS-22's elastic closed form to `mixed_partition_check_tolerance`; that is what catches a
wrong split or split tangent. Every method must complete every step and reproduce `code/check_reference.npz`
(untracked) bit for bit, at the `thread_count` it was saved with. For a refactor: `check.py save` before,
`check.py` after.

Not covered by `check.py`, so recheck by hand after touching them: the Fourier preconditioner must leave answers
alone (compare with it on and off), and every method must stop at the same accuracy (compare against a solve at
tolerance 1e-11).

`check.py` sets its `pinned_inputs` on the `config` module in memory, with a temporary cache folder. This works only
because **no module reads `config` at import time**: no `config.x` in default arguments or module-level constants.
A renamed config input must be renamed in `pinned_inputs` too.

`python` resolves to the untracked `.venv/` (Python 3.12); the `fenics-env`/`fenicsx-env` envs are unrelated.
Entry points import `config` before numpy, since config.py sets the thread environment variables; online FFTs use
`scipy.fft` with `workers=config.thread_count`.

TFA's offline step builds a dense P of (6 × partitions)² doubles (4.8 GB at 64² or 16³ partitions), cached in
`code/cache/` under a hash of its parameters (`fe.get_cache_path`). Bump `fe.cache_version` when a change alters E
or P. Estimate the cost of large configurations and run them in the background.

## Architecture

Imports go one way: `config` → `material` → `newton` → `fe`, `ls_model`, `tfa_model` → `run` → `check`. `plots.py`
imports no project module; only `run.py` calls it.

| Module | Contents |
|---|---|
| `config.py` | every input, `check_inputs()`, folders |
| `run.py` | `Lattice`, `Problem`, `get_problem()` (builds everything, times each method's offline step), `get_solvers`, `run_load_path`, `solve_all`, timing table, figures, `main` |
| `fe.py` | shapes (`is_inside_inclusion`, `shapes_constant_along_z`), `Discretization`, `Mesh`, `Materials`, `Clusters`/`get_clusters`, element matrices, `FEModel`, TFA's E and P (`compute_E_and_P`) and their cache (`get_E_and_P`) |
| `newton.py` | `solve_load_step` (the one Newton loop), `take_step` (line search), relative residual, `has_converged`, `solve_newton_correction` (GMRES), the `LoadPathAbandoned` errors |
| `ls_model.py` | `apply_on_lattice` (FFT convolution), C0, P0 and its partition average (`get_P0_transformed`), `get_fourier_preconditioner`, the mixed-partition split, `LSSolver` for CP and Spectral LS |
| `tfa_model.py` | TFA residual with the elastic reset, column-restricted P products, `get_tfa_solver` |
| `material.py` | vectorized J2 radial return, its sensitivity H_μ and algorithmic stiffness L_t |
| `plots.py` | the figures, `method_styles` keyed by method name |
| `check.py` | the checks and the laminate's exact solution |

`LSSolver` maps onto the LS note: `evaluate` → LS-18, `get_newton_correction` → LS-20 and LS-23, `solve_step` →
section 7.3, `residual_in_matrix_units` → LS-24.

## Non-obvious points

- **Strain convention (ER-2):** engineering shear, `(ε11, ε22, ε33, 2ε12, 2ε23, 2ε13)`.
- **Convergence (LS-24):** the LS residual is a strain of the reference medium, so convergence and the line search
  judge M_m⁻¹ r (the correction an all-matrix cell would need, `residual_in_matrix_units`), while the Newton
  correction still solves for r. `get_relative_residual` weights shear by ½ and divides by the imposed strain's norm,
  never by less than `residual_strain_scale_floor`.
- **No global lattice size online:** solvers, FFTs and the load path take sizes from the arrays they're given, never
  from `config.partition_*` or `config.element_*`. That is what lets CP and Spectral LS share code.
- **Lattices are 3D, indexed `[z, y, x]`.** A z-constant shape has one partition along z, so its LS is the 2D problem.
  `apply_on_lattice` transforms only axes longer than one (length-1 transforms are identities but slow).
- **Half spectrum:** `P0_transformed` holds only `rfftn`'s half (shape `(N_z, N, N//2 + 1, 6, 6)`), transformed back
  with `irfftn(..., s=(N_z, N, N))`. Anything new reading it must do the same.
- **Clusters** are ordered by partition, matrix first, so a mixed partition's inclusion cluster follows its matrix
  cluster. They have unequal volumes: every cell average goes through `Clusters.average`.
- **A new shape** goes in `is_inside_inclusion`, `check_inputs`, the E/P cache parameters (`fe.get_E_and_P`: every
  parameter of the shape, or a stale E and P is reused silently) and, if constant along z, `shapes_constant_along_z`.
- **TFA's P** is stored column-major so consecutive yielding clusters form a contiguous column block
  (`apply_P_to_partitions`). Don't fancy-index `P[:, columns]`: the copy costs more than the full product.
- **E is built only when needed:** by TFA, or by `reference_stiffness = "homogenized"`.
- **Fourier preconditioner (LS-23):** I − P̂0(ξ)⟨D⟩ with ⟨D⟩ the lattice mean of I − C0⁻¹ L_t, rebuilt at every Newton
  correction, applied on the right. On by default for CP, off for Spectral LS (where it changes nothing).
- **Iteration counts:** "Newton its" are corrections, "GMRES its" the inner iterations summed. "Online" is
  `solve_step` wall time; "Offline" is each method's operator build as if it ran alone (cached E and P report their
  stored `build_seconds`).
- **Failed load steps** raise a `LoadPathAbandoned` subclass; that method stops and the others carry on, except on
  the first step, which re-raises.
- **Known differences by design:** LS methods start each step from the previous strain plus the uniform increment;
  TFA starts from the exact elastic predictor, so LS spends Newton iterations on elastic steps.

## Status

The roadmap is in the README's "Status and roadmap"; keep the two in step. Things tried and rejected, not to redo
without the user asking: block-Jacobi preconditioning on top of the Fourier one (CP stopped less accurately),
geometric (interface-aware) split rules for mixed partitions, and single-phase-only partitions. Mixed partitions must
stay O(M log M).

Known behaviour (seen with the FE kernel, since removed): element-level LS rang along high-contrast interfaces, an
element-to-element alternation near the inclusion's corners not reduced by refinement, which a global measure
misses; inspect fields locally.
