# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase for FFT-accelerated Transformation Field Analysis (TFA): a reduced-order homogenization
method for elastoplastic composites (stiff circular/square inclusions in a softer J2-plastic matrix). Two reduced
models (LS, TFA) are each solved by two strategies (fixed point, Newton). LS is the model under development; TFA is
the baseline it is compared against. There is one entry point per strategy:

| Entry point | Solvers (TFA ones only with `include_tfa_model`) | Outputs prefix |
|---|---|---|
| `code/MAIN_FP.py` | LS FP; TFA Standard FP, TFA FFT FP | `FP_` |
| `code/MAIN_Newton.py` | LS Newton; TFA Newton, TFA FFT Newton | `Newton_` |

- **LS** is the partition-averaged Lippmann-Schwinger model of `notes/Partition_Lippmann_Schwinger_New_9_28_2026`,
  where the nonlocal operator is the P0 convolution alone. It is always solved.
- **TFA** is the actual E/P model (`notes/Formulation.md`, ER-4): residual `r = ε - Eε̄ - Pμ(ε)`. FP is Strategy 1
  (fixed point, optionally preconditioned by the FFT reference `M0 = I - P0 H_μ,0`); Newton is Strategy 2 (Newton
  with GMRES, optionally right-preconditioned by `M0⁻¹`). All four TFA solvers solve the same equation and must agree
  to `solver_agreement_tolerance`. `config.include_tfa_model` (default `True`) adds the two TFA solvers of the
  entry point's strategy as the baseline of its plots and checks.
- LS and TFA are *different* models: their difference is modelling error, printed as such, not a solver bug. Only
  under `matched_stiffness_control` are they the same equation; that is the LS regression gate, and it needs
  `include_tfa_model`.

Read the relevant note before changing solver math: code names (`E`, `P`, `P0_transformed`, `M0`, and `sensitivity`
for H_μ, …) map onto the notes' symbols and numbered equations (ER-*, LS-*), which code comments cite.

## Running the code

```bash
python code/MAIN_FP.py            # from any directory; run before MAIN_Newton.py, which cross-checks against it
python code/MAIN_Newton.py
python code/check_code.py         # after any change: static checks plus four smoke runs of both (~3 minutes)
```

There is no pytest suite and no way to run a single test; `code/check_code.py` is the test to run after changing
code. It fails on anything unused (including `NamedTuple` fields and config constants, but not imports), lines over
120 characters, trailing whitespace, or any entry point that exits with an error or prints a FAIL or SKIPPED check.
Each smoke run works on a temporary copy of `code/` whose `config.py` is shrunk (`smoke_run_overrides`: 9
partitions, 27 elements, 12 steps, 1 repeat, TFA included, no matched stiffness) and runs `MAIN_FP.py` then
`MAIN_Newton.py`. `smoke_runs` does this four times: monotonic; cyclic (`load_path_shape = "cyclic"`, the only test
of unloading, reverse yielding and zero imposed strain, where a solver abandoning the load path also fails, since
abandoning prints no FAIL); matched stiffness (the LS regression gate); and LS only. It never touches this folder's
config, cache or outputs, so it is safe to run with a local study configuration. Setting those values in
`config.py` and running one entry point by hand is the quick loop for a single solver (restore them afterwards;
that run does write this folder's cache and outputs). It checks consistency, not results: compare numbers against a
baseline by hand when a change is meant to leave them alone.

No package manifest or virtualenv. The default `python` (miniforge `base` conda env) has the
dependencies (`numpy`, `scipy`, `matplotlib`, `tqdm`); the `fenics-env`/`fenicsx-env` envs are unrelated.

**All configuration is `code/config.py`**, shared by every entry point and grouped by topic: geometry, materials,
reference stiffness, load path (`load_path_shape`: monotonic ramp, or cyclic 0 → +max → −max → 0 in
4 × `strain_increment_count` steps; `config.applied_macro_strain` holds the steps), convergence
(`convergence_tolerance` applies to every solver), `tfa_*`, `ls_*`
and `newton_*` solver settings, verification, timing. There are no CLI arguments. Check `git diff` before
committing, since it is often left at a local study configuration. "Default" in this file means the committed
values (`git show HEAD:code/config.py`); read the working copy before relying on a default-dependent statement.

Runs are slow: `main()` does `timing_repeat_count` interleaved full load paths per solver. LS FP needs ~55,000
iterations per path at the default 25×25 grid (~25 s per path). Estimate cost before running and background large
configurations.

**Checks every run makes, all printing PASS/FAIL and stopping the run on FAIL:**
- The independent residual check: after every step, outside the timings, `run_strain_path` recomputes the model
  residual from scratch at the returned strain (`get_ls_residual` for LS, `get_actual_residual` for TFA) and
  requires it converged. It catches solver bookkeeping bugs the solver's own residual cannot.
- With `include_tfa_model`: agreement between the two TFA solvers (`check_solver_agreement`), and, with
  `matched_stiffness_control = True`, LS must reproduce TFA to `matched_stiffness_tolerance`
  (`check_matched_stiffness`), the LS regression gate.
- `MAIN_Newton.py`: cross-checks every solver against the fixed-point result of the same model saved by
  `MAIN_FP.py` (`check_against_saved_results`). SKIPPED, with the reason, when that file is missing, has no result
  for that solver (it ran without TFA), or was saved for a different problem (`config.problem_parameter_names`) or
  `convergence_tolerance`. Run `MAIN_FP.py` first.
- With `verification_enabled`: reference-kernel checks, finite-difference Jacobian checks in `MAIN_Newton.py`
  (LS-20, and ER-18 with TFA), and, only when `partition_count <= verification_max_partitions` (default 256, so
  **skipped at the default 25×25 grid**), dense checks against a direct reference solve plus the LS elastic model
  error. Lower `partition_number_per_side` (e.g. 15 with 45 elements) to get them.

**Cache and output.** The offline step (factorizing the global stiffness to build `E`, `P`, `P0`) is cached in
`code/cache/` (gitignored), shared by both entry points, one file per parameter set named by a hash of its
parameters (`offline.get_cache_path`). `P0` is cached under the reference stiffness itself, so each
`reference_stiffness` choice gets its own file. Bump `offline.cache_version` whenever a code change alters the offline
operators, since parameters alone cannot detect that. `code/output/` is **tracked**: each run first deletes its own
`<prefix>*` files, then writes `<prefix>load_path_summary.png`, `<prefix>time_breakdown.png`,
`<prefix>von_mises_stress.png` (partition von Mises stress at the last completed step: LS, and with TFA also the
TFA FFT solver and LS − TFA FFT) and `<prefix>results.npz` (per-solver results plus the problem parameters, read by
the cross-checks; written last, only once every check has passed, so a failed run leaves none), plus the shared
`cross_section.png`. Any run therefore shows up in `git status`.

## Architecture

The entry points are thin (`main()` reads as an outline); the pipeline lives in modules under `code/`, imported in
one direction (each arrow points to the modules that may import what comes before it): `config` → `lattice_fft`,
`offline`, `online_timing` → `material` → `convergence` → `load_path` → `tfa_solvers`, `ls_solvers` (independent of
each other) → `verification`, `report`. Every module reads inputs as `config.<name>`, except `plots.py`, which
imports no project module and takes everything as arguments; only `report.py` imports it, passing the config values
in.

| Module | Contents |
|---|---|
| `config.py` | Inputs, derived values, geometry checks, `problem_parameter_names`, folders |
| `offline.py` | periodic hex mesh, materials, 8-node element matrices, `E`/`P`/`P0` influence functions, cache, `get_problem()` |
| `lattice_fft.py` | the partition-lattice FFT: `get_P0_transformed`, `apply_on_partition_lattice` and the inverse |
| `material.py` | vectorized J2 radial return and its sensitivity `H_μ` |
| `online_timing.py` | `timed_online` groups, each charged its own (exclusive) time |
| `convergence.py` | residual norm, `has_converged`, `StepResult`, failure exceptions |
| `load_path.py` | `Solver`, `LoadPathResult`, `run_strain_path` (with the independent check), interleaved repeats |
| `tfa_solvers.py` | TFA fixed-point and Newton-Krylov solvers; `get_tfa_fixed_point_solvers` / `get_tfa_newton_solvers` define the solver names |
| `ls_solvers.py` | LS residual, fixed point and Newton, and the warm-starting `LSSolver` |
| `verification.py` | kernel, dense and Jacobian checks, `run_verification` |
| `report.py` | timing aggregation, printed comparison, agreement and cross-checks, saved results, plot saving |
| `plots.py` | load-path summary and time breakdown for up to four solvers, cross-section, von Mises stress maps |

Non-obvious points:
- **Strain convention (ER-2):** engineering shear, `(ε11, ε22, ε33, 2ε12, 2ε23, 2ε13)`. `get_relative_residual`
  weights shear by ½ to undo the doubling, and divides by the imposed strain's norm but never by less than
  `residual_strain_scale_floor` (`Formulation.md` section 12).
- **Why the TFA solvers agree:** all four evaluate the residual through the same `reset_elastic_partitions`
  (material update, closed-form update of elastic partitions, actual residual), from the same starting strain
  `Eε̄ + Pμₙ`. Both FFT variants share `FFTReference`: `H_μ,0` (partition mean of `H_μ`), rebuilt only when the set of
  yielding partitions changes.
- **Induced strain:** within a step, Pμ = Pμₙ + PΔμ with Δμ zero outside the yielding partitions. `P` is stored
  column-major, so each run of consecutive yielding partitions is a contiguous block of columns, and
  `apply_P_to_partitions` multiplies only those blocks without copying. Newton's Jacobian products use the same
  function, since `H_μ` is zero outside the yielding partitions. Don't switch to fancy-indexing `P[:, columns]`:
  copying the columns costs more than the full product.
- **Half spectrum:** the kernel and every lattice field are real, so `P0_transformed` and `M0⁻¹` hold only the
  non-redundant half of the frequencies (`rfftn`, shape `(N, N//2 + 1, 6, 6)`), transformed back with
  `irfftn(..., s=(N, N))`. Anything new that reads `P0_transformed` must use this convention.
- **Lattice requirement:** the FFT needs a congruent, translated partition lattice, so there is one z-layer of
  partitions spanning the full thickness. The geometry checks also require a circular inclusion to contain a
  partition centre, and a square one to land exactly on partition boundaries.
- **Newton:** GMRES is *right*-preconditioned (solve `J M0⁻¹ y = -r`, then `δε = M0⁻¹ y`), so its residual is the
  true linear residual. Newton settings (`newton_*`) are shared by every Newton solver.
- **Iterations** count residual evaluations for every solver: a fixed point's corrections plus one, a Newton
  solver's Newton steps plus one, each Newton step including its whole GMRES solve. Time per iteration is
  therefore not comparable between fixed point and Newton; total time is.
- **Timing:** `timed_online` groups may nest, and each group is charged only its own (exclusive) time. So the P and
  P0 products inside a GMRES solve (`apply_P_to_partitions`, `get_P0_induced_strain`, both timed as induced
  strain) count as induced strain, the rest of the solve as correction solve, and nothing is counted twice. To time
  new work, decorate the function that does it; "Other" is whatever no group covers. Every check runs after a
  step's timings are recorded. Reports take the fastest repeat per step, over the steps every solver
  completed.
- **Failed load steps:** an unsolvable step raises a `LoadPathAbandoned` subclass (`SolverDidNotConverge`, or
  `MaterialFullySoftened` when a flow stress reaches zero and the return map would flip the stress sign).
  `run_strain_path` catches only these, so real bugs still stop the run; a failure on the first step re-raises.
  Under softening, a solver stopping early is not by itself a regression. `has_converged` calls a solver diverged
  when its relative residual exceeds the divergence limit times max(1, the step's first relative residual), not the
  limit alone: at a zero crossing of the cyclic path the first residual legitimately exceeds the imposed strain.
  `config.py` rejects a hardening modulus at or below −3G for any phase that can yield (no admissible return map).
- **Known differences between solvers, by design:** LS starts each step from its previous strain plus the uniform
  increment, while TFA starts from the exact elastic predictor, so LS spends iterations on elastic steps. TFA and LS
  fixed points have different divergence limits and iteration caps (`tfa_divergence_limit`, `tfa_max_iterations`
  against `ls_divergence_limit`, `ls_max_iterations`). Compare solvers by time: a column-restricted P product
  and a P0 convolution cost very different amounts, so work counts are not comparable across models.
- **Naming.** "Reference" always means the homogeneous reference medium C0 and what is built from it: P0
  (`get_reference_kernel`, `get_P0_induced_strain`), C0⁻¹ (`reference_compliance`), and the FFT preconditioner of
  TFA (`FFTReference`, `M0`, `H_μ,0`), as in both notes. The LS model's functions are `ls_*`, mapped to the LS note:

  | Code | LS note |
  |---|---|
  | `get_ls_state` (and `get_ls_residual`, from scratch) | LS-18 residual, with the equivalent eigenstrain μ* of LS-6 |
  | `get_P0_induced_strain` | LS-17, the P0 convolution |
  | `ls_fixed_point_iteration` | LS-19 (damped) |
  | `get_ls_jacobian_blocks`, `get_ls_newton_correction`, `ls_newton_iteration` | LS-20 and the load step of section 7.3 |

  In `report.py`, `plots.py` and the entry points, the solver a plot or check compares against is the "baseline"
  (`baseline_solver_name`): the plain TFA solver of the same strategy, or none in an LS-only run.
- **One reference stiffness C0 for both models.** `config.reference_stiffness` (`"homogenized"` ⟨L E⟩, `"voigt"`
  ⟨L⟩, or `"matrix"`) sets `Problem.reference_L`, from which the single `P0` is built (`offline.get_reference_L`).
  C0 is part of the LS model, so it changes the LS answer; for TFA it only preconditions the FFT solvers, so the
  plain TFA solvers are bit-identical under every choice and the FFT ones agree to solver tolerance. P0 is
  unchanged by scaling C0, so with equal phase Poisson ratios `"voigt"` and `"matrix"` give the same P0 and differ
  only through C0⁻¹ in the LS equivalent eigenstrain.

Measured at the default configuration (2026-09-30), useful before trying to speed things up: P products dominate
every TFA solver. TFA Newton needs ~5× fewer iterations than TFA FP but about as many P products, so it is not
faster (TFA FFT FP is the fastest TFA solver). The FFT reference does not reduce GMRES work for TFA Newton. With
strong softening (`matrix_hardening_modulus = -6e7`) every TFA solver stops at step 8; Newton has no line search.

## Notes directory

- `notes/Formulation.md` is the primary derivation (ER-1 … ER-25). Strategy 1 is
  `fixed_point_iteration` (with `P0_transformed`), Strategy 2 is `newton_krylov_iteration`. Sections 9–11 give
  convergence caveats to check before assuming a solver change is an improvement.
- `notes/verified_notes/` is a slower 7-part backing derivation; its `README.md` gives the reading order.
  `Formulation.md` cites part 7 for the J2 constitutive convention.
- `notes/Partition_Lippmann_Schwinger_New_9_28_2026` (no extension) is the LS model, and its LS-* labels are cited
  in `ls_solvers.py`. Its algebra holds. Three caveats are not yet written into the note:
  - Section 7.1's midpoint-Lamé reference stiffness measured worst of five candidates (55 % plastic error). It is
    a convergence heuristic for the linear elastic scheme, not a model choice.
  - The plastic branch of H_μ is never stated. The code derives it and checks it against a finite difference.
  - Section 5.4 leaves M-convergence open. Measured convergence is first order in partition size.
- `references/` holds the source PDFs the notes cite.
