# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase for reduced-order homogenization of elastoplastic composites (a stiff square, circular or
laminate inclusion in a softer J2-plastic matrix) on uniform partitions. The research question is how the operator
through which partitions interact decides accuracy and cost (README, "Research question";
`notes/1_research_context.md`). Three methods, all solved by Newton–GMRES, in this order everywhere (table, figures,
code):

- **CP**: the partition-averaged Lippmann–Schwinger (LS) equation of `notes/2_ls_model.md` on the partitions, the
  method under development. Its only nonlocal operator is the FFT convolution with the Moulinec–Suquet kernel P0,
  averaged over partitions (LS-21). Always solved.
- **Spectral LS** (`config.include_spectral_ls`): the same LS code on the element lattice (one partition per element
  column), i.e. the standard Moulinec–Suquet spectral method. A comparison, not a reference solution.
- **TFA** (`config.include_tfa_model`): the actual E/P model of `notes/3_tfa_baseline.md` (ER-4,
  `r = ε - Eε̄ - Pμ(ε)`), Strategy 2 (Newton–GMRES, unpreconditioned), the baseline.

CP and TFA are *different* models: their difference is modelling error, not a solver bug. Use these method names
exactly; "fine LS", "coarse LS", "LS Newton" and "MAIN_*" are retired names.

The code was cut back on 2026-10-26 (git tag `pre-cleanup` has what was removed: fixed-point solvers, FFT-preconditioned
TFA, the FE kernel for P0 and with it the matched-stiffness gate, the coarse LS lattice, verification checks, the
timing breakdown, saved results and cross-checks between entry points, `check_code.py`). Don't reintroduce any of it
without the user asking.

Read the relevant note before changing solver math: code names (`E`, `P`, `P0_transformed`, `sensitivity` for H_μ, …)
map onto the notes' symbols and numbered equations (ER-*, LS-*), which code comments cite.

## Running the code

```bash
python code/run.py           # solve config.py's problem with every enabled method; table + figures to code/output/
python code/check.py         # the test: two small problems, a few seconds
python code/check.py save    # the same, then save the results as the reference (only if all passed)
```

There is no pytest suite. `check.py` solves a cyclic laminate (every method must match the exact solution in every
partition, each phase on its own stress scale, to `laminate_tolerance`) and a square inclusion; on both every method
must complete every step and, once `code/cache/check_reference.npz` exists, reproduce it bit for bit. For a change
meant to leave results alone (a refactor), run `check.py save` before and `check.py` after. `check.py` pins every
problem and solver input in `pinned_inputs` and sets them on the `config` module in memory (then calls
`config.check_inputs()`), with a temporary cache folder, so it never touches the user's config, cache or outputs.
This only works because no module reads `config` at import time: keep it that way (no `config.x` in default
arguments or module-level constants). A renamed config input must be renamed in `pinned_inputs` too. The reference
is bit-for-bit only at the `thread_count` it was saved with, since threaded BLAS may sum in a different order.

Dependencies (`numpy`, `scipy`, `matplotlib`, `tqdm`) are in the repository's untracked `.venv/` (Python 3.12), which
is what `python` resolves to here. The `fenics-env`/`fenicsx-env` envs are unrelated.

**All settings live in `code/config.py`, and only there** (the user's explicit rule): problem, materials, load path,
methods, C0, every solver tolerance and limit, offline batch size, figure limits, check tolerances. A new knob goes in
config.py with a comment, never as a literal in another module. The deliberate exceptions: plot appearance (sizes,
colours) in `plots.py`, and check.py's own problem definitions. "Default" means the committed values
(`git show HEAD:code/config.py`); check `git diff` before committing, since config.py is often left at a local study
configuration. A run exercises plasticity only if a phase's yield stress is finite.

**Threads:** `config.thread_count` (default 1) caps every method alike: config.py sets the BLAS/OpenMP environment
variables before it imports numpy, and the online FFTs use `scipy.fft` with `workers=config.thread_count` (numpy's
FFT cannot thread). So `run.py` and `check.py` import `config` before numpy; keep any new entry point that way.
Offline kernel construction still uses `np.fft` (milliseconds, once).

Cost: with TFA, the offline step builds the dense P of (6 × partitions)² doubles (0.3 GB at 32² partitions, 4.8 GB
at 64²), cached in `code/cache/` under a hash of its parameters (`fe.get_cache_path`); bump `fe.cache_version` when a
code change alters E or P. Each cache file also stores `build_seconds`, the time it took to compute, so a cached run
still reports TFA's true offline cost; a file without it is recomputed. Estimate cost before running large
configurations and run them in the background.

## Architecture

Imports go one way: `config` → `material` → `newton` → `fe`, `ls_model`, `tfa_model` → `run` → `check`. `plots.py`
imports no project module; only `run.py` calls it.

| Module | Contents |
|---|---|
| `config.py` | every input, `check_inputs()`, folders |
| `run.py` | `get_applied_macro_strain`, `get_solvers` (method name → `Solver`), `run_load_path`, `solve_all`, timing table, figures, `main` |
| `fe.py` | `Discretization` (partition and spectral), periodic mesh, materials, 8-node element matrices, E and P, C0, the Moulinec–Suquet P0 and its partition average, cache, `get_problem()` → `Problem` |
| `newton.py` | `StepResult`, `Solver`, the `LoadPathAbandoned` errors, relative residual, `has_converged`, `solve_newton_correction` (GMRES with iteration count) |
| `ls_model.py` | LS residual, Jacobian and Newton step, the warm-starting `LSSolver`, `get_ls_solver(lattice, …)` for CP and Spectral LS |
| `tfa_model.py` | TFA residual with the elastic reset, column-restricted P products, Newton step, `get_tfa_solver` |
| `material.py` | vectorized J2 radial return and its sensitivity H_μ, von Mises stress |
| `plots.py` | stress–strain curves, cross-section, von Mises maps, `method_styles` keyed by method name |
| `check.py` | the checks, and the laminate's exact solution (Newton on ε11, γ12, γ13 of the matrix) |

Non-obvious points:
- **Strain convention (ER-2):** engineering shear, `(ε11, ε22, ε33, 2ε12, 2ε23, 2ε13)`. `get_relative_residual`
  weights shear by ½, and divides by the imposed strain's norm but never by less than `residual_strain_scale_floor`.
- **No global lattice size online:** solvers, the FFT and the load path take sizes from the arrays they are given
  (`reshape(-1, 6)`, the kernel's shape, `len(partition_materials.L)`), never from `config.partition_*` or
  `config.element_*`; that is what lets CP and Spectral LS run on different lattices with the same code.
- **Half spectrum:** P0 and every lattice field are real, so `P0_transformed` holds only `rfftn`'s half
  (shape `(N, N//2 + 1, 6, 6)`), transformed back with `irfftn(..., s=(N, N))`. Anything new reading it must too.
- **Induced strain (TFA):** within a step Pμ = Pμₙ + PΔμ, Δμ zero outside yielding partitions. P is stored
  column-major so each run of consecutive yielding partitions is a contiguous column block (`apply_P_to_partitions`);
  Newton's Jacobian products use the same function. Don't fancy-index `P[:, columns]`: the copy costs more than the
  full product.
- **Geometry and partitions:** geometry is resolved on element-column centres; `get_partition_material_ids` stops the
  run before any offline solve if a partition holds both phases. One z-layer of partitions spans the thickness.
- **E is built only when needed:** by TFA, or by `reference_stiffness = "homogenized"`. A CP/Spectral LS run with
  `"matrix"` or `"voigt"` C0 does no FE solve at all. C0 changes the LS answer; TFA doesn't use it.
- **Iterations:** "Newton its" are Newton corrections (residual evaluations − 1), "GMRES its" the inner GMRES
  iterations summed over them (scipy callback, `callback_type='pr_norm'`). "Online" is `solve_step` wall time only;
  "Offline" (`Problem.offline_seconds`) is what each method's operators cost to build as if it ran alone: the
  partition mesh and materials for all three, plus C0 and P0 for CP and Spectral LS (C0 includes E's build time
  under `"homogenized"`), plus E and P for TFA at the time they took to compute.
- **Failed load steps:** an unsolvable step raises a `LoadPathAbandoned` subclass (`SolverDidNotConverge`, or
  `MaterialFullySoftened` when a flow stress reaches zero); `run_load_path` stops that method there and the others
  carry on; a failure on the first step re-raises. Divergence means exceeding the limit times
  max(1, the step's first relative residual), since at a cyclic zero crossing the first residual legitimately exceeds
  the imposed strain.
- **Known differences by design:** LS methods start each step from the previous strain plus the uniform increment;
  TFA starts from the exact elastic predictor Eε̄ + Pμₙ (and resets elastic partitions exactly), so LS spends Newton
  iterations on elastic steps. LS and TFA have separate divergence limits. Compare methods by time: a
  column-restricted P product and a P0 convolution cost very different amounts.
- **Naming.** "Reference" means the homogeneous reference medium C0 and what is built from it (P0,
  `reference_compliance`). LS functions map to the LS note: `get_ls_state` → LS-18 with μ* of LS-6;
  `get_P0_induced_strain` → LS-17; `get_ls_jacobian_blocks`, `get_ls_newton_correction`, `ls_newton_iteration` →
  LS-20 and section 7.3. `newton_krylov_iteration` (TFA) is cited by name in `3_tfa_baseline.md`.

## Notes directory

The notes are numbered in reading order.

- `notes/1_research_context.md`: the research question, the P0 operator options (consistent, Moulinec–Suquet,
  filtered, FE), LS in the literature (clustered LS, SCA, Brisard–Dormieux, composite voxels), which papers were read
  cover to cover. Read it before changing P0 or framing results.
- `notes/2_ls_model.md`: the LS model (LS-* labels, cited in `ls_model.py`); section 9 lists its open points.
- `notes/3_tfa_baseline.md`: the TFA derivation (ER-1 … ER-25), both strategies and their convergence caveats
  (sections 9–11); the code now implements only unpreconditioned Strategy 2.
- `notes/background_influence_functions_and_J2/`: a 7-part derivation of E, P (parts 1–3) and the J2 return map
  (parts 4–7); its `README.md` gives the reading order.
- `references/` holds the cited PDFs; `references/reading_notes/` one note per paper read cover to cover.

## Status and next steps

The README's "Status and roadmap" section is the user-facing version of this list; keep the two in step.

Finding (FE kernel, since removed, 64 elements / 32 partitions square, 2026-10-05): element-level LS rang along
high-contrast interfaces, an element-to-element alternation of about ±30 % of the local stress near the inclusion's
corners, not reduced by refinement and absent from element-level TFA. A global spectral measure misses it; inspect
fields locally.

Roadmap, in order:
1. Thickness check: done (2026-10-12). `element_number_along_z = 1` reproduces `= 3` to round-off.
2. Moulinec–Suquet kernel: done (2026-10-12), LS-21, `notes/2_ls_model.md` section 6.5; the only kernel since
   2026-10-26. TFA keeps its FE operators, so a CP-vs-TFA difference includes the FE-vs-MS discretization gap. Never
   call CP "SCA", which also clusters by k-means and updates C0.
3. Gold-standard reference: a separate FE-consistent FFT solver (Ladecký et al., `references/`; displacement
   unknowns, plasticity at the Gauss points). For its error split, a derivation already checked: with every partition
   a k × k block of equal-volume reference cells and Π the block average, ‖f − g‖² = ‖f − Πg‖² + ‖g − Πg‖² at every
   step for any fixed quadratic norm (total² = scheme² + floor²); it holds per phase only while partitions are
   single-phase, and not for maxima over the load path.
4. TFA–LS study: Schneider (2019, §4) shows formally that TFA and clustered LS differ only in the Green operator (full
   against partition-averaged). Derive the exact relation between TFA's P and the LS operator, and measure TFA, CP and
   Spectral LS on identical partitions against the reference.
5. Cost at equal accuracy: time and memory against resolution for each method, compared at the same error.

Later: a mixing rule for partitions holding both phases (so circles run on coarse partitions; TFA will not get one;
composite voxels and Brisard–Dormieux's rule are the precedents), a reference stiffness that follows plastic softening
(as in SCA), a Newton line search (CP's Newton fails at step 1 on a square whose inclusion can yield at 20 MPa, though
the converged inclusion stays elastic, probably because the uniform-strain start pushes the stiff inclusion past
yield; not yet confirmed), a Fourier
preconditioner for the LS Newton solver of CP and Spectral LS (its GMRES is
unpreconditioned), and the consistent kernel (exact LS-7, the limit of the
averaged MS kernel as the element grid is refined; its alias series converges as 1/T, and assigning the tail to the
axis directions, whose sinc² weights sum to one in closed form, makes it 1/T², checked in a scratch test on
2026-10-12).
