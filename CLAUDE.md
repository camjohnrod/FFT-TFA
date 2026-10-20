# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase for reduced-order homogenization of elastoplastic composites (stiff circular/square inclusions,
or a laminate, in a softer J2-plastic matrix) on uniform partitions. Two reduced models, CP and TFA, differ only in
the operator through which partitions interact; the research question is how that operator decides accuracy and
cost (README, "Research question"; `notes/1_research_context.md`). CP is the model under development; TFA is the
baseline it is compared against. Each is solved by two strategies (fixed point, Newton), one entry point per strategy:

| Entry point | Solvers | Outputs prefix |
|---|---|---|
| `code/MAIN_FP.py` | CP FP; with `include_tfa_model`: TFA Standard FP, TFA FFT FP | `FP_` |
| `code/MAIN_Newton.py` | CP Newton; with `include_non_partitioned_ls`: LS Newton, fine and LS Newton, coarse; with `include_tfa_model`: TFA Newton, TFA FFT Newton | `Newton_` |

- **CP** (constant polarization) is the partition-averaged Lippmann-Schwinger (LS) model of `notes/2_ls_model.md`
  on the main partitions (the notes and the `ls_*` code name the model LS, as do the fine and coarse solvers),
  where the nonlocal operator is the P0 convolution alone. It is always solved.
- **TFA** is the actual E/P model (`notes/3_tfa_baseline.md`, ER-4): residual `r = ε - Eε̄ - Pμ(ε)`. FP is Strategy 1
  (fixed point, optionally preconditioned by the FFT reference `M0 = I - P0 H_μ,0`); Newton is Strategy 2 (Newton
  with GMRES, optionally right-preconditioned by `M0⁻¹`). All four TFA solvers solve the same equation and must agree
  to `solver_agreement_tolerance`. `config.include_tfa_model` (default `True`) adds the two TFA solvers of the
  entry point's strategy as the baseline of its plots and checks.
- CP and TFA are *different* models: their difference is modelling error, printed as such, not a solver bug. Only
  under `matched_stiffness_control` are they the same equation; that is the CP regression gate, and it needs
  `include_tfa_model` and the FE kernel (`reference_kernel = "fe"`; the default is Moulinec–Suquet).
- **Non-partitioned LS** (`config.include_non_partitioned_ls`, default `True`, `MAIN_Newton.py` only, since the LS
  fixed point at fine resolution takes too long): the same LS scheme on two more lattices, as comparisons. "LS
  Newton, fine" has one partition per element column, the finest lattice the mesh allows; "LS Newton, coarse" has
  one element per partition, as cheap as CP but without the fine mesh's information. With the FE kernel
  neither is a reference solution (see "Status"), so the code measures no solver against them; a gold-standard
  reference is roadmap step 3.

Read the relevant note before changing solver math: code names (`E`, `P`, `P0_transformed`, `M0`, and `sensitivity`
for H_μ, …) map onto the notes' symbols and numbered equations (ER-*, LS-*), which code comments cite.

## Running the code

```bash
python code/MAIN_FP.py            # from any directory; run before MAIN_Newton.py, which cross-checks against it
python code/MAIN_Newton.py
python code/check_code.py         # after any change: static checks plus nine smoke runs (~6 minutes)
python code/check_code.py save    # the same, then save the smoke runs' results as a baseline (only if all passed)
python code/check_code.py compare # the same, then require the results bit-identical to that baseline
```

There is no pytest suite and no way to run a single test; `code/check_code.py` is the test to run after changing
code. It fails on anything unused (including `NamedTuple` fields and config constants, but not imports), lines over
120 characters, trailing whitespace, or any entry point that exits with an error or prints a FAIL or SKIPPED check.
Each smoke run works on a temporary copy of `code/` whose `config.py` is shrunk (`smoke_run_overrides`: 9 partitions,
27 elements, 12 steps, 1 repeat, an aligned square, TFA included, no matched stiffness, a yielding matrix (yield
1 MPa, hardening 10 MPa), verification on, and every other problem, solver and verification input pinned to a fixed
value, so a local study configuration cannot change what they cover) and runs `MAIN_FP.py` then
`MAIN_Newton.py`. `smoke_runs` lists the runs, each with a comment saying why it exists: monotonic; cyclic
(`load_path_shape = "cyclic"`, the only test of unloading, reverse yielding and zero imposed strain, where a solver
abandoning the load path also fails, since abandoning prints no FAIL); matched stiffness (the CP regression gate); CP
only; laminate (every solver against the exact solution); circle (one partition per element); the Moulinec–Suquet
kernel on a laminate (exact) and on even grids (its highest-frequency convention); and mixed partitions, a negative
test that must stop with the mixed-partition error (`SmokeRun.expected_error`). It never touches this
folder's config, cache or outputs, so it is safe to run with a local study configuration. Setting those values in
`config.py` and running one entry point by hand is the quick loop for a single solver (restore them afterwards; that
run does write this folder's cache and outputs). The checks test consistency, not results: for a change meant to leave
every result alone (a refactor), run `check_code.py save` before it and `check_code.py compare` after it. The baseline
(`code/cache/check_code_baseline.npz`, gitignored) holds every saved array of every smoke run, so a difference names
the run, entry point, solver and array; a `problem/` difference means `config.py`'s inputs changed since the baseline
was saved.

No package manifest. The dependencies (`numpy`, `scipy`, `matplotlib`, `tqdm`) are installed in the repository's
`.venv/` (Python 3.12, untracked), which is what `python` resolves to in this workspace; `check_code.py` runs its
smoke runs with whichever interpreter launched it (`sys.executable`). The `fenics-env`/`fenicsx-env` envs are
unrelated.

**All configuration is `code/config.py`**, shared by every entry point and grouped by topic: geometry, materials,
reference stiffness, load path (`load_path_shape`: monotonic ramp, or cyclic 0 → +max → −max → 0 in
4 × `strain_increment_count` steps; `config.applied_macro_strain` holds the steps), convergence
(`convergence_tolerance` applies to every solver), `tfa_*`, `ls_*`
and `newton_*` solver settings, verification, timing. There are no CLI arguments. Check `git diff` before
committing, since it is often left at a local study configuration. "Default" in this file means the committed
values (`git show HEAD:code/config.py`); read the working copy before relying on a default-dependent statement.
A run exercises plasticity only if a phase's yield stress is finite (`np.inf` makes it elastic); the smoke runs
always yield.
Keep every input a single-line `name = value` at column 0: the smoke runs set inputs by regex-replacing those lines
(`^name\s*= .*$`), so a value split over several lines breaks the copy's `config.py`, and a renamed input fails with
"config.py has no line setting …" until `smoke_run_overrides`/`smoke_runs` are updated. A new input must be read
somewhere as `config.<name>`, or the unused check fails.

Runs are slow: `main()` does `timing_repeat_count` interleaved full load paths per solver. The CP fixed point needs
tens of thousands of iterations per path (~55,000, ~25 s per path, measured at a 25×25 grid), and `MAIN_Newton.py` with
`include_non_partitioned_ls` also solves LS on the element lattice. With `include_tfa_model`, the offline step builds
TFA's dense P, of (6 × partitions)² doubles: 0.3 GB at 32² partitions, 4.8 GB at 64²; a CP-only run builds only E
(6 load columns) and the P0 kernels. Estimate cost before running and background large configurations.

**Checks every run makes, all printing PASS/FAIL and stopping the run on FAIL:**
- The independent residual check: after every step, outside the timings, `run_strain_path` recomputes the model
  residual from scratch at the returned strain (`get_ls_residual` for CP, `get_actual_residual` for TFA) and
  requires it converged. It catches solver bookkeeping bugs the solver's own residual cannot.
- With non-partitioned LS and `partition_number_per_side = element_number_per_side`: the three LS lattices are one
  lattice, so LS Newton, fine and LS Newton, coarse must agree with CP Newton (the "circle" smoke run).
- With `include_tfa_model`: agreement between the two TFA solvers (`check_solver_agreement`), and, with
  `matched_stiffness_control = True`, CP must reproduce TFA to `matched_stiffness_tolerance`
  (`check_matched_stiffness`), the CP regression gate.
- With `inclusion_shape = "laminate"`: every solver's macroscopic stress and the stress of every partition (each
  phase on its own stress scale, since the inclusion's is about a hundred times the matrix's) must match the exact
  laminate solution (`laminate.py`) to `laminate_tolerance` (`check_laminate_solution`), and verification
  checks the FE homogenized stiffness against the exact laminate stiffness. It is the only check with an exact answer,
  and the only one that tests the plastic path rather than agreement between solvers.
- `MAIN_Newton.py`: cross-checks every solver against the fixed-point result of the same model saved by
  `MAIN_FP.py` (`check_against_saved_results`). SKIPPED, with the reason, when that file is missing, has no result
  for that solver (it ran without TFA), or was saved for a different problem (`config.problem_parameter_names`) or
  `convergence_tolerance`. Run `MAIN_FP.py` first.
- With `verification_enabled`: structural checks on the main, fine and coarse kernels (zero at ξ = 0, D̂0 Hermitian,
  eigenvalues real and in [0, 1]); finite-difference Jacobian checks in `MAIN_Newton.py` (LS-20, and ER-18 with TFA);
  and, only when `partition_count <= verification_max_partitions` (default 256, so **skipped at the default 32×32
  partitions**), the FFT convolution against the dense product, plus the CP elastic stiffness printed against the FE
  one. FE kernel only: the partitioned P0 rebuilt from the fine one (`get_partition_averaged_P0_offset_blocks`, exact by
  linearity) and the dense check against a direct reference solve; the MS kernel is built by that same averaging, and
  refuses to build unless its half spectrum is the transform of a real kernel. Lower `partition_number_per_side` to get them, keeping the inclusion aligned (e.g. 16 partitions with 32
  elements for the default 0.5 mm square).

**Cache and output.** The offline step (factorizing the global stiffness to build `E`, `P`, `P0`) is cached in
`code/cache/` (gitignored), shared by both entry points, one file per array and parameter set named by a hash of its
parameters (`offline.get_cache_path`, `get_cached`). `E` and `P` share one factorization of the stiffness but are
solved separately and cached in separate files, so E is bit-identical whether or not P is built, and a CP-only run
neither builds nor loads P. Each FE `P0` is cached under its discretization and the reference stiffness itself, so
each lattice and each `reference_stiffness` choice gets its own file; the Moulinec–Suquet kernel takes milliseconds and
is not cached. Bump `offline.cache_version` whenever a code change
alters the offline operators, since parameters alone cannot detect that. `code/output/` (gitignored): each run first
deletes its own `<prefix>*` files, then writes `<prefix>stress_strain.png` (every macroscopic stress component
against the applied strain, every solver but TFA FFT; a row of components zero for every solver is left out),
`<prefix>solver_performance.png` (iterations and time per iteration per step, total time per solver),
`<prefix>time_breakdown.png`, `<prefix>von_mises_stress.png` (von Mises stress at the last step every mapped solver
completed, one map per solver on its own lattice, from the finest LS lattice to the coarsest, then TFA) and
`<prefix>results.npz` (per-solver results plus the problem parameters, read by the cross-checks; written last, only
once every check has passed, so a failed run leaves none), plus the shared `cross_section.png`.

## Architecture

The entry points are thin: `main()` reads as an outline, and what both report and check after solving is one call
each, `report.report_results` and `report.check_results`, driven by `SolverRoles` (each solver's role: CP, TFA
baseline, TFA FFT, LS fine, LS coarse; `None` when not run). The pipeline lives in modules under `code/`, imported in
one direction (each arrow points to the modules that may import what comes before it): `config` → `lattice_fft` →
`offline`, `online_timing` → `material` → `convergence` → `load_path` → `tfa_solvers`, `ls_solvers`, `laminate`
(independent of each other) → `verification`, `report`. Every module reads inputs as `config.<name>`, except
`plots.py`, which imports no project module and takes everything as arguments; only `report.py` imports it, passing
the config values in.

| Module | Contents |
|---|---|
| `config.py` | Inputs, derived values, geometry checks, `problem_parameter_names`, folders |
| `offline.py` | `Discretization` (main, fine, coarse), periodic hex mesh, materials, 8-node element matrices, `E`/`P` influence functions, the FE and Moulinec–Suquet `P0` kernels and the partition averaging they share, cache, `get_problem()` |
| `lattice_fft.py` | the partition-lattice FFT: `get_P0_transformed`, `apply_on_partition_lattice` and the inverse |
| `material.py` | vectorized J2 radial return and its sensitivity `H_μ` |
| `online_timing.py` | `timed_online` groups, each charged its own (exclusive) time |
| `convergence.py` | residual norm, `has_converged`, `StepResult`, failure exceptions |
| `load_path.py` | `Solver` (self-contained: its model's operators bound in, plus its partitions' materials), `LoadPathResult`, `run_strain_path` (with the independent check), interleaved repeats |
| `tfa_solvers.py` | TFA fixed-point and Newton-Krylov solvers; `get_tfa_fixed_point_solvers` / `get_tfa_newton_solvers` define the solver names |
| `ls_solvers.py` | LS residual, fixed point and Newton, and the warm-starting `LSSolver` |
| `laminate.py` | exact solution of the laminate geometry: two uniform phase states, Newton on ε11, γ12, γ13 |
| `verification.py` | kernel, dense and Jacobian checks, `run_verification` |
| `report.py` | `SolverRoles`, `report_results`, `check_results`; timing aggregation, agreement, laminate and cross-checks, saved results, plots |
| `plots.py` | stress-strain curves, solver performance, time breakdown, cross-section, von Mises stress maps; one style and short name per solver role (`solver_role_styles`, `solver_role_labels`), shared by every figure |

Non-obvious points:
- **Strain convention (ER-2):** engineering shear, `(ε11, ε22, ε33, 2ε12, 2ε23, 2ε13)`. `get_relative_residual`
  weights shear by ½ to undo the doubling, and divides by the imposed strain's norm but never by less than
  `residual_strain_scale_floor` (`3_tfa_baseline.md` section 12).
- **Why the TFA solvers agree:** all four evaluate the residual through the same `reset_elastic_partitions`
  (material update, closed-form update of elastic partitions, actual residual), from the same starting strain
  `Eε̄ + Pμₙ`. Both FFT variants share `FFTReference`: `H_μ,0` (partition mean of `H_μ`), rebuilt only when the set of
  yielding partitions changes.
- **Induced strain:** within a step, Pμ = Pμₙ + PΔμ with Δμ zero outside the yielding partitions. `P` is stored
  column-major, so each run of consecutive yielding partitions is a contiguous block of columns, and
  `apply_P_to_partitions` multiplies only those blocks without copying. Newton's Jacobian products use the same
  function, since `H_μ` is zero outside the yielding partitions. Don't switch to fancy-indexing `P[:, columns]`:
  copying the columns costs more than the full product.
- **No global lattice size online:** solvers, the FFT and the load path take their sizes from the arrays they are
  given (`reshape(-1, 6)`, the kernel's shape, `len(partition_materials.L)`), never from `config.partition_count`.
  A `Solver` carries everything it needs: TFA binds E and P, CP and LS bind P0, and both carry their partitions' materials.
  This is what lets solvers on different lattices (CP, fine LS, coarse LS) run side by side in one run.
  `check_code.py` enforces it: no module in `online_modules` may read `config.partition_*`, `config.element_*` or
  `config.dof_count`.
- **Discretizations:** everything offline is built for an `offline.Discretization` (elements per side, elements
  along z, partitions per side, and the sizes derived from them), carried by its `Mesh` along with the element B
  matrix, so offline functions take just the mesh. There are three, one function each: the main one from `config`
  (TFA and CP), the fine one (the main mesh, one partition per element column: fine LS) and the
  coarse one (one element per main partition). All three P0 kernels share the one C0 and the one `reference_kernel`;
  the Moulinec–Suquet one is built on each discretization's element grid and averaged over its partitions, so fine
  and coarse LS (one element per partition) get the standard scheme and CP the SCA-style average.
- **Half spectrum:** the kernel and every lattice field are real, so `P0_transformed` and `M0⁻¹` hold only the
  non-redundant half of the frequencies (`rfftn`, shape `(N, N//2 + 1, 6, 6)`), transformed back with
  `irfftn(..., s=(N, N))`. Anything new that reads `P0_transformed` must use this convention.
- **Geometry and partitions:** the geometry is resolved on the elements (each element column's centre is tested,
  `offline.get_element_column_material_ids`), and each partition's material is read off its elements. TFA and
  CP give each partition one material law, so `get_partition_material_ids` stops the run, before any
  offline solve, if a partition holds both phases. A circle therefore runs only with one partition per element
  (`partition_number_per_side = element_number_per_side`); a square or laminate runs whenever its edges fall on
  partition edges. The FFT needs a congruent, translated partition lattice, so there is one z-layer of partitions
  spanning the full thickness.
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
- **Known differences between solvers, by design:** CP starts each step from its previous strain plus the uniform
  increment, while TFA starts from the exact elastic predictor, so CP spends iterations on elastic steps. TFA and CP
  fixed points have different divergence limits and iteration caps (`tfa_divergence_limit`, `tfa_max_iterations`
  against `ls_divergence_limit`, `ls_max_iterations`). Compare solvers by time: a column-restricted P product
  and a P0 convolution cost very different amounts, so work counts are not comparable across models.
- **Naming.** "Reference" always means the homogeneous reference medium C0 and what is built from it: P0
  (`get_reference_kernel`, `get_P0_induced_strain`), C0⁻¹ (`reference_compliance`), and the FFT preconditioner of
  TFA (`FFTReference`, `M0`, `H_μ,0`), as in both notes. The CP and LS models' functions are `ls_*`, mapped to the LS note:

  | Code | LS note |
  |---|---|
  | `get_ls_state` (and `get_ls_residual`, from scratch) | LS-18 residual, with the equivalent eigenstrain μ* of LS-6 |
  | `get_P0_induced_strain` | LS-17, the P0 convolution |
  | `ls_fixed_point_iteration` | LS-19 (damped) |
  | `get_ls_jacobian_blocks`, `get_ls_newton_correction`, `ls_newton_iteration` | LS-20 and the load step of section 7.3 |

  In `report.py` and the entry points, the solver a check compares against is the "baseline" (`tfa_baseline` in
  `SolverRoles`): the plain TFA solver of the same strategy, or none in a CP-only run. Figures draw and name every
  solver by its role, so the same role looks the same in every figure and in both entry points.
- **One reference stiffness C0 for both models.** `config.reference_stiffness` (`"homogenized"` ⟨L E⟩, `"voigt"`
  ⟨L⟩, or `"matrix"`) sets `Problem.reference_L`, from which the single `P0` is built (`offline.get_reference_L`).
  C0 is part of the CP model, so it changes the CP answer; for TFA it only preconditions the FFT solvers, so the
  plain TFA solvers are bit-identical under every choice and the FFT ones agree to solver tolerance. P0 is
  unchanged by scaling C0, so with equal phase Poisson ratios `"voigt"` and `"matrix"` give the same P0 and differ
  only through C0⁻¹ in the CP equivalent eigenstrain.

## Notes directory

The notes are numbered in reading order.

- `notes/1_research_context.md` states the research question, derives the P0 operator options (consistent,
  Moulinec–Suquet, filtered, FE) and what "consistent" means, places LS in the literature (clustered LS, SCA,
  Brisard–Dormieux, composite voxels), and records which papers were read cover to cover. Read it before changing P0
  or framing results.
- `notes/2_ls_model.md` is the LS model, and its LS-* labels are cited in `ls_solvers.py`. Its algebra holds; its
  section 9 lists the open points (reference choice, the plastic branch of H_μ, convergence in partition size, the FE
  kernel being non-consistent).
- `notes/3_tfa_baseline.md` is the TFA derivation (ER-1 … ER-25). Strategy 1 is
  `fixed_point_iteration` (with `P0_transformed`), Strategy 2 is `newton_krylov_iteration`. Sections 9–11 give
  convergence caveats to check before assuming a solver change is an improvement.
- `notes/background_influence_functions_and_J2/` is a slower 7-part backing derivation: the influence functions E
  and P (parts 1–3) and the J2 return map both models use (parts 4–7). Its `README.md` gives the reading order;
  `3_tfa_baseline.md` cites part 7 for the J2 constitutive convention.
- `references/` holds the source PDFs the notes cite, and `references/reading_notes/` one note per paper read cover to
  cover (equations, quotes with page numbers, relevance). `notes/1_research_context.md` lists further papers, read or
  not.

## Status and next steps

The README's "Status and roadmap" section is the user-facing version of this list; keep the two in step.

Finding (FE kernel, 64 elements / 32 partitions square, 2026-10-05): fine LS rings along high-contrast interfaces,
an element-to-element alternation of about ±30 % of the local stress in the inclusion's outer layers near the corners,
not reduced by refining from 32 to 64 elements, which element-level TFA (exact FE elastically) does not show. A global
spectral measure misses it; inspect fields locally.

Roadmap, in order:
1. Thickness check: done (2026-10-12). `element_number_along_z = 1` reproduces `= 3` to round-off.
2. Moulinec–Suquet kernel: done (2026-10-12), `reference_kernel = "moulinec_suquet"` (LS-21, `notes/2_ls_model.md`
   section 6.5); the default since 2026-10-13, with FE kept as an option and the only kernel for the matched-stiffness
   gate. Under MS, TFA keeps its FE operators, so a CP-vs-TFA difference includes the
   FE-vs-MS discretization gap; it is not reported separately (fine LS against TFA on elastic steps, where TFA is the
   exact FE answer, measures it). Never call CP "SCA", which also clusters by k-means and updates C0.
3. Gold-standard reference: a separate FE-consistent FFT solver (Ladecký et al., `references/`; displacement
   unknowns, plasticity at the Gauss points). For its error split, a derivation already checked: with every partition
   a k × k block of equal-volume reference cells and Π the block average, ‖f − g‖² = ‖f − Πg‖² + ‖g − Πg‖² at every
   step for any fixed quadratic norm (total² = scheme² + floor²); it holds per phase only while partitions are
   single-phase, and not for maxima over the load path.
4. TFA–LS study: Schneider (2019, §4) shows formally that TFA and clustered LS differ only in the Green operator (full
   against partition-averaged). Derive the exact relation between TFA's P and the LS operator, and measure TFA and
   every LS kernel on identical partitions against the reference.
5. Cost at equal accuracy: time and memory against resolution for each operator, compared at the same error.

Later: a mixing rule for LS partitions holding both phases (so circles run on coarse partitions; TFA will not get
one; composite voxels and Brisard–Dormieux's rule are the precedents), a reference stiffness that follows plastic
softening (as in SCA), a Newton line search for strong softening, a Fourier preconditioner for LS Newton (its GMRES
is unpreconditioned), and the consistent kernel (exact LS-7, the limit of the averaged MS kernel as the element grid
is refined; its alias series converges as 1/T, and assigning the tail to the axis directions, whose sinc² weights
sum to one in closed form, makes it 1/T², checked in a scratch test on 2026-10-12).
