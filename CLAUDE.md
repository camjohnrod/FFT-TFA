# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase for FFT-accelerated Transformation Field Analysis (TFA): a reduced-order homogenization
method for elastoplastic composites (e.g. stiff circular/square inclusions in a softer J2-plastic matrix).
There are two entry points. `code/MAIN.py` is the stable pipeline and compares two online solvers for the same
partition-strain formulation:

- **Standard fixed-point (Richardson) iteration** on the actual eigenstrain-influence relation.
- **FFT-preconditioned Richardson iteration**, which uses a homogeneous reference material's periodic Green's
  function (applied via FFT over the partition lattice) to accelerate/precondition the same fixed point.

Both solvers must converge to the same macroscopic stress. `check_solver_agreement` prints PASS/FAIL against
`solver_agreement_tolerance` and raises on FAIL, after all plots are saved. `main()` also compares solve time and
iteration count between the two.

A load step that cannot be solved no longer aborts the run. It raises a subclass of `LoadPathAbandoned`:
- `SolverDidNotConverge`, raised by `has_converged` on divergence or the iteration cap, or by
  `get_reference_fourier_inverse` when `M0` is singular.
- `MaterialFullySoftened`, raised by `check_flow_stress_is_positive` when a converged step leaves a partition's flow
  stress `σ_y + H·α` at or below zero. Past that point the return map flips the sign of the stress, and both solvers
  would otherwise agree on the wrong answer.

`run_strain_path` catches only `LoadPathAbandoned`, so genuine bugs (the online timer's nesting guard, a failed
conjugate gradient in post-processing) still fail loudly. It keeps the steps before the failure and records
`completed_step_count` and `failure_reason` on the `LoadPathResult`. Failing on the very first step re-raises,
since there is no partial result worth keeping. Downstream, timings and the agreement check cover only the steps
every solver completed, a stopped solver's label says where and why it stopped, and the load-path plot slices each
curve to its completed steps and marks the abandoned step with an x. The stress curve and von Mises plots come
from whichever solver got furthest (`get_most_complete_solver_name`). This matters mainly under softening
(negative `matrix_hardening_modulus`), where a solver stopping early is not by itself a regression.

`MAIN_Testing.py` shares all of this. Its reference solver also raises `SolverDidNotConverge` when GMRES fails
inside a Newton step. The von Mises model-difference plot is skipped when any solver stopped early, since final
stresses from different load steps are not comparable.

`code/MAIN_Testing.py` is a working copy of that pipeline which adds a **third solver for a different reduced
model** — the partition-averaged Lippmann-Schwinger formulation of
`notes/Partition_Lippmann_Schwinger_New_9_28_2026`, where a partitionwise-constant polarization interacts
through a homogeneous reference medium, so the whole nonlocal operator is a convolution applied by FFT over the
partition lattice. It is selectable as fixed-point or matrix-free Newton-GMRES via `reference_solver_method`.
Because it solves a *different* model, it does **not** agree with the other two to solver tolerance, and the
difference it reports is modelling error rather than a solver bug — the printed output labels these separately.
The one configuration where the models coincide exactly is matched phase stiffness
(`matched_stiffness_control`), which is the regression gate for that solver.

The theory is derived step-by-step in `notes/` (see below) — read the relevant note before changing solver
math, since the code's variable names (`E`, `P`, `P0_transformed`, `H_mu`, `M0`, etc.) map directly onto the
symbols defined there.

## Running the code

```bash
python code/MAIN.py            # stable pipeline, two solvers
python code/MAIN_Testing.py    # adds the reference Lippmann-Schwinger solver and its verification checks
python code/experiments/<name>.py   # runnable from any cwd; reads config from MAIN_Testing.py's constants
```

There is no package manifest (no `requirements.txt`/`pyproject.toml`), no pytest suite, and no virtualenv in
the repo. The default `python` (miniforge `base` conda env) already has the four dependencies: `numpy`,
`scipy`, `matplotlib`, `tqdm`. The `fenics-env`/`fenicsx-env` conda envs are unrelated to this project.

Configuration is changed by editing the module-level constants at the top of `MAIN.py` / `MAIN_Testing.py`
(there are no CLI arguments). The experiment scripts have no config of their own; edit `MAIN_Testing.py` to
change what they measure. Check `git diff` before committing, since those constants are often left at a local
study configuration.

`MAIN_Testing.py` does carry its own checks: with `verification_enabled` it runs tolerance checks (printed
PASS/FAIL, raising `RuntimeError` on any failure) before solving: four on reference-kernel structure, three on
the LS-20 Jacobian (finite-difference match, plastic branch reached, active set fixed), and — only when
`partition_count <= verification_max_partitions` (default 256) — three dense comparisons that need an extra full
offline solve, followed by the elastic model error. At the default 25×25 grid the dense checks and the model
error are **skipped** (7 checks, not 10); lower `partition_number_per_side` to get them. The two entry points keep **separate caches and output folders**
(`cache`/`output` versus `cache_testing`/`output_testing`); they were shared originally, and because the cache
is keyed on mesh and geometry parameters, alternating between two entry points at different configurations
forced a full offline recompute every switch.

`code/experiments/` holds standalone measurement scripts that import `MAIN_Testing`, build their inputs with
`m.get_problem()` (which also applies `matched_stiffness_control`) and read its module-level constants: `c0_sweep` (reference-stiffness accuracy/speed tradeoff), `elastic_only` (model error at one mesh,
for M-refinement studies — use a square inclusion aligned to partition boundaries or geometry drift confounds
it), `spectrum` (dense Jacobian spectrum and the damped-Richardson stability limit), `solver_cross_check`
(fixed point versus Newton reaching the same root), `group_breakdown` (per-timing-group cost), and
`mean_check` (LS-9 mean consistency plus an independent residual recompute). Each has a header explaining what
it measures. Several build dense `M x M` operators, so run them at small partition counts.

Runs are not fast: `main()` executes `timing_repeat_count` (default 5) interleaved full strain-path solves for
*each* solver, at `strain_increment_count` (default 60) load steps each, on a `partition_number_per_side`² grid
(default 25×25 = 625 partitions) with `element_number_per_side`² (default 75×75) elements. Estimate cost from
these input constants before running, and background/redirect output for large configurations. Offline setup
(assembling and factorizing the global stiffness matrix to build the influence functions `E`, `P`, and
`P0_offset_blocks`) is the expensive one-time step; it is cached to `code/cache/*.npz` keyed on the exact input
parameters that affect it plus a `cache_version` constant (increase it whenever a code change alters the offline
operators, since parameters alone cannot detect that), so changing unrelated parameters (e.g. load path, yield stress) reuses the cache
while changing mesh/geometry/reference-stiffness parameters triggers recomputation.

Output plots (cross-sections, load-path summary, timing breakdown, von Mises fields) are written to
`code/output/` (created if missing but never cleared, so a PNG a run did not regenerate — e.g. von Mises fields
with `post_process_element_stress` off — is stale from an earlier configuration). The caches are gitignored, but `code/output/` is **tracked** in git
(and `code/output_testing/` is neither tracked nor ignored), so any run shows up as changed PNGs in
`git status`.

## Architecture (`code/MAIN.py`, single file, organized top-to-bottom as a pipeline)

`MAIN_Testing.py` is `MAIN.py` plus additions, in the same section order. Every top-level definition that exists
in both files is **textually identical**, including the whole standard and FFT-preconditioned solver path (J2
update, `reset_elastic_partitions`, residual, convergence, corrections, `run_interleaved_repeats`), with only
these exceptions: the configuration constants (`matrix_hardening_modulus`, cache and output folders),
`OnlineTimer` (which also counts P0 applications nested inside the Newton Krylov solve), `get_solvers`,
`LoadPathResult`, `run_strain_path`, and the Main-section helpers. Keep it that way: it is a copy, not an import,
so a change to shared logic must be made identically in both files. A quick check is to parse both files with
`ast` and compare the source of every top-level name they share.

The additions are: a Reference Lippmann-Schwinger Solver section (`ReferenceSolver`, which keeps its warm start
between steps and is therefore built fresh for every load path by `get_solvers`), a Verification section, a
Problem Setup section (`Problem`/`get_problem`, shared with the experiments), and a total-time cost breakdown
that shows all operator applications as one segment.

The file runs as one linear script, in this order, and later sections depend on earlier ones by data, not by
class hierarchy — there is no object model beyond a few `NamedTuple`s (`Mesh`, `PartitionMaterials`, `PlasticState`,
`TrialState`, `LoadPathResult`):

1. **Inputs** — every physical/numerical parameter is a plain module-level constant at the top of the file
   (domain size, inclusion shape/size, mesh and partition resolution, material properties, load path,
   solver tolerances). This is the only place run configuration is changed.
2. **Calculated values and checks** — derived quantities (element/partition counts, DOF count, element size)
   and validation that the inclusion geometry is compatible with the partition grid (e.g. a circular inclusion
   must contain at least one partition centre; a square one must land exactly on partition boundaries).
3. **Mesh, geometry and materials** — builds a periodic hexahedral (8-node brick) mesh on a regular grid
   (`get_periodic_mesh`), assigns each element to a partition and a material (matrix=0, inclusion=1) via
   `get_partition_material_ids`/`is_inside_inclusion`. `get_grid_id` is the shared (i, j, k) → flat-index
   convention used throughout for elements, nodes, and partitions.
4. **Element matrices and assembly** — standard isoparametric 8-node hex FEM: shape function derivatives at 8 Gauss points
   (`get_shape_function_derivatives`), the strain-displacement matrix `B` (`get_B`, using the ER-2 engineering
   strain/stress convention: shear strains are doubled, `(ε11,ε22,ε33,2ε12,2ε23,2ε13)`), per-element
   stiffness `K_element`, and `assemble_element_blocks`, the one sparse assembly helper behind `K`, the load
   matrices `F_macrostrain`/`F_eigenstrain` (`F_μ` of ER-24) and the partition-averaging operator `A_ε`.
5. **Influence functions (offline)** — the expensive precomputation. `get_influence_functions` assembles the
   global stiffness `K`, solves once per macrostrain/eigenstrain load column (batched, `solve_influence_function`,
   via a sparse LU factorization reused across right-hand sides) and applies `A_ε`, following
   ER-24's `P = A_ε K⁻¹ F_μ`, to get partition-average-strain response matrices `E` (macrostrain → partition strain) and `P` (partition eigenstrain → partition strain) — these are
   the *actual* heterogeneous-material influence operators from the TFA relation `ε^B = E^B ε̄ + Σ_A P^{BA} μ^A`
   (see `notes/Formulation.md` Eq. ER-1). `get_P0_offset_blocks`/`get_P0_transformed` build the analogous
   influence operator `P0` for a *homogeneous reference material* on the same partition lattice, and Fourier
   transform it over the 2D partition grid (`np.fft.fftn`) — this is what makes the second solver "FFT" and
   requires partitions to form a congruent, translated lattice (hence one z-layer of partitions spanning the
   full domain thickness, `element_number_along_z` not required to equal `partition_number_per_side`).
6. **Cache** — `load_cache`/`save_cache`/`get_offline_operators` wrap steps above: results are stored in
   `code/cache/E_P.npz` and `code/cache/P0_offset_blocks.npz` alongside the exact parameter dict that produced
   them; a cache is only reused if every recorded parameter matches exactly.
7. **Online timing** — `OnlineTimer`/`timed_online` is a lightweight instrumentation decorator that accumulates
   wall-clock time per named phase (`material_update`, `induced_strain`, `reference_sensitivity`,
   `reference_inverse`, `correction_solve`) across an online solve, used later to build the per-iteration timing
   breakdown plot. Decorated calls must not nest (it raises if they do), since nested timing would double-count.
8. **J2 material** — vectorized (all partitions at once) J2 plasticity with linear isotropic hardening and
   closed-form radial return (`get_trial_state`, `get_plastic_eigenstrain`) plus its analytic eigenstrain
   sensitivity `H_μ` (`get_eigenstrain_sensitivity`), used both for the actual stress-update and (averaged) as
   the FFT solver's frozen reference sensitivity `H_{μ,0}`.
9. **Solvers** — `standard_richardson_iteration` (direct fixed point, correction = `-residual`) and
   `fft_preconditioned_richardson_iteration` (correction solved via the reference operator in Fourier space,
   `get_fft_correction`, rebuilt only when the set of yielding partitions changes — see
   `get_reference_sensitivity`/`get_reference_fourier_inverse`). Both share `reset_elastic_partitions`, which evaluates
   the material update, updates non-yielding partitions' strain in closed form, and computes the actual
   residual; this shared logic is why solver correctness parity is expected. Convergence is measured by
   `get_relative_residual` as in `Formulation.md` section 12: the RMS tensor norm of the residual (shear
   components weighted by 1/2, undoing the engineering doubling), divided by the imposed strain's norm but never
   by less than `residual_strain_scale_floor`. `relaxation_factor` scales the
   correction in either solver.
10. **Load path** — `run_strain_path` steps a prescribed macroscopic strain path (`max_macro_strain` scaled
    linearly over `strain_increment_count` steps) through a solver function passed in by name (`get_solvers`), carrying plastic history
    forward between steps; `run_interleaved_repeats` runs every solver `timing_repeat_count` times interleaved
    (to average out system noise fairly) for the final timing comparison.
11. **Post-processing** — `get_element_stress` recovers full per-element stress fields by solving one more
    (fine-mesh, CG-based) elastic problem with the converged partition eigenstrains as body-force-equivalent
    loads, only run if `post_process_element_stress` is set; used to plot fine-resolution von Mises fields
    against the coarser partition-averaged von Mises field.
12. **Timing report** — picks the fastest repeat per step and sums timings over the comparable steps.
13. **Main** — `main()` reads as an outline: build mesh/materials, get cached offline operators, run the
    interleaved solvers, then `print_comparison`, `save_load_path_plots`, `save_von_mises_plots` and finally
    `check_solver_agreement`.

`code/plots.py` holds all matplotlib figure-generation logic (cross-section geometry plot, load-path summary
with elastic-regime shading, timing breakdown, von Mises cross-sections) and has no solver logic of its own.
`code/plots_testing.py` is its counterpart for `MAIN_Testing.py`: the two are deliberately separate so the test
pipeline can change figures without breaking `MAIN.py`. It generalises the load-path summary to a list of
solvers and adds a model-difference panel, replaces the per-iteration breakdown with a total-time
`plot_cost_breakdown` (an iteration is not a comparable unit once a Newton step contains a whole Krylov solve,
so it also annotates nonlocal operator applications), and adds a von Mises model-difference field.

## Notes directory (theory — read before touching solver math)

- `notes/Formulation.md` — the primary, current derivation. Walks from the actual heterogeneous partition-strain
  relation (Eq. ER-1) through the classical Moulinec–Suquet reference-split/FFT idea, to two solution
  strategies for the *reduced* TFA problem: Strategy 1, reference fixed-point iteration (implemented in
  `MAIN.py` as `fft_preconditioned_richardson_iteration`), and reference-preconditioned Newton–Krylov (not yet implemented —
  described as Strategy 2, a possible future direction). Defines every symbol used in the code
  (`E`, `P`, `P0`, `H_μ`, `M0`, anchor/sensitivity choices) with numbered equations (ER-1 … ER-25) that are
  referenced directly in code comments/commit history. Section 9–11 give convergence caveats worth checking
  before assuming a solver change is an improvement (e.g. elastic steps have `H_μ = 0` so the FFT solver's
  reference matches exactly; fixed-point convergence is not guaranteed just because the reference is
  invertible).
- `notes/verified_notes/` — an earlier, more slowly-paced 7-part derivation sequence (see its own `README.md`
  for the reading order) covering displacement influence functions, partitioned eigenstrain, the partition
  strain transformation, nonlinear FEM/Newton structure, and the full J2 return-mapping algorithm in detail.
  `Formulation.md` cites part 7 of this sequence for the constitutive convention; treat these as the
  detailed backing derivations when `Formulation.md`'s summary isn't enough.
- `notes/Partition_Lippmann_Schwinger_New_9_28_2026` — a newer formulation (no file extension), now
  **implemented** as the third solver in `MAIN_Testing.py`; read it before touching that solver, since its
  equation labels (`LS-7`, `LS-14`, `LS-18`, `LS-20`) are cited directly in the code. Its algebra was checked
  equation by equation and holds. Three caveats worth knowing, none of them yet written into the note itself:
  its section 7.1 recommends a midpoint-Lame reference stiffness, which measured worst of five candidates
  (55 % plastic error) because that is a convergence heuristic for the linear elastic basic scheme rather than
  a model choice; the plastic branch of `H_mu` is never stated even though section 1.4 gives the full return
  map, so the implementation derives it (and verifies it against a finite difference); and the note leaves
  M-convergence open in section 5.4, where measurement shows first order in partition size.
- `references/` — source PDFs cited by the notes (Moulinec–Suquet, Fish–Cui eigenstate-based homogenization,
  Ladecký et al. FFT-preconditioned FE solver, Dvorak, computational plasticity), referenced by the notes'
  attribution sections.
