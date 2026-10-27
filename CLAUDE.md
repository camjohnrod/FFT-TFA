# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase for reduced-order homogenization of elastoplastic composites (a stiff circular fiber, or a
laminate layer for the exact checks, in a softer J2-plastic matrix, in a periodic cube). Three grids: elements (the FE
mesh, which resolves the geometry); CP's uniform partitions (k × k element columns through the thickness for a shape in
`fe.shapes_constant_along_z`, k × k × k element blocks otherwise); and clusters, each partition's part in each phase
(`fe.Clusters`, built by the shared `fe.get_clusters`: one cluster per single-phase partition, two per mixed one).
Every method keeps one material state per cluster. The research question is how the operator through which material
states interact decides accuracy and cost (README, "Research question"; `notes/1_research_context.md`). Three methods,
all solved by Newton–GMRES with a backtracking line search, in this order everywhere (table, figures, code):

- **CP**: the partition-averaged Lippmann–Schwinger (LS) equation of `notes/2_ls_model.md` on the partitions, the
  method under development. Its only nonlocal operator is the FFT convolution with the Moulinec–Suquet kernel P0,
  averaged over partitions (LS-21). Unknowns are partition strains; a mixed partition's strain is shared between its
  two clusters at equal polarization σ − C0 ε (`ls_model.get_cluster_strain`, LS-22), slide 14 of the user's
  `references/10.20.2026 Progress.pptx`. **One C0 for everything, constant polarization in every partition** (the
  user's decision, 2026-10-27): the same C0 builds P0 and splits mixed partitions. C0 =
  `reference_stiffness_scale` × the named stiffness (default 0.01 × matrix) and must stay below every cluster's
  tangent, or the split has no solution. GMRES is preconditioned by the Fourier preconditioner (LS-23,
  `ls_model.get_fourier_preconditioner`), which cancels the cost of a small C0. Don't reintroduce a separate C0 for the split
  (the equal-stress hybrid of 2026-10-26) without the user asking. Always solved.
- **Spectral LS** (`config.include_spectral_ls`): the same LS code on the element lattice (one partition per element,
  per element column for a z-constant shape), i.e. the standard Moulinec–Suquet spectral method. A comparison, not a
  reference solution. It has no mixed partitions, and its converged answer does not depend on its reference stiffness
  (P0 is a projection there), so that is a solver setting, its own (`spectral_ls_reference_stiffness`, default
  Moulinec–Suquet's midpoint (L_m + L_f)/2), not CP's C0: giving it CP's 0.01 × matrix was unfair (2× slower). The
  same Fourier preconditioner as CP's is available to it (`spectral_ls_fourier_preconditioner`, off by default):
  on its compatible-strain Krylov space it equals a change of reference medium, so it measurably changes nothing
  (11,484 against 11,487 GMRES its) and only adds an FFT pair per iteration.
- **Fairness between CP and Spectral LS** is a standing requirement of the user ("equal chances to shine"): each
  method gets its best setting from the same principles, justified from the LS literature only (no FE papers, e.g.
  not Ladecký et al.), and both stop at the same accuracy (the C0-independent residual measure, LS-24).
- **TFA** (`config.include_tfa_model`): the actual E/P model of `notes/3_tfa_baseline.md` (ER-4,
  `r = ε - Eε̄ - Pμ(ε)`), Strategy 2 (Newton–GMRES, unpreconditioned), the baseline. Its regions (the notes'
  "partitions") are CP's clusters, so E and P are built on clusters and CP and TFA share their material states.

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
python code/check.py         # the test: four small problems, a few seconds
python code/check.py save    # the same, minus the old-reference comparison, then save the new reference
```

There is no pytest suite. `check.py` solves a cyclic laminate aligned with the partitions, a laminate whose
interfaces cut partitions loaded in transverse shear, a laminate with interfaces normal to z on a 9³ mesh with 3³
partitions (the only test of the 3D lattice and the kernel's ξ_z frequencies), and a circle with mixed partitions. On
the laminates every method must match the exact solution in every cluster, each phase on its own stress scale, to
`laminate_tolerance`, except CP on the cut laminate (`inexact_methods`: equal polarization at C0 ≠ 0 is not exact
there; its difference is printed as INFO). Where CP has mixed partitions, their elastic stress and tangent must match
LS-22's elastic closed form C*_A = C0 + [c_m (L_m − C0)⁻¹ + c_f (L_f − C0)⁻¹]⁻¹ to `mixed_partition_check_tolerance`:
that is what catches a wrong split or split tangent (a wrong tangent alone only slows Newton). check.py doesn't test
that the Fourier preconditioner leaves answers alone; it was checked by hand (2026-10-27, both toggles on and off:
every check problem agrees to the Newton tolerance): recheck after changing it. Nor does it test that every method
stops at the same accuracy; that was checked by hand too (2026-10-27, each method and check problem against its own
solution at tolerance 1e-11: all within ~2e-6), recheck after touching a stopping test. On all, every method
must complete every step and, once `code/check_reference.npz` exists (untracked, outside the cache folder so that
clearing the cache keeps it), reproduce it bit for bit. For a change
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
Offline kernel construction still uses `np.fft` (about 0.5 s at 64³ elements, once).

Cost: with TFA, the offline step builds the dense P of (6 × partitions)² doubles (0.3 GB at 32² partitions, 4.8 GB
at 64² or 16³: a z-varying shape has partition_number_per_side³ partitions and an element_number_per_side³ mesh),
cached in `code/cache/` under a hash of its parameters (`fe.get_cache_path`); bump `fe.cache_version` when a
code change alters E or P. Each cache file also stores `build_seconds`, the time it took to compute, so a cached run
still reports TFA's true offline cost; a file without it is recomputed. Estimate cost before running large
configurations and run them in the background.

## Architecture

Imports go one way: `config` → `material` → `newton` → `fe`, `ls_model`, `tfa_model` → `run` → `check`. `plots.py`
imports no project module; only `run.py` calls it.

| Module | Contents |
|---|---|
| `config.py` | every input, `check_inputs()`, folders |
| `run.py` | `Lattice`, `Problem` and `get_problem()` (builds everything and times each method's offline step), `get_applied_macro_strain`, `get_solvers` (method name → solver), `run_load_path` (per-cluster stress and accumulated plastic strain), `solve_all`, timing table, labels, figures (element maps via `get_element_map`, each method's cells via `get_cell_maps`), `main` |
| `fe.py` | shapes (`is_inside_inclusion`, `shapes_constant_along_z`), `Discretization` (partition and spectral), the periodic `Mesh`, `Materials`, `Clusters` and `get_clusters`; 8-node element matrices, `FEModel` (K and the ER-24 operators), TFA's E and P on clusters (`compute_E_and_P`), the cache (`get_E_and_P`) |
| `newton.py` | `solve_load_step` (the one Newton loop) and `take_step` (line search), `Evaluation`, `StepResult`, `Solver`, the `LoadPathAbandoned` errors, relative residual, `has_converged`, `solve_newton_correction` (GMRES with iteration count) |
| `ls_model.py` | `apply_on_lattice` (FFT convolution); the reference medium: CP's C0 and Spectral LS's, the Moulinec–Suquet kernel P0 and its partition average (`get_P0_transformed`), the Fourier preconditioner (`get_fourier_preconditioner`); the partition stress from its clusters (`get_cluster_strain`: equal-polarization split of mixed partitions against C0; `get_partition_stress`, `get_partition_tangent`), and `LSSolver(lattice, matrix_L)` for CP and Spectral LS: residual, Newton correction, load step |
| `tfa_model.py` | TFA residual with the elastic reset, column-restricted P products, Newton step, `get_tfa_solver` (on CP's clusters) |
| `material.py` | vectorized J2 radial return, its sensitivity H_μ and algorithmic stiffness L_t, von Mises stress |
| `plots.py` | stress–strain curves, cross-section (one panel per method with its cells), field maps (`plot_field_maps`: von Mises and plastic strain), `method_styles` keyed by method name |
| `check.py` | the checks, and the laminate's exact solution (Newton on the matrix's three interface-normal strains: ε11, γ12, γ13 for interfaces normal to x; ε33, γ23, γ13 normal to z) |

Non-obvious points:
- **Strain convention (ER-2):** engineering shear, `(ε11, ε22, ε33, 2ε12, 2ε23, 2ε13)`. `get_relative_residual`
  weights shear by ½, and divides by the imposed strain's norm but never by less than `residual_strain_scale_floor`.
  The LS methods' residual is a strain of their reference medium (it scales as 1/C0 for the same equilibrium error),
  so convergence and the line search judge M_m⁻¹ r instead: the strain correction an all-matrix cell would need to
  cancel r, with M̂_m(ξ) = I − P̂0(ξ)(I − C0⁻¹ C_m) built once per lattice (LS-24, `LSSolver.residual_in_matrix_units`,
  passed as `measure_residual` to `newton.solve_load_step`); the Newton correction still solves for r itself. For
  Spectral LS it equals the earlier C_m⁻¹ C0 r (bit-identical results); for CP it also judges in full the part of r
  that C_m⁻¹ C0 r shrank by α, which let CP stop 30× less accurately on the cut laminate. **The user has said they
  don't really understand LS-24:** when it comes up, explain it from the LS note's section 7.5 (written step by step
  with equations for that reason) before building on it.
- **C0 against the tangent:** `ls_model.check_reference_below_tangents` stops CP with the cause ("lower
  `reference_stiffness_scale`") when L_t − C0 is not positive definite in a mixed partition's cluster, where LS-22
  has no solution; before, that surfaced as a misleading GMRES failure. It fires at step 1 for `"voigt"` × 0.01,
  `"matrix"` × 0.1 and a perfectly plastic matrix, and a step-1 failure still aborts the whole run.
- **No global lattice size online:** solvers, the FFT and the load path take sizes from the arrays they are given
  (`reshape(-1, 6)`, the kernel's shape, the `Clusters` they carry), never from `config.partition_*` or
  `config.element_*`; that is what lets CP and Spectral LS run on different lattices with the same code.
- **Lattices are 3D, indexed `[z, y, x]`** (`get_grid_id`: i fastest). A z-constant shape has one partition along
  z (N_z = 1), so its LS is the 2D problem; any number of element layers then gives the same kernel to round-off,
  since averaging over the full thickness keeps only ξ_z = 0. The lattice is square in-plane (the cell is a cube);
  that is how the half spectrum's shape gives back the lattice's.
- **Half spectrum:** P0 and every lattice field are real, so `P0_transformed` holds only `rfftn`'s half over axes
  (0, 1, 2) (shape `(N_z, N, N//2 + 1, 6, 6)`), transformed back with `irfftn(..., s=(N_z, N, N))`
  (`ls_model.get_offset_blocks`). Anything new reading it must too. Online, `ls_model.apply_on_lattice`
  transforms only axes longer than one, read from the lattice's shape (no geometric assumption): a length-1
  transform is exactly the identity but made every convolution on a lattice one partition thick 1.5× slower (found
  2026-10-27, after the 3D change had cost CP and Spectral LS ~15 % online). 3D lattices transform all three axes.
- **Induced strain (TFA):** within a step Pμ = Pμₙ + PΔμ, Δμ zero outside yielding clusters. P is stored
  column-major so each run of consecutive yielding clusters is a contiguous column block (`apply_P_to_partitions`);
  Newton's Jacobian products use the same function. Don't fancy-index `P[:, columns]`: the copy costs more than the
  full product.
- **Geometry, partitions, clusters:** geometry is resolved on element centres (`fe.is_inside_inclusion` takes
  (x, y, z)). A new shape goes in `is_inside_inclusion`, `check_inputs`, the E/P cache parameters (`fe.get_E_and_P`)
  and, if its geometry does not vary along z, `fe.shapes_constant_along_z` (`get_element_material_ids` stops if a
  listed shape does). Clusters are ordered by partition, matrix first, so a mixed partition's inclusion cluster
  directly follows its matrix cluster. Clusters have
  unequal volumes: every cell average goes through `Clusters.average` (volume-weighted; written so that with one
  cluster per partition it is bit-identical to a plain mean), including the macroscopic stress and TFA's residual
  norm. Results (`LoadPathResult`, maps) are per cluster; maps paint each element with its cluster's value, and
  unset plastic strain colour limits come from the Spectral LS map (clipping the others). Maps show the element
  layer at mid-thickness.
- **Line search:** `newton.solve_load_step` is the one Newton loop; `take_step` tries the full step and halves it until
  the relative residual drops by `line_search_sufficient_decrease` × step. Where full steps work it changes nothing
  (bit-identical on every earlier test problem); it fixed the step-1 failure when the inclusion can yield and the
  stalls with mixed partitions. Backtracking evaluations cost residual evaluations that show only in the time.
- **E is built only when needed:** by TFA, or by `reference_stiffness = "homogenized"`. A CP/Spectral LS run with
  `"matrix"` or `"voigt"` C0 does no FE solve at all. C0 changes CP's answer (not Spectral LS's); TFA doesn't use it.
- **Fourier preconditioner (LS-23, LS note section 7.4):** `get_fourier_preconditioner(P0_transformed,
  jacobian_blocks)`, one function for both LS methods: the best Fourier-diagonal approximation of the method's own
  Jacobian J = I − P0 D, i.e. I − P̂0(ξ)⟨D⟩ with ⟨D⟩ the lattice mean of D = I − C0⁻¹ L_t (the mean current tangent),
  rebuilt at every Newton correction, applied on the right so GMRES still measures the true linear residual. Toggles
  `cp_fourier_preconditioner` (default on) and `spectral_ls_fourier_preconditioner` (default off). Default problem:
  CP 40,705 → 7,146 GMRES its (23 → 6 s). Rejected: the earlier fixed version (C_p = matrix stiffness, offline;
  10,618 its), and block-Jacobi on top (5,538 its but CP then stops at 6e-6 / 1.2e-3 of its tight solution instead of
  2e-8 / 9e-7).
- **Iterations:** "Newton its" are Newton corrections (residual evaluations − 1), "GMRES its" the inner GMRES
  iterations summed over them (scipy callback, `callback_type='pr_norm'`). "Online" is `solve_step` wall time only;
  "Offline" (`Problem.offline_seconds`) is what each method's operators cost to build as if it ran alone: the
  partition mesh and clusters, C0 and P0 for CP (C0 includes E's build time under
  `"homogenized"`); the element mesh, its clusters, its reference stiffness and P0 for Spectral LS; the partition
  mesh and clusters plus E and P for TFA, at the time they took to compute.
- **Failed load steps:** an unsolvable step raises a `LoadPathAbandoned` subclass (`SolverDidNotConverge`, or
  `MaterialFullySoftened` when a flow stress reaches zero); `run_load_path` stops that method there and the others
  carry on; a failure on the first step re-raises. Divergence means exceeding the limit times
  max(1, the step's first relative residual), since at a cyclic zero crossing the first residual legitimately exceeds
  the imposed strain.
- **Known differences by design:** LS methods start each step from the previous strain plus the uniform increment;
  TFA starts from the exact elastic predictor Eε̄ + Pμₙ (and resets elastic partitions exactly), so LS spends Newton
  iterations on elastic steps. LS and TFA have separate divergence limits. Compare methods by time: a
  column-restricted P product and a P0 convolution cost very different amounts.
- **Naming.** "Reference" means the homogeneous reference medium C0 (`reference_L`) and what is built from it
  (P0). `LSSolver` maps to the LS note: `evaluate` → LS-18 with μ* of LS-6 (P0 by `apply_on_lattice`, LS-17);
  `get_newton_correction` → LS-20 and LS-23; `solve_step` → section 7.3; `residual_in_matrix_units` → LS-24.
  `newton_krylov_iteration` (TFA) is cited by name in `3_tfa_baseline.md`. Code comments cite equations and papers,
  never the history of a decision (that lives here and in the notes).

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

Mixed partitions, done 2026-10-26: tests in a scratch prototype first. On a laminate whose interfaces cut partitions,
every split rule without geometry is badly wrong in one direction (equal stress across the interface, equal strain
along it), and slide 14 with C0 = matrix or C_hom fails or gives negative stiffness. On circles (loads mixed, uniaxial,
cyclic; contrast 100 and 10; radius 0.35 and 0.25 mm; 32 and 48 elements) equal stress gave macroscopic errors of
1–5 % against element-level TFA, about Spectral LS's; local errors next to the fiber matched Spectral LS at
2 × 2-element partitions and were 2–5× larger at 4 × 4, worst for the small fiber. The user rejects geometric
(laminate) rules, because a partition may later hold several interfaces, and single-phase-only partitions, and
requires O(M log M).

Fair CP vs Spectral LS, done 2026-10-27: Spectral LS's own reference (midpoint), the Fourier preconditioner (LS-23, CP only by default), and the
C0-independent stopping test (LS-24). Default problem: CP 6.2 s, Spectral LS 16.9 s, TFA 7.5 s online; CP and
Spectral LS within 2.0e-8 / 1.2e-8 (macroscopic) and 8.7e-7 / 5.8e-7 (worst cluster) of their own tight solutions, TFA 1.1e-8 / 2.2e-7.

One C0, done 2026-10-27 (after a detour): on 2026-10-26 the split was equal stress with matrix C0 for the
interaction (the hybrid); the slides' one C0 = 0.01 × matrix was tried, found 6× slower, undone, then rebuilt on
2026-10-27 with the FFT preconditioner, at the user's insistence on constant polarization. Measured on check.py's
circle against element-level TFA (CP macroscopic error): hybrid 2.4 %; one C0 at 0.01 / 0.05 / 0.07 × matrix
4.3 / 4.5 / 4.8 %, 0.1 no solution once the matrix yields; stiff C0 (2–10 × fiber) 64–66 %; bulk-matrix with
sub-tangent shear 16 %. The accuracy loss is mostly the interaction's (equal-stress split with C0 = 0.01 × matrix:
4.1 %): matrix partitions have zero polarization only at C0 = matrix. Default problem (64²/32²), online: CP 12.5 s
with the preconditioner (49 s without, hybrid 8.5 s); Spectral LS 43 s (31.6 s without: it gains nothing, see
Non-obvious points). Not adopted: block-Jacobi after the FFT preconditioner (CP 1792 → 441 GMRES its on the check
circle; the user asked for the FFT one only), CG on Brisard–Dormieux's symmetric polarization form (not tried), and
a Hashin–Shtrikman split, ε + P(C0) τ equal in both clusters with P the direction average of Γ0(n; C0) (one C0 =
matrix: 3.9 %, worst cluster 26 % against 48 %, as fast as the hybrid, but not constant polarization; slide 14 is
P → ∞, equal stress P = C0⁻¹).

3D cells, done 2026-10-26: the user wants more complex RVEs later, always cubic. z-constant shapes keep
`element_number_along_z` and column partitions; z-varying ones get an `element_number_per_side`³ mesh and cubic
partitions (the user's choices). Checked: z laminate exact for every method; a circle as columns at 1 and 3 layers
and as cubes agrees to 1e-14 at a tight Newton tolerance (1e-8 at the default, where Newton stops). Mid-plane maps
show little of a z-varying geometry; an x–z slice may be wanted later.

Later: a reference stiffness that follows plastic softening (as in SCA; also the only way CP could run a perfectly
plastic phase with one C0), CG on the symmetric polarization form (block-Jacobi was rejected: CP lost accuracy),
and the consistent kernel (exact LS-7, the limit of the
averaged MS kernel as the element grid is refined; its alias series converges as 1/T, and assigning the tail to the
axis directions, whose sinc² weights sum to one in closed form, makes it 1/T², checked in a scratch test on
2026-10-12).
