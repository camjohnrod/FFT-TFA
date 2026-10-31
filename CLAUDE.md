# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase comparing reduced-order methods for elastoplastic composites; the README explains the problem,
the three grids (elements, partitions, clusters) and the methods. Use the method names exactly, and in this order
everywhere (table, figures, code): **FFT-TFA**, **Spectral LS**, **TFA**, **CP**. FFT-TFA is the method under
development and always runs; CP is the earlier method, off by default (`include_cp`), kept separate so it can be
removed. "fine LS", "coarse LS", "LS Newton" and "MAIN_*" are retired names. FFT-TFA and TFA share their material
model (clusters, eigenstrains, update at the average strain) and differ in how the clusters interact: FFT-TFA averages
the LS operator over its regions with the matrix as C0, exact in the matrix but with the inclusion's polarization
uniform per piece (TFA resolves it), truncates the interaction to nearby partitions, and uses the Moulinec–Suquet
kernel instead of FE. At full reach FFT-TFA is SCA's clustered LS equation with C0 = L_m (LS note, section 10.4). CP and
TFA are different models. Neither difference is a bug.

Read the relevant note before changing solver math. Code names map onto the notes' symbols and numbered equations
(FT-* in `notes/2_ls_model.md` section 10, LS-* in its sections 1–9, ER-* in `notes/3_tfa_baseline.md`), which code
comments cite. Comments cite equations
and papers, never the history of a decision.

## The user's standing decisions

- **All settings live in `code/config.py`, and only there**: a new knob goes there with a comment, never as a literal
  elsewhere. config.py runs from the settings changed most to those changed least (problem, methods, figures,
  materials, solver details, checks), after a table of what each grid means to each method. A setting of one method
  carries its prefix (`fft_tfa_`, `spectral_ls_`, `tfa_`, `cp_`, `k_means_`), and the same setting for several
  methods sits as a group. Exceptions: plot appearance in `plots.py`, check problems in `check.py`. config.py is often left at a local
  study configuration; "default" means `git show HEAD:code/config.py`. The user edits it while you work: change it by
  targeted edits, never by rewriting it from an earlier read.
- **FFT-TFA is separate from CP** (`fft_tfa_model.py`), so CP can be deleted without touching it: FFT-TFA uses only
  `ls_model`'s shared kernel code (`apply_on_lattice`, the Moulinec–Suquet kernel, `get_P0_transformed`, the block
  helpers), never CP's. Its reference medium is always the matrix (no setting), its regions are the matrix clusters
  and the inclusion pieces (FT-1), and its truncated interaction is FT-2/FT-3 as approved; the consistent variant of
  section 10.5 is measured, not adopted. Neighbour blocks are a setting, on by default.
- **One C0 for everything in CP, constant polarization in every partition.** The same C0 builds P0 and sets how a
  partition's strain is shared among its pieces (LS-25; LS-22 for partition-phase clusters). C0 must stay below the
  tangent of every coupled cluster or LS-26 has no solution (`check_reference_below_tangents` stops with that
  message). Don't reintroduce a separate C0 for the split.
- **Clusters are a per-method setting** (`fft_tfa_clusters`, `tfa_clusters`, `cp_clusters`: `"partition phases"`,
  `"partition k-means"` or `"k-means"`). The setting only chooses how `fe.Clusters` is built; nothing downstream
  branches on it. Spectral LS stays per element, on its own elements when `spectral_ls_elements_per_side` is set
  (`fe.get_spectral_ls_discretization`; figures draw each method on its own elements). CP keeps its partitions and one polarization each; k-means changes
  only where its eigenstrains live.
- **Spectral LS's reference stiffness is its own** (`spectral_ls_reference_stiffness`): its answer doesn't depend on
  it, so it is a solver setting, not CP's C0.
- **Fairness between methods:** each gets its best setting from the same principles, justified from the LS
  literature only (no FE papers), all share the Newton–Krylov settings and thread count, and all stop at the same
  accuracy (LS-24; FFT-TFA's residual is already in matrix units, FT-5), which `check.py` checks. The papers are in
  `references/`, with one note per paper read cover to cover in `references/reading_notes/`; cite from those, never
  from a title or abstract.
- **Small codebase:** don't add modules (`fft_tfa_model.py` was asked for), and don't bring back what the 2026-10-26
  cut-back removed (git tag `pre-cleanup`: fixed-point solvers, FFT-preconditioned TFA, the FE kernel for P0, the
  coarse LS lattice, extra checks, saved results) without the user asking. The cached reference
  (`include_reference`) was asked for: it saves only the macroscopic stress at every step and the fields at the last
  step, for the figures; error measures against it were left for later (the user chose to wait: a fair field error
  across different meshes is not settled, see the README's roadmap).
- **The user has said they don't really understand LS-24.** When it comes up, explain it from the LS note's
  section 7.5 before building on it.

## Running the code

```bash
python code/run.py           # solve config.py's problem with every enabled method; table + figures to code/output/
python code/check.py         # the test: seven small problems, about a minute
python code/check.py save    # the same, then save the results as the new reference
```

There is no pytest suite and no way to run one check problem alone. `check.py` solves `"laminate"`, `"cut laminate"`,
`"fiber"`, `"z laminate"`, `"sphere"`, `"k-means laminate"` and `"k-means sphere"` with every method (CP pinned on).
On the laminates every method must match the exact solution to `laminate_tolerance` (Spectral LS on its own coarser
elements on the laminate, finer on the z laminate, both resolving the layer exactly), except CP and FFT-TFA on the cut
laminate and FFT-TFA on the k-means laminate (`inexact_methods`: printed as INFO); FFT-TFA solved at full reach
(`solve_at_full_reach`, its interaction then exact) must match it on every laminate. On every problem FFT-TFA's
interaction at full reach must equal the element-level LS operator averaged over its regions and clusters
(`get_interaction_error`, to `interaction_check_tolerance`): that catches a wrong offset, pair count, correction block
or average. On the fiber and the k-means sphere every method must stop within `stopping_accuracy_tolerance` of its
own solve at `stopping_check_tolerance` (not on partition k-means with few-element clusters, whose clustered problem
can have several solutions, for TFA too). CP's sparse LS-25 matrices (E, F, Λ, D_e) must match LS-25 computed piece by piece in `check.py`, with
no solver code, to `piece_law_check_tolerance`; that is what catches a wrong piece law or piece indexing. The
coupled-cluster terms are tested only where a coupled cluster has a piece filling a partition (the k-means sphere has
such pieces). On the fiber the reference is on at the working resolution and must equal Spectral LS bit for bit and
reload from its cache unchanged (`is_reference_spectral_ls`). Every method must complete every step and reproduce `code/check_reference.npz` (untracked) bit for bit, at
the `thread_count` it was saved with; with none saved, check.py skips that test and says so, so run `check.py save` on
a clean tree first. For a refactor: `check.py save` before, `check.py` after.

Not covered by `check.py`, so recheck by hand after touching them: the Fourier preconditioner and FFT-TFA's block-Jacobi
preconditioner must leave answers alone (compare with them on and off, or against the identity), FFT-TFA's Newton
product must match finite differences of its residual (FT-6), its assembled diagonal blocks (FT-7) those of the
Jacobian probed column by column, and stopping accuracy on the default problem (against a
solve at tolerance 1e-11). No check problem uses the `"random fibers"` shape, or partitions along z for a z-constant
shape.

`check.py` sets its `pinned_inputs` on the `config` module in memory, with a temporary cache folder. This works only
because **no module reads `config` at import time**: no `config.x` in default arguments or module-level constants.
A renamed config input must be renamed in `pinned_inputs` too.

`python` is the base miniforge environment's (Python 3.12, with `numpy`, `scipy`, `matplotlib` and `tqdm`); the
`fenics-env`/`fenicsx-env` conda envs are unrelated.
Entry points import `config` before numpy, since config.py sets the thread environment variables; online FFTs use
`scipy.fft` with `workers=config.thread_count`.

TFA's offline step builds a dense P of (6 × TFA's clusters)² doubles (4.8 GB at 64² or 16³ partitions), cached in
`code/cache/` under a hash of its parameters (`fe.get_cache_path`). Bump `fe.cache_version` when a change alters E
or P. The reference (`run.get_reference`) is cached the same way, under the shared geometry key
(`fe.get_geometry_parameters`) plus its mesh, the materials, the load path and the solver inputs that change its
answer; bump `run.reference_cache_version` when a change alters it. A new input that changes the reference must be
added to its key, or a stale one is reused silently. Estimate the cost of large configurations and run them in the background.

## Architecture

Imports go one way: `config` → `material` → `newton` → `fe`, `ls_model`, `tfa_model` → `fft_tfa_model` (from
`ls_model`) → `run` → `check`. `plots.py`
imports no project module; only `run.py` calls it.

| Module | Contents |
|---|---|
| `config.py` | every input, `check_inputs()`, folders |
| `run.py` | `Lattice`, `Problem`, `get_problem()` (builds everything, each method's clusters, times each method's offline step), `get_cluster_labels`, `get_solvers`, `run_load_path`, `solve_all`, the cached reference (`Reference`, `get_reference`), timing table, macroscopic stress errors against the reference (`get_loaded_components`, `get_macro_stress_errors`, `errors.txt`), figures, `main` |
| `fft_tfa_model.py` | FFT-TFA: `Interaction`, its regions (`get_regions`, FT-1), correction blocks (`get_correction_blocks`, FT-3; stored as 6 × 6 BSR blocks), `get_interaction`, `get_strain_change` (FT-2 with FT-4), the block-Jacobi preconditioner's diagonal blocks (`get_strain_change_within_groups`, FT-7), `FFTTFASolver` (FT-5, FT-6) |
| `fe.py` | shapes (`is_inside_inclusion`, `shapes_constant_along_z`, `get_fiber_centres`), `Discretization`, `Mesh`, `Materials`, `Clusters` (clusters and their pieces), `label_partition_phases`, `label_partition_k_means`, `label_k_means`, `get_clusters`, element matrices, `FEModel`, TFA's E and P (`compute_E_and_P`) and their cache (`get_E_and_P`) |
| `newton.py` | `solve_load_step` (the one Newton loop), `take_step` (line search), relative residual, `has_converged`, `solve_newton_correction` (GMRES), the `LoadPathAbandoned` errors |
| `ls_model.py` | `apply_on_lattice` (FFT convolution), C0, P0 and its partition average (`get_P0_transformed`), LS-24, `get_fourier_preconditioner`, `LatticeSolver` (Newton–Krylov shared by CP and Spectral LS), `SpectralLSSolver` (its corrections by CG, LS-28), `get_elastic_strain_concentration` (k-means features), CP's piece law (`PieceLaw`, LS-25) and `CPSolver` |
| `tfa_model.py` | TFA residual with the elastic reset, column-restricted P products, `get_tfa_solver` |
| `material.py` | vectorized J2 radial return, its sensitivity H_μ and algorithmic stiffness L_t |
| `plots.py` | the figures: `method_styles` keyed by method name; the cross-section's `boundary_styles` (voxel, cluster, partition lines) and k-means cluster colours |
| `check.py` | the checks and the laminate's exact solution |

The LS solvers map onto the LS note: `evaluate` → LS-18, `get_newton_correction` → LS-20 and LS-23 with each
subclass's `get_D` (CP's is LS-27; Spectral LS's correction is solved by CG in C0's inner product, LS-28), `solve_step` → section 7.3, `residual_in_matrix_units` → LS-24,
`CPSolver.solve_eigenstrain` → LS-26.

## Non-obvious points

- **Strain convention (ER-2):** engineering shear, `(ε11, ε22, ε33, 2ε12, 2ε23, 2ε13)`.
- **Convergence (LS-24):** the LS residual is a strain of the reference medium, so convergence and the line search
  judge M_m⁻¹ r (the correction an all-matrix cell would need, `residual_in_matrix_units`), while the Newton
  correction still solves for r. `get_relative_residual` weights shear by ½ and divides by the imposed strain's norm,
  never by less than `residual_strain_scale_floor`.
- **No global lattice size online:** solvers, FFTs and the load path take sizes from the arrays they're given, never
  from `config.partitions_*` or `config.elements_*`. That is what lets CP and Spectral LS share code, and FFT-TFA reuse
  the kernel.
- **FFT-TFA's unknowns are region strains** (matrix clusters, inclusion pieces), not cluster strains: a matrix
  cluster spanning partitions is one region, an inclusion cluster has one region per piece. Its correction blocks
  come from FFT cross-correlations of the piece indicators on twice the partition, against the element kernel on the
  mesh's own element lattice (`mesh.discretization`, also for z-constant shapes partitioned in layers).
- **Lattices are 3D, indexed `[z, y, x]`.** A z-constant shape has `partitions_along_z` partitions along z
  (1: columns, the 2D problem; more layers give the same answer to solver tolerance, a useful consistency test, except
  FFT-TFA without neighbour blocks, whose truncation then differs by a few %; with them it agrees).
  `apply_on_lattice` transforms only axes longer than one (length-1 transforms are identities but slow).
- **Half spectrum:** `P0_transformed` holds only `rfftn`'s half (shape `(N_z, N, N//2 + 1, 6, 6)`), transformed back
  with `irfftn(..., s=(N_z, N, N))`. Anything new reading it must do the same.
- **Clusters and pieces:** `fe.Clusters` holds the clusters (material states; `volume` in partition volumes) and
  their pieces (cluster ∩ partition, ordered by partition). Clusters have unequal volumes: every cell average goes
  through `Clusters.average`. Every method reports stress and plastic state per cluster. A **coupled** cluster has a
  piece in a partition holding several; CP solves only those clusters' eigenstrains iteratively (LS-26), by Newton
  with sparse LU and the outer Newton's line search, with no FFT.
- **A new shape** goes in `is_inside_inclusion`, `check_inputs`, the E/P cache parameters (`fe.get_E_and_P`: every
  parameter of the shape, or a stale E and P is reused silently) and, if constant along z, `shapes_constant_along_z`.
  `"random fibers"` keys its cache on the placed radius and centres (`get_fiber_centres`), which stand for the count,
  volume fraction, gap and seed; a new input that changes the geometry some other way must be added to the key.
- **TFA's P** is stored column-major so consecutive yielding clusters form a contiguous column block
  (`apply_P_to_partitions`). Don't fancy-index `P[:, columns]`: the copy costs more than the full product.
- **E is built only when needed:** by TFA, or by `cp_reference_stiffness = "homogenized"`, on TFA's clusters when TFA
  runs. The E/P cache key includes every element's cluster label, so each clustering finds its own cache.
- **k-means** runs on each element's elastic strain concentration from six elastic Spectral LS steps on the mesh's
  elements (whatever `spectral_ls_elements_per_side`), smoothed by [1, 2, 1] / 4 within each phase
  (`fe.smooth_within_phases`: removes the interface ringing that dominated the stiff fibers' features, and leaves a
  uniform phase exactly uniform), per phase (over
  the cell) or per partition's part in each phase (partition k-means; one cluster per part is exactly partition
  phases), with a fixed seed; a phase or part gets no more clusters than distinct features, and empty clusters are
  dropped. Its time counts
  in the offline time of each method that uses it.
- **Fourier preconditioner (LS-23):** I − P̂0(ξ)⟨D⟩ with ⟨D⟩ the lattice mean of D (for CP, of D's response to uniform
  fields, LS-27), rebuilt at every Newton correction, applied on the right (for Spectral LS, as CG's
  preconditioner). On by default for CP, off for Spectral LS (where it changes nothing).
- **Iteration counts:** "Newton its" are corrections, "Krylov its" the inner (GMRES, or CG for Spectral LS)
  iterations summed. "Online" is
  `solve_step` wall time; "Offline" is each method's operator build as if it ran alone (cached E and P report their
  stored `build_seconds`).
- **Failed load steps** raise a `LoadPathAbandoned` subclass; that method stops and the others carry on, except on
  the first step, which re-raises.
- **Known differences by design:** LS methods (FFT-TFA included) start each step from the previous strain plus the
  uniform increment; TFA starts from the exact elastic predictor, so LS spends Newton iterations on elastic steps.

## Status

The roadmap is in the README's "Status and roadmap"; keep the two in step. Mixed partitions must stay O(M log M).
Tried and rejected, not to redo without the user asking: block-Jacobi preconditioning on top of the Fourier one (CP
stopped less accurately), geometric (interface-aware) split rules for mixed partitions, single-phase-only partitions,
and CP's cluster eigenstrains as Newton unknowns with a polarization-form correction (more Newton iterations, a 20–50×
less accurate stop; LS note section 7.6). Declined: polarization shapes from the elastic strain concentration plus
one polarization per piece (needs offline fine-mesh solves, and a near/far split of the interaction to be fast); the
user wants a richer polarization that needs neither (README, "Open").

Known behaviour, model error rather than bugs (FFT-TFA's in the LS note, section 10.5):
- **FFT-TFA's truncated interaction fails the patch test** where a partition holds several pieces (a uniform
  polarization gets a spurious strain, 16 % with own blocks on the check fiber); neighbour blocks can make results
  worse (k-means laminate 2 % → 43 % worst cluster, on a matrix split k-means drew from GMRES's solver noise; with CG
  that laminate's uniform matrix gets one cluster, see the README's open points); the cut laminate is 25 % off in its worst cluster. The
  consistent variant (section 10.5) passes the patch test but is less accurate with neighbours on the 126-fiber cell
  and the sphere (2026-10-29 study, same cost and stopping accuracy); not adopted unless the user decides otherwise.
- **CP's σ33 is low along a z-invariant fiber** (about 9 % at 16² partitions, 2 % at 32²): mixed partitions share
  strain near equal stress, but ε33 is uniform there. C0 cannot fix it (it must stay below the matrix's tangent);
  finer partitions do. Imposing each partition's ε33 on its pieces would fix it for z-invariant shapes only; the user
  declined that: they want a fix that holds for any geometry, if one exists.
- **k-means clusters break the cell's mirror symmetry,** so normal loading gives a small spurious macroscopic shear
  (about 3e-4 of the normal stress).
- **Element-level LS rang along high-contrast interfaces** (seen with the FE kernel, since removed): an alternation
  near the inclusion's corners that refinement did not reduce and a global measure misses; inspect fields locally.
