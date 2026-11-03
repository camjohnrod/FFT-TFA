# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase comparing reduced-order methods for elastoplastic composites; the README explains the problem,
the three grids (elements, partitions, clusters) and the methods. Use the method names exactly, and in this order
everywhere (table, figures, code): **FFT-TFA**, **Spectral LS**, **TFA**. FFT-TFA is the method under development. CP,
the earlier method, was removed (git tag `pre-cleanup-2`); "fine LS", "coarse LS", "LS Newton" and "MAIN_*" are retired
names, and so are "piece" and "region", since every cluster now lies in one partition. FFT-TFA and TFA share their
material model (clusters, eigenstrains, update at the average strain) and differ in how the clusters interact: FFT-TFA
averages the LS operator over its clusters with the matrix as C0, exact in the matrix but with the inclusion's
polarization uniform per cluster (TFA resolves it), truncates the interaction to nearby partitions, and uses the
Moulinec–Suquet kernel instead of FE. At full reach FFT-TFA is SCA's clustered LS equation with C0 = L_m (LS note,
section 5.4). Neither difference is a bug.

Read the relevant note before changing solver math. Code names map onto the notes' symbols and numbered equations
(FT-* in `notes/2_ls_model.md` section 5, LS-* in its sections 2–4, ER-* in `notes/3_tfa_baseline.md`), which code
comments cite. Comments cite equations and papers, never the history of a decision.

## The user's standing decisions

- **All knobs live in `code/config.py`, and only there**: a new knob goes there with a comment, never as a literal
  elsewhere, solver internals and check tolerances included. Exceptions: plot appearance in `plots.py`, check problems
  in `check.py`. config.py is meant to be understood at a glance: its first screen is the problem (one name from each list in section 4)
  and what to compare; then results, the lists, solver details, checks. config.py is often left at a local study
  configuration; "default" means `git show HEAD:code/config.py`. The user edits it while you work: change it by targeted
  edits, never by rewriting it from an earlier read.
- **What runs is a list of entries per method** (`fft_tfa`, `spectral_ls`, `tfa`), empty to switch a method off. Each
  entry holds that method's own settings, so nothing downstream reads a global partition or cluster count.
  `elements_per_side` is one fine mesh shared by FFT-TFA and TFA; Spectral LS's elements are its own.
- **FFT-TFA is separate from Spectral LS**: `fft_tfa_model.py` uses only `ls_model`'s shared kernel code
  (`apply_on_lattice`, the Moulinec–Suquet kernel, `get_P0_transformed`, the block helpers). Its reference medium is
  always the matrix (no setting), and its truncated interaction is FT-2/FT-3 as approved; the consistent variant of LS
  note section 5.5 is measured, not adopted. `reach` is per entry.
- **Clusters lie inside partitions**: k-means inside each partition's part in each phase, or those parts themselves. No
  smoothing of the features. Cell-wide k-means was removed: it breaks the memory advantage that is FFT-TFA's selling
  point, by fragmenting into many parts per partition.
- **Spectral LS's reference stiffness is a choice** (`C0` in `spectral_ls`, `"midpoint"` or `"matrix"`), both kept so as
  not to flatter FFT-TFA: its answer does not depend on it, but `"matrix"` was 17 % slower at contrast 100. So LS-24 stays.
- **Fairness between methods:** each gets its best setting from the same principles, justified from the LS
  literature only (no FE papers), all share the Newton–Krylov settings and thread count, and all stop at the same
  accuracy (LS-24; FFT-TFA's residual is already in matrix units, FT-5), which `check.py` checks. The papers are in
  `references/`, with one note per paper read cover to cover in `references/reading_notes/`; cite from those, never
  from a title or abstract.
- **Small codebase:** one file per method and concept, the current ten; don't add modules without asking, and don't bring
  back what the cleanups removed (tags `pre-cleanup` and `pre-cleanup-2`: fixed-point solvers, FFT-preconditioned TFA, the
  FE kernel for P0, the coarse LS lattice, CP, the Fourier preconditioner, softening) without the user asking. Field error
  measures against the reference were left for later (a fair field error across different meshes is not settled, see the
  README's roadmap).
- **The user has said they don't really understand LS-24.** When it comes up, explain it from the LS note's section 4.3
  before building on it.

## Running the code

```bash
python code/run.py                # solve config.py's entries; a new folder in code/output/
python code/run.py preview        # what that would need, and the cross-section; solves nothing
python code/run.py replot latest  # redraw a saved run with config.py's Figures settings
python code/run.py cache          # list code/cache/
python code/check.py              # the test, about a minute
python code/check.py save         # the same, then save the results
```

There is no pytest suite and no way to run one check problem alone. `check.py` solves `"laminate"`, `"cut laminate"`,
`"fiber"`, `"z laminate"`, `"sphere"`, `"k-means sphere"` and `"random fibers"` with every method. On the laminates every
method must match the exact solution to `laminate_tolerance` (Spectral LS on its own coarser elements on the laminate,
finer on the z laminate, both resolving the layer exactly), except FFT-TFA on the cut laminate (`inexact_methods`: printed
as INFO); FFT-TFA at full reach must match it on every laminate, and with TFA also on the laminate with each partition's
matrix part cut into two clusters (`get_split_clusters`). On every problem FFT-TFA's interaction at full reach must equal
the element-level LS operator averaged over its clusters (`get_interaction_error`, to `interaction_check_tolerance`): that
catches a wrong offset, pair count, correction block or average. On the fiber and the sphere every method must stop
within `stopping_accuracy_tolerance` of its own solution at `stopping_check_tolerance` (not with k-means clusters of a few
elements, whose clustered problem can have several solutions, for TFA too). On the fiber the reference is on at the
working resolution and must equal Spectral LS bit for bit and reload from its cache unchanged (`check_reference`). Every
method must complete every step and reproduce `code/check_saved_results.npz` (untracked) bit for bit, at the
`thread_count` it was saved with; with none saved, check.py skips that test and says so, so run `check.py save` on a clean
tree first. For a refactor: `check.py save` before, `check.py` after.

Not covered by `check.py`, so recheck by hand after touching them: FFT-TFA's block-Jacobi preconditioner must leave
answers alone (compare with it on and off), FFT-TFA's Newton product must match finite differences of its residual (FT-6),
its assembled diagonal blocks (FT-7) those of the Jacobian probed column by column, stopping accuracy on the default
problem (against a solve at tolerance 1e-11), and `run.py`'s commands, saving and figures (`check.py` draws nothing).

`check.py` sets its `pinned_inputs` on the `config` module in memory, with a temporary cache folder. This works only
because **no module reads `config` at import time**: no `config.x` in default arguments or module-level constants.
A renamed config input must be renamed in `pinned_inputs` too.

`python` is the base miniforge environment's (Python 3.12, with `numpy`, `scipy`, `matplotlib` and `tqdm`); the
`fenics-env`/`fenicsx-env` conda envs are unrelated. Entry points import `config` before numpy, since config.py sets the
thread environment variables (and so never imports numpy itself); online FFTs use `scipy.fft` with `workers=config.thread_count`. Linux only (`/proc` for
peak memory, `fork` for the entries' processes).

TFA's offline step builds a dense P of (6 × TFA's clusters)² doubles, cached in `code/cache/` under a hash of its
parameters (`cell.get_cache_path`). Bump `tfa_model.cache_version` when a change alters E or P. The reference
(`run.get_reference`) is cached the same way, under the shared geometry key (`cell.get_geometry_parameters`) plus its
mesh, the materials, the load path and the solver inputs that change its answer; bump `run.reference_cache_version` when
a change alters it. A new input that changes the reference must be added to its key, or a stale one is reused silently.
Estimate the cost of large configurations (`preview`) and run them in the background.

## Architecture

Imports go one way: `config` → `material` → `newton` → `cell` → `ls_model`, `tfa_model` → `fft_tfa_model` (from
`ls_model`) → `run` → `check`. `plots.py` imports no project module; only `run.py` calls it.

| Module | Contents |
|---|---|
| `config.py` | every input, `strain`, `check_inputs()`, folders |
| `run.py` | `Entry`, `get_entries`, the builders, `get_load_path`, `run_load_path`, `run_in_own_process`, `solve_all`, the cached reference (`Reference`, `get_reference`), `save_run`, `get_saved_run`, `runs.csv`, tables, macroscopic stress errors, `draw_run`, and the commands (`run`, `preview`, `replot`, `list_cache`) |
| `fft_tfa_model.py` | FFT-TFA: `Interaction`, correction blocks (`get_correction_blocks`, FT-3; stored as 6 × 6 BSR blocks), `get_interaction`, `get_strain_change` (FT-2 with FT-4), the block-Jacobi preconditioner's diagonal blocks (`get_own_strain_change`, FT-7), `FFTTFASolver` (FT-5, FT-6) |
| `cell.py` | shapes (`is_inside_inclusion`, `shapes_constant_along_z`, `get_fiber_centres`), `Discretization`, `Mesh`, `Materials`, `Clusters`, `label_partition_phases`, `label_partition_k_means`, `get_clusters`, `get_memory_MB`, the cache (`get_cached`) |
| `newton.py` | `Solver` (the one Newton loop and line search), `solve_by_gmres`, relative residual, `SolverDidNotConverge` |
| `ls_model.py` | `apply_on_lattice` (FFT convolution), C0, P0 and its partition average (`get_P0_transformed`), LS-24, `Lattice`, `SpectralLSSolver` (its corrections by CG, LS-28), `get_elastic_strain_concentration` (k-means features) |
| `tfa_model.py` | element matrices, `FEModel`, E and P (`compute_E_and_P`) and their cache (`get_E_and_P`), column-restricted P products, `TFASolver` with the elastic reset |
| `material.py` | vectorized J2 radial return, its sensitivity H_μ and algorithmic stiffness L_t, `apply_blocks` |
| `plots.py` | the figures: `method_styles` keyed by method name, `get_line_styles` (tints for a method's several entries); the cross-section's `boundary_styles` (voxel, cluster, partition lines) and k-means cluster colours |
| `check.py` | the checks and the laminate's exact solution |

A method is a `newton.Solver` subclass with `evaluate` (its residual: LS-18, FT-5, ER-4) and `get_newton_correction` (LS-20
with LS-28, FT-6, ER-21). `start_step` (TFA's elastic predictor), `measure_residual` (Spectral LS's LS-24) and
`divergence_limit` (TFA's) may change. Figures are drawn only from what a run saved (`run.json`, `results.npz`), which is
why `replot` needs no solve.

## Non-obvious points

- **Strain convention (ER-2):** engineering shear, `(ε11, ε22, ε33, 2ε12, 2ε23, 2ε13)`.
- **Convergence (LS-24):** Spectral LS's residual is a strain of its reference medium, so convergence and the line search
  judge M_m⁻¹ r (the correction an all-matrix cell would need, `measure_residual`), while the Newton correction still solves
  for r. `get_relative_residual` weights shear by ½ and divides by the imposed strain's norm, never by less than
  `residual_strain_scale_floor`.
- **No global lattice size online:** solvers, FFTs and the load path take sizes from the arrays they're given, never from
  config.
- **Each entry runs in its own process** (`run_in_own_process`): that is what makes its peak memory its own and lets
  `memory_limit_GB` stop only it. Reported peaks are above the process's start, leaving out Python and its modules. A
  cached E and P report the seconds and memory they took when computed.
- **Correction blocks** come from FFT cross-correlations of the clusters' indicators on twice the partition, against the
  element kernel on the mesh's own element lattice.
- **Lattices are 3D, indexed `[z, y, x]`.** A z-constant shape has one element layer and columns of partitions.
  `apply_on_lattice` transforms only axes longer than one (length-1 transforms are identities but slow).
- **Half spectrum:** `P0_transformed` holds only `rfftn`'s half (shape `(N_z, N, N//2 + 1, 6, 6)`), transformed back
  with `irfftn(..., s=(N_z, N, N))`. Anything new reading it must do the same.
- **Clusters:** `cell.Clusters` holds the material states (`volume` as a share of the cluster's partition), ordered by
  partition. Clusters have unequal volumes: every cell average goes through `Clusters.average`. Every method reports stress
  and plastic state per cluster.
- **A new shape** goes in `is_inside_inclusion`, `check_geometry`, the geometry cache parameters
  (`cell.get_geometry_parameters`: every parameter of the shape, or a stale E, P or reference is reused silently) and, if
  constant along z, `shapes_constant_along_z`. `"random fibers"` keys its cache on the placed radius and centres
  (`get_fiber_centres`), which stand for the count, volume fraction, gap and seed.
- **TFA's P** is stored column-major so consecutive yielding clusters form a contiguous column block
  (`apply_P_to_clusters`). Don't fancy-index `P[:, columns]`: the copy costs more than the full product.
- **k-means** runs on each element's elastic strain concentration from six elastic Spectral LS steps on the mesh's elements,
  per partition's part in each phase (one cluster per part is exactly partition phases, with no features needed), with a
  fixed seed; a part gets no more clusters than distinct features, and empty clusters are dropped. Its time counts in the
  offline time of each entry that uses it.
- **Iteration counts:** "Newton its" are corrections, "Krylov its" the inner (GMRES, or CG for Spectral LS) iterations
  summed. "Online" is `solve_step` wall time; "Offline" is each entry's operator build as if it ran alone.
- **A failed load step** raises `SolverDidNotConverge`; that entry stops and the others carry on.
- **Known differences by design:** LS methods (FFT-TFA included) start each step from the previous strain plus the
  uniform increment; TFA starts from the exact elastic predictor, so LS spends Newton iterations on elastic steps.

## Status

The roadmap is in the README's "Status and roadmap"; keep the two in step. Mixed partitions must stay O(M log M).
Memory matters as much as time: FFT-TFA must keep needing less than SCA, TFA and Spectral LS, which holds only for
clusters local to partitions (README, "Memory"); weigh new features by how their memory scales, in 3D too.
Tried and rejected, not to redo without the user asking: for FFT-TFA (README, open points), one fiber part per fiber in
each partition (stiffer and less accurate), face neighbours only (well short of all neighbours locally, no help in 3D),
and the consistent variant again, against full reach (no better). Declined: polarization shapes from the elastic strain
concentration plus one polarization per cluster (needs offline fine-mesh solves, and a near/far split of the interaction
to be fast); the user wants a richer polarization that needs neither (README, "Open").

Known behaviour, model error rather than bugs (LS note, section 5.5):
- **FFT-TFA's truncated interaction fails the patch test** where a partition holds several clusters (a uniform
  polarization gets a spurious strain, 16 % at reach 0 on the check fiber); the cut laminate is 25 % off in its worst
  cluster. The consistent variant passes the patch test but is less accurate with neighbours on the 126-fiber cell and the
  sphere; not adopted unless the user decides otherwise.
- **k-means clusters break the cell's mirror symmetry,** so normal loading gives a small spurious macroscopic shear
  (about 3e-4 of the normal stress).
- **Element-level LS rang along high-contrast interfaces:** an alternation near the inclusion's corners that refinement did
  not reduce and a global measure misses; inspect fields locally.
