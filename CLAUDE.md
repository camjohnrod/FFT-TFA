# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A teaching and research codebase comparing reduced-order methods for elastoplastic composites; the README explains
the problem, the three grids (elements, partitions, clusters) and the methods. Use the method names exactly, and in
this order everywhere (table, figures, code): **FFT-TFA**, **SCA**, **Spectral LS**, **TFA**. FFT-TFA is the method
under development. "piece" and "region" are not used: every cluster lies in one partition.

FFT-TFA and TFA share their material model (clusters, eigenstrains, update at the average strain) and differ in how
the clusters interact: FFT-TFA averages the LS operator over its clusters with the matrix as C0, exact in the matrix
but with the inclusion's polarization uniform per cluster (TFA resolves it), summarizes the other partitions by
polynomial moments up to a `degree`, and uses an LS element kernel instead of FE. At full degree FFT-TFA is SCA's
clustered LS equation with C0 = L_m (lecture, sections 5–7). Neither difference is a bug. SCA solves that clustered
equation exactly, on k-means clusters over the whole cell (one partition), through a dense D (lecture, appendix E).

The theory is one note, `notes/fft_tfa_lecture.md`: FFT-TFA in its body; the material law, Spectral LS, TFA, SCA and
stress-free directions in its appendix. Read the relevant section before changing solver math; keep it one file. Code
names map onto its symbols and its numbered equations (LS-*, FT-*, HS-*, ML-*, TFA-*, SC-*, SF-*), which code comments cite.
`notes/research_log.md` is the dated record of findings, open questions and what was tried and set aside; read it
before proposing a direction, and don't redo what it sets aside without a new reason.

## Rules for changes

- **The code is for teaching:** correct first, then obvious to use and to read, nothing convoluted; reuse general
  helpers; optimize only where the code stays readable; a fair comparison between methods. Each module opens with a
  docstring saying what it does in plain words, functions and classes carry docstrings, and names are short but say
  what a thing is.
- **Docs and comments stand on their own:** comments cite equations and papers, never the history of a decision, and
  no doc but the research log carries dates or the story of how something came about.
- **All knobs live in `code/config.py`, and only there:** a new knob goes there with a comment, never as a literal
  elsewhere, solver internals and check tolerances included. Exceptions: plot appearance in `plots.py`, check problems
  in `check.py`. config.py's first screen is the problem (one name from each list in its section 4) and what to
  compare; then results, the lists, solver details, checks. The committed config.py is a small, fast first run; the
  working copy is often a local study configuration, so "default" means `git show HEAD:code/config.py`. Change it by
  targeted edits, never by rewriting it from an earlier read: it may have been edited meanwhile.
- **What runs is a list of entries per method** (`fft_tfa`, `sca`, `spectral_ls`, `tfa`), empty to switch a method
  off. Each entry holds that method's own settings, so nothing downstream reads a global partition or cluster count. A
  list of lists splits a method into schemes (`config.get_schemes`; names from the settings, `run.get_entries`): the
  error-against-time plot draws one line per scheme in the order listed, the other line figures each scheme's last
  (finest) entry only; a scheme's colour is a shade of its method's (`plots.scheme_shades`). An entry listed in two
  schemes is solved once. `clustered_mesh` is one fine mesh shared by FFT-TFA, SCA and TFA; Spectral LS's elements
  are its own.
- **Every method is a `newton.Solver`** built from its clusters and its own settings (`FFTTFASolver(clusters,
  degree)`, `SCASolver(clusters, C0)`, `SpectralLSSolver(clusters, C0)`, `TFASolver(clusters)`); keep new methods and
  features to that shape.
- **FFT-TFA is separate from Spectral LS:** `fft_tfa_model.py` uses only `ls_model`'s shared code (the element kernel,
  `apply_on_lattice`, `to_fourier_blocks`) and `newton`'s block helpers. Its reference medium is always the matrix (no
  setting; lecture, section 4). Its interaction: each partition's own clusters exactly (the corrections, FT-2), every
  other partition through its polynomial moments up to `degree` (`"full"` is exact), by one FFT; the corrections act
  on the polarization's deviation from its cell mean, so a uniform polarization causes no strain.
- **Clusters lie inside partitions:** k-means inside each partition's part in each phase, or those parts themselves,
  with no smoothing of the features. Cell-wide clusters fragment into many parts per partition, and FFT-TFA's cost
  grows with the parts (research log). SCA's clusters are cell-wide by definition: `run.get_entry_clusters` gives it
  one partition. `k_means_method` applies to every method alike: `"plain"`, or `"nested"` (bisecting, far faster,
  clusters nest under refinement); check.py pins `"plain"` and runs one problem nested.
- **One element kernel for every LS method** (`element_kernel`: `"Willot"`, the default, or `"Moulinec-Suquet"`;
  `ls_model.get_element_kernel`, lecture appendix B.1): FFT-TFA, SCA, Spectral LS, the reference and the k-means
  features share it, never one per entry, so the comparison stays fair. TFA is FE whatever it says. check.py runs
  every problem on Willot's and the laminate and the fiber again on Moulinec–Suquet's.
- **Spectral LS's reference stiffness is a setting** (`C0`, `"midpoint"` or `"matrix"`): its answer does not depend
  on it, but its speed does, so both are kept and LS-10 judges convergence the same way for either.
- **Fairness between methods:** each gets its best setting from the same principles, justified from the LS
  literature only (no FE papers), all share the Newton–Krylov settings and thread count, and all stop at the same
  accuracy (LS-10; FFT-TFA's residual is already in matrix units, FT-5; SCA's is judged as it is), which `check.py`
  checks. SCA uses Newton–GMRES too, though Liu–Bessa–Liu solve directly: GMRES is faster past a few dozen clusters
  (lecture E.4). The papers are in `references/`; cite only papers read cover to cover, never from a title or
  abstract.
- **Stress-free directions** (`unloaded_directions`; lecture appendix F, SF-*): `"fixed"` is strain control, the
  default; `"free"` makes the macroscopic stress zero in the components the load path never strains. Only zero
  stress, no prescribed values. It is one term per method, the mean strain the cell takes there (SF-2), with no new
  unknown and no outer loop; k-means features stay strain-controlled. `"fixed"` must stay bit for bit what it was:
  the term is then an exact zero.
- **Small codebase:** one file per method and concept, the current twelve; don't add modules without asking, and
  don't bring back what the cleanups removed (research log, "Tried and set aside") without being asked. Field error
  measures against the reference are not settled (a fair field error across different meshes).
- **LS-10 is the least intuitive part of the solvers.** When it comes up, explain it from the lecture's appendix B.3
  before building on it.

## Running the code

From the repository root:

```bash
python code/run.py                # solve config.py's entries; a new folder in code/output/
python code/run.py preview        # what that would need, and the cross-section; solves nothing
python code/run.py replot latest  # redraw a saved run with config.py's figure settings (section 3)
python code/run.py cache          # list code/cache/
python code/check.py              # the test, under a minute
python code/check.py save         # the same, then save the results
python notes/figures/make_figures.py  # the lecture's figures (under a minute)
```

There is no pytest suite and no way to run one check problem alone; check.py's docstring lists its checks. Its last
check compares every result with `code/check_saved_results.npz` (untracked) bit for bit, at the `thread_count` it was
saved with, and is skipped if none is saved. For a refactor: `check.py save` on a clean tree before, `check.py` after.

Not covered by `check.py`, so recheck by hand after touching them: FFT-TFA's block-Jacobi preconditioner must leave
answers alone (compare with it on and off), FFT-TFA's Newton product must match finite differences of its residual
(FT-6), its assembled diagonal blocks (FT-7) those of the Jacobian probed column by column, stopping accuracy on the
default problem (against a solve at tolerance 1e-11), and `run.py`'s commands, saving and figures (`check.py` draws
nothing).

`check.py` sets its `pinned_inputs` on the `config` module in memory, with a temporary cache folder. This works only
because **no module reads `config` at import time**: no `config.x` in default arguments or module-level constants.
A renamed config input must be renamed in `pinned_inputs` too.

`python` is the base miniforge environment's (Python 3.12, with `numpy`, `scipy`, `matplotlib` and `tqdm`); the
`fenics-env`/`fenicsx-env` conda envs are unrelated. Entry points import `config` before numpy, since config.py sets
the thread environment variables (and so never imports numpy itself); online FFTs use `scipy.fft` with
`workers=config.thread_count`. Linux only (`/proc` for peak memory, `fork` for the entries' processes).

TFA's offline step builds a dense P of (6 × TFA's clusters)² doubles, cached in `code/cache/` under a hash of its
parameters (`cell.get_cache_path`). Bump `tfa_model.cache_version` when a change alters E or P. The reference
(`run.get_reference`) is cached the same way, under the shared geometry key (`cell.get_geometry_parameters`) plus its
mesh, the materials, the load path and the solver inputs that change its answer; bump `run.reference_cache_version`
when a change alters it. A new input that changes the reference must be added to its key, or a stale one is reused
silently. Estimate the cost of large configurations (`preview`) and run them in the background.

## Architecture

Imports go one way: `config` → `material` → `newton` → `cell` → `ls_model`, `tfa_model` → `fft_tfa_model`,
`sca_model` (from `ls_model`) → `report` → `run` → `check`. `plots.py` imports no project module; only `report.py`
calls it.

The README's "Reading the code" table says what each module holds, in reading order. The shared types: `cell.Mesh`
(elements and partitions, shapes along [z, y, x]), `cell.Clusters` (its mesh, the material states, ordered by
partition), `run.LoadPathResult` (a solver's results along the load path; the reference's too).

A method is a `newton.Solver` subclass with `evaluate` (its residual: LS-7, FT-5, SC-1, TFA-3) and
`get_newton_correction` (LS-8 with LS-9, FT-6, SC-4, TFA-5). `start_step` (TFA's elastic predictor),
`measure_residual` (Spectral LS's LS-10), `divergence_limit` (TFA's) and `solve_step` (SCA's self-consistent loop) may
change. `run.py` solves; `report.py` saves a run and draws it, only from what was saved (`run.json`, `results.npz`),
which is why `replot` needs no solve: `results.npz` holds the macroscopic stress at every step and maps of the
mid-thickness layer (cluster, partition, phase; stress and accumulated plastic strain at the turning points), so any
field in `report.fields` can be added and redrawn. Runs saved in an older format, with per-cluster arrays, cannot
be replotted.

## Non-obvious points

- **Naming:** an element is matrix or inclusion, its `phase` (0, 1), never "material id" or "fiber". Arrays named
  `<owner>_<what>` hold the `<what>` of every `<owner>`: `element_phase`, `element_partition`, `element_cluster`,
  `cluster_partition`. Cell averages are `macro_strain`, `macro_stress`. `Clusters.share` is each cluster's share of
  its partition's volume, the lecture's v(a); `Clusters.element_counts` its elements. Entry labels write `m=` and `i=`
  for the matrix's and inclusion's clusters per partition.
- **Cache keys and cached arrays keep their own names:** the key strings in `cell.get_geometry_parameters`,
  `run.get_reference_parameters` and `tfa_model.get_E_and_P` (`"domain_side_length"`, `"elements_per_side"`,
  `"discretization"` for `element_kernel`, ...) are hashed into the cache file names, and the cached reference stores
  `material_ids` and `macroscopic_stress`. Renaming any of them, or changing a value's type (they are Python ints),
  orphans every existing cache (a reference can take 30 min); change them only with a version bump, on purpose.
  `report.draw_run` reads run.json files without `unloaded_directions` or `element_kernel` as `"fixed"` and
  Moulinec–Suquet's.
- **Strain convention (lecture, appendix I):** engineering shear, `(ε11, ε22, ε33, 2ε12, 2ε23, 2ε13)`; `newton.tensor_norm_weights`
  and `stress_norm_weights` give tensor norms of strains and stresses.
- **Spectral LS projects each Newton correction by P0** once CG stops: outside P0's range J is not symmetric, and the
  round-off CG leaves there would build up over Newton iterations until CG stalls (strongly plastic steps with lateral
  contraction, tight tolerances).
- **Under stress-free loading** the stiff inclusion carries far less stress, so a strain residual is a larger share
  of its stress: `check.py`'s stress-free problems run at a tighter `convergence_tolerance`. SCA's self-consistent C0
  with one cluster per phase settles too slowly there (more than `sca_max_reference_updates` solves).
- **Convergence (LS-10):** Spectral LS's residual is a strain of its reference medium, so convergence and the line
  search judge M_m⁻¹ r (the correction an all-matrix cell would need, `measure_residual`), while the Newton correction
  still solves for r. `get_relative_residual` weights shear by ½ and divides by the imposed strain's norm, never by
  less than `residual_strain_scale_floor`.
- **FFT-TFA's moment kernels** hold one 6p × 6p block per frequency of the partition lattice (`rfftn`'s half, over
  the axes longer than one), p the polynomials of the degree: memory grows as p² × partitions, independent of the
  clusters per partition; in 3D p = 4, 10, 20 for degrees 1–3 (`preview` reports it). The basis is orthonormal on a
  partition's elements, constant first, so the degree-0 moment is a scaled mean. It is built as products of
  orthonormal polynomials along each axis (by recurrence, `get_axis_polynomials`), never by Gram–Schmidt on monomials,
  which loses the high degrees to round-off: at full degree it must hold one polynomial per element, which `check.py`
  checks on partitions up to 16 across.
- **No global lattice size online:** solvers, FFTs and the load path take sizes from the arrays they're given, never
  from config.
- **Each entry runs in its own process** (`run_in_own_process`): that is what makes its peak memory its own and lets
  `memory_limit_GB` stop only it. Reported peaks are above the process's start, leaving out Python and its modules.
- **Lattices are 3D, indexed `[z, y, x]`.** A z-constant shape has one element layer and columns of partitions.
  `apply_on_lattice` transforms only axes longer than one (length-1 transforms are identities but slow).
- **Half spectrum:** Fourier blocks (Spectral LS's `P0_transformed`, FFT-TFA's `moment_kernels`) hold only `rfftn`'s
  half (shape `(N_z, N, N//2 + 1, b, b)`), transformed back with `irfftn(..., s=(N_z, N, N))`; `apply_on_lattice`
  does both. Anything new reading them must do the same.
- **Clusters have unequal volumes:** every cell average goes through `Clusters.average`. Every method reports stress
  and plastic state per cluster.
- **A new shape** goes in `is_inside_inclusion`, `check_geometry`, the geometry cache parameters
  (`cell.get_geometry_parameters`: every parameter of the shape, or a stale E, P or reference is reused silently) and,
  if constant along z, `shapes_constant_along_z`. `"random fibers"` and `"random spheres"` key their cache on the
  placed radius and centres (`get_random_centres`), which stand for the count, volume fraction, gap and seed.
- **TFA's P** is stored column-major so consecutive yielding clusters form a contiguous column block
  (`apply_P_to_clusters`). Don't fancy-index `P[:, columns]`: the copy costs more than the full product.
- **Iteration counts:** "Newton its" are corrections, "Krylov its" the inner (GMRES, or CG for Spectral LS) iterations
  summed. "Online" is `solve_step` wall time; "Offline" is building the entry's solver (its clusters included), work
  loaded from the cache counted as when it was computed (`Solver.offline_from_cache`, set by TFA's E and P).
- **Known differences by design:** LS methods (FFT-TFA included) start each step from the previous strain plus the
  uniform increment; TFA starts from the exact elastic predictor, so LS spends Newton iterations on elastic steps.
- **SCA's residual is in increments** from the last accepted step (SC-1, Liu–Bessa–Liu's eq. 28), not total: with a
  self-consistent C0 the total form makes C0 move the whole solution, the fit then has no root and the loop runs away.
  Its self-consistent C0 is isotropic with Liu's two Lamé constants (their Box II), so D is built once per fixed
  kernel (two parts with Willot's kernel, three with Moulinec–Suquet's, `sca_model.get_kernel_parts`) and assembled at
  each refit; the pure-shear case keeps λ. Its Newton and Krylov counts sum every re-solve of a step. D is not cached.

## Status

Findings, open points and the roadmap live in `notes/research_log.md` only. Two constraints hold for any new feature:
mixed partitions must stay O(M log M), and memory matters as much as time: FFT-TFA must keep needing less than SCA,
TFA and Spectral LS, which holds only for clusters local to partitions; weigh new features by how their memory scales,
in 3D too.

Known behaviour, model error rather than bugs: FFT-TFA's far field converges with the degree, not with the partitions
alone, and partitions should be at most about half an inclusion across (lecture, section 10). Also:
- **k-means clusters break the cell's mirror symmetry,** so normal loading gives a small spurious macroscopic shear
  (about 3e-4 of the normal stress).
- **Element-level LS rings along high-contrast interfaces:** an alternation near the inclusion's corners that
  refinement does not reduce and a global measure misses; inspect fields locally.
