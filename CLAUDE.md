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
polarization uniform per cluster (TFA resolves it), summarizes the other partitions by polynomial moments up to a
`degree`, and uses the Moulinec–Suquet kernel instead of FE. At full degree FFT-TFA is SCA's clustered LS equation with
C0 = L_m (`notes/fft_tfa_lecture.md`, sections 6–7). Neither difference is a bug.

The theory is one note, `notes/fft_tfa_lecture.md`: FFT-TFA in its body, the material law, Spectral LS and TFA in its
appendix. Read the relevant section before changing solver math; keep it one file. Code names map onto its symbols and
its numbered equations (FT-*, LS-*, ER-*), which code comments cite. Comments cite equations and papers, never the history of a decision. The code is meant to read as an
introduction for someone new to it: each module opens with a docstring saying what it does in plain words, functions and
classes carry docstrings, and names are short but say what a thing is (the user's words: "tight and concise, but it
should communicate what it does").

## The user's standing decisions

- **All knobs live in `code/config.py`, and only there**: a new knob goes there with a comment, never as a literal
  elsewhere, solver internals and check tolerances included. Exceptions: plot appearance in `plots.py`, check problems
  in `check.py`. config.py is meant to be understood at a glance: its first screen is the problem (one name from each list in section 4)
  and what to compare; then results, the lists, solver details, checks. config.py is often left at a local study
  configuration; "default" means `git show HEAD:code/config.py`. The user edits it while you work: change it by targeted
  edits, never by rewriting it from an earlier read.
- **What runs is a list of entries per method** (`fft_tfa`, `spectral_ls`, `tfa`), empty to switch a method off. Each
  entry holds that method's own settings, so nothing downstream reads a global partition or cluster count. A list of
  lists splits a method into schemes, chosen by the user (`config.get_schemes`; names from the settings, `run.get_entries`):
  the error-against-time plot draws one line per scheme in the order listed, the other line figures each scheme's last
  (finest) entry only, never several shades of one scheme; a scheme's colour is a shade of its method's
  (`plots.scheme_shades`). An entry listed in two schemes is solved once.
  `clustered_mesh` (elements per side) is one fine mesh shared by FFT-TFA and TFA; Spectral LS's elements are its own.
- **The code is for teaching** (the user's goals, 2026-11-10): correct first, then obvious to use and to read, nothing
  convoluted; reuse general helpers; optimize only where the code stays readable; a fair comparison between methods.
  Every method is a `newton.Solver` built from its clusters and its own settings (`FFTTFASolver(clusters, degree)`,
  `SpectralLSSolver(clusters, C0)`, `TFASolver(clusters)`); keep new methods and features to that shape.
- **FFT-TFA is separate from Spectral LS**: `fft_tfa_model.py` uses only `ls_model`'s shared code (the Moulinec–Suquet
  kernel, `apply_on_lattice`) and `newton`'s block helpers. Its reference medium is always the matrix (no setting). Its
  interaction: each partition's own clusters exactly (the corrections, FT-3), every other partition through its
  polynomial moments up to `degree` (per entry; `"full"` is exact), by one FFT; the corrections act on the
  polarization's deviation from its cell mean, so a uniform polarization causes no strain.
- **Clusters lie inside partitions**: k-means inside each partition's part in each phase, or those parts themselves. No
  smoothing of the features. Cell-wide k-means was removed: it breaks the memory advantage that is FFT-TFA's selling
  point, by fragmenting into many parts per partition.
- **Spectral LS's reference stiffness is a choice** (`C0` in `spectral_ls`, `"midpoint"` or `"matrix"`), both kept so as
  not to flatter FFT-TFA: its answer does not depend on it, but `"matrix"` was 17 % slower at contrast 100. So LS-24 stays.
- **Fairness between methods:** each gets its best setting from the same principles, justified from the LS
  literature only (no FE papers), all share the Newton–Krylov settings and thread count, and all stop at the same
  accuracy (LS-24; FFT-TFA's residual is already in matrix units, FT-5), which `check.py` checks. The papers are in
  `references/`; cite only papers read cover to cover, never from a title or abstract.
- **Small codebase:** one file per method and concept, the current ten; don't add modules without asking, and don't bring
  back what the cleanups removed (tags `pre-cleanup`, `pre-cleanup-2` and `pre-cleanup-3`: fixed-point solvers,
  FFT-preconditioned TFA, the FE kernel for P0, the coarse LS lattice, CP, the Fourier preconditioner, softening, FFT-TFA's
  truncation to neighbouring partitions, `reach`) without the user asking. Field error
  measures against the reference were left for later (a fair field error across different meshes is not settled, see the
  README's roadmap).
- **The user has said they don't really understand LS-24.** When it comes up, explain it from the lecture's appendix B.3
  before building on it.

## Running the code

```bash
python code/run.py                # solve config.py's entries; a new folder in code/output/
python code/run.py preview        # what that would need, and the cross-section; solves nothing
python code/run.py replot latest  # redraw a saved run with config.py's Figures settings
python code/run.py cache          # list code/cache/
python code/check.py              # the test, under a minute
python code/check.py save         # the same, then save the results
python notes/figures/make_figures.py  # the lecture's figures (a few minutes)
```

There is no pytest suite and no way to run one check problem alone; check.py's docstring lists its checks. Its last
check compares every result with `code/check_saved_results.npz` (untracked) bit for bit, at the `thread_count` it was
saved with, and is skipped if none is saved. For a refactor: `check.py save` on a clean tree before, `check.py` after.

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

The README's "Reading the code" table says what each module holds, in reading order. The shared types: `cell.Mesh`
(elements and partitions, shapes along [z, y, x]), `cell.Clusters` (its mesh, the material states, ordered by partition),
`run.LoadPathResult` (a solver's results along the load path; the reference's too).

A method is a `newton.Solver` subclass with `evaluate` (its residual: LS-18, FT-5, ER-4) and `get_newton_correction` (LS-20
with LS-28, FT-6, ER-21). `start_step` (TFA's elastic predictor), `measure_residual` (Spectral LS's LS-24) and
`divergence_limit` (TFA's) may change. Figures are drawn only from what a run saved (`run.json`, `results.npz`), which is
why `replot` needs no solve: `results.npz` holds the macroscopic stress at every step and maps of the mid-thickness
layer (cluster, partition, phase; stress and accumulated plastic strain at the turning points), so any field in
`run.fields` can be added and redrawn. Runs saved before 2026-11-10 stored per-cluster arrays and cannot be replotted.

## Non-obvious points

- **Naming:** an element is matrix or inclusion, its `phase` (0, 1), never "material id" or "fiber". Arrays named
  `<owner>_<what>` hold the `<what>` of every `<owner>`: `element_phase`, `element_partition`, `element_cluster`,
  `cluster_partition`. Cell averages are `macro_strain`, `macro_stress`. `Clusters.share` is each cluster's share of its
  partition's volume (c_p). Entry labels write `m=` and `i=` for the matrix's and inclusion's clusters per partition.
- **Cache keys and cached arrays keep old names:** the key strings in `cell.get_geometry_parameters`,
  `run.get_reference_parameters` and `tfa_model.get_E_and_P` (`"domain_side_length"`, `"elements_per_side"`, ...) are
  hashed into the cache file names, and the cached reference stores `material_ids` and `macroscopic_stress`. Renaming
  any of them, or changing a value's type (they are Python ints), orphans every existing cache (a reference can take 30
  min); change them only with a version bump, on purpose.
- **Strain convention (ER-2):** engineering shear, `(ε11, ε22, ε33, 2ε12, 2ε23, 2ε13)`.
- **Convergence (LS-24):** Spectral LS's residual is a strain of its reference medium, so convergence and the line search
  judge M_m⁻¹ r (the correction an all-matrix cell would need, `measure_residual`), while the Newton correction still solves
  for r. `get_relative_residual` weights shear by ½ and divides by the imposed strain's norm, never by less than
  `residual_strain_scale_floor`.
- **FFT-TFA's moment kernels** hold one 6p × 6p block per frequency of the partition lattice (`rfftn`'s half, over the
  axes longer than one), p the polynomials of the degree: memory grows as p² × partitions, independent of the clusters
  per partition; in 3D p = 4, 10, 20 for degrees 1–3, so degree 3 is already expensive (`preview` reports it). The
  basis is orthonormal on a partition's elements, constant first, so the degree-0 moment is a scaled mean and degree 0
  reproduces the partition-averaged kernel. It is built as products of orthonormal polynomials along each axis (by
  recurrence, `get_axis_polynomials`), never by Gram–Schmidt on monomials, which loses the high degrees to round-off: at
  full degree it must hold one polynomial per element, which `check.py` checks on partitions up to 16 across.
- **No global lattice size online:** solvers, FFTs and the load path take sizes from the arrays they're given, never from
  config.
- **Each entry runs in its own process** (`run_in_own_process`): that is what makes its peak memory its own and lets
  `memory_limit_GB` stop only it. Reported peaks are above the process's start, leaving out Python and its modules.
- **Lattices are 3D, indexed `[z, y, x]`.** A z-constant shape has one element layer and columns of partitions.
  `apply_on_lattice` transforms only axes longer than one (length-1 transforms are identities but slow).
- **Half spectrum:** Fourier blocks (Spectral LS's `P0_transformed`, FFT-TFA's `moment_kernels`) hold only `rfftn`'s
  half (shape `(N_z, N, N//2 + 1, b, b)`), transformed back with `irfftn(..., s=(N_z, N, N))`; `apply_on_lattice` does
  both. Anything new reading them must do the same.
- **Clusters:** `cell.Clusters` holds its mesh and the material states, ordered by partition. Clusters have unequal
  volumes: every cell average goes through `Clusters.average`. Every method reports stress and plastic state per cluster.
- **A new shape** goes in `is_inside_inclusion`, `check_geometry`, the geometry cache parameters
  (`cell.get_geometry_parameters`: every parameter of the shape, or a stale E, P or reference is reused silently) and, if
  constant along z, `shapes_constant_along_z`. `"random fibers"` keys its cache on the placed radius and centres
  (`get_fiber_centres`), which stand for the count, volume fraction, gap and seed.
- **TFA's P** is stored column-major so consecutive yielding clusters form a contiguous column block
  (`apply_P_to_clusters`). Don't fancy-index `P[:, columns]`: the copy costs more than the full product.
- **Iteration counts:** "Newton its" are corrections, "Krylov its" the inner (GMRES, or CG for Spectral LS) iterations
  summed. "Online" is `solve_step` wall time; "Offline" is building the entry's solver (its clusters included), work
  loaded from the cache counted as when it was computed (`Solver.offline_from_cache`, set by TFA's E and P).
- **Known differences by design:** LS methods (FFT-TFA included) start each step from the previous strain plus the
  uniform increment; TFA starts from the exact elastic predictor, so LS spends Newton iterations on elastic steps.

## Status

The roadmap is in the README's "Status and roadmap"; keep the two in step. Mixed partitions must stay O(M log M).
Memory matters as much as time: FFT-TFA must keep needing less than SCA, TFA and Spectral LS, which holds only for
clusters local to partitions (README, "Memory"); weigh new features by how their memory scales, in 3D too.
Tried and rejected, not to redo without the user asking: for FFT-TFA (README, open points), one fiber part per fiber in
each partition (stiffer and less accurate). Not worth it: a per-partition block-Jacobi preconditioner (fewer GMRES
iterations, slower overall); removing the exact corrections inside each partition (polynomials cannot follow the sharp
near field: 14–68 % off at practical partition sizes). Declined: polarization shapes from the elastic strain
concentration plus one polarization per cluster (needs offline fine-mesh solves, and a near/far split of the interaction
to be fast); the user wants a richer polarization that needs neither (README, "Open").

Known behaviour, model error rather than bugs: FFT-TFA's far field converges with the degree, not with the partitions
alone, and partitions should be at most about half an inclusion across (lecture, section 10). Also:
- **k-means clusters break the cell's mirror symmetry,** so normal loading gives a small spurious macroscopic shear
  (about 3e-4 of the normal stress).
- **Element-level LS rang along high-contrast interfaces:** an alternation near the inclusion's corners that refinement did
  not reduce and a global measure misses; inspect fields locally.
