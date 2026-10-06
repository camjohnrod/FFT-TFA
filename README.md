# FFT-TFA

Reduced-order homogenization of elastoplastic composites: a stiff inclusion (square, circle or laminate layer) in a
softer J2-plastic matrix, in a periodic cell. The cell is meshed with 8-node hexahedra and divided into
*partitions*, columns of elements through the thickness; each partition carries one uniform strain and plastic
state. Two reduced models are compared:

- **LS**, the model under development: a partition-averaged Lippmann–Schwinger equation in which partitions interact
  only through a homogeneous reference medium (stiffness C0), so the nonlocal operator is one FFT convolution with a
  kernel P0 (`notes/Partition_Lippmann_Schwinger_New_9_28_2026`).
- **TFA**, the baseline: Transformation Field Analysis, whose interaction operator P is computed exactly from the
  real composite but is dense, so its cost grows with the square of the number of partitions (`notes/Formulation.md`).

The research question is whether partitioned LS approaches the accuracy of a fine, non-partitioned solve at the
cost of a coarse one.

## Quick start

```bash
python code/MAIN_FP.py        # fixed-point solvers; run first
python code/MAIN_Newton.py    # Newton–Krylov solvers; also cross-checks against MAIN_FP.py's saved results
python code/check_code.py     # after any code change: static checks and seven small smoke runs (~5 minutes)
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `tqdm` (no package file; the miniforge `base` environment has them).
Every setting is in `code/config.py`; there are no command-line options. The first run of a configuration factorizes
the global stiffness and caches the result in `code/cache/` (gitignored), which takes minutes and, for TFA, a dense
matrix of (6 × partitions)² doubles: 0.3 GB at the default 32 × 32 partitions.

## What a run does

Each entry point loads the macroscopic strain path step by step (monotonic or cyclic) and solves every load step with
each of its solvers, several times over for fair timings:

| Entry point | Solvers |
|---|---|
| `MAIN_FP.py` | LS fixed point; with `include_tfa_model`, TFA fixed point plain and FFT-preconditioned |
| `MAIN_Newton.py` | LS Newton; with `include_non_partitioned_ls`, LS on the element lattice ("fine") and on a coarse mesh with one element per partition ("coarse"); with `include_tfa_model`, TFA Newton plain and FFT-preconditioned |

Outputs, in `code/output/` (prefix `FP_` or `Newton_`): `load_path_summary.png` (macroscopic deviatoric stress,
iterations and time per iteration per step), `time_breakdown.png` (where the time goes), `von_mises_stress.png` (one
map per solver at the last step, finest lattice first) and `results.npz`, plus `cross_section.png`.

## Configuration essentials

- **Geometry** (`inclusion_shape`, `element_number_per_side`, `partition_number_per_side`): the geometry is resolved
  on the elements, and every partition must be a single phase, or the run stops before any solve. A square or a
  laminate must have its edges on partition edges; a circle needs one partition per element.
- **Reference stiffness** (`reference_stiffness`): C0 is part of the LS model, so it changes LS's answer; for TFA it
  only affects the FFT preconditioner's speed.
- **Load path** (`max_macro_strain`, `strain_increment_count`, `load_path_shape`).
- **Checks with a known answer**: `matched_stiffness_control = True` makes LS and TFA the same equation, so they must
  agree; `inclusion_shape = "laminate"` has an exact solution (`code/laminate.py`) that every solver must match.

## How the code is checked

Every run prints PASS/FAIL checks and stops on a failure: the converged residual recomputed from scratch after every
step, agreement between solvers that solve the same equation, cross-checks between the two entry points, and, before
solving, checks of the operators (kernel structure, finite-difference Jacobians, dense reference solves on small
grids). `code/check_code.py` runs both entry points on seven small configurations, each with a comment saying what it
tests. For a change that must leave every result unchanged, run `python code/check_code.py save` before it and
`python code/check_code.py compare` after it: every solver's stress path and iteration counts must be bit-identical.

## Code layout

| File | Contents |
|---|---|
| `code/config.py` | every input, its checks and derived values |
| `code/MAIN_FP.py`, `code/MAIN_Newton.py` | entry points, each `main()` an outline |
| `code/offline.py` | mesh, materials, finite-element operators, E, P and P0 for any discretization, cache |
| `code/lattice_fft.py` | FFT over the partition lattice |
| `code/material.py` | J2 return map and its sensitivity |
| `code/tfa_solvers.py`, `code/ls_solvers.py` | the two models' solvers |
| `code/laminate.py` | exact laminate solution |
| `code/load_path.py`, `code/convergence.py`, `code/online_timing.py` | stepping a solver along the path, convergence, timing |
| `code/verification.py`, `code/report.py`, `code/plots.py` | pre-solve checks, reports and post-solve checks, figures |
| `code/check_code.py` | the test script |
| `notes/` | derivations: `Formulation.md` (TFA, equations ER-*), the LS note (LS-*), `verified_notes/` |
| `references/` | the papers the notes cite |

`CLAUDE.md` holds the detailed conventions and design decisions (strain convention, half-spectrum FFT, timing rules,
naming).

## Status and roadmap

All solvers, checks and figures above work and are tested. The fine LS solver is a comparison, **not** a reference
solution: it rings along high-contrast interfaces (an element-to-element alternation near the inclusion corners of
about ±30 %, which refinement does not remove and which element-level TFA does not show), so no accuracy figures are
computed against it.

Next, in order:
1. Build TFA's dense P only when TFA is included, so LS-only runs at fine resolution stop paying for it.
2. Check whether one element layer through the thickness gives the same results as three (it should, since nothing
   varies through the thickness). One layer is already the default, for about 3× cheaper offline solves.
3. A cost study: time and memory against resolution for partitioned, fine and coarse LS and TFA.
4. A gold-standard reference: an FE-consistent FFT solver (Ladecký et al., in `references/`), and then the accuracy
   study, splitting each field error into what any one-value-per-partition scheme must miss and what the scheme adds.
5. Later: a rule for LS partitions holding both phases (circles on coarse partitions), a Moulinec–Suquet kernel option
   for LS, and a Newton line search for strongly softening materials.
