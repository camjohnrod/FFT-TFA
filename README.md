# FFT-TFA

Reduced-order homogenization of elastoplastic composites: a stiff inclusion (square, circle or laminate layer) in a
softer J2-plastic matrix, in a periodic cell. The cell is meshed with 8-node hexahedra and divided into
*partitions*, columns of elements through the thickness; each partition carries one uniform strain and plastic
state. Two reduced models are compared:

- **LS**, the model under development: a partition-averaged Lippmann–Schwinger equation in which partitions interact
  only through a homogeneous reference medium (stiffness C0), so the nonlocal operator is one FFT convolution with a
  kernel P0 (`notes/2_ls_model.md`).
- **TFA**, the baseline: Transformation Field Analysis, whose interaction operator P is computed exactly from the
  real composite but is dense, so its cost grows with the square of the number of partitions (`notes/3_tfa_baseline.md`).

**Research question.** On identical uniform partitions, how does the interaction operator decide accuracy and cost?
TFA uses the real composite's operator: exact in elasticity, but dense. LS replaces it with a reference-medium
operator applied by FFT, built either exactly (the consistent partition average of the Green operator) or
approximately (Moulinec–Suquet, or the FE kernel used now). The aim is to measure each against a full-field
reference, and to find the exact relation between the TFA and LS operators. `notes/1_research_context.md` places this in the
literature.

## Quick start

```bash
python code/MAIN_FP.py        # fixed-point solvers; run first
python code/MAIN_Newton.py    # Newton–Krylov solvers; also cross-checks against MAIN_FP.py's saved results
python code/check_code.py     # after any code change: static checks and seven small smoke runs (~5 minutes)
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `tqdm` (no package file; the miniforge `base` environment has them).
Every setting is in `code/config.py`; there are no command-line options. The first run of a configuration factorizes
the global stiffness and caches the result in `code/cache/` (gitignored), which takes minutes and, only when TFA is
included, a dense matrix of (6 × partitions)² doubles: 0.3 GB at the default 32 × 32 partitions.

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
| `notes/` | in reading order: `1_research_context.md` (research question, P0 operators, literature), `2_ls_model.md` (LS, equations LS-*), `3_tfa_baseline.md` (TFA and its solvers, ER-*), `background_influence_functions_and_J2/` (derivations of E, P and the J2 return map) |
| `references/` | the papers the notes cite; `reading_notes/` has one note per paper read cover to cover |

`CLAUDE.md` holds the detailed conventions and design decisions (strain convention, half-spectrum FFT, timing rules,
naming).

## Status and roadmap

All solvers, checks and figures above work and are tested. The fine LS solver is a comparison, **not** a reference
solution: it rings along high-contrast interfaces (an element-to-element alternation near the inclusion corners of
about ±30 %, which refinement does not remove and which element-level TFA does not show), so no accuracy figures are
computed against it.

**Position in the literature** (details in `notes/1_research_context.md`). LS is the clustered Lippmann–Schwinger equation
that self-consistent clustering analysis (SCA) also solves, here on uniform partitions so that it runs by FFT. For
linear elasticity this is Brisard–Dormieux's scheme (2010, 2012). With plasticity, no numerical study on uniform
partitions was found among the papers read, and no numerical comparison of LS and TFA on identical partitions.

Roadmap, in order:
1. **Thickness check.** Confirm that one element layer through the thickness reproduces three to round-off. One layer
   is already the default, so every later result rests on this.
2. **Analytic P0 kernels.** Build P0 from the partition lattice alone: the consistent operator (the exact partition
   average, LS-7), Moulinec–Suquet and filtered. The FE kernel stays as an option. With the consistent kernel,
   partitioned and coarse LS coincide.
3. **Gold-standard reference.** A full-field FE-consistent FFT solver (Ladecký et al., in `references/`) with
   plasticity at the Gauss points, so that errors become absolute rather than relative to another model.
4. **TFA–LS study.** Derive the exact relation between TFA's P and the consistent LS operator, and measure TFA and
   every LS kernel on identical partitions against the reference, across contrast, partition count and load path.
5. **Cost at equal accuracy.** Time and memory against resolution for each operator, compared at the same error.

Later: a rule for LS partitions holding both phases (circles on coarse partitions), a reference stiffness that
follows plastic softening (as in SCA), and a Newton line search for strongly softening materials.
