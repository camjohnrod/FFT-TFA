# FFT-TFA

Reduced-order homogenization of elastoplastic composites: a stiff circular fiber in a softer J2-plastic matrix, in a
periodic cell. Three grids describe the cell:

- **elements:** the fine mesh of 8-node hexahedra, which resolves the geometry;
- **partitions:** CP's uniform cells, k × k columns of elements through the thickness;
- **clusters:** each partition's part in each phase. A partition inside the fiber or the matrix is one cluster; a
  partition the fiber's edge cuts through is two, its matrix part and its fiber part.

Every method keeps one material state (strain, plastic strain) per cluster. Three methods solve the same problem:

| Method | What it is |
|---|---|
| **CP** | The method under development: the partition-averaged Lippmann–Schwinger (LS) equation, in which partitions interact only through a homogeneous reference medium (stiffness C0), so the interaction is one FFT convolution with the Moulinec–Suquet kernel P0 (`notes/2_ls_model.md`). The unknowns are the partition strains; in a partition holding both phases, the strain is shared between its two clusters so that both carry the same stress. |
| **Spectral LS** | The same equation with one partition per element: the standard Moulinec–Suquet spectral method on the element grid. |
| **TFA** | The baseline, Transformation Field Analysis, on CP's clusters: clusters interact through the exact FE influence function P of the composite, exact in elasticity but dense, so its cost grows with the square of the number of clusters (`notes/3_tfa_baseline.md`). |

CP and TFA have the same material states; they differ in how those states interact. Every method solves each load
step by Newton's method with a backtracking line search, with GMRES for each Newton correction.

**Research question.** On identical material states, how does the interaction operator decide accuracy and cost?
`notes/1_research_context.md` places this in the literature.

## Quick start

```bash
python code/run.py      # solve the problem in code/config.py with every method
python code/check.py    # check the code (a few seconds); run after any change to it
```

The default problem (64 × 64 elements, 32 × 32 partitions, 60 load steps) takes about half a minute once its FE
operators are cached. The first run of a new configuration with TFA builds and caches them in `code/cache/`, which
takes minutes and a dense matrix of (6 × partitions)² doubles: 0.3 GB at 32 × 32 partitions, 4.8 GB at 64 × 64.

Dependencies: `numpy`, `scipy`, `matplotlib`, `tqdm` (installed in the untracked `.venv/`).

## Changing the problem

**Every setting is in `code/config.py`**, grouped and commented: geometry and mesh, materials, load path, which
methods to run, the reference stiffness C0, the Newton–GMRES settings, the number of CPU threads, the offline batch
size, the figures' colour scale and the check tolerances. There are no command-line options and no settings anywhere
else.

`thread_count` (default 1) caps every method alike, TFA's dense products and the FFTs of CP and Spectral LS, so that
no method gets more cores than another.

The geometry is a circular fiber (`inclusion_shape = "circle"`), or a laminate layer, which `check.py` uses because
its exact solution is known. Partitions may hold both phases, so any partition size that divides the element grid
works.

## Output

`code/run.py` prints its progress and a timing table, and saves to `code/output/`:

| File | Contents |
|---|---|
| `timing.txt` | each method's offline time (building its operators) and online time (solving the load path), Newton iterations, GMRES iterations and load steps completed |
| `stress_strain.png` | macroscopic stress against applied strain, every method |
| `cross_section.png` | one panel per method that ran: the fiber on the element grid, with that method's cells drawn on top (CP's partitions, Spectral LS's elements, TFA's clusters) and its settings |
| `von_mises_stress.png` | von Mises stress at the last load step, one map per method, each element showing its cluster's stress |
| `plastic_strain.png` | accumulated plastic strain at the last load step, drawn the same way |

```
Method        Offline (s)   Online (s)   Newton its   GMRES its   Steps
CP                   0.02         6.29          180       11396   60/60
Spectral LS          0.03        17.29          184       13924   60/60
TFA                  9.54         7.84          112         757   60/60
```

Offline is the time to build each method's operators: for TFA, the FE influence functions E and P, reported at the
time they took to compute even when they now come from the cache. Online is the time to solve the load path. On a
shared or busy machine, single runs vary by up to about 20 %.

A method that cannot solve a load step stops there, and the table says why; the others carry on.

## Checking the code

```bash
python code/check.py         # run the checks
python code/check.py save    # run the checks and, if they pass, save the results as the reference
```

`check.py` solves three small problems with every method, without touching your `config.py`, cache or outputs:

- **laminate**, loaded, unloaded and loaded in reverse: every method must reproduce the exact solution in every
  cluster.
- **cut laminate**, whose interfaces cut through partitions, in transverse shear, where sharing a mixed partition's
  strain at equal stress is exact: again every method must match the exact solution.
- **circle**, under tension and shear, with a band of mixed partitions around the fiber.

On all three, every method must complete every load step and, once a reference is saved, reproduce it bit for bit.
Before a change that should not alter any result (a cleanup), run `check.py save`; after it, `check.py` must pass.
After a change meant to alter results, check them, then save a new reference.

## Code layout

| File | Contents |
|---|---|
| `code/config.py` | every setting |
| `code/run.py` | the entry point: builds the problem, runs each method along the load path, writes the table and figures |
| `code/check.py` | the checks, with the laminate's exact solution |
| `code/fe.py` | mesh, partitions and clusters, element matrices, the influence functions E and P, the kernel P0, the cache |
| `code/ls_model.py` | the LS equation, the stress of a partition from its clusters, and its Newton solver (CP and Spectral LS) |
| `code/tfa_model.py` | the TFA equation and its Newton solver |
| `code/newton.py` | what both share: the Newton loop with its line search, residual norm, convergence test, GMRES solve |
| `code/material.py` | J2 return map and its sensitivity |
| `code/plots.py` | the figures |
| `notes/` | in reading order: `1_research_context.md` (research question, P0 operators, literature), `2_ls_model.md` (LS, equations LS-*), `3_tfa_baseline.md` (TFA and its solvers, ER-*), `background_influence_functions_and_J2/` (derivations of E, P and the J2 return map) |
| `references/` | the papers the notes cite; `reading_notes/` has one note per paper read cover to cover |

`CLAUDE.md` holds the detailed conventions (strain convention, half-spectrum FFT, naming).

## Status and roadmap

All three methods, the figures and the checks work and are tested. The code was cut back to these on 2026-10-26; the
fixed-point solvers, the FFT-preconditioned TFA solvers, the FE kernel for P0 and the coarse LS lattice are at git tag
`pre-cleanup`; the square geometry was removed the same day and is in the git history.

**Position in the literature** (details in `notes/1_research_context.md`). LS is the clustered Lippmann–Schwinger
equation that self-consistent clustering analysis (SCA) also solves, here on uniform partitions so that it runs by FFT.
For linear elasticity this is Brisard–Dormieux's scheme (2010, 2012). With plasticity, no numerical study on uniform
partitions was found among the papers read, and no numerical comparison of LS and TFA on identical partitions.

Roadmap, in order:
1. **Thickness check (done).** One element layer through the thickness reproduces three to round-off.
2. **Moulinec–Suquet kernel (done).** One rule for every LS lattice: Moulinec–Suquet on the element grid, averaged
   over partitions as SCA does (Liu et al. 2016). TFA keeps its FE operators, so a CP-vs-TFA difference includes the
   FE-vs-Moulinec–Suquet discretization gap.
3. **Gold-standard reference.** A full-field FE-consistent FFT solver (Ladecký et al., in `references/`) with
   plasticity at the Gauss points, so that errors become absolute rather than relative to another method.
4. **TFA–LS study.** Derive the exact relation between TFA's P and the LS operator, and measure TFA, CP and Spectral
   LS against the reference, across contrast, partition count and load path.
5. **Cost at equal accuracy.** Time and memory against resolution for each method, compared at the same error.

Done since: partitions holding both phases (2026-10-26). CP shares a mixed partition's strain between its two
clusters at equal stress, TFA works on those clusters directly, and every method's Newton solver has a backtracking
line search. Tested on circles against element-level TFA: macroscopic stress within 1–5 % in every case tried, about
as close as Spectral LS; the stress and plastic strain next to the fiber are as close as Spectral LS's with
2 × 2-element partitions and 2–5 times further off with 4 × 4.

Later: a reference stiffness that follows plastic softening (as in SCA), a Fourier preconditioner for the Newton
solver of CP and Spectral LS (its GMRES is unpreconditioned), and the consistent kernel (the exact partition average,
Brisard–Dormieux), the limit of the averaged Moulinec–Suquet kernel as the element grid is refined.
