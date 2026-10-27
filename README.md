# FFT-TFA

Reduced-order homogenization of elastoplastic composites: a stiff circular fiber in a softer J2-plastic matrix, in a
periodic cell. Three grids describe the cell:

- **elements:** the fine mesh of 8-node hexahedra, which resolves the geometry;
- **partitions:** CP's uniform cells, k × k columns of elements through the thickness for a geometry constant along z
  (the circular fiber, the laminate normal to x), k × k × k blocks of elements otherwise;
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

The cell is a periodic cube. The geometry is a circular fiber (`inclusion_shape = "circle"`), or a laminate layer
with interfaces normal to x (`"laminate"`) or z (`"laminate_z"`), which `check.py` uses because their exact solution
is known. A geometry constant along z is meshed with `element_number_along_z` element layers (one is exact) and
partitioned in columns; any other is meshed and partitioned alike along all three axes, so TFA's dense P holds
(6 × `partition_number_per_side`³)² doubles: 4.8 GB at 16³ partitions. Partitions may hold both phases, so any
partition size that divides the element grid works. The maps show the element layer at mid-thickness.

The reference stiffness is set in two parts, `C0 = reference_stiffness_scale × reference_stiffness`, e.g.
`"matrix"` and `0.01` for one hundredth of the matrix stiffness (the default). The same C0 is used everywhere in CP, as
in the slides: for the interaction between partitions, and for how a partition holding both phases shares its strain
between them (equal polarization σ − C0 ε, so the polarization is constant in every partition). C0 must stay below
the tangent stiffness of every phase all along the load path, or that sharing has no solution: a matrix that yields
without hardening needs a different approach. A small C0 would make GMRES slow; the Fourier preconditioner removes
that cost without changing the answer. It is the best Fourier-space approximation of each method's own Newton
Jacobian, the same rule for CP and Spectral LS, switched on separately for each (`cp_fourier_preconditioner`,
`spectral_ls_fourier_preconditioner`); by default only CP uses it, since for Spectral LS it changes nothing.

Spectral LS has its own reference stiffness (`spectral_ls_reference_stiffness`, default Moulinec–Suquet's midpoint
of the two phases): its answer does not depend on it, so it is only a solver setting, unlike CP's C0, which is part
of CP's model. Both methods judge convergence on the strain correction an all-matrix cell would need to cancel their
residual, not on the residual itself, so `convergence_tolerance` means the same accuracy for both whatever their
reference stiffness (LS note, section 7.5).

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
CP                   0.03         6.58          181        7146   60/60
Spectral LS          0.02        18.31          183       11487   60/60
TFA                 12.58         7.49          112         757   60/60
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

`check.py` solves four small problems with every method, without touching your `config.py`, cache or outputs:

- **laminate**, loaded, unloaded and loaded in reverse: every method must reproduce the exact solution in every
  cluster.
- **cut laminate**, whose interfaces cut through partitions, in transverse shear: Spectral LS and TFA must match the
  exact solution. CP's difference is only printed, since its strain sharing at equal polarization is exact here only
  as C0 → 0.
- **circle**, under tension and shear, with a band of mixed partitions around the fiber.
- **z laminate**, the laminate with interfaces normal to z, so the cell is meshed and partitioned in 3D: every method
  must match the exact solution.

On all four, every method must complete every load step and, once a reference is saved, reproduce it bit for bit.
Where CP has mixed partitions, their elastic stress and tangent must also match the closed form of slide 15 of
`references/10.20.2026 Progress.pptx`.
Before a change that should not alter any result (a cleanup), run `check.py save`; after it, `check.py` must pass.
After a change meant to alter results, check them, then save a new reference.

## Code layout

| File | Contents |
|---|---|
| `code/config.py` | every setting |
| `code/run.py` | the entry point: builds the problem, runs each method along the load path, writes the table and figures |
| `code/check.py` | the checks, with the laminate's exact solution; its reference results are saved next to it |
| `code/fe.py` | the cell's grids (mesh, partitions, clusters) and finite elements: TFA's influence functions E and P, their cache |
| `code/ls_model.py` | the LS equation (CP and Spectral LS): the reference medium C0, the kernel P0, the Fourier preconditioner, the stress of a partition from its clusters, and the Newton solver |
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
2 × 2-element partitions and 2–5 times further off with 4 × 4. That was with equal stress in mixed partitions and the
matrix C0 for the interaction. CP now follows the slides with one C0 = 0.01 × matrix for both (2026-10-27), so the
polarization is constant in every partition: on the small circle of `check.py` its macroscopic error is 4.3 % instead
of 2.4 %, and with the Fourier preconditioner its online time on the default problem is 6.6 s (8.5 s before,
23 s without the preconditioner).

Also done (2026-10-26): geometries that vary through the thickness. The cell is a cube; such a geometry is meshed and
partitioned in 3D, with a 3D kernel and FFT, while one constant along z keeps columns through the thickness. A
laminate with interfaces normal to z matches its exact solution with every method, and a circle gives the same
results as columns or as cubes. Only laminates and the circle exist so far; a new shape is one function in `fe.py`.

Later: a reference stiffness that follows plastic softening (as in SCA; with one C0, CP cannot run a perfectly
plastic phase until then), conjugate gradients on Brisard–Dormieux's symmetric form (block-Jacobi preconditioning
was tried and rejected: CP stopped less accurately), and the consistent kernel (the exact partition average,
Brisard–Dormieux), the limit of the averaged Moulinec–Suquet kernel as the element grid is refined.
