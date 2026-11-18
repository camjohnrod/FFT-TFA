# FFT-TFA

Reduced-order homogenization of elastoplastic composites. A composite here is stiff elastic inclusions (fibers,
spheres, a layer) in a softer matrix that yields plastically, and we study one periodic cubic cell of it under a
prescribed average strain. A full-field solution has one unknown strain per element of a fine mesh. A reduced-order
model keeps far fewer *material states*, and the question this code explores is how those states should interact:
accurately, quickly and in little memory. It compares four methods on the same problem, side by side, and draws the
results.

The theory, derived step by step, is in [`notes/fft_tfa_lecture.md`](notes/fft_tfa_lecture.md). This README is about
using and reading the code.

## The idea in one page

The cell is seen at three scales:

- **elements:** a fine regular mesh that resolves the geometry; each element is matrix or inclusion, its *phase*;
- **partitions:** uniform blocks of elements, the cells of FFT-TFA's coarse FFT (columns through the thickness when
  the geometry does not vary along z);
- **clusters:** groups of elements of one phase in one partition, each with **one material state** (one strain, stress
  and plastic strain): each partition's part in each phase, or that part split further by k-means.

Every method keeps one material state per cluster. They differ in how the clusters interact:

| Method | How clusters interact |
|---|---|
| **FFT-TFA** | The method under development. The Lippmann–Schwinger (LS) equation of a homogeneous medium with the matrix's stiffness. Inside each partition the clusters interact exactly; beyond it, each partition is summarized by the polynomial moments of its polarization up to a `degree`, and every partition acts on every other through one FFT on the partitions. Raising the degree converges to SCA's equation with the matrix as reference medium. |
| **SCA** | Self-consistent clustering analysis (Liu, Bessa & Liu 2016): the same clustered LS equation solved exactly, on k-means clusters over the whole cell, through a dense interaction matrix. |
| **Spectral LS** | The LS equation on the elements, each its own cluster (Moulinec & Suquet 1998): the standard full-field FFT method. |
| **TFA** | Transformation field analysis: clusters interact through exact finite-element influence functions, which are dense. |

All four solve each load step by the same Newton loop, line search and tolerances, so their times compare fairly. A
**reference**, Spectral LS on a fine mesh, gives the answer the others are measured against.

## Getting started

It needs Linux (or WSL on Windows) and Python 3.12 or later:

```bash
pip install -r requirements.txt
```

Every command runs from the repository's folder:

```bash
python code/run.py preview          # what the settings would need, and a drawing of each method's cells; solves nothing
python code/run.py                  # solve what code/config.py lists; results and figures go to a new folder
python code/run.py replot latest    # redraw a saved run with config.py's figure settings; solves nothing
python code/run.py cache            # what is cached in code/cache/
python code/check.py                # check the code (under a minute)
```

### Your first run

`code/config.py` comes set to an elastic example: 100 glass fibers, 35 % of the cell, in a polyamide matrix softened
by heat (contrast 90), pulled across the fibers to 1 % and then sheared, in uniaxial stress (the other directions
free). It compares untrained FFT-TFA (one cluster per phase in each partition) at degrees 0, 1 and 2 with Spectral LS
on coarser grids. The clustered mesh and the reference share one 1024² image, so every FFT-TFA error is its model's
alone (see "Measuring errors" below). The first run takes about half an hour, most of it the reference, which later
runs load from the cache; for a quick look, set `clustered_mesh` and the reference's `elements` to 256 and the
partitions to at most 64, which takes a few minutes.

1. `python code/run.py preview` prints each entry's cluster count and memory, and draws
   `code/output/preview_cross_section.png`: every method's elements, clusters and partitions on the cross-section.
2. `python code/run.py` solves the reference (cached for later runs), then every entry with a progress bar, and saves
   everything to `code/output/<date>_<results_name>/` (`code/output/latest` points at it).
3. Open the folder:
   - `summary.txt`: each entry's cost (offline time to build it, online time to solve the load path, iterations,
     peak memory) and its error against the reference.
   - `stress_strain.png`: the macroscopic stress along the load path, the reference in black.
   - `error_against_time.png`: error against online time, one line per method through its entries, coarse to fine.
     Lower and further left is better: this is the figure that compares methods.
   - `macro_stress_error.png`: each method's signed error along the path; above zero the method is too stiff.
   - `von_mises_stress.png`, and `plastic_strain.png` when the matrix yields: the fields as maps, and along one line
     (`*_profile.png`).
   - `cross_section.png`: how each entry divided the cell.

### Measuring errors

A clustered method groups the elements of `clustered_mesh`, so its answer can be wrong in two ways: its model (the
clusters, the far-field summary) and the mesh itself, which is only an approximation of the true geometry. With a
reference finer than the clustered mesh, the error measured holds both, and in elasticity they have opposite signs:
clustering makes FFT-TFA soft (lecture, 5.1), while a coarse voxel mesh is stiff, so the two can cancel and a coarse
result can look better than its model is. Setting the reference's `elements` equal to `clustered_mesh` makes the
reference the exact answer on the very image the clustered methods group, so their errors are their models' alone,
while Spectral LS on coarser grids shows what coarsening the image costs. An FFT-TFA, SCA or TFA entry may also set
its own `elements` instead of `clustered_mesh`: on a Spectral LS entry's elements, both carry the same mesh error, and
what separates them is the clustered method's model (one scheme per mesh draws one line each).

### Changing what runs

Everything is set in `code/config.py`, and its first screen is all you normally touch:

- **the problem:** a geometry, materials and a load path, each one name from the lists in its section 4, where you can
  add your own; and `unloaded_directions`, `"fixed"` (the cell cannot contract sideways) or `"free"` (zero stress in
  the directions the load path does not strain, so a path along one component is a tensile test);
- **what to compare:** a list of entries per method, each a line in the table. An empty list switches a method off.
  Several lists make several *schemes* of one method, each a line of its own in the error-against-time plot, for
  example FFT-TFA at degree 2 against degree 3:

  ```python
  fft_tfa = [[dict(partitions=p, matrix_clusters=4, inclusion_clusters=1, degree=2) for p in (8, 16, 32)],
             [dict(partitions=p, matrix_clusters=4, inclusion_clusters=1, degree=3) for p in (8, 16, 32)]]
  ```

Each setting is explained where it is set. Run `preview` before a large configuration: the memory of SCA and TFA
grows with the square of their clusters, and a fine reference can take half an hour (it is cached once solved).

## Words used throughout

| Word | Meaning |
|---|---|
| phase | matrix (0) or inclusion (1) |
| macro strain, macro stress | the cell's average strain and stress; the load path prescribes the macro strain |
| eigenstrain μ | the strain the material no longer resists elastically: here the plastic strain |
| reference medium C0 | the homogeneous stiffness the LS equation is written for; the difference from the real material is the polarization |
| polarization τ | the stress the reference medium gets wrong, τ = σ − C0 ε; it drives every interaction |
| degree | of the polynomials FFT-TFA summarizes each partition by; `"full"` is exact |
| entry, scheme | one configured run of a method (a line in the table); a list of a method's entries, drawn as one line |
| offline, online | building a method (its clusters and operators), and solving the load path with it |
| LS-*, FT-*, ML-*, TFA-*, SC-*, SF-* | equation labels in the lecture, which the code's comments cite |

## Reading the code

Each file opens by saying what it does. In reading order:

| File | What it holds |
|---|---|
| `code/config.py` | every setting |
| `code/material.py` | the J2 material law: radial return and its derivative |
| `code/newton.py` | what every method shares: the Newton loop, line search, GMRES, 6 × 6 block helpers, stress-free directions |
| `code/cell.py` | the cell: shapes, mesh and partitions, materials, clusters and k-means, the cache |
| `code/ls_model.py` | the LS equation's element kernel (Willot's or Moulinec–Suquet's), and Spectral LS, the simplest method |
| `code/fft_tfa_model.py` | FFT-TFA: its interaction (polynomial moments by FFT, exact corrections inside each partition) and its solver |
| `code/sca_model.py` | SCA: its dense interaction by FFT, its reference stiffness, and its solver |
| `code/tfa_model.py` | TFA: finite elements, the influence functions E and P, and its solver |
| `code/run.py` | the entry point: the entries, building and solving them, the reference |
| `code/report.py` | saving a run, and its tables and figures |
| `code/plots.py` | how the figures look |
| `code/check.py` | the checks |

Every method is a `newton.Solver` built from its clusters and its own settings, `FFTTFASolver(clusters, degree)`,
`SCASolver(clusters, C0)`, `SpectralLSSolver(clusters, C0)` and `TFASolver(clusters)`, and says two things: its
residual (`evaluate`) and its Newton correction (`get_newton_correction`). `newton.Solver.solve_step` does the rest.
To follow one method from end to end, start at `run.get_solver` and read that method's solver class.

## Checking a change

`python code/check.py` solves small problems with every method, against exact solutions (a laminate, whose phases
each have one uniform state) and against each other, and prints PASS or FAIL for each check; the top of
`code/check.py` lists them. For a change that should leave every result alone, such as a cleanup, run
`python code/check.py save` before it and `python code/check.py` after: the last check then compares every result
bit for bit.

## Status and known limits

FFT-TFA, SCA, Spectral LS and TFA run on 2D and 3D cells, with partitions holding both phases and k-means clusters
inside partitions. Known limits, explained in the lecture's section 10:

- FFT-TFA's summary of the other partitions converges with the degree, not with the partitions alone. Partitions work
  best at most about half an inclusion across.
- FFT-TFA's moment kernels hold (6p)² numbers per partition, p the polynomials of the degree (in 3D, 10 at degree 2
  and 20 at degree 3), so a high degree on many partitions costs memory; `preview` reports it.
- One material state per cluster limits every clustered method once the matrix yields: its error grows with the
  plastic flow.
- TFA's finite-element influence functions limit it to small 3D meshes.
- Errors are measured on the macroscopic stress; there is no field error against the reference yet.

The research behind the code (its goals, findings and open questions) is logged in
[`notes/research_log.md`](notes/research_log.md). The lecture's figures are drawn by
`python notes/figures/make_figures.py`.
