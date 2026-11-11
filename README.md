# FFT-TFA

Reduced-order homogenization of elastoplastic composites: stiff inclusions (one fiber or many, a sphere, a layer)
in a softer J2-plastic matrix, in a periodic cube. The cell is seen at three scales:

- **elements:** the fine mesh, which resolves the geometry;
- **partitions:** uniform blocks of elements, the cells of FFT-TFA's FFT (columns through the thickness if the geometry
  does not vary along z);
- **clusters:** groups of elements of one phase in one partition, each with one material state: each partition's part
  in each phase, or k-means clusters of the elements' elastic strain concentration inside each such part.

Every method keeps one material state per cluster. They differ only in how those states interact:

| Method | How clusters interact |
|---|---|
| **FFT-TFA** | The method under development. TFA's clusters, interacting through the Lippmann–Schwinger (LS) equation of a homogeneous medium of the matrix's stiffness: exact for the matrix's polarization, uniform per cluster in the inclusion. Inside each partition the clusters interact exactly; beyond it, each partition's polarization is summarized by its polynomial moments up to a `degree`, and every partition acts on every other through one FFT on the partitions. Raising the degree converges to the exact clustered equation. |
| **Spectral LS** | The LS equation on the elements, each its own cluster: the standard Moulinec–Suquet method, as a comparison. |
| **TFA** | The baseline. Clusters interact through the composite's exact FE influence functions E and P: exact in elasticity, but P is dense. |

All three solve each load step by the same Newton loop, line search and tolerances.

**Research question.** With the same material states, how does the interaction decide accuracy and cost? At full degree
FFT-TFA is SCA's clustered equation with the matrix as reference medium. The theory, step by step, is
`notes/fft_tfa_lecture.md`.

## Quick start

```bash
python code/run.py preview          # what the settings would need, and the cross-section; solves nothing
python code/run.py                  # solve what code/config.py lists; results and figures to a new folder
python code/run.py replot latest    # redraw a saved run with config.py's figure settings; solves nothing
python code/run.py cache            # what is cached in code/cache/
python code/check.py                # check the code (under a minute)
```

It needs Linux (or WSL), and Python 3.12 with `numpy`, `scipy`, `matplotlib` and `tqdm`.

**What to run:** every setting is in `code/config.py`, with what it means. Its first screen is all you normally touch:
the problem (a geometry, materials and a load path, each one name from the lists further down, where you can add your
own) and the entries of each method to compare, which you can group into schemes, say degree 2 against degree 3.

**What a run gives:** a folder in `code/output/` with a summary table, figures, and the results; the top of
`code/run.py` lists its files.

**Checking the code:** `check.py` solves small problems with every method, against exact solutions and against each
other; the top of `code/check.py` lists the checks. For a change that should leave the results alone, run
`python code/check.py save` before it and `python code/check.py` after.

## Reading the code

Each file opens by saying what it does. In reading order:

| File | What it holds |
|---|---|
| `code/config.py` | every setting |
| `code/material.py` | the J2 material law: radial return and its derivative |
| `code/newton.py` | what every method shares: the Newton loop, line search, GMRES, 6 × 6 block helpers |
| `code/cell.py` | the cell: shapes, mesh and partitions, materials, clusters and k-means, the cache |
| `code/ls_model.py` | the LS equation's kernel, and Spectral LS, the simplest method |
| `code/fft_tfa_model.py` | FFT-TFA: its interaction (polynomial moments by FFT, exact corrections inside each partition) and its solver |
| `code/tfa_model.py` | TFA: finite elements, E and P, and its solver |
| `code/run.py` | the entry point: the entries, solving and timing them, saving and drawing a run |
| `code/plots.py` | the figures |
| `code/check.py` | the checks |

Every method is a `newton.Solver` built from its clusters and its own settings, `FFTTFASolver(clusters, degree)`,
`SpectralLSSolver(clusters, C0)`, `TFASolver(clusters)`, and says two things: its residual (`evaluate`) and its Newton
correction (`get_newton_correction`). Equation labels in the comments (FT-, LS-, ER-) are the lecture's.

## Status and roadmap

Working: FFT-TFA, Spectral LS and TFA on 2D and 3D cells, partitions holding both phases, k-means clusters inside
partitions, and the checks.

1. **Gold-standard reference.** A full-field FE solution on a conforming (body-fitted) mesh with plasticity at the Gauss
   points (a gmsh mesh, our own solver reusing `material.py`), so that errors become absolute rather than relative to
   another method. Spectral LS on a fine voxel mesh, the reference until then, shares every method's staircase geometry.
2. **TFA–LS study.** Derive the exact relation between TFA's P and the LS operator, and compare FFT-TFA with TFA on the
   same clusters: their difference splits into the inclusion's polarization taken uniform per cluster (FFT-TFA at full
   degree against TFA), the far-field summary (FFT-TFA against itself at full degree) and the Moulinec–Suquet kernel
   against TFA's FE. Against the reference, each method adds the error of one eigenstrain per cluster, which FFT-TFA and
   TFA share; the two errors can have opposite signs and cancel, so a method measured only against the reference can
   look better than its interaction is.
3. **Cost at equal accuracy.** Time and memory against resolution for each method, at the same error, in 2D and 3D. An
   FFT on n lattice points costs about n log n, with n the elements for Spectral LS and the partitions for FFT-TFA, so
   Spectral LS on a mesh as coarse as FFT-TFA's partitions costs about as much per iteration: the two differ in accuracy,
   FFT-TFA seeing the fine geometry through each partition's clusters and its corrections. Memory counts as much as
   time: FFT-TFA's case rests on needing less than SCA and TFA, whose interaction is dense in the clusters, and than
   Spectral LS, which stores every element. Spectral LS here keeps several 6 × 6 arrays per element (its kernel, the
   LS-24 operator, stiffness, tangent); a fair comparison counts a leaner one.

Open for FFT-TFA (the lecture's section 10):

- **Far field by polynomial moments:** the interaction converges with the degree to the exact clustered equation, so
  refining the partitions pays off (100 fibers, 32² partitions: 1.25 % against 512² Spectral LS at degree 4), and the
  cut laminate is exact once the degree spans a partition's elements in x. Partitions about an inclusion across need a
  high degree, since the sharp near field then reaches past a partition. Open: the kernels' memory grows with the
  square of the polynomials (in 3D, 10 at degree 2 and 20 at degree 3); halving it by their symmetry, or keeping only
  pairs of total degree up to the degree, are untried.
- **Inclusion polarization:** one polarization per inclusion cluster; σ33 along z-invariant fibers runs low. Splitting a
  partition's fiber part by fiber makes FFT-TFA stiffer and less accurate, with the exact interaction too: with the
  matrix as C0, a coarser fiber polarization errs soft (Brisard–Dormieux 2010's Hashin–Shtrikman bound), offsetting the
  stiff error of one eigenstrain per matrix cluster, which dominates. To do: refine the matrix clusters until refining
  the fiber helps, so that accuracy does not rest on the two cancelling. Wanted: a richer inclusion polarization that
  works for any geometry, needs no offline solve and keeps one interaction operator.
- **Material states:** how FFT-TFA's and TFA's accuracy and cost change with the number of clusters; with clusters of a
  few elements the clustered model can have several solutions (TFA too).
- **Memory:** online, FFT-TFA stores the moment kernels, (6p)² numbers per frequency of the partition lattice with p
  the polynomials of its degree, and one 6 × 6 correction block per pair of clusters in the same partition, so its
  memory grows with the partitions times p² and with the square of the clusters per partition, not with the elements.
  Offline, `get_correction_blocks` cross-correlates the clusters of every partition at once, which in 3D with k-means
  takes more than a Spectral LS solve of the whole load path. To do before any 3D study: build them a chunk of
  partitions at a time (a chunk size or memory budget in `config.py`). Every k-means setting also needs six elastic
  Spectral LS solves on the elements, the floor of its offline memory.

Later: a reference stiffness for FFT-TFA that follows plastic softening (needed for a perfectly plastic phase), and the
consistent kernel (the exact partition average).

Field errors against the reference are not computed yet (macroscopic errors are). A measure fair across meshes of
different resolution is not settled: candidates are block averages at a scale every mesh resolves (on points chosen by
the reference's phases alone), and element errors on the fine grid split into elements whose phase matches the
reference's and elements whose phase does not.
