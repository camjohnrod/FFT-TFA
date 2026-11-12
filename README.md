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

**The goal: a paper (CMAME or IJNME).** Its claim: an interaction for clustered models that is exact inside each
partition and summarizes every other partition by polynomial moments carried by one FFT, converging with the degree to
SCA's clustered equation with C0 = L_m, at a cost and memory that grow with the partitions × p², never with the clusters
squared. TFA, SCA, Spectral LS and FFT-TFA become one family with three dials: the material states, the inclusion's
polarization and the degree. Shown so far: convergence with the degree (the cut laminate is exact from degree 5), local
fields about 2–3× faster than Spectral LS at equal error in 2D, less memory than TFA and SCA. Not yet: macroscopic
stress beats Spectral LS only at moderate accuracy in 2D, and not on a single 3D sphere; the reference is Spectral LS
itself; 3D with many inclusions, where FFT-TFA should win, is untested. In order:

1. **Literature.** The scheme is the precorrected-FFT idea (and kin of the fast multipole method and particle–mesh
   Ewald) applied to clusters. Find and read, cover to cover, those methods' foundational papers and any use of them for
   SCA, TFA or LS homogenization, before claiming anything new. The cheapest step, and it decides the paper's framing.
2. **Error measure.** The macroscopic error as the relative tensor norm ‖Σ − Σ_ref‖ / ‖Σ_ref‖ (shears counted twice),
   Schneider 2019's eq. (5.1) extended to the whole tensor, at the turning points and the first (elastic) step. It
   replaces today's per-component rms, which depends on the axes and inflates small components.
3. **A modern element kernel.** Every method here uses Moulinec–Suquet's original discretization of the LS equation,
   which rings at high-contrast interfaces and has been superseded (Willot's rotated staggered grid, the staggered grid
   of Schneider, Ospald and Kabel, finite-element-based kernels; Schneider 2021's review). FFT-TFA works with any
   element kernel Γ0, so the paper can give FFT-TFA and Spectral LS the same modern one, a fair comparison, and show
   that FFT-TFA does not depend on the discretization. Wanted: a drop-in scheme that changes only the kernel's formula
   per frequency, chosen by a setting in `config.py`, with Moulinec–Suquet kept as the other choice. Willot 2015's
   rotated scheme may be one (a modified frequency vector in the same closed form); read it before deciding.
4. **Gold-standard reference.** A full-field FE solution on a conforming (body-fitted) mesh with plasticity at the Gauss
   points (a gmsh mesh, our own solver reusing `material.py`), in generalized plane strain for z-constant cells, since
   their loads include ε33, with a convergence study showing its own error is well below the errors compared. Until
   then the reference is Spectral LS on a fine voxel mesh, which shares every method's staircase geometry and kernel and
   favours Spectral LS, which converges to it.
5. **Error decomposition.** Derive the exact relation between TFA's P and the LS operator, and split FFT-TFA's error
   into its parts: one eigenstrain per cluster (stiff, shared with TFA: TFA against the reference), the inclusion's
   polarization uniform per cluster (soft: FFT-TFA at full degree against TFA), the far-field summary (FFT-TFA against
   itself at full degree) and the element kernel against TFA's FE. The stiff and soft parts cancel in part, so a method
   measured only against the reference can look better than its interaction is: the paper shows that.
6. **Cost at equal accuracy, in 3D.** Online time against error for each method, in 2D and 3D, on many inclusions: a
   random-spheres shape, and `get_correction_blocks` built a chunk of partitions at a time ("Memory" below). Spectral LS
   over a range of resolutions answers "why not a full-field solve?"; TFA (and SCA, if added) on the same clusters
   isolate the interaction. An FFT on n lattice points costs about n log n, with n the elements for Spectral LS and the
   partitions for FFT-TFA, so Spectral LS on a mesh as coarse as FFT-TFA's partitions costs about as much per
   iteration: the two differ in accuracy, FFT-TFA seeing the fine geometry through each partition's clusters and its
   corrections. Also: offline cost reported apart, the number of load paths after which a method's offline cost pays
   off, and load paths other than the one the clusters were built from. **Memory** (proposed, the user leaning this
   way): a short discussion with a table of what each method stores, by formula and counted for the examples (TFA and
   SCA dense in the clusters, FFT-TFA partitions × p², Spectral LS a few fields per element with Γ̂0 evaluated on the
   fly as production codes do), not a measured comparison: a measured one would have to count a leaner Spectral LS than
   ours, which stores several 6 × 6 arrays per element.

The paper's examples: the cut laminate (an exact solution, so the interaction's error alone), random fibers in 2D (error
against cost, field maps), random spheres in 3D. Undecided: SCA as a fourth method (reviewers will ask for it). With C0
= L_m its accuracy is already FFT-TFA's at full degree, so a dense implementation would add only its cost; a reviewer
may call that a weak SCA, since Liu–Bessa–Liu update C0 self-consistently, a different method and a new module; whether
the inclusion polarization below is fixed first or reported as a limitation (proposed: reported); the journal (CMAME
would want an error estimate for the far field in the degree, which might also explain the floor far away, lecture
section 10).

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
