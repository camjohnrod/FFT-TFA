# FFT-TFA

Reduced-order homogenization of elastoplastic composites: stiff inclusions (one fiber or many, a sphere, a layer)
in a softer J2-plastic matrix, in a periodic cube. The cell is seen at three scales:

- **elements:** the fine mesh of 8-node hexahedra, which resolves the geometry;
- **partitions:** uniform blocks of elements, the cells of FFT-TFA's FFT: k × k columns through the thickness if the
  geometry does not vary along z, k × k × k blocks otherwise;
- **clusters:** regions of one phase in one partition, with one material state (one uniform plastic eigenstrain) each:
  each partition's part in each phase (one cluster if the partition lies in one phase, two if an interface cuts
  through it), or k-means clusters of the elements' elastic strain concentration inside each such part.

Every method keeps one material state per cluster. They differ only in how those states interact:

| Method | How clusters interact |
|---|---|
| **FFT-TFA** | The method under development. TFA's clusters, interacting through the Lippmann–Schwinger (LS) equation of a homogeneous medium of the matrix's stiffness: exact for the matrix's polarization, uniform per cluster in the inclusion. The interaction is one FFT on the partitions, on each partition's mean polarization, plus exact corrections between the clusters of each partition (and, optionally, of its neighbours). (`notes/2_ls_model.md`, section 5) |
| **Spectral LS** | The LS equation on the elements, each its own cluster: the standard Moulinec–Suquet spectral method, as a comparison. (`notes/2_ls_model.md`, section 4) |
| **TFA** | The baseline. Clusters interact through the composite's exact FE influence functions E and P: exact in elasticity, but P is dense. (`notes/3_tfa_baseline.md`) |

All of them solve each load step by Newton's method, with a Krylov method for each correction (GMRES; conjugate
gradients for Spectral LS, whose Newton system is symmetric) and a backtracking line search.

**Research question.** With the same material states, how does the interaction operator decide accuracy and cost?
FFT-TFA and TFA share their material model, so their difference is how the clusters interact: FFT-TFA takes the
inclusion's polarization uniform per cluster (exact in the matrix, with the matrix as reference medium), truncates the
interaction to nearby partitions, and uses the Moulinec–Suquet kernel where TFA uses FE. With the truncation removed,
FFT-TFA is SCA's clustered equation with the matrix as reference medium. (`notes/1_research_context.md`,
`notes/2_ls_model.md` section 5.4)

## Quick start

```bash
python code/run.py preview          # what the settings would need, and the cross-section; solves nothing
python code/run.py                  # solve what code/config.py lists; results and figures to a new folder
python code/run.py replot latest    # redraw a saved run with config.py's Figures settings; solves nothing
python code/run.py cache            # what is cached in code/cache/
python code/check.py                # check the code (about a minute)
```

It needs Linux (or WSL), and Python 3.12 with `numpy`, `scipy`, `matplotlib` and `tqdm`.

## Choosing what to run

Every setting is in `code/config.py`, commented, from those changed most to those changed least; there are no other
inputs. Its first screen is what you will change most. To compare methods, list them; an empty list switches a method
off, and any method may have several entries, drawn in tints of its colour, so list them coarse to fine:

```python
run_name    = "biaxial cyclic"
fft_tfa     = [dict(partitions=8,  matrix_clusters=8, inclusion_clusters=1, reach=1),
               dict(partitions=16, matrix_clusters=8, inclusion_clusters=1, reach=1)]
spectral_ls = [dict(elements=64, C0="midpoint"), dict(elements=128, C0="midpoint")]
tfa         = []
```

- **FFT-TFA and TFA entries** share one fine mesh, `elements_per_side`; `partitions` must divide it. One cluster of each
  kind per partition needs no k-means. `reach` is how far FFT-TFA's exact interaction goes: 0 each partition's own
  clusters, 1 its neighbours too, `"full"` every partition.
- **Spectral LS entries** set their own `elements`, coarser or finer than the mesh. `C0` is `"midpoint"` or `"matrix"`,
  a solver setting: the answer does not depend on it.
- **Geometry:** `inclusion_shape` is `"fiber"`, `"random fibers"` (`fiber_count` non-overlapping fibers along z,
  `fiber_volume_fraction` in all), `"sphere"`, `"laminate"` (interfaces normal to x) or `"laminate_z"` (normal to z).
- **Materials:** `materials` names an entry of `material_sets`; add your own. A phase stays elastic if its yield stress
  is `np.inf`.
- **Load path:** `load` names an entry of `load_paths`, a list of turning points written like
  `strain(e11=0.02, g12=0.03)`: straight lines from zero through each in turn, so it need not be proportional.
  `strain_increment_count` steps reach the first, and the other lines keep that step length.
- **Reference:** `include_reference` adds a fine solution to compare with: Spectral LS on `reference_elements_per_side`
  elements per side (a multiple of the mesh's), solved once and cached in `code/cache/`. Any change of geometry,
  materials, load path or solver tolerance solves it again, and a first solve can take tens of minutes, so run it in the
  background. It is still Moulinec–Suquet on voxels (the same staircase, ringing at high contrast), until a conforming FE
  reference exists (roadmap step 1).
- **Cost:** `thread_count` caps every method alike; `timing_repeats` runs each entry several times, in turn, keeping its
  fastest time; `memory_limit_GB` stops an entry that needs more, with an error, while the others carry on.

TFA builds E and P on a new configuration and caches them. P takes (6 × clusters)² doubles, and in 3D the FE
factorization limits the mesh too (with 7 GB a 24³ mesh works, 36³ does not); `preview` says what P would need.

## Results

Every run gets its own folder in `code/output/` (`code/output/latest` points at the last), and a row per entry in
`code/output/runs.csv`, to compare runs at a glance:

- `config.py`, `code_version.txt`: the settings, and the code's commit and uncommitted changes;
- `run.json`, `results.npz`: what each entry did, and its results (macroscopic stress at every step, stress and plastic
  strain per cluster at each turning point, the reference's maps), read by `run.get_saved_run`;
- `timing.txt`: each entry's clusters, offline time (building its operators), online time (solving the load path), Newton
  and Krylov iterations, load steps completed, and peak memory offline and online (above Python's own, about 100 MB);
- `stress_strain.png`: macroscopic stress against applied strain, or against the load step on a path that is not
  proportional;
- `cross_section.png`: each entry's discretization, k-means clusters coloured in;
- `von_mises_stress.png`, `plastic_strain.png`: maps at turning point `map_turning_point` (on a cyclic path the last is
  back at zero strain, so residual stress), one per entry, each cluster at its average, with the reference's at its own
  resolution, which then sets the colour scale;
- `plastic_strain_profile.png`: the accumulated plastic strain along the middle element row, every entry and the
  reference on one plot, the inclusion shaded, to verify the contours;
- with the reference on, `macro_stress_error.png` and `errors.txt`: each entry's macroscopic stress error in every stress
  component the load path produces, in % of that component's largest reference value, along the path;
- with the reference on and several entries of a method, `error_against_time.png`.

An entry that cannot solve a load step, or runs out of memory, stops there and the table says why; the others carry on.

## Checking the code

```bash
python code/check.py         # run the checks
python code/check.py save    # run them and, if they pass, save the results
```

`check.py` solves small problems with every method, using its own settings (it never touches your `config.py`, cache or
outputs): a laminate, a laminate whose interfaces cut through partitions, a fiber, a laminate normal to z, a sphere with
and without k-means clusters, and random fibers. On the laminates every method must match the exact solution (FFT-TFA on
the cut laminate only up to its modelling error, printed for information), and FFT-TFA must match it with its interaction
made exact, also with clusters dividing each partition's matrix part. FFT-TFA's interaction must match the element-level
LS operator, and every method must stop at the same accuracy. The fine reference, solved at the working resolution, must
equal Spectral LS bit for bit. Once results are saved, every result must reproduce them bit for bit. For a change that
shouldn't alter results, run `check.py save` before it and `check.py` after.

## Code layout

| File | Contents |
|---|---|
| `code/config.py` | every setting |
| `code/run.py` | the entry point: entries, the load path, each entry in its own process, saving a run, tables and figures |
| `code/check.py` | the checks, with the laminate's exact solution |
| `code/cell.py` | the cell: shapes, mesh and partitions, materials, clusters and k-means, the cache |
| `code/fft_tfa_model.py` | FFT-TFA: its interaction (partition FFT and correction blocks) and its Newton solver |
| `code/ls_model.py` | the LS equation: the kernel P0, LS-24, Spectral LS with its conjugate gradients |
| `code/tfa_model.py` | TFA: finite elements, E and P, and its Newton solver |
| `code/newton.py` | what all methods share: the Newton loop, line search, GMRES |
| `code/material.py` | the J2 return map |
| `code/plots.py` | the figures |
| `notes/` | the theory, in reading order: research context, the LS model, the TFA baseline, background derivations |
| `references/` | the papers the notes cite; `reading_notes/` has one note per paper read cover to cover |

## Status and roadmap

Working: FFT-TFA, Spectral LS and TFA on 2D and 3D cells, partitions holding both phases, k-means clusters inside
partitions, and the checks.

1. **Gold-standard reference.** A full-field FE solution on a conforming (body-fitted) mesh with plasticity at the Gauss
   points (a gmsh mesh, our own solver reusing `material.py`), so that errors become absolute rather than relative to
   another method. Spectral LS on a fine voxel mesh, the reference until then, shares every method's staircase geometry.
2. **TFA–LS study.** Derive the exact relation between TFA's P and the LS operator, and compare FFT-TFA with TFA on the
   same clusters: their difference splits into the inclusion's polarization taken uniform per cluster (FFT-TFA at full
   reach against TFA), the truncation (FFT-TFA against itself at full reach) and the Moulinec–Suquet kernel against TFA's
   FE. Against the reference, each method adds the error of one eigenstrain per cluster, which FFT-TFA and TFA share; the
   two errors can have opposite signs and cancel, so a method measured only against the reference can look better than
   its interaction is.
3. **Cost at equal accuracy.** Time and memory against resolution for each method, at the same error, in 2D and 3D. An FFT
   on n lattice points costs about n log n, with n the elements for Spectral LS and the partitions for FFT-TFA, so Spectral
   LS on a mesh as coarse as FFT-TFA's partitions costs about as much per iteration: the two differ in accuracy, FFT-TFA
   seeing the fine geometry through each partition's clusters and its corrections. Memory counts as much as time:
   FFT-TFA's case rests on needing less than SCA and TFA, whose interaction is dense in the clusters, and than Spectral
   LS, which stores every element. Spectral LS here keeps several 6 × 6 arrays per element (its kernel, the LS-24 operator,
   stiffness, tangent); a fair comparison counts a leaner one.

Open for FFT-TFA (`notes/2_ls_model.md`, section 5.5):

- **Truncated interaction:** it does not leave a uniform polarization alone where a partition holds several clusters (the
  patch test), and neighbour blocks can make that worse; a consistent variant that corrects only each partition's
  deviation from its mean passes it, but is less accurate with neighbour blocks on the 126-fiber cell and the sphere (as
  accurate without). Coherent structures that cut partitions (a laminate along its layers) keep a far-field error the
  corrections do not reach. Against FFT-TFA at full reach (the truncation alone), neighbour blocks help the matrix
  clusters in 2D and 3D, and face neighbours alone fall well short of all neighbours; in 3D the inclusion's error can grow
  with more neighbours (the sphere under shear). Untested idea: keep the current corrections and add one term per cluster
  that cancels exactly the leftover a uniform polarization leaves (a 6 × 6 block per cluster, known offline).
- **Inclusion polarization:** one polarization per inclusion cluster; σ33 along z-invariant fibers runs low. Splitting a
  partition's fiber part by fiber makes FFT-TFA stiffer and less accurate, at full reach too: with the matrix as C0, a
  coarser fiber polarization errs soft (Brisard–Dormieux 2010's Hashin–Shtrikman bound), offsetting the stiff error of
  one eigenstrain per matrix cluster, which dominates. To do: refine the matrix clusters until refining the fiber helps,
  so that accuracy does not rest on the two cancelling. Wanted: a richer inclusion polarization that works for any
  geometry, needs no offline solve and keeps one interaction operator.
- **Material states:** how FFT-TFA's and TFA's accuracy and cost change with the number of clusters; with clusters of a
  few elements the clustered model can have several solutions (TFA too).
- **Memory:** online, FFT-TFA stores the partition kernel and one 6 × 6 correction block per pair of clusters in the same
  or (with neighbour blocks) neighbouring partitions, so its memory grows with the partitions times the square of the
  clusters per partition (27 times more with neighbour blocks in 3D), not with the elements. Offline, `get_correction_blocks`
  cross-correlates the clusters of every partition at once, which in 3D with k-means takes more than a Spectral LS solve
  of the whole load path. To do before any 3D study: build them a chunk of partitions at a time (a chunk size or memory
  budget in `config.py`). Every k-means setting also needs six elastic Spectral LS solves on the elements, the floor of
  its offline memory.

Later: a reference stiffness for FFT-TFA that follows plastic softening (needed for a perfectly plastic phase), and the
consistent kernel (the exact partition average).

Field errors against the reference are not computed yet (macroscopic errors are, above). A measure fair across meshes of
different resolution is not settled: candidates are block averages at a scale every mesh resolves (on points chosen by
the reference's phases alone), and element errors on the fine grid split into elements whose phase matches the
reference's and elements whose phase does not.
