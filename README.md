# FFT-TFA

Reduced-order homogenization of elastoplastic composites: stiff inclusions (one fiber or many, a sphere, a layer)
in a softer J2-plastic matrix, in a periodic cube. The cell is seen at three scales:

- **elements:** the fine mesh of 8-node hexahedra, which resolves the geometry;
- **partitions:** uniform blocks of elements, the cells of FFT-TFA's (and CP's) FFT: k × k columns through the
  thickness, or layers of them, if the geometry does not vary along z, k × k × k blocks otherwise;
- **clusters:** regions of one phase with one material state (one uniform plastic eigenstrain) each: each partition's
  part in each phase (one cluster if the partition lies in one phase, two if an interface cuts through it), k-means
  clusters of the elements' elastic strain concentration inside each such part, or k-means clusters over the whole
  cell, which may span many partitions. A cluster's part in one partition is a **piece**.

Every method keeps one material state per cluster. They differ only in how those states interact:

| Method | How clusters interact |
|---|---|
| **FFT-TFA** | The method under development. TFA's clusters, interacting through the Lippmann–Schwinger (LS) equation of a homogeneous medium of the matrix's stiffness: exact for the matrix's polarization, uniform per piece in the inclusion. The interaction is one FFT on the partitions, on each partition's mean polarization, plus exact corrections between the pieces of each partition (and, optionally, of its neighbours). (`notes/2_ls_model.md`, section 10) |
| **Spectral LS** | The LS equation on the elements: the standard Moulinec–Suquet spectral method, as a comparison. |
| **TFA** | The baseline. Clusters interact through the composite's exact FE influence functions E and P: exact in elasticity, but P is dense. (`notes/3_tfa_baseline.md`) |
| **CP** | The earlier method, off by default, to be removed once FFT-TFA has replaced it. The partition-averaged LS equation with one polarization σ − C0 ε per partition and a reference stiffness C0 below every tangent. (`notes/2_ls_model.md`, sections 1–9) |

All of them solve each load step by Newton's method, with a Krylov method for each correction (GMRES; conjugate
gradients for Spectral LS, whose Newton system is symmetric) and a backtracking line search.

**Research question.** With the same material states, how does the interaction operator decide accuracy and cost?
FFT-TFA and TFA share their material model, so their difference is how the clusters interact: FFT-TFA takes the
inclusion's polarization uniform per piece (exact in the matrix, with the matrix as reference medium), truncates the
interaction to nearby partitions, and uses the Moulinec–Suquet kernel where TFA uses FE. With the truncation removed,
FFT-TFA is SCA's clustered equation with the matrix as reference medium. (`notes/1_research_context.md`,
`notes/2_ls_model.md` section 10.4)

## Quick start

```bash
python code/run.py      # solve the problem in code/config.py with every method
python code/check.py    # check the code (about a minute)
```

The first run with TFA on a new configuration builds E and P and caches them in `code/cache/`; later runs reuse
them. P takes (6 × TFA's clusters)² doubles: with partition phases 0.3 GB at 32² partitions and 4.8 GB at 64² or 16³,
far less with k-means clusters. In 3D the FE factorization limits the mesh too (with 7 GB a 24³ mesh works, 36³ does
not); FFT-TFA and Spectral LS have no such limit, so switch TFA off for finer 3D meshes.

Dependencies: Python 3.12 with `numpy`, `scipy`, `matplotlib` and `tqdm`.

## Changing the problem

Every setting is in `code/config.py`, commented, from the settings changed most to those changed least; there are no
command-line options. Its header has a table of what the mesh, the partitions and the clusters mean to each method.
The ones you will change most:

- **Geometry:** `inclusion_shape` is `"fiber"` (a circular fiber along z), `"random fibers"` (`fiber_count`
  non-overlapping fibers along z at random periodic positions, `fiber_volume_fraction` in all), `"sphere"`,
  `"laminate"` (interfaces normal to x) or `"laminate_z"` (normal to z). `elements_per_side` sets the mesh;
  `partitions_per_side` must divide it. The fibers and the x laminate are constant along z: they use
  `elements_along_z` element layers (one is enough) and `partitions_along_z` partition layers (one gives
  columns; more give the same answer, except for FFT-TFA without neighbour blocks, whose truncation then changes it:
  keep columns). The sphere and the z laminate are meshed and partitioned in 3D.
- **Materials:** stiffness, yield stress and hardening per phase. A phase stays elastic if its yield stress is
  `np.inf`.
- **Load path:** `max_macro_strain`, `load_path_shape` (`"monotonic"` or `"cyclic"`), `strain_increment_count`.
- **Methods:** FFT-TFA always runs; `include_spectral_ls` and `include_tfa` (default on) and `include_cp` (default off)
  add the others. `spectral_ls_elements_per_side` (default `None`: the mesh's) runs Spectral LS on its own elements,
  coarser or finer, without changing the other methods or the k-means features.
- **Reference:** `include_reference` (default on) adds a fine solution to compare with: Spectral LS on
  `reference_elements_per_side` elements per side (a multiple of the mesh's), solved once and cached in
  `code/cache/` with its macroscopic stress at every step and its fields at the last one. Any change of geometry,
  materials, load path or solver tolerance solves it again; a first solve can take tens of minutes, so run it in the
  background. It is still Moulinec–Suquet on voxels (the same staircase, ringing at high contrast), until a conforming
  FE reference exists (roadmap step 3).
- **Clusters:** `fft_tfa_clusters` and `tfa_clusters` (default `"partition k-means"`) and `cp_clusters`, each
  `"partition phases"`, `"partition k-means"` (up to `k_means_matrix_clusters_per_partition` and
  `k_means_inclusion_clusters_per_partition` in each partition's part in each phase) or `"k-means"` (over the cell, up
  to `k_means_matrix_clusters` and `k_means_inclusion_clusters`), on the elements' elastic strain concentration,
  smoothed within each phase to remove the spectral ringing at interfaces.
  One inclusion cluster is enough for an elastic inclusion. Spectral LS keeps one cluster per element.
- **FFT-TFA's neighbour blocks:** `fft_tfa_neighbour_blocks` (default on) keeps the exact interaction between
  neighbouring partitions too, at about 9 (columns) or 27 (3D) times the cost of the corrections.
- **CP's reference stiffness:** C0 = `cp_reference_stiffness_scale` × `cp_reference_stiffness` (default 0.05 ×
  `"matrix"`). C0 must stay below the tangent stiffness of every cluster sharing a partition; CP stops and says so if
  it doesn't. FFT-TFA's reference medium is always the matrix.
- **Figures:** colour ranges of the maps (unset ends follow the reference's map, or Spectral LS's when no map is
  finer), and
  `von_mises_matrix_only` (default on) to grey out the inclusion in the von Mises maps.
- **Threads:** `thread_count` (default 1) caps every method alike.

## Output

`run.py` prints a table of each method's offline time (building its operators), online time (solving the load path),
Newton and Krylov iterations and load steps completed, and saves to `code/output/`:

- `stress_strain.png`: macroscopic stress against applied strain (the reference, when on, solid black);
- `cross_section.png`: each method's discretization of the cross-section: FFT-TFA's (and CP's) partitions (blue) and
  clusters (black), Spectral LS's voxels, TFA's clusters, with k-means clusters coloured in and a legend;
- `von_mises_stress.png`, `plastic_strain.png`: maps at the last load step, one per method (each cluster at its
  average), and the reference's at its own resolution, which then sets the colour scale;
- `plastic_strain_profile.png`: the accumulated plastic strain along the middle element row (the horizontal line
  through the maps' middle), left to right, every method and the reference on one plot, one step per element, the
  fibers shaded;
- `timing.txt`: the table, and the reference's mesh and build time (it is not timed against the methods);
- with the reference on, `macro_stress_error.png`: each method's signed macroscopic stress error in every stress
  component the load path produces (reaching `loaded_stress_fraction` of the largest), in % of that component's
  largest reference value, against the load step; and `errors.txt`: its largest value and root mean square over the
  path, per component.

A method that cannot solve a load step stops there and the table says why; the others carry on.

## Checking the code

```bash
python code/check.py         # run the checks
python code/check.py save    # run them and, if they pass, save the results as the reference
```

`check.py` solves seven small problems with every method, using its own settings (it never touches your `config.py`,
cache or outputs): a laminate, a laminate whose interfaces cut through partitions, a fiber, a laminate normal to z and
a sphere (the last two in 3D), then the laminate and the sphere with k-means clusters. On the laminates every method
must match the exact solution (FFT-TFA and CP on the cut laminate, and FFT-TFA on the k-means laminate, only up to
their modelling error, printed for information), and FFT-TFA must match it on every laminate with its interaction
made exact. FFT-TFA's interaction must match the element-level LS operator, and every method must stop at the same
accuracy. The fine reference, solved at the working resolution, must equal Spectral LS bit for bit. Once a reference
is saved, every result must reproduce it bit for bit. For a change that shouldn't alter
results, run `check.py save` before it and `check.py` after.

## Code layout

| File | Contents |
|---|---|
| `code/config.py` | every setting |
| `code/run.py` | the entry point: builds the problem, runs each method along the load path, writes the table and figures |
| `code/check.py` | the checks, with the laminate's exact solution |
| `code/fe.py` | the mesh, partitions and clusters; finite elements and TFA's E and P, with their cache |
| `code/fft_tfa_model.py` | FFT-TFA: its regions, its interaction (partition FFT and correction blocks) and its Newton solver, preconditioned by block-Jacobi |
| `code/ls_model.py` | the LS equation (Spectral LS and CP): the kernel P0, LS-24, the Fourier preconditioner, the lattice Newton solver, Spectral LS's conjugate gradients, CP's reference stiffness and piece law |
| `code/tfa_model.py` | the TFA equation and its Newton solver |
| `code/newton.py` | what all methods share: the Newton loop, line search, convergence test, GMRES |
| `code/material.py` | the J2 return map |
| `code/plots.py` | the figures |
| `notes/` | the theory, in reading order: research context, the LS model, the TFA baseline, background derivations |
| `references/` | the papers the notes cite; `reading_notes/` has one note per paper read cover to cover |

## Status and roadmap

Working: FFT-TFA, Spectral LS, TFA and CP on 2D and 3D cells, partitions holding both phases, partition and
cell-wide k-means clusters for every method but Spectral LS, and the checks.

1. **Thickness check (done).** One element layer through the thickness gives the same answer as three, to round-off.
2. **Moulinec–Suquet kernel (done).** TFA keeps its FE operators, so a CP-vs-TFA difference includes the
   FE-vs-Moulinec–Suquet discretization gap.
3. **Gold-standard reference.** A full-field FE solution on a conforming (body-fitted) mesh with plasticity at the
   Gauss points (a gmsh mesh, our own solver reusing `material.py`), so that errors become absolute rather than
   relative to another method. Spectral LS on a fine voxel mesh, the reference until then, shares every method's
   staircase geometry.
4. **TFA–LS study.** Derive the exact relation between TFA's P and the LS operator, and compare FFT-TFA with TFA on
   the same clusters: their difference splits into the inclusion's polarization taken uniform per piece (FFT-TFA at
   full reach against TFA), the truncation (FFT-TFA against itself at full reach) and step 2's kernel gap. Against the
   reference, each method adds the error of one eigenstrain per cluster, which FFT-TFA and TFA share; the two errors
   can have opposite signs and cancel, so a method measured only against the reference can look better than its
   interaction is.
5. **Cost at equal accuracy.** Time and memory against resolution for each method, at the same error, in 2D and 3D.
   An FFT on n lattice points costs about n log n, with n the elements for Spectral LS and the partitions for FFT-TFA,
   so Spectral LS on a mesh as coarse as FFT-TFA's partitions costs about as much per iteration: the two differ in
   accuracy, FFT-TFA seeing the fine geometry through each partition's pieces and its corrections.
6. **Remove CP** once FFT-TFA has replaced it: `include_cp`, its settings, its part of `ls_model.py` and its checks.

Open for FFT-TFA (`notes/2_ls_model.md`, section 10.5):

- **Truncated interaction:** it does not leave a uniform polarization alone where a partition holds several pieces
  (the patch test), and neighbour blocks can make that worse; a consistent variant that corrects only each
  partition's deviation from its mean passes it, but is less accurate with neighbour blocks on the 126-fiber cell
  and the sphere (as accurate without). Coherent structures
  that cut partitions (a laminate along its layers) keep a far-field error the corrections do not reach.
- **Inclusion polarization:** one polarization per inclusion piece; σ33 along z-invariant fibers runs low as for CP,
  less so.
- **Material states:** how FFT-TFA's and TFA's accuracy and cost change with the number of clusters; with clusters
  of a few elements in one partition the clustered model can have several solutions (TFA too).
- **k-means laminate check:** a laminate's matrix strain concentration is uniform, so k-means can divide the matrix
  only on solver noise. Since Spectral LS solves by conjugate gradients the noise is at round-off and the matrix gets
  one cluster, so the check no longer tests matrix clusters that divide partitions; it needs a split that does not
  depend on noise.

Open for CP (`notes/2_ls_model.md`, section 9):

- **Material states:** how CP's accuracy and cost change with the number of clusters.
- **Reference stiffness:** C0 changes CP's answer, but every C0 below every tangent splits a mixed partition at
  equal stress or softer (section 4.4), so no allowed C0 removes that bias.
- **Polarization inside a partition:** CP's main interaction error in fine microstructures. Under in-plane loads it
  comes from the plastic strain varying inside each piece (bands between close fibers), not from the jump at the
  interface: one polarization per piece barely reduces it. Wanted: a richer polarization that works for any
  geometry, needs no offline solve and keeps one interaction operator.

Later: a reference stiffness that follows plastic softening (needed for a perfectly plastic phase), conjugate
gradients on Brisard–Dormieux's symmetric form, and the consistent kernel (the exact partition average).

Field errors against the reference are not computed yet (macroscopic errors are, above). A measure fair across
meshes of different resolution is not settled: candidates are block averages at a scale every mesh resolves (on
points chosen by the reference's phases alone), and element errors on the fine grid split into elements whose phase
matches the reference's and elements whose phase does not. The earlier regional comparisons (prototype studies) kept
only elements in the matrix on both meshes, which favours coarser meshes; their macroscopic errors are unaffected.
