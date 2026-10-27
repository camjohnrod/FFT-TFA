# FFT-TFA

Reduced-order homogenization of elastoplastic composites: a stiff fiber in a softer J2-plastic matrix, in a periodic
cube. The cell is seen at three scales:

- **elements:** the fine mesh of 8-node hexahedra, which resolves the geometry;
- **partitions:** uniform blocks of k × k element columns (k × k × k blocks if the geometry varies along z);
- **clusters:** regions of one phase with one material state (one uniform plastic eigenstrain) each. By default each
  partition's part in each phase: one cluster if the partition lies in one phase, two if the fiber's edge cuts
  through it. CP and TFA can each take k-means clusters instead, which may span many partitions; a cluster's part in
  one partition is a **piece**.

Every method keeps one material state per cluster. They differ only in how those states interact:

| Method | How clusters interact |
|---|---|
| **CP** | The method under development. The partition-averaged Lippmann–Schwinger (LS) equation: partitions interact through a homogeneous reference medium of stiffness C0, which is one FFT convolution with the Moulinec–Suquet kernel. Every piece of a partition carries the partition's polarization σ − C0 ε, and every piece of a cluster the cluster's eigenstrain, updated at the cluster's average strain. (`notes/2_ls_model.md`) |
| **Spectral LS** | The same equation with one partition per element: the standard Moulinec–Suquet spectral method, as a comparison. |
| **TFA** | The baseline. Clusters interact through the composite's exact FE influence functions E and P: exact in elasticity, but P is dense. (`notes/3_tfa_baseline.md`) |

All three solve each load step by Newton's method, with GMRES for each correction and a backtracking line search.

**Research question.** With the same material states, how does the interaction operator decide accuracy and cost?
(`notes/1_research_context.md`)

## Quick start

```bash
python code/run.py      # solve the problem in code/config.py with every method
python code/check.py    # check the code (about 15 seconds)
```

The first run with TFA on a new configuration builds E and P and caches them in `code/cache/`. That takes minutes and
memory of (6 × TFA's clusters)² doubles: 0.3 GB with 32² partitions, 4.8 GB with 64² or 16³ (far less with k-means
clusters). Later runs reuse the cache.
In 3D the FE factorization is the limit too: with 7 GB of memory a 24³ mesh works and 36³ runs out of memory.
CP and Spectral LS have no such limit; switch TFA off for finer 3D meshes.

Dependencies: `numpy`, `scipy`, `matplotlib`, `tqdm` (in the untracked `.venv/`).

## Changing the problem

Every setting is in `code/config.py`, commented, from the settings changed most to those changed least; there are no
command-line options. Its header has a table of what the mesh, the partitions and the clusters mean to each method.
The ones you will change most:

- **Geometry:** `inclusion_shape` is `"circle"` (a fiber along z), `"sphere"`, `"laminate"` (interfaces normal to
  x) or `"laminate_z"` (normal to z). `element_number_per_side` sets the mesh; `partition_number_per_side` must
  divide it. The circle and the x laminate are constant along z, so they use `element_number_along_z` layers (one is
  enough) and partitions that are columns; the sphere and the z laminate are meshed and partitioned in 3D.
- **Materials:** stiffness, yield stress and hardening per phase. A phase stays elastic if its yield stress is
  `np.inf`.
- **Load path:** `max_macro_strain`, `load_path_shape` (`"monotonic"` or `"cyclic"`), `strain_increment_count`.
- **Methods:** CP always runs; `include_spectral_ls` and `include_tfa` add the others.
- **Clusters:** `cp_clusters` and `tfa_clusters`, each `"k-means"` (default) or `"partition phases"`: k-means clusters
  of the elements' elastic strain concentration, found in each phase apart, up to `k_means_matrix_clusters` and
  `k_means_inclusion_clusters` (one is enough for an elastic inclusion). Spectral LS keeps one cluster per element.
- **CP's reference stiffness:** C0 = `cp_reference_stiffness_scale` × `cp_reference_stiffness` (default 0.05 ×
  `"matrix"`). C0 must stay below the tangent stiffness of every cluster sharing a partition; CP stops and says so if
  it doesn't.
- **Figures:** colour limits of the maps, and `von_mises_matrix_only` (default on) to grey out the inclusion in the von
  Mises maps so their scale follows the matrix.
- **Threads:** `thread_count` (default 1) caps every method alike.

## Output

`run.py` prints a table of each method's offline time (building its operators), online time (solving the load path),
Newton and GMRES iterations and load steps completed, and saves to `code/output/`:

- `stress_strain.png`: macroscopic stress against applied strain;
- `cross_section.png`: each method's discretization of the cross-section: CP's partitions (blue) and clusters (black),
  Spectral LS's voxels, TFA's clusters, with k-means clusters coloured in and a legend;
- `von_mises_stress.png`, `plastic_strain.png`: maps at the last load step, one per method (each cluster at its
  average);
- `timing.txt`: the table.

A method that cannot solve a load step stops there and the table says why; the others carry on.

## Checking the code

```bash
python code/check.py         # run the checks
python code/check.py save    # run them and, if they pass, save the results as the reference
```

`check.py` solves seven small problems with every method, using its own settings (it never touches your `config.py`,
cache or outputs): a laminate, a laminate whose interfaces cut through partitions, a circle, a laminate normal to z and
a sphere (the last two in 3D), then the laminate and the sphere with k-means clusters. On the laminates every method
must match the exact solution (CP on the cut laminate only up to its modelling error, printed for information). Once a
reference is saved, every result must reproduce it bit for bit. For a change that shouldn't alter results, run `check.py
save` before it and `check.py` after.

## Code layout

| File | Contents |
|---|---|
| `code/config.py` | every setting |
| `code/run.py` | the entry point: builds the problem, runs each method along the load path, writes the table and figures |
| `code/check.py` | the checks, with the laminate's exact solution |
| `code/fe.py` | the mesh, partitions and clusters; finite elements and TFA's E and P, with their cache |
| `code/ls_model.py` | the LS equation (CP and Spectral LS): C0, the kernel P0, the Fourier preconditioner, the Newton solver |
| `code/tfa_model.py` | the TFA equation and its Newton solver |
| `code/newton.py` | what all methods share: the Newton loop, line search, convergence test, GMRES |
| `code/material.py` | the J2 return map |
| `code/plots.py` | the figures |
| `notes/` | the theory, in reading order: research context, the LS model, the TFA baseline, background derivations |
| `references/` | the papers the notes cite; `reading_notes/` has one note per paper read cover to cover |

## Status and roadmap

Working: the three methods on 2D and 3D cells, partitions holding both phases, k-means clusters for CP and TFA, and
the checks.

1. **Thickness check (done).** One element layer through the thickness reproduces three to round-off.
2. **Moulinec–Suquet kernel (done).** TFA keeps its FE operators, so a CP-vs-TFA difference includes the
   FE-vs-Moulinec–Suquet discretization gap.
3. **Gold-standard reference.** A full-field FE-consistent FFT solver (Ladecký et al.) with plasticity at the Gauss
   points, so that errors become absolute rather than relative to another method.
4. **TFA–LS study.** Derive the exact relation between TFA's P and the LS operator, and measure TFA, CP and Spectral
   LS against the reference.
5. **Cost at equal accuracy.** Time and memory against resolution for each method, at the same error.

Open: how CP's and TFA's accuracy and cost change with the number of k-means clusters (`notes/2_ls_model.md`,
section 9).

Later: a reference stiffness that follows plastic softening (needed for a perfectly plastic phase), conjugate
gradients on Brisard–Dormieux's symmetric form, and the consistent kernel (the exact partition average).
