# Research context: the operator question and the literature

Rewritten 2026-10-06. **Read cover to cover** (every section, figure and example): Moulinec–Suquet (1998),
Brisard–Dormieux (2010, 2012), Schneider (2019), Schneider's review (2021), Liu–Bessa–Liu (2016), Zhang et al.
(2019), Kabel et al. (2016). Everything stated about these papers is taken from them; quotations were transcribed from
the pages, and the per-paper notes are in `references/reading_notes/`. Papers marked *(not read)* are named only so
they can be checked; nothing here relies on them. Section 5 lists how prior work was searched for.

## 1. One equation, many methods

Pick a homogeneous **reference stiffness** $\mathbf C_0$ and call the stress it gets wrong the **polarization**,
$\boldsymbol\tau=\boldsymbol\sigma-\mathbf C_0\boldsymbol\varepsilon$. Equilibrium then becomes the exact
Lippmann–Schwinger equation (Moulinec–Suquet 1998, eq. 7):

$$
\boldsymbol\varepsilon(\mathbf x)=\bar{\boldsymbol\varepsilon}-\int_Y\boldsymbol\Gamma_0(\mathbf x-\mathbf y)\,
\boldsymbol\tau(\mathbf y)\,d\mathbf y,
$$

where the **Green operator** $\boldsymbol\Gamma_0$ gives the strain the reference medium develops under a
polarization. In Fourier space it is one $6\times6$ block $\widehat{\boldsymbol\Gamma}_0(\mathbf k)$ per wavevector
$\mathbf k$, in closed form for any $\mathbf C_0$ (their eq. 4 and Appendix A), and it depends only on the direction
of $\mathbf k$.

Taking $\boldsymbol\tau$ constant on each partition and averaging over partition $B$ gives our LS model (LS-7):

$$
\boldsymbol\varepsilon^B=\bar{\boldsymbol\varepsilon}-\sum_A\mathbf D^{BA}\boldsymbol\tau^A,\qquad
\mathbf D^{BA}=\frac{1}{|\Omega_B|}\int_{\Omega_B}\int_{\Omega_A}\boldsymbol\Gamma_0(\mathbf x-\mathbf y)\,d\mathbf y\,
d\mathbf x .
$$

The code writes it with $\mathbf P_0=\mathbf D\mathbf C_0$ acting on $\boldsymbol\mu^*=-\mathbf C_0^{-1}
\boldsymbol\tau$ (LS-8). The same equation is solved by:

- **SCA** (Liu–Bessa–Liu 2016, eqs. 16, 22), where the partitions are **clusters**: regions of any shape, possibly
  disconnected, found by k-means on the elastic strain-concentration tensor. $\mathbf D$ is a dense matrix, computed
  by Fourier transform on the fine voxel grid (eq. 27).
- **Brisard–Dormieux** (2010, 2012), where the partitions are identical boxes ("cells"). Then $\mathbf D^{BA}$
  depends only on the offset between $A$ and $B$, so it is a convolution applied by FFT. This is our setting.

So **the methods differ only in how they compute $\mathbf D$**. Schneider (2019) analyses this "clustered
Lippmann–Schwinger equation" (his eq. 2.16) for nonlinear materials, under two assumptions worth knowing: each
cluster lies in one phase (eq. 2.10, our single-phase restriction), and the material hardens (strongly monotone
stress, eqs. 2.1–2.2), which excludes our softening cases. He also notes that "In contrast to the continuous case,
the solutions of the Lippmann–Schwinger equation (2.16) depend on the reference material", which is why
$\mathbf C_0$ changes the LS answer.

**LS and TFA.** Schneider (2019, §4) writes both in the same form for a time step $n\to n+1$, with
$\boldsymbol\varepsilon^p$ the plastic strain:

$$
\text{TFA: }\ \boldsymbol\varepsilon_{n+1}+\boldsymbol\Gamma_0:\big(\mathbf C:(\boldsymbol\varepsilon_{n+1}-
\boldsymbol\varepsilon^p_n)-\mathbf C_0:\boldsymbol\varepsilon_{n+1}\big)=\bar{\boldsymbol\varepsilon}_{n+1},
\qquad
\text{SCA: }\ \text{the same with }\boldsymbol\Gamma_0\to\boldsymbol\Gamma_0^N .
$$

In TFA only the plastic variables are constant per partition and the strain field stays unrestricted, so the full
$\boldsymbol\Gamma_0$ acts; in SCA the strain is also constant per partition, which replaces $\boldsymbol\Gamma_0$ by
its partition average $\boldsymbol\Gamma_0^N$. In his words: "the only difference between the self-consistent
clustering analysis and the transformation field analysis ... is whether the clustered or the non-clustered
Eshelby–Green operator is used." He also notes (§1) that TFA "captures the elastic behavior of the composite
correctly. However, in the nonlinear regime, the stress predicted by the TFA could be much larger than the
'full-field' stress." His §4 argument is formal (not a theorem), and he compares neither method numerically with the
other.

## 2. Computing $\mathbf D$ for box partitions

Let the partitions be boxes of side $h=L/n$ spanning the thickness. A field that is constant on each box has Fourier
coefficients at every in-plane wavevector, not just $n$ per direction. Brisard–Dormieux (2010, p. 666) give them: the
coefficient at $\tfrac{2\pi}{L}(\boldsymbol\xi+n\mathbf m)$ is the lattice's discrete Fourier transform at
$\boldsymbol\xi$ times a sinc factor from the box shape ($\operatorname{sinc}x=\sin x/x$):

$$
\widehat{\boldsymbol\tau}\!\left(\tfrac{2\pi}{L}(\boldsymbol\xi+n\mathbf m)\right)\;\propto\;
\operatorname{sinc}\frac{\pi(\xi_1+nm_1)}{n}\,\operatorname{sinc}\frac{\pi(\xi_2+nm_2)}{n}\;
\mathrm{DFT}(\boldsymbol\tau)(\boldsymbol\xi).
$$

The wavevectors $\boldsymbol\xi+n\mathbf m$ are the **aliases** of $\boldsymbol\xi$: sampled at the partition
centres they look identical, but together they make up the jumps at partition edges. Averaging the response over
the receiving box brings a second sinc, so the exact operator is a weighted sum over aliases (Brisard–Dormieux 2010
eq. 14, 2012 eq. 28; the thickness-spanning form, with only in-plane aliases, is our derivation):

$$
\widehat{\mathbf D}(\boldsymbol\xi)=\sum_{\mathbf m\in\mathbb Z^2}w_{\mathbf m}(\boldsymbol\xi)\,
\widehat{\boldsymbol\Gamma}_0\!\left(\tfrac{2\pi}{L}(\boldsymbol\xi+n\mathbf m)\right),\qquad
w_{\mathbf m}=\operatorname{sinc}^2\frac{\pi(\xi_1+nm_1)}{n}\,\operatorname{sinc}^2\frac{\pi(\xi_2+nm_2)}{n},
\qquad\sum_{\mathbf m}w_{\mathbf m}=1 .
$$

The weights sum to one (Brisard–Dormieux 2012, eq. 52), so $\widehat{\mathbf D}$ is a weighted average of
$\widehat{\boldsymbol\Gamma}_0$ over the alias directions. The known operators are different choices of weights:

| Operator | Aliases kept | Character |
|---|---|---|
| **Consistent** (Brisard–Dormieux) | all, exact $w_{\mathbf m}$ | Exactly LS-7. Infinite sum, precomputed once per grid. |
| **Moulinec–Suquet** | $\mathbf m=\mathbf 0$ only, weight 1 | Closed form. Wrong near the highest lattice frequency $\xi=n/2$, where the true weight is $(2/\pi)^2\approx0.41$ per direction. |
| **Filtered** (Brisard–Dormieux 2012, eqs. 44–45) | 2 per direction, $\cos^2$ weights | MS on a grid twice as fine, averaged back. |
| **Fine grid, averaged** (how SCA computes $\mathbf D$) | $k$ per direction | Tends to consistent as $k$ grows. MS is $k=1$, filtered $k=2$ (our observation). |
| **FE kernel** (our code, LS-14) | — | The FE discretization's Green operator, averaged. Not of this form. |

Notes on each:

- **Consistent.** There is no closed form; Brisard–Dormieux (2010, §3.4) evaluate it numerically, and report that
  the alias series "converge very slowly". Their Appendix C fixes this with a tail estimate (truncated sum plus the
  mean of two integral bounds): "in the worst case, the error scales as T⁻¹ in the initial scheme, and as T⁻² in our
  improved scheme", for T retained terms, and the method extends to double sums. They also show the operator matters
  even at fine resolution: the consistent and plain operators can differ by up to 30 % "for M = N = 1024", which
  "indicates that introducing the periodized Green operator is the appropriate way of accounting for finite-resolution
  effects". In 3D the series converge "very slowly" (2012, §4.2); our partitions span the thickness, which makes our
  sum two-dimensional like theirs.
- **Moulinec–Suquet.** Their own view (1998, §3.3.3) is that the discrete Fourier transform is exact only for fields
  with a cut-off frequency, and "a discontinuous field has no cut-off frequency and there is no discretization able to
  capture this discontinuity". They expect convergence as resolution increases. With plasticity, resolution matters:
  at 32 × 32 pixels per fibre their flow-stress error is "about 15 % for the square array of fibers in an
  elastic-perfectly plastic matrix under tension at 0°" (§3.4; elastic stiffness errors stay below 1 %). Our partition
  lattice is at that resolution. Separately, they report oscillations for grids below 128 from the highest frequency
  of even grids, which they removed by redefining the operator there (§2.4.2); this is a fixed implementation detail,
  not a property of coarse grids. The review's Fig. 2 (linear elastic sphere, 128³ voxels) shows that "In the
  inclusion, Moulinec–Suquet's discretization shows ringing artifacts" (ringing: spurious oscillations near interfaces).
- **FE kernel.** The review lists FFT solvers for "trilinear hexahedral elements on a regular periodic grid" (its
  ref. 99) as one discretization among others; our kernel is that discretization's Green operator, averaged over
  partitions (our reading). The LS note says these operators "are not asserted to equal the continuum operators at
  finite N". It is kept from the TFA work and will be phased out unless it shows an advantage. It is the only kernel
  under which LS and TFA become the same equation (`matched_stiffness_control`), since TFA's $\mathbf P$ comes from the
  same FE mesh.

## 3. Consistent against non-consistent

Brisard–Dormieux (2012) view both schemes as **Galerkin approximations**: write the LS equation as
$a(\boldsymbol\tau,\boldsymbol\varpi)=\ell(\boldsymbol\varpi)$ for every test field $\boldsymbol\varpi$ (their eq.
10, the "weak form of the Lippmann–Schwinger equation"), then restrict $\boldsymbol\tau$ and $\boldsymbol\varpi$ to
fields constant on each cell. The schemes differ in whether $a$ is then evaluated exactly:

$$
\text{consistent: } a_h=a,\qquad\text{non-consistent: } a_h\neq a .
$$

- **Consistent** (§4.2): "operators a and ℓ are computed exactly on V^h. In other words the solution τ ∈ V of the
  exact problem (10) satisfies the approximate problem (21)." The only error is that the true $\boldsymbol\tau$ is not
  constant per cell; convergence as cells shrink follows from Céa's lemma.
- **Non-consistent, MS** (§4.3): "the bilinear form a is not computed exactly and the approximation is in fact
  non-consistent." The extra operator error vanishes only as cells shrink ("asymptotically consistent", Theorem 7);
  convergence follows from Strang's lemma.

The source of MS's error, in their words: MS "suggest that ε^{h,n}_β should be understood as a point-wise estimate"
of the strain at the cell centre, whereas "it is more natural to consider that ε^{h,n}_β is the step value on Ω^h_β
of a cell-wise constant function". A point sample of a smooth field has no high aliases; a step function has all of
them. The review summarizes this as MS being "an under-integrated variant of the conforming Galerkin discretization".

**What it changes in practice** (2012, §5; 2D plane strain, square inclusion, shear):
- At finite contrast (inclusion 100 times softer than the matrix, comparable to our 100:1): "the consistent method is
  slightly more accurate than the non-consistent method, both methods being approximately of order one in h."
- At infinite contrast (a pore): the non-consistent stress field "exhibits a 'checkerboard' pattern" while the
  consistent one is smooth, and "the shortcomings of the non-consistent scheme originate in an inaccurate treatment of
  the highest frequencies". The filtered operator gives results "practically undistinguishable" from the consistent one.

Their analysis assumes isotropic phases and reference (§3.2); our `"homogenized"` $\mathbf C_0$ is anisotropic, so it
falls outside it. All their results are linear elastic.

**For us:** with the consistent operator, partitioned LS adds no error beyond the LS model itself. How much that
gains over MS at our contrast, under plasticity, is not known from the literature read.

## 4. What has been done

| Work | Partitions | Material | Operator for $\mathbf D$ | Difference from ours |
|---|---|---|---|---|
| Brisard–Dormieux 2010, 2012 | uniform cells, may hold several phases | linear elastic, 2D examples | consistent, filtered, MS | no plasticity |
| Schneider 2019 | any shape, single-phase | nonlinear, hardening (theory) | numerics: k-means clusters, staggered-grid FFT | no uniform-partition numerics, no TFA comparison |
| SCA, Liu–Bessa–Liu 2016 | k-means clusters | J2 plasticity, contrast 5:1 | Fourier transform on the fine grid, dense | not uniform, no FFT online |
| Zhang et al. 2019 | clusters (self-organizing maps in the examples), via a uniform grid | J2 plasticity, contrast 5:1 and 10:1 | analytic cell–cell integrals of the infinite-medium Green function | not periodic; grid only an intermediate step |
| Composite voxels, Kabel et al. 2016 | uniform coarse voxels, mixed phases | J2 plasticity | the coarse grid's own (staggered grid) | fine scale enters only the material law |

The closest, in their own words:

- **Brisard–Dormieux** handle cells holding several phases: "When a real microstructure is discretized into a
  relatively small number of pixels (coarse grid), it is highly probable that each pixel contains more than one
  phase." Their rule (2010 eq. 10, 2012 eq. 23) averages the inverse stiffness contrast over the cell,
  $(\mathbf C_h-\mathbf C_0)^{-1}=\langle(\mathbf C-\mathbf C_0)^{-1}\rangle_{\text{cell}}$, and "depends on the
  composition of the heterogeneous cell, but not on the spatial organization". With the reference equal to one phase
  they enforce zero polarization in that phase (2010, §4.1); our reading is that the rule then gives any cell
  containing that phase the reference stiffness, so it would not serve as our mixing rule with
  `reference_stiffness = "matrix"`. Elastic only.
- **Schneider 2019** extends "the work of Brisard–Dormieux [45,46], who investigated the clustered Lippmann–Schwinger
  equation for identically shaped voxel clusters, to arbitrary cluster shape and nonlinear material behavior".
  Uniform partitions with hardening plasticity are covered by his convergence theory, not by his examples.
- **Liu–Bessa–Liu 2016** test spatial clusters against their mechanics-based ones (Fig. 6): "the convergence using
  the position-based clusters is poor and the accuracy of the prediction does not change significantly even
  considering 256 clusters for phase 1. On the contrary, with the same number of clusters the predictions from A-based
  clustering reproduce the DNS almost exactly." Uniform partitions are spatial clusters. They also find a matrix
  reference too stiff once plasticity spreads, hence their self-consistent update of $\mathbf C_0$ (Fig. 5), matching
  our observation that LS error grows with plasticity.
- **Zhang et al. 2019**: "We first cast a cubic/rectangular coarse grid over the representative volume element.
  Using analytical expressions for the integral of the Green's functions, we then calculate interaction tensors on
  the coarse grid." The cell–cell integral "only depends on the relative position" of the cells. But they use an
  infinite surrounding medium ("Instead of using the periodic boundary conditions, we introduce a fictitious
  homogeneous isotropic material ... surrounding the original domain"), and the grid only serves to approximate the
  interaction tensors of clusters.
- **Composite voxels** (Kabel–Fink–Ospald–Schneider 2016): "sub-voxels are merged into bigger voxels to which an
  effective material law based on laminates is assigned". For plasticity each phase's law is linearized, the
  linearized stiffnesses are mixed, and "The internal variables of the constituents are updated due to the
  deformation" of the composite voxel. The coarse grid is solved with its own operator. At half resolution the
  relative error they plot for the tensile curve starts near 20 %, and "working on a coarser resolution for physically nonlinear problems is only
  acceptable in combination with laminate mixing."

## 5. What was not found, and how it was searched

**Search.** (1) The eight papers above, read cover to cover. (2) The titles of all 345 papers that Semantic Scholar
lists as citing Brisard–Dormieux 2010, Brisard–Dormieux 2012 or Schneider 2019 (October 2026), all scanned by eye.
Titles only flag candidates; none was read.

**Not found in the papers read:**
- **Our scheme with plasticity**: uniform partitions, a partition-averaged Green operator, FFT on the partition
  lattice, J2 plasticity. It is the natural extension of Brisard–Dormieux and is covered by Schneider's theory for
  hardening materials, so it would read as an extension rather than a new method.
- **LS and TFA compared numerically on identical partitions.** Schneider relates them formally (section 1); nobody in
  the papers read computes both.

**Candidates from the citation scan, not read** (titles only; their content is unknown):
- "Numerical homogenization method for heterogeneous materials based on Hill tensor for cuboids and Fourier
  Transform" (2026). The title suggests box-shaped cells with analytic interactions, close to section 2.
- "Model order reduction of nonlinear homogenization problems using a Hashin–Shtrikman type finite element method"
  (Wulfinghoff, Cavaliere, Reese, CMAME 330, 2018). Schneider (2019) states it is equivalent to SCA.
- "Mathematical foundations of FEM-cluster based reduced order analysis method and a spectral analysis algorithm for
  improving the accuracy" (Li, Nie, Cheng, Comput. Mech. 2022). TFA-type cluster method; may compare with SCA.
- "A variational form of the equivalent inclusion method for numerical homogenization" (2014) and "Efficient
  reduced-order model based on n-point bounds for homogenization of elastic composites" (2026).
- "Adaptive selection of reference stiffness in virtual clustering analysis" (2021).
- "Accurate and consistent composite voxel methods for digital images in computational micromechanics" (2025), and
  Kabel–Fink–Schneider, "The composite voxel technique for inelastic problems" (CMAME 322, 2017).
- Eloh–Jacques–Berbenni, "Development of a new consistent discrete Green operator ..." (Int. J. Plasticity 116, 2019).
  The review groups it with "further finite difference discretization schemes [90–94]".
- Buryachenko, "Transformation field analysis as a background of clustering discretization methods" (2023); not in
  the citation lists above, found by web search.

## 6. Consequences for this project

- With single-phase partitions the partition grid already resolves the geometry, so the fine mesh enters LS only
  through $\mathbf P_0$. With the consistent operator, $\mathbf P_0$ contains no mesh at all and partitioned and
  coarse LS coincide. Comparing them then measures the operator, which Brisard–Dormieux did for linear elasticity
  (section 3).
- Two cautions from the literature. Spatial clusters converged poorly in Liu–Bessa–Liu's test, and coarse
  resolution costs accuracy once plasticity starts (Moulinec–Suquet: about 15 % flow-stress error at 32 pixels per
  fibre; Kabel et al.: relative error near 20 % at half resolution). Uniform partitions will likely need many
  partitions; the FFT is what makes many affordable.
- What follows for the project, adopted as its research question and roadmap (README, "Status and roadmap"):
  1. **Analytic kernels** (roadmap step 2): consistent, filtered and MS, extending Brisard–Dormieux 2012 beyond
     elasticity.
  2. **The TFA–LS relationship** (step 4), the main thread: Schneider's §4 gives the formal link (full against
     averaged Green operator), and nobody in the papers read measures it. Can TFA be written as LS plus a correction
     for the strain varying inside partitions, how large is it, and do the two converge at the same rate? A first
     data point from a scratch test: no $\mathbf C_0$ reproduced the translation average of TFA's $\mathbf P$.
  3. **Cost at equal accuracy** (step 5): many uniform partitions with an FFT, $O(M\log M)$, against a few adaptive
     clusters with a dense $\mathbf D$, $O(K^2)$.

## Sources

Read cover to cover:
- Moulinec, Suquet (1998), CMAME 157, 69–94. `references/`
- Brisard, Dormieux (2010), Comput. Mater. Sci. 49, 663–671. `references/`
- Brisard, Dormieux (2012), CMAME 217–220, 197–212. `references/`
- Schneider (2019), On the mathematical foundations of the self-consistent clustering analysis, CMAME 354, 783–801.
  `references/`
- Schneider (2021), A review of nonlinear FFT-based computational homogenization methods, Acta Mech. 232, 2051–2100.
  `references/`
- Liu, Bessa, Liu (2016), Self-consistent clustering analysis, CMAME 306, 319–341 (author preprint).
  [preprint](https://zeliangliu.com/publication/liu-2016-self/preprint.pdf)
- Zhang, Tang, Yu, Zhu, Liu (2019), Fast calculation of interaction tensors in clustering-based homogenization,
  Comput. Mech. 64, 351–364. [PDF](https://par.nsf.gov/servlets/purl/10103937)
- Kabel, Fink, Ospald, Schneider (2016), Nonlinear composite voxels and FFT-based homogenization, ECCOMAS Congress.
  [PDF](https://files.eccomasproceedia.org/papers/eccomas-congress-2016/5977.pdf?mtime=20170308161923)

Not read: the candidates in section 5; Nasirov–Oskay (2024), IJNME (abstract and introduction only); Cheng, Li, Nie,
Li (2019), CMAME 348, 157–184 (known only from a later paper's summary). The LS note also cites Brisard–Legoll (2015)
and Gehrig–Schneider (2025), which were not read for this review.
