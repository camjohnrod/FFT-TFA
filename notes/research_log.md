# Research log

The state of the research behind the code: what works, the goal (a paper), the findings so far and what is still
open, with dates. Nothing here is needed to use or read the code; the README and `notes/fft_tfa_lecture.md` are.

Working: FFT-TFA, SCA, Spectral LS and TFA on 2D and 3D cells, partitions holding both phases, k-means clusters inside
partitions, and the checks.

**The goal: a paper (CMAME or IJNME).** Its claim: an interaction for clustered models that is exact inside each
partition and summarizes every other partition by polynomial moments carried by one FFT, converging with the degree to
SCA's clustered equation with C0 = L_m, at a cost and memory that grow with the partitions × p², never with the clusters
squared. TFA, SCA, Spectral LS and FFT-TFA become one family with three dials: the material states, the inclusion's
polarization and the degree. Shown so far: convergence with the degree (the cut laminate is exact from degree 5), local
fields about 2–3× faster than Spectral LS at equal error in 2D, less memory than TFA and SCA. Not yet: macroscopic
stress beats Spectral LS only at moderate accuracy in 2D, and not on a single 3D sphere; the reference is Spectral LS
itself; 3D with many inclusions, where FFT-TFA should win, has only a first small test (below).

**Direction (2026-11-13).** The claim narrows to two points: *SCA at scale*, SCA's clustered equation with 10⁴–10⁵
clusters, which dense SCA cannot store; and *no training*, each partition's phase parts as its clusters
(`matrix_clusters = inclusion_clusters = 1`), with no k-means and no offline solves. The examples are 3D only (random
spheres), 2D staying for checks and teaching. Proposed: the voxel image is the problem, and the reference is Spectral LS
on that same image, exact for it, so every method's error is its model's alone. The baseline is SCA as published, with
its self-consistent C0; SCA with C0 = L_m is FFT-TFA's own equation solved densely, not a competitor. First result
(32³ image, 16 spheres at 25 %, glass/PA, tension then shear, rms σ̄11 against the same image): FFT-TFA with k-means at
8³ partitions 1.5 % in 11 s online; with no training at 8³ 4.7 % in 6 s and 0.2 s offline; self-consistent SCA at
512 + 512 clusters 5.6 % in 63 s online and 88 s offline; Spectral LS on a 16³ image 8.0 % in 6.5 s. At 32³ the exact
solve takes only 59 s; 64³ is where the advantage has to show. In order:

**Findings (2026-11-14): plasticity limits every clustered method.** On Schneider 2019's §5.3 setting (100 fibers at
50 %, glass/PA, transverse to 2.5 %) under uniaxial stress (`unloaded_directions = "free"`, new that day), against a
1024² Spectral LS reference, the σ̄11 error at the end of the path was 36 / 13.9 / 4.4 % for Spectral LS on 64² /
128² / 256²; 9.1 % (32²) and 4.8 % (64²) for FFT-TFA with 16 k-means matrix clusters per partition, about level with
Spectral LS at equal online time; 12.3 % for self-consistent SCA at 256 + 256 (Schneider reports 22.9 % at 32 + 32,
ours 30 %). Without training (one cluster per phase part) FFT-TFA's elastic step is 20–100× more accurate than Spectral
LS on the same grid (0.72 / 0.15 / 0.02 % at 64² / 128² / 256²), but at the end of the path it loses at equal time
(21.5 / 12.3 / 5.9 %). The error is one plastic state per cluster: the plastic strain covers most of the matrix with
steep gradients (90 % of it in 57 % of the matrix), so no placement of clusters gains much (allocating them by k-means
misfit: under 0.4 points; SCA's cell-wide clusters on the same equation: about 2×, which is the cost of keeping clusters
inside partitions). Under strain control the same runs looked far better, since the mostly elastic mean stress dilutes
the plastic error. Rejected as directions (2026-11-15): FFT-TFA as a geometry-aware coarse FFT solver
(composite-voxel framing), FFT-TFA as a preconditioner for the element-level solve, and richer (linear) plastic states
per cluster (convoluted, and not specific to FFT-TFA). Wanted: a new solver. Open.

**Findings (2026-11-15): FFT-TFA refined to 128² partitions, and SCA with C0 = L_m.** The same setting, on the 512²
clustered mesh against the 1024² reference, tensor-norm error at the end of the path. FFT-TFA with 8 k-means matrix
clusters per partition at degree 3: 31.6 / 24.2 / 15.6 / 11.9 / 7.2 / 2.7 % on 4² to 128² partitions, in 0.4 to 208 s
online and 44 to 74 s offline; its online memory, 286 MB at 64² and 959 MB at 128² (the degree-3 moment kernels), is
above Spectral LS 256²'s 183 MB. Spectral LS on 64² / 128² / 256²: 36.3 / 13.9 / 4.4 % in 3.8 / 17 / 82 s. SCA on 8 to
512 clusters: self-consistent 42 → 12.3 % in 0.2 to 16 s; with C0 = L_m 37 → 17.1 % in 0.1 to 2.9 s. FFT-TFA leads
Spectral LS at equal online time below about 30 s and ties it beyond; with the offline time counted, Spectral LS leads
on a single load path. SCA with C0 = L_m is the fastest below 1 s and level with FFT-TFA near 3 s; at equal cluster
count it is about 1.4× more accurate than FFT-TFA (cell-wide clusters against clusters inside partitions), but it
converges slowly in its clusters (37 → 17 % for 64 times the clusters) while D grows with their square. Its error is
soft over the first, elastic steps (−0.5 %, as the Hashin–Shtrikman bound with C0 = L_m predicts), then grows stiff in
proportion to the plastic flow and passes the self-consistent C0's after about step 16 of 25: a C0 that follows the
matrix's softening slows the stiff plastic error.

**Direction (2026-11-15).** The paper: FFT-TFA as SCA's clustered equation at scale (10⁴–10⁵ clusters, cost and
memory in partitions × p², converging with the degree), with the error analysis as its second half (the
Hashin–Shtrikman / Galerkin reading, the error split by assumption, and the cancellation behind clustered models'
accuracy). The examples show where FFT-TFA's advantages matter, with the scope stated and Schneider's case as the
stated limit, not hidden. 2D first: Schneider's glass/PA fibers in uniaxial stress at a lower fiber fraction (35 %;
his §6 sees clustered methods suited to lower filler fractions), at 1 % and 2.5 % strain, the full map of fraction ×
strain shown; FFT-TFA with k-means clusters and untrained (one cluster per phase part), both at degree 2. Then 3D
random spheres. A self-consistent C0 for FFT-TFA was considered and not pursued: it would put the matrix under A2, need
increments and repeated solves, and SCA's own numbers show no gain at equal time. Parked for a later paper: adaptive
clustering (start untrained, split clusters where the plastic strain jumps, as Ferreira et al. 2022 do for SCA; for
FFT-TFA a split changes only its partition's corrections, not the moment kernels).

**Findings (2026-11-15, later): material states, not the interaction, limit plasticity.** On the 35 % fibers (1 %
then γ12 1 %, uniaxial stress, 1024² reference), error against the number of material states: in the elastic step
the methods differ by how they see the geometry (untrained FFT-TFA about 20× better than Spectral LS per state);
after yield FFT-TFA, k-means or untrained, lies on Spectral LS's line, so states inside partitions are worth no more
than voxels, while SCA's cell-wide clusters are about 10× better per state at a few hundred. The rates differ: about
0.7 for voxels, 0.4–0.5 for clusters inside partitions, and 0.25–0.35 for cell-wide clusters, falling as they are
refined. Cell-wide clusters for FFT-TFA were estimated and dropped: memory stays under 1 GB, but 256–4096 clusters
fragment into 41k–111k parts per partition lattice, and FFT-TFA's cost grows with the parts, 3–4× more than today's
clusters for the same accuracy. Nested (bisecting) k-means against plain, for SCA with C0 = L_m on cell-wide
clusters: about 15 % more error at equal count (7.03 → 8.22 % at 256, 4.99 → 5.71 % at 1024), the same rate, and
clustering in 4.6 / 6.4 s instead of 65 / 915 s; added as `k_means_method`. Rerun of the 35 % case with nested clusters: FFT-TFA within 1–3 % of plain but no faster offline (its
k-means runs per small part, and the elastic features dominate), SCA's offline about halved; the default stays
"plain".

**Findings (2026-11-15, evening): the 2D fiber example, and contrast.** 35 % fibers, 1 % then γ12 1 %, uniaxial
stress, plain k-means, 1024² references. Contrast 34 (glass/PA) against 90 (a provisional hot glass/PA: the matrix's
E, yield stress and hardening scaled by 0.8 / 2.1, so only the contrast changes): Spectral LS's errors after yield
grew 5–7 % (its elastic error 20 %) and its online time 25–30 % (CG iterations +56 %); FFT-TFA's errors grew 7 % and
its time 0–4 %. At equal online time FFT-TFA with k-means leads by about 1.8× at 1 % with contrast 34 and 2.2× with
contrast 90, and is about level with Spectral LS after the shear at both; untrained FFT-TFA trails both. Contrast is
a weak lever: it slows Spectral LS, but FFT-TFA's plastic error grows as much. Conclusion for the paper's 2D fiber
example: report it as it is, about 2× online at moderate plasticity and level under strong plastic flow, offline
amortized over load paths, explained by the error against material states. Open: whether FFT-TFA's online cost per
material state can be cut in a way that stays fair to every method (each iteration multiplies 6p × 6p blocks per
partition frequency; the kernels' symmetry could halve that, but an optimization only FFT-TFA gets invites an equally
optimized Spectral LS). 3D, where Spectral LS's cost grows fastest, remains the most likely place for a clear time
result.

**Findings (2026-11-16): elasticity, and measuring errors without cancellation.** Elastic hot glass/PA (contrast 90),
35 % fibers, 1 % then γ12 1 %, uniaxial stress. Against a 1024² reference, untrained FFT-TFA on a 256² clustered mesh
went −1.9, −0.7, +0.003, +0.4 % at 16² to 128² partitions: against its own mesh (Spectral LS 256²) it was soft and
monotone (−3.5 → −1.2 %), as HS-3 says, while that mesh was +1.7 % stiff against the reference; the near-zero at 64²
was the two cancelling. With the clustered mesh equal to the reference (1024², the image is the problem), errors are
the model's alone: degrees 1 and 2 soft and monotone in tension and shear (degree 1 at 64²: −0.55 / −0.98 % in 7.5 s,
against Spectral LS 256²'s +1.69 / +1.53 % in 50 s); degree 1 as accurate as degree 2 at half the cost; degree 0 errs
stiff in tension (+0.8 %) from 32² on, its far field outweighing the clustering, and −7 % in shear at 16². Errors level
off near −0.3 % (tension) and −0.5 % (shear) from 128² to 256² partitions at degrees 1 and 2, cause not yet confirmed
(likely one polarization per inclusion part). k-means and untrained FFT-TFA give identical elastic answers (the
matrix carries no polarization). A conforming FE reference would not remove the cancellation (every voxel method then
carries a staircase error); it answers a different question (absolute accuracy). Self-consistent SCA's fit gave an
indefinite C0 at the elastic shear step: the cell's volume change there was 0.4 % of the shear, from the random cell's
anisotropy, too little to fit λ to but above the round-off threshold. A relative threshold (0.05) was tried and
dropped: under uniaxial stress the plastically flowing cell is nearly incompressible, so its volume change is just as
small while λ still reflects a bulk response, and the threshold changed SCA there. Instead, only when the fit comes
out indefinite, λ keeps its last value and μ is fitted alone (as under pure shear); everywhere else SC-3 is unchanged. C0 at the midpoint for FFT-TFA was considered: SCA's two C0 already show it
(between the phases: +4.4 → +1.5 % stiff; C0 = L_m: −2.6 → −1.0 % soft), so a midpoint would lose the sign and the
monotone convergence and put the matrix under A2.

**Findings (2026-11-17): in elasticity the error is A2, not the far field.** Tried: a degree-free FFT-TFA whose
residual uses the exact clustered interaction (each cluster's polarization on its elements, Γ0 by one FFT on the
element grid, averaged over each cluster: FFT-TFA at full degree, or SCA with C0 = L_m on clusters inside partitions),
while the Newton correction uses today's partition operator, so the degree sets only the speed. A prototype outside the
repository: on check.py's problems with an elastic matrix (contrast 100) it converges to FFT-TFA at full degree and to
SCA with C0 = L_m (1e-9), and to the cut laminate's exact solution. As a fixed point (exact residual, cheap Jacobian)
it contracts only 0.69 per iteration at degree 0 and 0.38 at degree 1; as flexible GMRES on the exact Jacobian,
preconditioned by a loose solve (1e-1) with today's degree-1 operator, it takes 12–18 element-grid FFT pairs per step.
On a 256² image of the 35 % fibers (uniaxial stress, untrained, 16² to 128² partitions, contrast 34 and 90) Spectral LS
on the same image took 53–86, rising with contrast, while this stayed flat. But the far field (A4: today's degree 1
against the exact clustered answer) was only 0.0–0.7 % of a 1–5 % error (degree 2: at most 0.15 %), and it often
offsets part of A2, so today's degree 1 is usually closer to the image's answer than the exact clustered one. A2 (one
polarization per inclusion part) is nearly all of the elastic error and converges slowly as the partitions halve
(3.1 → 2.1 → 1.45 → 1.0 %, contrast 34, tension). This confirms 2026-11-16's "likely" cause, on a 256² image. The
exact-residual solver stays useful as a diagnostic, since it isolates A2; the lever for elastic accuracy is a richer
inclusion polarization.

**Tried and set aside** (not to redo without a new reason):
- For FFT-TFA, one fiber part per fiber in each partition: stiffer and less accurate.
- A per-partition block-Jacobi preconditioner: fewer GMRES iterations, slower overall.
- A degree-free FFT-TFA, its residual by the exact clustered interaction on the element grid and its correction by
  today's operator, 2026-11-17 (findings above): correct and fast against Spectral LS, but it removes only the far
  field's error, a small share of the elastic error, at 5–20× today's cost on coarse partitions and with fields on the
  element grid.
- Removing the exact corrections inside each partition: polynomials cannot follow the sharp near field, 14–68 % off at
  practical partition sizes.
- Polarization shapes from the elastic strain concentration, plus one polarization per cluster: needs offline
  fine-mesh solves, and a near/far split of the interaction to be fast. Wanted instead: a richer polarization that
  needs neither ("Inclusion polarization" below).
- Building FFT-TFA's interaction by applying Γ0 to each cluster's or polynomial's unit polarization (as SCA's D is
  built), 2026-11-12: 2–6× the time and memory in 3D of grouping element pairs by offset; it wins only past about 36
  clusters in a partition, which is SCA's case.
- A C0 stiffer than every phase, 2026-11-13 (lecture, section 4): every error part stiff, but 8–25 times less
  accurate and 2–3 times slower; the setting was removed.
- Cell-wide k-means for FFT-TFA and TFA: it fragments into many parts per partition, and FFT-TFA's cost grows with
  the parts (re-estimated 2026-11-15 with the polynomial far field: 3–4× the cost of today's clusters at equal
  accuracy).
- SCA's residual in total form with a self-consistent C0, 2026-11-12: C0 then moves the whole solution, the fit has no
  root and the loop runs away; hence SC-1 in increments.
- Removed in the cleanups (git tags `pre-cleanup`, `pre-cleanup-2`, `pre-cleanup-3`): fixed-point solvers,
  FFT-preconditioned TFA, the FE kernel for P0, the coarse LS lattice, CP (the earlier method), the Fourier
  preconditioner, softening, and FFT-TFA's truncation to neighbouring partitions (`reach`).

1. **Literature.** The scheme is the precorrected-FFT idea (and kin of the fast multipole method and particle–mesh
   Ewald) applied to clusters. Find and read, cover to cover, those methods' foundational papers and any use of them for
   SCA, TFA or LS homogenization, before claiming anything new. The cheapest step, and it decides the paper's framing.
   Read so far (2026-11-13, cover to cover):
   - Zhang, Tang, Yu, Zhu & Liu, Comput. Mech. 64 (2019) 351–364: the closest prior art. SCA's interaction tensors
     from a coarse grid of cells, the cell-averaged Green operator weighted by each cluster's share of a cell (FFT-TFA's
     degree 0), but inside a cell only each voxel with itself (no exact near field), cell-wide clusters and a dense D
     (only its offline build is faster). Coarser cells answered stiffer. FFT-TFA must cite it and state what it adds:
     clusters inside partitions (no dense D, one FFT), exact corrections, moments converging to the exact equation.
   - Ferreira, Andrade Pires & Bessa, CMAME 2022: adaptive SCA (split clusters where the plastic strain jumps,
     during the solve). Nothing new on C0 (Liu's self-consistent fit). Updating D dominated their adaptivity's cost,
     which FFT-TFA's partition-local interaction would avoid.
   - Gierden, Waimann, Svendsen & Reese, CMAME 2021: reduced Fourier modes chosen from the phase map's spectrum; a
     different reduction (the material law still at every voxel), to cite in an FFT framing.
   No paper read gives a principled C0; Zhang et al. choose it arbitrarily, the others fit it self-consistently. To
   read: Wulfinghoff, Cavaliere & Reese, CMAME 330 (2018), and Nie, Cheng et al., Comput. Mech. 64 (2019), the
   variational (Hashin–Shtrikman, complementary energy) clustered methods.
   - **The variational framework** (read 2026-11-13: Brisard & Dormieux 2010 and 2012, Schneider 2019 §2–4). The
     clustered equation is the Galerkin discretization of the Hashin–Shtrikman weak form with the polarization
     constant per cluster, the internal variables included (Schneider §2.2, §4): one state per cluster is the Galerkin
     space, not a further approximation. It has a unique solution for any positive-definite C0 and converges on
     cluster refinement, in plasticity with hardening too (Schneider eq. 3.9, a Céa-type bound). A *bound* on the
     energy needs C0 on one side of every phase and the exactly averaged Green operator (Brisard); then refinement on
     nested spaces is monotone. For FFT-TFA against Spectral LS on the same voxel grid that holds in elasticity with
     C0 = L_m: the elastic step of every C0 = L_m entry of the 64³ spheres run is soft and monotone in partitions and
     clusters, while self-consistent SCA and coarse Spectral LS are stiff. In plasticity C0 = L_m lies above the
     matrix's plastic tangent, so the functional is no longer one-sided: convergent but unsigned, which is the stiff
     matrix part the decomposition found. The far field at finite degree adds a consistency error on top (Strang),
     exact at full degree. This is the theory the lecture should be written against; see the error decomposition
     (step 5) and the "clusters and degree" question in the status.
2. **Error measure.** Done 2026-11-13: the macroscopic error is the relative tensor norm ‖Σ − Σ_ref‖ / ‖Σ_ref‖
   (shears counted twice), Schneider 2019's eq. (5.1) extended to the whole tensor, reported at the first (elastic)
   step and at every turning point away from zero strain (at zero strain Σ_ref is only the residual stress); the
   error-against-time plot draws one panel per such turning point. It replaced the per-component rms, which depends on
   the axes and inflates small components; that stays in `summary.txt` and `macro_stress_error.png` to show where an
   error comes from. The rms figures quoted elsewhere in this log predate it.
3. **A modern element kernel.** Done 2026-11-12: `element_kernel` in `config.py` chooses Willot 2015's rotated
   staggered grid (the default) or Moulinec–Suquet's, for every LS method and the reference alike. Willot's is the
   standard alternative (Schneider 2021's review, §2.5) and equals trilinear hexahedra with one Gauss point (TFA's
   have eight). On config.py's 100 fibers (glass/PA, biaxial cyclic), against the two kernels' common limit
   (Richardson from 128², 256², 512²; the two limits agree to 0.03 %), Spectral LS's rms macroscopic error was 4.8,
   1.6, 0.47, 0.13 % at 64², 128², 256², 512² with Willot's and 5.3, 1.9, 0.62, 0.21 % with Moulinec–Suquet's, at
   the same iterations and time, with far less ringing near interfaces. FFT-TFA barely moved (1.11 against 1.14 % on
   32² partitions): its error is the clustered model's. The old 512² Moulinec–Suquet reference is itself 0.2 % from
   the limit, which a paper's error tables must not ignore. To do: the same comparison in 3D.
4. **Gold-standard reference.** A full-field FE solution on a conforming (body-fitted) mesh with plasticity at the Gauss
   points (a gmsh mesh, our own solver reusing `material.py`), in generalized plane strain for z-constant cells, since
   their loads include ε33, with a convergence study showing its own error is well below the errors compared. Until
   then the reference is Spectral LS on a fine voxel mesh, which shares every method's staircase geometry and kernel and
   favours Spectral LS, which converges to it.
5. **Error decomposition.** Derive the exact relation between TFA's P and the LS operator, and split FFT-TFA's error
   into its parts: one eigenstrain per cluster (stiff, shared with TFA: TFA against the reference), the inclusion's
   polarization uniform per cluster (soft: FFT-TFA at full degree against TFA), the far-field summary (FFT-TFA against
   itself at full degree) and the element kernel against TFA's FE. The stiff and soft parts cancel in part, so a method
   measured only against the reference can look better than its interaction is: the paper shows that. **Measured for
   FFT-TFA** (lecture, section 10): with C0 = L_m the inclusion part errs soft, the matrix part and the far field stiff,
   and they add up. A C0 stiffer than every phase makes every part err stiff, but was 8–25 times less accurate and
   2–3 times slower (lecture, section 4), so C0 = L_m stays. For SCA: with C0 = L_m, SCA
   and FFT-TFA err soft in elasticity and stiff once the matrix yields, and fewer fiber clusters look better only
   after yield; with SCA's self-consistent C0 both parts are stiff and add up. Neither SCA paper (Liu–Bessa–Liu 2016,
   Schneider 2019) separates the parts. Plan: SCA against TFA on the same cell-wide clusters (the soft part), TFA
   against the reference (the stiff part), on the elastic step first (where the stiff part is absent) and then along
   the path, for SCA and FFT-TFA alike; the kernel difference between TFA's FE and the LS kernel must be told apart
   from the soft part.
6. **Cost at equal accuracy, in 3D.** Online time against error for each method, in 2D and 3D, on many inclusions: the
   `"random spheres"` shape (done), and `get_correction_blocks` built a chunk of partitions at a time ("Memory" below). Spectral LS
   over a range of resolutions answers "why not a full-field solve?"; TFA and SCA (with C0 = L_m) on the same clusters
   isolate the interaction. An FFT on n lattice points costs about n log n, with n the elements for Spectral LS and the
   partitions for FFT-TFA, so Spectral LS on a mesh as coarse as FFT-TFA's partitions costs about as much per
   iteration: the two differ in accuracy, FFT-TFA seeing the fine geometry through each partition's clusters and its
   corrections. Also: offline cost reported apart, the number of load paths after which a method's offline cost pays
   off, and load paths other than the one the clusters were built from. **Memory** (proposed): a short discussion with a table of what each method stores, by formula and counted for the examples (TFA and
   SCA dense in the clusters, FFT-TFA partitions × p², Spectral LS a few fields per element with Γ̂0 evaluated on the
   fly as production codes do), not a measured comparison: a measured one would have to count a leaner Spectral LS than
   ours, which stores several 6 × 6 arrays per element.

The paper's examples (before the 2026-11-13 direction above, which keeps only the 3D ones): the cut laminate (an exact
solution, so the interaction's error alone), random fibers in 2D (error against cost, field maps), random spheres in 3D.
SCA (`sca_model.py`, lecture appendix E), with C0 = L_m or self-consistent, is there to show where FFT-TFA is worth it. Open: on 2D transverse tension of config.py's 100 fibers,
SCA with C0 = L_m matched FFT-TFA down to moderate accuracy, and both owe part of their accuracy to a soft error (the
cluster-uniform polarization) cancelling a stiff one (one eigenstrain per cluster); steps 2 and 5 are how to judge
methods without relying on that cancellation, and 3D is where FFT-TFA's case is expected.
Undecided: whether the inclusion polarization below is fixed first or reported as a limitation (proposed: reported); the
journal (CMAME
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
  Offline, `get_correction_blocks` cross-correlates every partition's clusters with one of its clusters at a time,
  holding about 2³ × the elements × the most clusters in a partition numbers, which in 3D with k-means can still exceed
  a Spectral LS solve of the whole load path. To do before any 3D study: also build them a chunk of partitions at a
  time (a chunk size or memory budget in `config.py`). Every k-means setting also needs six elastic
  Spectral LS solves on the elements, the floor of its offline memory.

Later: a reference stiffness for FFT-TFA that follows plastic softening (needed for a perfectly plastic phase), and the
consistent kernel (the exact partition average).

Field errors against the reference are not computed yet (macroscopic errors are). A measure fair across meshes of
different resolution is not settled: candidates are block averages at a scale every mesh resolves (on points chosen by
the reference's phases alone), and element errors on the fine grid split into elements whose phase matches the
reference's and elements whose phase does not.

## Measurements behind the lecture (moved from it, 2026-11-15)

The lecture states these qualitatively; the numbers, as they were measured (several on older code, before the
self-consistent SCA fitted two Lamé constants), are kept here.

- **C0 above every phase (lecture, section 4).** Measured on 100 glass fibers in PA (contrast 34), biaxial cyclic, degree 3: stiff at every step, but 8–25 times further from Spectral LS on the same mesh than with $\mathbf L_M$ (62 % against 2.5 % rms in σ̄11 with 4² partitions, 6.3 % against 0.81 % with 32²), and 2–3 times slower, since the Jacobian is positive only by about $\mathbf L_T/\mathbf C_0$; a far field below degree 3 can even leave a step without a solution.
- **The elastic bound in practice (lecture, 5.1).** Measured (16 spheres at 25 %, glass/PA, 64³ elements, the first load step, $\bar\varepsilon_{11}=0.08\,\%$; $\bar{\boldsymbol\varepsilon}^{\mathsf T}\Delta\bar{\boldsymbol\sigma}$ against Spectral LS on the same mesh, and $\Delta\bar\sigma_{11}/\bar\sigma_{11}$); the self-consistent SCA row is from the old fit
  C0 = c I:

| Entry | $\bar{\boldsymbol\varepsilon}^{\mathsf T}\Delta\bar{\boldsymbol\sigma}$ | $\Delta\bar\sigma_{11}/\bar\sigma_{11}$ (%) |
|---|---|---|
| FFT-TFA, phase parts, degree 3, 2³ / 4³ / 8³ / 16³ partitions | soft at each | −3.84, −3.55, −2.34, −1.76 |
| SCA, $\mathbf C_0=\mathbf L_M$, 64 / 256 / 1024 k-means clusters | soft at each | −2.93, −2.21, −1.71 |
| SCA, self-consistent $\mathbf C_0=c\,\mathbf I$, 64 / 256 / 1024 clusters | stiff at each | +5.18, +3.25, +1.98 |
| Spectral LS on 16³ / 32³ elements | stiff at each | +12.98, +2.35 |

  Every $\mathbf C_0=\mathbf L_M$ entry is soft and monotone, as HS-3 says (the k-means sequence happens to be, with no guarantee); the self-consistent $\mathbf C_0$ lies between the phases, so $W$ has no sign there; the coarse meshes are not restrictions of the 64³ image's space, so no bound applies to them, and their error is the mesh's. FFT-TFA with four matrix clusters per partition gives the same four numbers as with one: in elasticity the matrix carries no polarization, so its clusters are invisible (step 2 of section 4).
- **A cut laminate (lecture, section 10).** A layer whose interfaces cut partitions 6 elements across (`check.py`'s cut laminate refined to 54² elements, 9² partitions) has its worst cluster 24 % off at degree 0, 8.5 % at degree 1, 3.0 % at degree 3, 0.6 % at degree 4, and exact from degree 5. The polarization steps across the cut partitions, which line up along the layer, so their errors add up instead of decaying. Degree 5 follows any profile across 6 elements.
- **The far floor (lecture, section 10).** With Moulinec–Suquet's kernel the floor is ten times higher, a few hundredths: it comes from the element kernel's high-frequency content, which varies from element to element even far from its source and which polynomials smooth over, and Willot's kernel has less of it.
- **Opposite signs (lecture, section 10).** Measured on 100 random glass fibers in PA (contrast 34), biaxial cyclic, 64² partitions of 4 × 4 elements, against Spectral LS on the same mesh, each part alone by refining the others to one cluster per element at full degree, and signed by each step's stiffness. One polarization per inclusion cluster (A2) errs soft at 96–100 % of the steps (0.48 % rms in σ̄11, −1.9 % in every elastic step); one state per matrix cluster (A1) errs stiff at 84–96 % (0.15 %); the far field at degree 3 errs stiff at every step (0.10 %). The parts add up, to 0.06 % rms.
- **GMRES against a direct solve for SCA (lecture, E.4).** On 100 fibers in glass/PA over 25 steps, GMRES and the direct solve take under 0.2 s each at 16 clusters, and GMRES is 8 times faster at 256 clusters and 15 times at 512, with the same answers to the stopping accuracy.
