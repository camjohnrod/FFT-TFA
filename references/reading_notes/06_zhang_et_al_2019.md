# Zhang, Tang, Yu, Zhu, Liu (2019), Comput. Mech. 64, 351–364 — READ COVER TO COVER (page images, all 14 pages)

## What it is
Speed-up of interaction-tensor computation for VCA (virtual clustering analysis, Tang–Zhang–Liu 2018); "may also be
used in SCA". Also finite-strain LS for VCA (Appendix C).

## VCA setting (§2.1)
- "Instead of using the periodic boundary conditions, we introduce a fictitious homogeneous isotropic material, also
  termed as comparison material, with stiffness tensor C0, surrounding the original domain. We load it with a uniform
  strain ε0 at infinity." (6)–(8). LS (10) with boundary terms; enlarged domain + Saint-Venant → boundary-term-free
  (13) ε − ε0 + Φ*(σ − C0:ε) = 0 over Ω only. Φ from infinite-space fundamental solutions (Kelvin).
- Clustering: "These voxels are grouped into k clusters according to their responses under several selected loadings.
  ... The clustering is performed by a machine learning technique such as k-means or Self-Organizing Map (SOM)."
  (In the numerics: SOM, per phase.)
- (15) cluster equations; (16) D^IJ = (1/|Ω^I|) ∫_{Ω^I}∫_{Ω^J} Φ(x − x̃) dx̃ dx.
- §2.2: 2D explicit Fourier transforms (21)–(24), biharmonic fundamental solution (25), Φ via fourth derivatives
  (28)–(35); analytic indefinite double integrals (36)–(38) (singular, numerical limit along (1,1)). S(A,B) = ∫_A∫_B Φ.
  (40) D^IJ = sum over voxel pairs → O(n²).
- §2.3 fast method: coarse cubic/rectangular grid Q_p (m cells) over a box Q ⊇ RVE; compute S(Q_p,Q_q) analytically;
  approximate S(ω_i, ω_j) by volume ratios (41) (kernel ~constant over two different cells; self-cell term ∝ volume,
  cross terms within the same cell set to 0 (42)); (43) D^IJ ≈ composition-ratio-weighted sum; (44) α^I_p = volume
  fraction of cluster I in cell p. "It is remarked that we usually make an equidistant partition in each space
  dimension. The kernel function Φ(x − x̃) depends on the relative position between x and x̃. Hence, the integral
  S(Q_p, Q_q) also only depends on the relative position of Q_p and Q_q." → O(m) storage (< 2^d m distinct). Cost of D:
  O(k² m²) vs O(n²) voxel-based vs O(k² n log n) "FFT method used in SCA" (Table 1).
- §3 numerics: J2 piecewise-linear hardening matrix; 2D: 600×600, E1 = 100 MPa, E2 = 500 MPa (5:1), C0: E0 = 20 MPa,
  ν0 = 0.4; m = 200²…30²: deviation from DNS 1.59–1.98% (Table 2), stiffer as m decreases. 3D: 41³, E2/E1 = 10,
  m = 21³…3³, deviation 0.55–0.87%. Hyperelastic finite strain example.
- Appendix A: "VCA uses Green's functions with zero strain boundary condition at infinity, while SCA uses Fourier
  series of them to express interaction tensors." "In SCA, periodic boundary conditions are applied to the RVE problem
  and FFT is used to calculate interaction tensors."
- Conclusion: converges to DNS "when voxels/pixels and coarse grid cells are all cubes/rectangles and coincide".

## Relevance
- Closest to "uniform grid + analytic averaged Green operator": they compute exact cell–cell double integrals of the
  INFINITE-medium Green function on a uniform grid (translation invariant), but use them only to assemble the dense
  D^IJ of SOM/k-means clusters (no FFT on the grid, no periodicity). Plasticity (J2), contrasts 5 and 10.
- Not our setting: non-periodic; clusters not the grid cells; D dense k×k.
- Note: the coarse-grid cells themselves are never used as the clusters in their examples.
