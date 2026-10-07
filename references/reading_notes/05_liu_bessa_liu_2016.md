# Liu, Bessa, Liu (2016), CMAME 306, 319–341 — READ COVER TO COVER (arXiv/author preprint, all 24 pages, page images)
(Preprint "submitted to CMAME", dated Sept 16, 2019 in footer; content = the SCA paper. Equation numbers are the
preprint's; may differ from the journal version.)

## Method
- Piecewise uniform approximation of all local variables in clusters (1); "every local variable β(x) within each
  material cluster is uniform".
- (5) σ = C0:ε + p (p = polarization stress); isotropic linear elastic reference C0. (7)–(8) LS with Green's function
  Φ0 and far-field strain ε0 (not necessarily ε̄; constraint (9) or (10)). (12) incremental form.
- (13)–(15): average LS over each cluster; (16) interaction tensor D^IJ = 1/(c^I|Ω|) ∫∫ χ^I(x) χ^J(x′) Φ0(x,x′) dx′dx
  "written as an integral of Green's function Φ0(x, x′) in a RUC domain Ω with periodic boundary conditions".
  (17) cluster equations, (18) macro constraint.
- Remark after (17)/(18): "Note that equations (17) and (18) do not depend on the strain concentration tensor or on any
  additional parameters or constitutive laws. This contrasts with approaches such as TFA [22] and NTFA [23, 39]."
  Stresses between clusters not continuous; only global equilibrium.
- §2.2.1 k-means on strain concentration tensor A(x) from elastic DNS (3 loadings in 2D, 6 in 3D); clusters can be
  disconnected (Fig. 3). Remark 2: "a simpler metric would be to group data points according to their spatial
  proximity (disregarding their mechanical similarity), as was done later in this article for comparison (Figure 6)."
- §2.2.2 (22) same D^IJ; (23) c^I D^IJ = c^J D^JI; (24) isotropic Φ̂0; split (25)–(26) Φ̂0 = Φ̂1/(4μ0) +
  (λ0+μ0)/(μ0(λ0+2μ0)) Φ̂2 → only coefficients change when C0 changes (enables self-consistent C0 cheaply).
  (27) ∫χ^I Φ0 = F⁻¹(χ̂^I Φ̂0): computed by Fourier transform on the discretization grid ("Only a one-time calculation
  is needed"). Appendix A: DNS for A(x) by FFT (MS) "The finite element method could also be used".
- §2.3 Remark 4: "The solution of the discrete Lippmann-Schwinger equations ... is actually affected by the choice of
  the reference stiffness C0." Box 2.1 Newton with constant C0 (Jacobian M^IJ = δ_IJ I + D^IJ:(C^J_alg − C0)) —
  same structure as our LS-20. §2.3.2 self-consistent C0: isotropic (λ0, μ0) fitted to macro tangent each increment
  by least squares (41)–(44); D^IJ updated (Box 2.2).
- §2.4 materials: E1 = 100 MPa ν1 = 0.3 (matrix, plastic), E2 = 500 MPa ν2 = 0.19 (contrast 5:1). J2, piecewise
  linear (47) or power law (48) hardening. Validated against FE DNS.

## Results
- §3.1 elastic: accuracy improves with clusters; with/without SC similar for fibre composite; SC converges much faster
  for amorphous (C0 = matrix w/o SC). Footnote: in linear elasticity SC C0 = C_macro.
- §3.2 Fig. 5: without SC (C0 = C_matrix), plastic response much too stiff, especially power-law hardening; SC much
  better. "When plastic yielding happens ... C^matrix is no longer a good choice for the tangent stiffness C0".
  (Matches our finding: LS error grows with plasticity; C0 elastic.)
- Fig. 6: position-based clustering (spatial Voronoi-like clusters, 32 in matrix): "the convergence using the
  position-based clusters is poor and the accuracy of the prediction does not change significantly even considering
  256 clusters for phase 1. On the contrary, with the same number of clusters the predictions from A-based clustering
  reproduce the DNS almost exactly." (with SC.)
  => Directly relevant: spatial partitions (like our uniform partitions) converged poorly in their test.
- Fig. 7/8: k1 = 1, 16, 256; local fields more diffuse than DNS. Fig. 9 timing: k1=16: 3 s, k1=256: 59 s (MATLAB) vs
  DNS 1420 s; time ~ proportional to number of clusters. Fig. 10 cyclic/complex path good. 3D (64³, 80³) examples.
- Conclusion: softening "requires special attention, since the effective modulus becomes negative ... self-consistent
  reference material needs to be modified".
- Appendix B: new elastic properties online without redoing offline (ratio 0.01–50), SC good.

## Relevance
- SCA = clustered LS with periodic Green operator; D computed by Fourier transform on the fine grid (eq. 27).
- Reference: isotropic only; self-consistent update.
- Their position-based (spatial) clustering result is a caution for uniform spatial partitions.
- No uniform/box clusters, no comparison with TFA numerically (only remark that eqs don't need strain concentration).
- Contrast 5:1 (ours 100:1).
