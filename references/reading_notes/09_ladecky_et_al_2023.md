# Ladecký, Leute, Falsafi, Pultarová, Pastewka, Junge, Zeman (2023), Appl. Math. Comput. 446, 127835 — READ COVER TO COVER (all 19 pages, from extracted text: equations came through partly garbled, figures not seen)

"An optimal preconditioned FFT-accelerated finite element solver for homogenization". PDF:
`../An-optimal-preconditioned-FFT-accelerated-finite-element-solver-for-homogenization.pdf`. Read 2026-10-27 to
choose preconditioners for CP and Spectral LS.

## Scope
Displacement-based (DB) FE discretization on a regular periodic grid (any space-filling stencil: bilinear quads,
linear triangles, trilinear hexes, several nodes per stencil), small-strain nonlinear elasticity (finite strain in
§7.3), Newton's method, linear systems by preconditioned CG. Mandel notation (not engineering shear).

## Key equations and statements
- (8) discrete equilibrium Dᵀ W σ(e + D ũ, g) = 0; (9) Newton: K δũ = b with K = Dᵀ W C D, C the algorithmic tangent
  at quadrature points. "The system matrix K(i) is symmetric and positive definite for the symmetric algorithmic
  tangent C(i), which renders the CG method as the method of choice ... For non-symmetric algorithmic tangents, an
  alternative iterative linear solver must be used, e.g., the generalized minimal residual method (GMRES)."
- (11) preconditioner M = K_ref = Dᵀ W C_ref D with C_ref "spatially uniform material data"; (12) (K_ref)⁻¹ K δũ =
  (K_ref)⁻¹ b, the "reference material-based preconditioned problem"; pseudo-inverse because of rigid-body modes.
- §4.2, (14): K_ref is block-circulant, so block-diagonal in Fourier space and cheaply (pseudo-)invertible, one
  dN_n × dN_n block per frequency; assembled matrix-free by applying K_ref to unit impulses (Algorithm 2).
- §4.3: every eigenvalue of (K_ref)⁻¹K is bounded element by element by λ_min and λ_max of (C_ref(x))⁻¹ C(x) over the
  support of each basis function (refs [33–35]); the bounds are independent of the element size h, so the iteration
  count is (almost) mesh independent. "The ratio between the maximum and minimum eigenvalues of the preconditioned
  problem (15) will increase with an increasing ratio between extreme eigenvalues of C(i) (so-called material
  contrast) and decrease as the data for the reference material C_ref(i) approach the data for the material C(i)".
  "Geometrically optimal": keeping the discretization and changing only the preconditioner's data could give
  condition number 1, at a higher inversion cost.
- §5.2: "It may be useful to reassemble the preconditioner with updated C_ref(i), whenever C(i) significantly changes
  with respect to the previous Newton step".
- §6.1, (19): the DB scheme is equivalent to the strain-based (SB) one: multiplying by D gives Γ0 C δ∂ũ = −Γ0 σ with
  Γ0 = D (Dᵀ C_ref D)⁻¹ Dᵀ, "the discretized periodic Green's operator"; DB and SB "generate equivalent solutions in
  every step of the CG in exact arithmetic". (Shown for equal quadrature weights; extension to a general reference
  material left to future work, §8.)
- §6.2: their preconditioner has the form of the fundamental solution inside Γ0 of Schneider et al.'s FFT-Q1Hex and of
  Leuschner–Fritzen's FANS; "The closer the reference material is to the real material of the sample, the better the
  conditioning of the discretized problem."
- (20) PCG stopped on the (K_ref)⁻¹-norm of the residual, which approximates the error's energy norm when
  (K_ref)⁻¹K ≈ I.

## Numerical results
- §7.1 (thermal, 815² pixels, contrast 100): Fourier–Galerkin oscillates; fully integrated FE does not; the reference
  material "has no effect on the discretization artifacts".
- §7.2 (3D Hashin coated sphere, contrast ρ up to 10³ and beyond, trilinear FE): two references compared, C_ref = I_s
  and C_ref_mean = (1/|Y|) Σ_Q C(x_Q) w_Q, "the mean stiffness matrices over Y". "The preconditioner with mean reference
  material C_ref_mean exhibits better performance in all studied cases, see Fig. 9 ... C_ref_mean significantly reduced
  the phase contrast sensibility, especially for ρ > 1 (softer sphere core)." Iterations grow slowly with N_I, then
  stabilize.
- §7.3, Table 1 (2D finite-strain elasto-plastic dual-phase steel, 5 increments, η = 1e-5): total PCG steps with
  C_ref = I / I_s / C_ref_mean: Fourier–Galerkin SB 1012 / 781 / 585; FE projection SB 861 / 609 / 457; DB linear
  FE 861 / 609 / 457 (equal to SB, as claimed); DB bilinear 761 / 540 / 407; under-integrated DB bilinear
  1014 / 779 / 586.

## Relevance (our reading)
- A displacement-based FE scheme, not a Lippmann–Schwinger one: the user's decision (2026-10-27) is not to cite it
  for the preconditioners of CP and Spectral LS. Their justification rests on the LS derivation itself
  (`notes/2_ls_model.md` section 7.4) and on the LS literature.
- Relevant to roadmap step 3, the FE-consistent gold-standard reference (displacement unknowns, plasticity at the
  Gauss points): Algorithms 1–2 and (14) are a recipe for its Newton–PCG solver with an FFT-inverted reference
  preconditioner; the mean-stiffness reference beat the identity in every test (Fig. 9, Table 1).
- Their eigenvalue bounds assume a Galerkin (consistent) FE discretization with PCG.
