# Brisard & Dormieux (2012), CMAME 217–220, 197–212 — READ COVER TO COVER (page images, all 16 pages)

## Scope
Linear elasticity, periodic BCs, d = 2 or 3 in the theory; isotropic phases and isotropic reference assumed
(§3.2: "For convenience, the present analysis is restricted to isotropic composites"; C0 isotropic μ0, ν0).
Numerical examples: 2D plane strain only, square inclusion a = L/2, simple shear.

## Framework
- (9) HS stationarity; (10) weak form: find τ ∈ V with a(τ, ϖ) = ℓ(ϖ) ∀ϖ; (11a) a(τ,ϖ) = ⟨ϖ:(C−C0)⁻¹:τ⟩ + ⟨ϖ:(Γ0*τ)⟩,
  (11b) ℓ(ϖ) = ⟨E:ϖ⟩. "(10) is in fact the weak form of the Lippmann–Schwinger equation".
- Assumption 1: C − C0 ± λI semidefinite (no C = C0); extension: C = C0 on Ω0 allowed with τ = 0 there.
  Assumption 2: κ, μ bounded below (excludes pores).
- Thm 3/Cor. 1 (Willis): bounds (15). Thm 4 + Cor. 2: (10) well-posed for ANY C0 (pos-def), via Banach–Nečas–Babuška.
- §4.1: V^h = cell-wise constant fields on a regular grid of cubic cells of size h (19). Thm 5: approximability.
  "the present analysis is greatly simplified by two facts. First, the spaces of trial and test functions coincide,
  and are included in the initial space V ... Second, unlike the bilinear form a, the linear form ℓ is computed exactly
  in both schemes on V^h."
- §4.2 consistent: (21) find τ^h ∈ V^h with a(τ^h,ϖ^h) = ℓ(ϖ^h). "It is conformal, since V^h ⊂ V, and consistent, since
  operators a and ℓ are computed exactly on V^h. In other words the solution τ ∈ V of the exact problem (10)
  satisfies the approximate problem (21)". (22)–(23) consistent equivalent stiffness (C^h_β − C0)⁻¹ = cell average of
  [C(x) − C0]⁻¹; "even in the frequent case of an heterogeneous cell". (24)–(28): F(K) = sinc(K1/2)…sinc(Kd/2) (26);
  consistent DGO Γ̂^{h,c}_{0,b} = Σ_{n∈Z^d} [F(h k_{b+nN})]² Γ̂0(k_{b+nN}) (28); "this operator was referred to as the
  periodized Green operator in [13]". "pre-computed and stored ... It can be evaluated very efficiently in plane strain
  elasticity.[footnote 2: The required formulas will be reported in a paper to come.] It is noted however that in
  the three-dimensional case, the infinite series involved in (28) converge very slowly, making the numerical
  evaluation of the consistent discrete Green operator rather involved."
  Thm 6 → well-posed for any C0; Céa's lemma → L² convergence of τ^h to τ.
- §4.3 non-consistent (MS): Green operator "simply approximated by a truncated Fourier series"; "the bilinear form a
  is not computed exactly and the approximation is in fact non-consistent. We then show that the approximate bilinear
  form a^h is asymptotically consistent, which leads to convergence results of the approximate solution."
  MS "suggest that ε^{h,n}_β should be understood as a point-wise estimate of ε^n at point x_β ... it is more natural to
  consider that ε^{h,n}_β is the step value on Ω^h_β of a cell-wise constant function ε^{h,n} which approximates ε^n in
  the L²-sense, rather than a point-wise estimate". Local stiffness: "to the best of our knowledge, no consistent rule
  has yet been proposed ... The analysis below shows that the consistent discretization C^h of C defined by (23)
  should be used." (29) Γ̂^{h,nc}_{0,b+nN} = Γ̂0(k_b) for b in lowest-frequency set (30), N_i even.
  "unlike its consistent counterpart (28), the non-consistent discrete Green operator is known in closed-form".
  (33) always has a unique solution regardless of C0; weakness of basic scheme = fixed-point solver, not discretization.
  Thm 7 (37)/(38): asymptotically consistent (Ern–Guermond Def. 2.15); Strang's lemma → L² convergence.
- §4.4: "The basic scheme ... is a non-consistent Galerkin approximation of problem (10), while the energy-based scheme
  ... is a consistent approximation of the same problem." "In our experience, the benefit of the latter over the
  former resides in the fact that it leads to generally better behaved numerical solutions at high contrast, where
  spurious oscillations might develop with the non-consistent discrete Green operator [27]. The reason for this is
  obvious on Fig. 1 ... while the consistent operator is smooth in Fourier space, the non-consistent operator exhibits
  a strong discontinuity at the highest frequencies." Rigorous bounds only when C0 stiffer/softer than all phases.
  Rule (23) "depends on the composition of the heterogeneous cell, but not on the spatial organization".
- Appendix B: (52) Σ_n [F(h k_{b+nN})]² = 1 (weights sum to one, stated explicitly). (53) best approximation in V^h is
  the cell average; (54) Fourier coefficients of cell average.

## Numerical results (§5, 2D, square inclusion a = L/2, shear, grids 4²…1024², reference 2048² consistent)
- §5.1 finite contrast μi = 0.01 μm: "Fig. 3 clearly shows the h-convergence of both consistent (C01) and
  non-consistent (NC01) approaches. It is experimentally observed that the consistent method is slightly more accurate
  than the non-consistent method, both methods being approximately of order one in h." Fig. 4: NC01/NC02 and C01/C03
  "barely distinguishable" → h-convergence insensitive to C0 (even C0 violating the classic conditions).
  Iteration counts Table 1 similar for NC and C.
- §5.2 infinite contrast (pore): NC04, AL04 (augmented Lagrangian), C04, FNC04 all converge, order ~1 in h (exploratory,
  no theory). "the consistent approach (C04) is much better behaved, and can be seen to converge in less than 100
  iterations for any refinement h" and slightly more accurate. "From the purely numerical perspective, the consistent
  scheme is superior ... However, the major drawback of the former lies in the complexity of the calculation of the
  consistent discrete Green operator (28)." Fig. 6 (32×32, pore): NC04 stress "exhibits a 'checkerboard' pattern",
  C04 smooth, FNC04 close to C04. "the shortcomings of the non-consistent scheme originate in an inaccurate treatment
  of the highest frequencies".
- Filtered (44)–(45): MS on grid h/2, averaged to h; Γ̂^{h,fnc}_{0,b} = Σ_{n∈{−1,0}^d} [G(h k_{b+nN})]² Γ̂0(k_{b+nN}),
  G(K) = cos(K1/4)…cos(Kd/4); 2^d terms; asymptotically consistent; "C04 and FNC04 are practically undistinguishable".
- Conclusion: "simple examples indicate that at high contrast, the numerical solution exhibits undesirable oscillations
  ('checkerboard' pattern). This is to be attributed to the discretization of the Green operator, which poorly
  reproduces the high-frequencies." "The main drawback of the energy-based scheme is the necessary precomputation of
  the consistent discrete Green operator." Optimal reference at fixed h is open.

## Relevance
- The weights-sum identity is their eq. (52).
- At finite contrast (100:1, their 0.01) the consistent operator is only "slightly more accurate"; the large
  qualitative difference (checkerboard vs smooth) appears at infinite contrast (pore). Our contrast is 100:1.
- Analysis assumes isotropic phases/reference; our "homogenized" C0 is anisotropic → outside their theory.
- Linear elastic only; no plasticity.
