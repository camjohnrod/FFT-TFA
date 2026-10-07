# Brisard & Dormieux (2010), Comput. Mater. Sci. 49, 663–671 — READ COVER TO COVER (page images, all 9 pages)

## Scope
2D plane strain linear elasticity ONLY ("For the sake of simplicity and illustrative purposes, the framework of 2D
(plane strain) elasticity will be adopted. The results presented in this paper can readily be generalized to 3D
elasticity, albeit at the expense of a slightly increased notational complexity." p.664). Nonlinear listed as future
work in the conclusion: "extensions to non-linear behaviors, similar to the basic scheme [8,9] should be considered."

## Method
- Hashin–Shtrikman energy principle (7): for c0 ≥ c(x) (stiffer reference), upper bound on elastic energy for any τ;
  reversed for softer reference. Willis: quadratic form positive (negative) definite.
- §3.2: pixel-wise constant polarization (8) τ(x,y) = Σ χ_αβ τ_αβ over an M×N pixel grid.
- (10) equivalent stiffness of a pixel: (c_αβ − c0)⁻¹ = (MN/WH) ∫ χ_αβ [c(x,y) − c0]⁻¹ d²x. "Incidentally, (10)
  provides an averaging rule, consistent with our energetic approach, to compute the equivalent stiffness of an
  heterogeneous pixel."
- Fourier coefficients of a pixel-wise constant τ (p.666): τ̂(k_ab) = (1/MN) sinc(πa/M) sinc(πb/N) exp[−iπ(a/M + b/N)]
  τ̂_ab (τ̂_ab = DFT). => (13), rearranged using (M,N)-periodicity of the DFT.
- (14) periodized Green operator: Γ̂_ab = Σ_m Σ_n sinc²(π(a+mM)/M) sinc²(π(b+nN)/N) Γ̂(k_{a+mM,b+nN}).
- (16): "which is exact, in the sense that no series truncation has been performed".
- (17) discrete bound; minimizer = "the M × N piecewise constant approximation of the real polarization field arising
  in the periodic elasticity problem (2)". Strain ε_αβ = E − η_αβ.
- §3.3: solved by conjugate gradient; one CG iteration costs the same FFTs as one basic-scheme iteration. "It should
  however again be emphasized that no approximation has been made to arrive at this result."

## §3.4 periodized Green operator (p.667–668)
- Appendix B: Γ̂_ab → Γ̂(k_ab) as resolution → ∞ (classical and periodized coincide in the limit).
- "Our numerical experiments show, however, that this convergence is very slow (the relative difference can reach 30%
  for M = N = 1024); this indicates that introducing the periodized Green operator is the appropriate way of
  accounting for finite-resolution effects."
- Notes Willot–Pellegrini modified (discrete) Green operator improves the basic scheme.
- "There is no closed-form expression for Γ̂_ab, which must be estimated numerically ... it needs to be carried out
  only once for each value of the grid size".
- Identity sinc(π(a+mM)/M) = (−1)^m a (a+mM)⁻¹ sinc(πa/M) used to rewrite (19) N^II, (20) N^IV (isotropic case).
- "Numerical experiments show that the series for N^II and N^IV converge very slowly, making their estimation
  difficult. The Poisson summation formula ... we were not however able to apply this formula to all components ...
  we devised another estimation method, based on the approximation of integrals by Riemann sums (see Appendix C)".
- Appendix C: tail of a positive decreasing series bounded between integrals; estimate = truncated sum + mean of
  the two integral bounds; error improves from T⁻¹ to T⁻² ("in the worst case, the error scales as T⁻¹ in the
  initial scheme, and as T⁻² in our improved scheme"). "Generalization to doubly infinite sums ... as well as
  bivariate functions is straightforward."
  => A tail correction is given in this paper. BD2012 still defers "the required formulas" for efficient
  evaluation (see note 03).

## Applications (§4, 2D, square grids 8²…128²)
- Square inclusion (Fig. 2, contrast μi/μm = 0.1): polarization-based energy monotone in resolution, "hardly differs
  from the finite element estimate"; basic scheme non-monotone, no bound at low resolution ("attributed to truncation
  errors").
- Fig. 3: iterations vs contrast: basic scheme grows as max(μi/μm, μm/μi); polarization-based bounded.
- Reference = matrix: "Having the reference medium coincide with one of the phases of the composite requires a
  special numerical treatment, due to the term (c_αβ − c0) in (17), which becomes singular. This is accounted for by
  enforcing that the corresponding local value of the polarization field be zero."
- Fig. 4: void inclusion, 32×32 polarization-based vs FEM 128×128 quadratic: very good agreement.
- §4.2 diamond (void) inclusion, composite boundary pixels: strategy 1 (eq. 10) vs strategy 2 (boundary pixels void).
  Strategy 1 gives the upper bound; strategy 2 fails ("emphasizes the importance of attributing energetically
  consistent properties to composite pixels"). Better reference (μ0 = 17.5, ν0 = −1) tightens bound much.
  Our inference (not stated): with c0 = matrix, eq. (10) sends any pixel containing matrix to c_αβ = c0 (integrand
  infinite on the matrix part), i.e. τ = 0 there.
- Conclusion: the polarization-based linear operator is definite → CG; converges faster even at infinite contrast;
  rigorous bound at any resolution; composite voxels accounted for.

## Relevance
- Exactly our partition scheme (uniform cells, constant polarization, exact cell-averaged Green operator, FFT),
  2D elastic. Cells are the pixels of their grid; they explicitly consider coarse grids with heterogeneous pixels.
- They did not do plasticity; they did not do 3D.
- They give the tail correction (App. C) we would need.
