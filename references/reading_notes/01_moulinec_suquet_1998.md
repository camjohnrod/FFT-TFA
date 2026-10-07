# Moulinec & Suquet (1998), CMAME 157, 69–94 — READ COVER TO COVER (page images, all 26 pages)

## What it is
FFT-based solution of the periodic Lippmann–Schwinger equation for nonlinear (J2) composites, directly on images.
All examples are 2D (generalized plane strain), unidirectional fibres.

## Equations
- (1)–(3): auxiliary problem, homogeneous c0 + polarization τ; ε(u*) = −Γ0 * τ; ε̂(ξ) = −Γ̂0(ξ):τ̂(ξ), ε̂(0)=0.
- (4): isotropic Γ̂0 (λ0, μ0): Γ̂0_khij = 1/(4μ0|ξ|²)(δ_ki ξ_h ξ_j + δ_hi ξ_k ξ_j + δ_kj ξ_h ξ_i + δ_hj ξ_k ξ_i)
  − (λ0+μ0)/(μ0(λ0+2μ0)) ξ_iξ_jξ_kξ_h/|ξ|⁴. Depends only on direction of ξ (degree 0) — visible from formula.
- (6) τ = δc : ε, δc = c − c0. (7) periodic LS eq: ε = −Γ0 * τ + E; ε̂(ξ) = −Γ̂0(ξ):τ̂(ξ), ε̂(0)=E.
- (8) continuous basic scheme; (9) simplified using Γ0*(c0:ε)=ε; (10) discrete algorithm on N1×N2 grid (pixel values,
  discrete frequencies ξ_d; FFT). (12) nonlinear step algorithm with radial return.
- Appendix A: general anisotropic Γ̂0 via acoustic tensor K0_ik = c0_ijkh ξ_h ξ_j, N0 = K0⁻¹, (A.4)
  Γ̂0_khij = 1/4 (N0_hi ξ_j ξ_k + N0_ki ξ_j ξ_h + N0_hj ξ_i ξ_k + N0_kj ξ_i ξ_h). Isotropic special case.
- Appendix B: imposing macroscopic stress direction. Appendix C: radial return (positive hardening assumed).

## Key statements (with page)
- p.73 §2.4.2: "When the spatial resolution is low and when the number N_j of discretization point is even, a special
  attention must be paid to the highest frequencies ξ_j = ±(N_j/2 − 1)/T_j". Reason: FFT packages use a single
  term (cos or exp) at these frequencies instead of the pair, and "Γ̂0 is neither even nor odd with respect to each
  individual component ξ_j". "Oscillations were observed when (4) was used with relatively small values of N_j
  (lower than 128). This problem was fixed by using a different expression of Γ̂0 in algorithm (10) at these
  frequencies. Γ̂0 = (c0)⁻¹." "In other terms, the stress σ is forced to 0 by the algorithm at these frequencies".
  => The <128 oscillations are a specific even-N highest-frequency issue, FIXED by them. NOT a general statement
  about coarse resolution or about aliasing.
- p.74 §3.1: reference medium choice (14): λ0 = ½(inf λ + sup λ), μ0 = ½(inf μ + sup μ) — "the best rate of
  convergence". Iterations grow with contrast; infinite contrast: no convergence.
- p.75–76 §3.2: cost per iteration k1 N ≤ t ≤ k2 N log2 N; real-field symmetry halves storage (like our rfft).
- p.76–77 §3.3: laminate 32×32: "shows no oscillation" and coincides with exact solution. Circular fibre dilute:
  1024²: "Except from little undulations inside the inclusion, there is no significant oscillations at the fiber
  boundary where the field ε12 is discontinuous." Discrepancy "should not be attributed to a Gibbs phenomenon ...
  This oscillation is attached to the summation of the Fourier series which is not what the discrete inverse Fourier
  transform performs."
- p.77 §3.3.3: DFT is exact Fourier transform if (C1) periodic and (C2) cut-off frequency below half sampling
  frequency (Shannon). "condition (C2) is not met in general. In particular a discontinuous field has no cut-off
  frequency and there is no discretization able to capture this discontinuity. It is however expected that the
  solution of the discrete problem approaches the solution of the continuous problem when the image sampling
  (number of pixels) increases."
  => This is MS's own statement that the discretization truncates a non-band-limited field.
- p.80–81 §3.4 Tables 1–6, remarks: elastic overall stiffness error < 1% even at 32×32 pixels/fibre. Elastic-plastic:
  sensitive to resolution; "about 15% for the square array of fibers in an elastic-perfectly plastic matrix under
  tension at 0°, with a resolution of 32 × 32 pixels/fiber"; hardening modulus error "about 7.5% with a resolution
  of 32 × 32 pixels/fiber". They use 128×128 pixels/fibre thereafter. (Errors relative to finest resolution.)
- p.89 conclusion limitations: convergence not ensured with voids/rigid inclusions; many DOF (1024² for 64 fibres).

## Relevance
- Source of the standard LS eq (7), Γ̂0 (4)/(A.4), basic scheme.
- Coarse-resolution data point: with plasticity, 32 pixels per fibre gives up to ~15% flow-stress error (MS operator,
  pixel-level). Our partitions are pixel-like at 32×32.
- No statement about aliases weights; the "truncated Fourier series" framing is Brisard–Dormieux's, MS's own framing
  is the Shannon condition (C2).
