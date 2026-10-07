# Schneider (2019), CMAME 354, 783–801 — READ COVER TO COVER (page images, all 19 pages)

## Scope
Mathematical analysis of SCA (Liu–Bessa–Liu) "and, independently, by Wulfinghoff–Cavaliere–Reese" for small-strain,
NON-SOFTENING materials (strongly monotone + Lipschitz stress, (2.1)–(2.2): "a positive, uniform lower bound on the
eigenvalues of the material tangent"). Our matrix_hardening_modulus may be negative (softening) → outside his theory.
Any spatial dimension, any (anisotropic) fixed reference C0.

## Content
- §1: TFA (Dvorak–Benveniste): "The (visco)plastic strains are approximated as piecewise-constant on a pre-chosen
  partition of the cell ... By construction, the TFA captures the elastic behavior of the composite correctly. However,
  in the nonlinear regime, the stress predicted by the TFA could be much larger than the 'full-field' stress."
  (cf. our observation TFA too stiff in plasticity.) NTFA (Michel–Suquet).
- SCA: Liu–Bessa–Liu "introduced (an incremental version of) the clustered Lippmann–Schwinger equation in an ad-hoc
  manner", k-means of strain concentration tensor, self-consistent reference. Tang–Zhang–Liu: 1D convergence proof;
  "they introduced a variant of the SCA, where the unit cell is embedded into a larger reference medium" (= VCA, Zhang
  2019 builds on this). Wulfinghoff–Cavaliere–Reese: HS-based, "turned out to be equivalent to the SCA"; clustering
  from nonlinear snapshots; reference = secant stiffness.
- Contributions: "we extend the one-dimensional analysis of Tang–Zhang–Liu [42] to higher dimensions and the work of
  Brisard–Dormieux [45,46], who investigated the clustered Lippmann–Schwinger equation for identically shaped voxel
  clusters, to arbitrary cluster shape and nonlinear material behavior."
  "Both SCA and TFA can be brought into a form where the two schemes are identical up to a single operator (of
  Eshelby–Green type): for SCA this operator is clustered, for TFA, it is non-clustered."
- §2.1: (2.7) LS ε + Γ0:(σ(ε) − C0:ε) = E; solution independent of C0. (2.8)/(2.9) HS functional.
- §2.2: clusters Ω_i, "We do not assume the clusters to be connected or convex." (2.10) compatibility: each cluster
  in one phase ("if σ is homogeneous on each Y_k, then (2.10) ensures that for each Ω_i there is a Y_k, s.t.
  Ω_i ⊆ Y_k") — i.e. single-phase clusters (same as our restriction). (2.11) V_N cluster-wise constant; (2.12) Q_N
  cluster averaging, orthogonal projector for any C0. (2.14) HS on V_N; (2.15) Γ0_N = Q_N Γ0 Q_N; (2.16) clustered LS
  ε_N + Γ0_N:(σ(ε_N) − C0:ε_N) = E.
  Remarks: "The clustered Lippmann–Schwinger equation (2.16) is well-defined for any reference material C0";
  "for three spatial dimensions, the clustered Eshelby–Green operator Γ0_N can be represented by a 6N × 6N matrix,
  whose entries can be precomputed numerically (depending only on the clustering and not on the material law)."
  "In contrast to the continuous case, the solutions of the Lippmann–Schwinger equation (2.16) depend on the reference
  material. This is, in part, caused by Γ0_N failing to be a projector, i.e. Γ0_N : Γ0_N = Γ0_N does not hold for
  clusters of finite size." (2.17) Γ0_N:C0 non-expansive. Best HS-type estimates with reference "close" to material.
- §3: clustered Eyre–Milton reformulation (3.6); existence/uniqueness for any C0 (3.7); convergence upon refinement:
  (3.9) (1−ρ0)‖P* − P*_N‖ ≤ ‖(Id − Q_N)P*‖ with P = σ(ε) + C0:ε; "for fixed reference material, the convergence
  rates of ‖P* − P*_N‖ and ‖(Id − Q_N)P*‖ are identical" → error controlled by the best piecewise-constant
  approximation of P (our scheme-vs-floor split is analogous). Conclusion: only logarithmic convergence can be expected
  for piecewise-constant approximation (Davydov).
- §4 (non-rigorous) relation to TFA for generalized standard elasto-viscoplastic materials: TFA = Galerkin with
  cluster-wise constant internal variables (ε^p, q), u unrestricted: (4.2) div C:(ε_{n+1} − ε^p_n) = 0, (4.3) cluster
  laws driven by cluster-averaged strain. Written as LS: ε_{n+1} + Γ0:(C:(ε_{n+1} − ε^p_n) − C0:ε_{n+1}) = E_{n+1};
  SCA: ε_{n+1} + Γ0_N:(C:(ε_{n+1} − ε^p_n) − C0:ε_{n+1}) = E_{n+1}. "the only difference between the self-consistent
  clustering analysis and the transformation field analysis for small strain elasto-viscoplastic generalized standard
  materials is whether the clustered or the non-clustered Eshelby–Green operator is used."
  => This IS a theoretical TFA–LS link (exactly the operator difference our code isolates). Precise: TFA keeps the full
  Γ0 with the fine strain field unrestricted (only internal variables clustered); SCA projects strain onto clusters.
  Note his TFA eq. has ε^p at step n (explicit in written form) — it's a formal statement.
- §5 numerics: staggered grid FFT code; k-means (Lloyd + Elkan) on strain localization tensor; Γ_N components
  "computed by FFT-based methods and stored in a database"; reference materials ∝ identity only; Newton; reference
  closest to effective tangent. J2 with exponential-linear hardening (PA matrix), E-glass elastic.
  5.2 sphere 32³ voxels: SCA overestimates stress after yielding; 160 clusters < 5% up to 4% strain; logarithmic
  convergence in cluster count.
  5.3 continuous fibres 512²: longitudinal < 0.25%; transverse: "Assuming piecewise constant strains (1 + 1 clusters)
  overpredicts the stresses almost by a factor of 2", 64+64 clusters still 22.89%; "the large errors for the
  transverse loading already originate from the linear elastic behavior, which is insufficiently captured by the
  clustering." Oscillations at interfaces "a well-known phenomenon in FFT-based micromechanics".
- §6: SCA suited for "materials with low elastic contrasts or low filler fraction". TFA "similar in spirit (and,
  possibly, the formulation), but is restricted to small strain elasto-viscoplasticity".
- NO numerical comparison of SCA vs TFA. No uniform/box clusters in numerics.

## Relevance
- Our LS = clustered LS (2.16) with box clusters; our single-phase restriction = his (2.10).
- His theory needs hardening (strong monotonicity); softening is outside it.
- §4 gives the TFA–LS operator relation (formal); numerics comparing them: none here.
- Logarithmic convergence + elastic-stage errors with few clusters: relevant to expectations for coarse partitions.
