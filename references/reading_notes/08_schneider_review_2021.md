# Schneider (2021), A review of nonlinear FFT-based computational homogenization methods, Acta Mech 232, 2051–2100
READ COVER TO COVER (page images, all 50 pages, references included).

## pp. 2051–2070
- §2.1 LS (2.12) ε = ε̄ − Γ0:(∂w/∂ε − C0:ε); Γ0 Fourier (2.13) for C0 = 2μ0 Id; basic scheme (2.15).
- §2.2 MS discretization (2.19); Nyquist frequencies (2.21) "require special attention to ensure that the resulting
  fields are real-valued": set ε̂ = 0 or ε̂ = −(C0)⁻¹ : τ̂ there. Trigonometric interpolation view (2.23)–(2.24):
  "the stress is evaluated only at the grid points (2.18), and interpolated thereafter ... For the Moulinec–Suquet
  discretization, the balance of linear momentum is satisfied on the entire cell Y, but the stress–strain relationship
  is approximated."
- p.2057: "Following this rationale, Brisard and Dormieux [49,50] used the Hashin–Shtrikman variational principle
  [51–53] to get insights into the Moulinec–Suquet discretization, see Sect. 2.4 for more details. In a nutshell,
  Brisard and Dormieux regarded the Moulinec–Suquet discretization as an under-integrated variant of the conforming
  Galerkin discretization of the Hashin–Shtrikman variational principle with voxel-wise constant strains as ansatz
  functions." Convergence needs averaging rule (2.26). HS critical points on a subspace depend on C0, unlike MS.
  Collocation interpretation (2.28) (trigonometric collocation). Vondřejc: under-integrated Fourier–Galerkin
  (trapezoidal rule) (2.30) — "natural" setting; displacement is the basic quantity; reference material auxiliary.
- §2.3 Fourier–Galerkin (2.35)–(2.39): P_N projection instead of Q_N; bounds; limited to linear (needs Fourier
  coefficients of C).
- §2.4 HS-based: (2.40), Galerkin on V_N voxel-wise constant (2.41); (2.42) (C_{N,0} − C0)⁻¹:τ_N + Γ0_N:τ_N = ε̄,
  Γ0_N = P_N:Γ0:P_N (P_N here = voxel-average projector), averaging (2.26). Advantages: symmetric; Γ0_N Fourier
  multiplier with explicit expression ("as both operators P_N and Γ0 are translation-invariant on the voxel grid and
  their Fourier coefficients are known explicitly"); convergence proof [50]; triggered composite voxels.
  Disadvantages: "the solution τ_N of the Brisard–Dormieux type Lippmann–Schwinger equation depends on the reference
  material ... a judicious choice of C0 is necessary"; "Although explicit expressions for the Fourier coefficients of
  the operator Γ0_N are available, the involved series converges rather slowly in three spatial dimensions [50]. In
  particular, computing the operator Γ0_N on the fly is not recommended, and the Fourier coefficients need to be
  pre-computed and stored." Strain ε_N not compatible in general (Brisard [69] remedy).
  "The discretized variational principle (2.41) was also utilized for more general decompositions of the unit cell Y
  (2.1), for instance obtained from a clustering analysis of solution fields. The resulting method, called
  self-consistent clustering analysis [70–72], represents a state-of-the-art data-driven model order reduction approach
  ... Please note that the self-consistent clustering analysis concerns the Lippmann–Schwinger equation (2.12), but may
  also be computed via finite elements. In this context, the convergence analysis of Brisard–Dormieux [50] was
  generalized to strongly convex free energies w with Lipschitz-continuous gradient, also covering cluster
  decompositions more general than voxelizations." B-splines [73].
- §2.5 FD/FE/FV: Müller central differences "the well-known ringing artifacts associated with spectral and
  pseudo-spectral discretizations of non-smooth problems [76,77] can be avoided". Willot rotated staggered grid
  (2.43)–(2.49) = trilinear hexahedra with reduced (1-point) integration (p.2066); staggered grid (2.51); high-order
  CD; Table (2.52) k(ξ) for each scheme. FEM: FFT solvers for trilinear hex with 8 Gauss points (Schneider et al.
  [99]); Fritzen–Leuschner FANS [102]. "As finite element methods on regular meshes may be interpreted as specific
  finite difference stencils, it is possible to construct FFT-based solvers for finite element discretizations"
  "it does not make sense to 'compare' FFT-based methods to finite element discretizations".
  => Our FE-derived P0 (trilinear hex, 8 Gauss pts) corresponds to the review's "FEM" discretization class (Schneider
  et al. [99]) averaged over partitions — inference.
- Table 1 (p.2066): Brisard–Dormieux discretization: Nonlinear +, Anisotropic +, Heterogeneous +, Bounds +, Pores +,
  Explicit Γ −, Low memory −. MS: bounds −, pores −, explicit Γ +.
- Fig. 2 (p.2067): sphere glass/PA, 128³, LINEAR elastic, 5% ext.: "For all discretization methods, there are distinct
  artifacts at the interface. In the matrix, all discretizations look rather similar. In the inclusion,
  Moulinec–Suquet's discretization shows ringing artifacts, whereas Willot's scheme leads to checkerboarding. The
  central difference discretizations lead to an oscillatory behavior away from the interface which increases for
  increasing order". Table 2: effective moduli similar across schemes at fixed resolution.
- Porous sand (Fig. 3, Table 3): MS & high-order CD CG stall; staggered grid converges fast.
- §3.1 basic scheme; MS reference choice; Neumann series; Milton nonlinear contraction for α0 > α+²/(2α−).

## pp. 2071–2100
- §3.1–3.6 solvers: basic scheme as gradient descent; Barzilai–Borwein; Krylov (Brisard–Dormieux used CG/MINRES on
  (3.10) polarization form; Zeman CG on strain form, symmetric on compatible fields; LS = preconditioned balance eq.);
  fast gradient; Newton/Newton–Krylov (3.24) "For an arbitrary reference material C0, the linearized equilibrium
  equation (3.23) may also be rewritten in Lippmann–Schwinger form"; Anderson; §3.5 polarization methods = Eyre–Milton,
  ADMM (Michel–Moulinec–Suquet), Monchiet–Bonnet — SOLVERS, not discretizations. Convergence criterion (3.41)–(3.44).
  Table 4 solvers' memory.
- §4.1 dual scheme; §4.2 mixed BCs; §4.3 composite voxels (quoted before): "Brisard and Dormieux [49], based on
  earlier FEM work [189], first proposed the idea of considering sub-voxel scale information on the microstructure.
  This idea was further refined by Gélébart and Ouaki [190] and by Kabel et al. [191] ... It was found that furnishing
  the composite voxels by the effective stiffness of a two-phase laminate material ... led to the most accurate results
  for finitely contrasted media. When intersecting pores, Voigt averaging turned out to be advantageous, whereas Reuss
  averaging is best when intersecting a rigid inclusion. The composite voxel method was subsequently extended to finite
  strains [192] and to inelastic problems [193]. For the latter, the idea is to introduce internal variables for all
  the phases in a composite voxel, and to solve the evolution equations on the laminate, considered as a sub-voxel
  scale microstructure." "Often, dedicated composite voxel methods permit to reduce the number of degrees of freedom
  while retaining the accuracy of the computed results, even for inelastic and nonlinear constitutive behavior."
  Applications [195]–[198] incl. [195] Mareau & Robert 2017 "Different composite voxel methods ... inelastic materials".
- §5 applications (look-up), §6 conclusions: future = sparse sampling, tensor methods, AI surrogates.
- References: [94] Eloh, Jacques, Berbenni 2019 is cited in §2.5 among "further finite difference discretization
  schemes [90–94]" => per the review, Eloh et al.'s operator is a voxel-level discretization of the full-field
  problem (classification by Schneider; I have not read Eloh). [71] Wulfinghoff–Cavaliere–Reese CMAME 330 (2018);
  [99] Schneider–Merkert–Kabel IJNME 109 (2017) FFT-based homogenization for linear hexahedral elements;
  [69] Brisard IJNME 109 (2017) displacement reconstruction; [73] Tu et al. 2020 B-splines HS.
- The review does NOT mention uniform-cluster / coarse-partition SCA, nor TFA vs SCA numerics.
