# Kabel, Fink, Ospald, Schneider (2016), ECCOMAS Congress, pp. 2099–2109 — READ COVER TO COVER (11 pages)

## Content
- Abstract: "This work focuses on the composite voxel technique, where sub-voxels are merged into bigger voxels to which
  an effective material law based on laminates is assigned. Due to the down-sampled grid, both the memory requirements
  and the computational effort are severely reduced, while retaining the original accuracy." "In contrast to
  conventional model order reduction methods, our approach does neither rely upon an 'offline phase' nor on
  preselected 'modes'."
- §2: composite voxel W = voxel with more than one phase. Voigt rule C_W = ⟨C⟩ (upper bound). Laminate rule (Milton
  §9.5): (P + λ(C_W − λId)⁻¹)⁻¹ = ⟨(P + λ(C − λId)⁻¹)⁻¹⟩ with P built from the interface normal n, λ > largest
  eigenvalue of C(x).
  Nonlinear extension in three steps: 1. linearize each phase's law at the deformed state ε^n of the composite voxel;
  2. "Mixing: The effective linearized stiffness C_W is calculated according to the consistent mixing rules for the
  linear elastic case. The nonlinear behavior of the composite voxel is approximated by the affine linear function
  σ(ε^{n+1}, ε^n, ...) = σ^n + C_W : (ε^{n+1} − ε^n)"; 3. "Internal variables: The internal variables of the
  constituents are updated due to the deformation ε^{n+1} of the composite voxel."
  NB: each phase in the composite voxel sees the SAME voxel strain for the internal-variable update (my reading of
  step 3; not more detailed in the paper).
- §3 LFT (long-fibre PP/glass), 8000×300×27 voxels, "discretized on a staggered grid [17]" (194.4 M DOF); downsampled
  by 2 and 4. PP: J2, isotropic piecewise-linear hardening (softening of measured curve neglected). Glass/PP contrast
  72 GPa / 1.25 GPa ≈ 58.
- Elastic: laminate underestimates stiffness (max error < 7% half, < 24% quarter resolution); Voigt overestimates
  (< 8%, < 20%). Plastic (half resolution only): both start ~20% error; Voigt grows to 37%, laminate decreases
  monotonically. Conclusion: "working on a coarser resolution for physically nonlinear problems is only acceptable in
  combination with laminate mixing." Laminate fails at quarter resolution "when the interface directions calculated in
  the downsampling process are no longer meaningful".
- Speed: half resolution reduces runtime 45% (laminate)/58% (Voigt) elastic; plastic 85%/90%. Composite voxels reduce
  nonlinear iteration counts (Fig. 6). Parallel scaling §4.
- Future: "direct laminate mixing of nonlinear material laws in a future work [10]" (= Kabel–Schneider, in
  preparation; presumably the 2017 CMAME paper — inference, not stated).

## Relevance
- Coarse uniform voxels with mixed phases, nonlinear, solved with the coarse grid's own discretization (staggered
  grid); fine information only via mixing law/interface normal. Not a partition-averaged operator.
- Mixing rules for mixed partitions + plasticity: laminate (with per-phase internal variables) is the precedent.
- Error at 2× downsampling ≈ 20% for that LFT case — coarse resolution is costly in accuracy there.
