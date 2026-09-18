# Two reference solvers for actual E/P transformation field analysis

## Purpose and route

Consider a composite with stiff inclusions in a softer material. We prescribe the average strain of a small representative sample and want its stress and plastic history. Different regions generally experience different strains, so a material update in each region must agree with their mechanical interactions.

The starting point is Fish–Cui's partition strain transformation, Eq. (8) of their paper, written in the matrix notation of the influence-function notes [1].

**Equation (ER-1) — Actual heterogeneous E/P relation**

$$
\boxed{
\boldsymbol\varepsilon^B
=\mathbf E^B\bar{\boldsymbol\varepsilon}
+\sum_{A=1}^{M}\mathbf P^{BA}\boldsymbol\mu^A
}
$$

The missing quantities are the partition eigenstrains, which change with plastic deformation. This lecture develops two ways to solve that coupled problem: a reference fixed-point iteration inspired by Moulinec–Suquet, and reference-preconditioned Newton–Krylov inspired by Ladecký et al. **Both retain actual E/P. They are two solution strategies for the same reduced material model.**

The literature supplies the transformation and full-field solver principles. Their combination below is a proposed construction; algebraic checks do not establish performance or novelty.

**Companion code.** The [3D teaching implementation](../numerical_tests/actual_ep_fft_benchmark/README.md) implements strategy 1 and a dense-Newton baseline for the same actual E/P model. Its [equation map](../numerical_tests/actual_ep_fft_benchmark/EQUATION_MAP.md) connects these ER labels to code and converts this lecture's engineering coordinates to Mandel coordinates. The [offline chapter](../numerical_tests/actual_ep_fft_benchmark/OFFLINE_ELASTICITY.md) derives the matrix-free FFT elastic solves used to build E/P. Its offline GMRES is distinct from strategy 2's proposed online Newton–Krylov solve.

## 1. Setting, supplied data, and unknowns

The representative volume element (RVE), denoted $\Theta$, is a rectangular periodic sample: displacement fluctuations match on opposite faces. Its imposed mean strain is $\bar{\boldsymbol\varepsilon}$. Assume small strains, perfect bonding, no body forces, and fixed, positive-definite elastic stiffness $\mathbf L(\mathbf y)$ at position $\mathbf y$. Remove arbitrary rigid translation by setting the average displacement fluctuation to zero.

Divide the RVE into $M$ fixed partitions. For this lecture, each contains one material with constant elastic stiffness $\mathbf L^A$. Approximate plastic eigenstrain as constant within each partition, and use its **average total strain** to perform one local material update. This last choice is a constitutive approximation; an update at average strain need not equal the average of fine-point updates. Total strain can still vary within a partition in the elastic influence calculation.

Actual E/P comes from a resolved heterogeneous elastic discretization. It is exact for that discrete elastic problem and the selected eigenstrain representation, up to its solver tolerance; it is not an exact continuum or unrestricted plasticity solution. Direct FFT inversion of the *reference* introduced later additionally requires congruent, translated partitions on a periodic regular lattice. The E/P relation itself does not require that geometry.

Use engineering strain and physical stress vectors, so their dot product is mechanical work.

**Equation (ER-2) — Three-dimensional vector convention**

$$
\begin{aligned}
\boldsymbol\varepsilon
&=(\varepsilon_{11},\varepsilon_{22},\varepsilon_{33},
2\varepsilon_{12},2\varepsilon_{23},2\varepsilon_{13})^{\mathsf T}\\
\boldsymbol\sigma
&=(\sigma_{11},\sigma_{22},\sigma_{33},
\sigma_{12},\sigma_{23},\sigma_{13})^{\mathsf T}
\end{aligned}
$$

Eigenstrain uses the strain convention. Every operator must use these same shear conventions.

| Symbol | Meaning and size | Units |
|---|---|---|
| $A$, $B$ | Source and observation partition, respectively | Index |
| $\boldsymbol\varepsilon^A$, $\boldsymbol\mu^A$ | Average total strain and uniform plastic eigenstrain; six entries each | Strain |
| $\mathbf E^B$, $\mathbf P^{BA}$ | Actual-material influence blocks; $6\times6$ | Dimensionless |
| $\mathbf L^A$, $\mathbf C^0$ | Actual partition stiffness and homogeneous reference stiffness; $6\times6$ | Stress |
| $\mathbf z_n^A$ | Accepted history: plastic strain and accumulated plastic strain at the previous endpoint | Strain |
| $\mathbf H_\mu^A$ | Derivative of the local eigenstrain update; $6\times6$ | Dimensionless |
| $\mathbf H_{\mu,*}^A$ | Actual material derivative evaluated at the chosen anchor $\boldsymbol\varepsilon_*^A$ | Dimensionless |
| $\mathbf H_{\mu,0}$ | Chosen uniform reference sensitivity; generally different from $\mathbf H_{\mu,*}^A$ | Dimensionless |
| $\mathbf r$, $\delta\boldsymbol\varepsilon$ | Stacked residual and numerical strain correction; $6M$ entries | Strain |
| $\mathbf J$, $\mathbf M_0$ | Actual Jacobian and reference matrix; $6M\times6M$ | Dimensionless |

The subscript on $\mathbf H_\mu$ distinguishes material sensitivity from displacement influences and hardening. “Strain” distinguishes the dimensionless reduced residual from a force imbalance.

For sensitivities, $*$ means evaluation at the chosen anchor, $k$ at the current Newton iterate, and $\mathrm{sol}$ at the converged solution. The subscript $0$ instead identifies the **reference operator**, not an evaluation point or iteration number. Without a partition superscript, actual sensitivities denote their stacked block-diagonal matrices. For brevity, $\mathbf H_{\mu,0}$ denotes either the common $6\times6$ reference block or its $6M\times6M$ block-diagonal repetition, according to whether it acts on a local or stacked strain vector.

At each load step, E/P, elastic properties, the new imposed mean strain, and $\mathbf z_n^A$ are supplied. The $6M$ partition total strains are the online unknowns. This is the partition-strain formulation; it does not reproduce Fish–Cui's further reduction to eigenstate unknowns or their complete EBH algorithms.

An iteration changes a guess within one fixed loading step. A numerical correction $\delta\boldsymbol\varepsilon$ is not the physical strain increment between two accepted loading endpoints.

## 2. Turn the transformation into a solvable equation

Given a trial partition strain and the accepted history, a stress-update routine returns candidate plastic strain, stress, and history. For J2 plasticity with linear isotropic hardening, that local update can use a closed-form radial return. It need not contain an iterative solver.

Make the strain dependence explicit, then compute stress from elastic strain.

**Equation (ER-3) — Local material update and stress**

$$
\boldsymbol\mu^A=\boldsymbol\mu^A(\boldsymbol\varepsilon^A;\mathbf z_n^A),
\qquad
\boldsymbol\sigma^A=\mathbf L^A
[\boldsymbol\varepsilon^A-\boldsymbol\mu^A]
$$

The semicolon reminds us that accepted history stays fixed while trial strains change. For constant stiffness and eigenstrain within a partition, ER-3 also gives its average stress.

Stack the partition vectors in partition order, assemble P from its $BA$ blocks, and stack the E blocks vertically. Define $\mathbf b$ as the known elastic loading contribution. The residual is guessed strain minus the strain predicted by ER-1.

**Equation (ER-4) — Actual reduced residual**

$$
\mathbf b=\mathbf E\bar{\boldsymbol\varepsilon},
\qquad
\boxed{
\mathbf r(\boldsymbol\varepsilon)
=\boldsymbol\varepsilon-\mathbf b
-\mathbf P\boldsymbol\mu(\boldsymbol\varepsilon)
=\mathbf0
}
$$

History arguments are suppressed from now on for readability, not changed. E/P remains fixed; the material update makes this equation nonlinear.

After convergence, accept the candidate histories and return the average stress. Let $c^A=|\Theta^A|/|\Theta|$ be the partition volume fraction.

**Equation (ER-5) — Macroscopic output**

$$
\bar{\boldsymbol\sigma}
=\sum_{A=1}^{M}c^A\mathbf L^A
[\boldsymbol\varepsilon^A-\boldsymbol\mu^A]
$$

Changing a solver should not change this converged prediction, provided it reaches the same solution branch and tolerance.

## 3. The original Moulinec–Suquet scheme, before partition reduction

To understand the reference idea, first solve the full spatial problem for **linear elasticity without eigenstrain** [2, Eqs. (1)–(5)]. No TFA partition approximation enters this section. Strain must both produce an equilibrated stress and come from a displacement. Here $\mathbf u$ is the periodic displacement fluctuation and $\nabla^s$ takes its symmetric gradient. Differential expressions use tensor fields; matrix products use their ER-2 representations.

**Equation (ER-6a) — Equilibrium, elasticity, and compatible strain**

$$
\nabla\cdot\boldsymbol\sigma=\mathbf0,
\qquad
\boldsymbol\sigma=\mathbf L(\mathbf y)\boldsymbol\varepsilon,
\qquad
\boldsymbol\varepsilon=\bar{\boldsymbol\varepsilon}+\nabla^s\mathbf u
$$

Choose uniform reference stiffness $\mathbf C^0$. Add and subtract its stress; call the remaining actual-minus-reference stress **polarization**.

**Equation (ER-6) — Exact reference-stress split**

$$
\begin{aligned}
\boldsymbol\sigma
&=\mathbf L\boldsymbol\varepsilon
+\mathbf C^0\boldsymbol\varepsilon-\mathbf C^0\boldsymbol\varepsilon\\
&=\mathbf C^0\boldsymbol\varepsilon+\boldsymbol\tau,
\qquad
\boldsymbol\tau=(\mathbf L-\mathbf C^0)\boldsymbol\varepsilon
\end{aligned}
$$

Nothing has been approximated. Substitute this split into equilibrium, then insert the compatible strain. The constant mean strain and constant reference stiffness contribute zero divergence.

**Equation (ER-6b) — The reference equilibrium problem**

$$
\begin{aligned}
\nabla\cdot(\mathbf C^0\boldsymbol\varepsilon)
&=-\nabla\cdot\boldsymbol\tau\\
\nabla\cdot(\mathbf C^0\nabla^s\mathbf u)
&=-\nabla\cdot\boldsymbol\tau\\
\mathcal L_0\mathbf u
:=\nabla\cdot(\mathbf C^0\nabla^s\mathbf u)
&=\mathbf f,
\qquad \mathbf f:=-\nabla\cdot\boldsymbol\tau
\end{aligned}
$$

If polarization were known, this would be homogeneous elasticity with a known source. We keep the minus sign in $\mathbf f$ and define $\mathcal L_0$ with **positive divergence**. The signed source density has units force per volume; its contribution from a small volume is $d\mathbf F_{\mathrm{src}}=\mathbf f(\widehat{\mathbf y})dV_{\widehat{\mathbf y}}$. In the usual physical balance $\nabla\cdot\boldsymbol\sigma+\mathbf b_{\mathrm{body}}=\mathbf0$, a body force would instead appear as $\mathbf b_{\mathrm{body}}=-\mathbf f$.

### From equilibrium to its Green response

Let $\mathbf G^0(\mathbf y,\widehat{\mathbf y})$ describe displacement at observation point $\mathbf y$ due to a unit signed source in ER-6b at $\widehat{\mathbf y}$. It inverts the positive-divergence operator just defined, with periodic conditions and rigid translation removed. The periodic unit source is balanced by a uniform opposite source; these balancing contributions cancel for a zero-integral input, such as $\mathbf f=-\nabla\cdot\boldsymbol\tau$.

Linearity lets us add the responses to all small forces. The defining inverse property of G₀ ensures that applying the equilibrium operator returns the supplied force density.

**Equation (ER-6c) — Displacement solution and equilibrium check**

$$
\mathbf u(\mathbf y)
=\int_\Theta\mathbf G^0(\mathbf y,\widehat{\mathbf y})
\mathbf f(\widehat{\mathbf y})\,dV_{\widehat{\mathbf y}},
\qquad
\mathcal L_0\mathbf u
=\int_\Theta(\mathcal L_{0,\mathbf y}\mathbf G^0)
\mathbf f\,dV_{\widehat{\mathbf y}}
=\mathbf f(\mathbf y)
$$

Thus equilibrium has not disappeared: this integral solves it. Now insert $\mathbf f=-\nabla\cdot\boldsymbol\tau$ and recover strain by differentiating displacement. Integration by parts moves the source derivative onto G₀; periodic boundary terms cancel. Denote the mixed source derivative and symmetric observation derivative of G₀ by $\boldsymbol\Gamma^0$.

**Equation (ER-6d) — Derive strain response before simplifying its kernel**

$$
\begin{aligned}
\boldsymbol\varepsilon(\mathbf y)-\bar{\boldsymbol\varepsilon}
&=-\nabla^s_{\mathbf y}\int_\Theta
\mathbf G^0(\mathbf y,\widehat{\mathbf y})
\nabla_{\widehat{\mathbf y}}\cdot\boldsymbol\tau(\widehat{\mathbf y})
\,dV_{\widehat{\mathbf y}}\\
&=+\int_\Theta\boldsymbol\Gamma^0(\mathbf y,\widehat{\mathbf y})
\boldsymbol\tau(\widehat{\mathbf y})\,dV_{\widehat{\mathbf y}}
\end{aligned}
$$

The minus sign in the source cancels the minus sign from integration by parts. The strain integral is a **derived solution**, not a new strain assumption. G₀ solves for displacement; Γ₀ includes strain recovery. With this convention, $\boldsymbol\Gamma^0$ is the negative of the usual Moulinec–Suquet strain Green operator: $\boldsymbol\Gamma^0_{\mathrm{MS}}=-\boldsymbol\Gamma^0$. The physical response is identical; keep this sign conversion when using published Fourier formulas. Singular continuum kernels are understood weakly; computation uses their specified discrete counterparts.

### Why the kernel depends only on separation

Shift a displacement by a constant vector $\mathbf a$, writing $\mathbf z=\mathbf y-\mathbf a$. The chain rule shifts its derivatives by the same amount. Because $\mathbf C^0$ is constant, the whole reference equation shifts with them.

**Equation (ER-6e) — Translation leaves the reference problem unchanged**

$$
\frac{\partial u_i(\mathbf y-\mathbf a)}{\partial y_j}
=\left.\frac{\partial u_i(\mathbf z)}{\partial z_j}\right|_{\mathbf z=\mathbf y-\mathbf a},
\qquad
\mathcal L_0[\mathbf u(\cdot-\mathbf a)](\mathbf y)
=(\mathcal L_0\mathbf u)(\mathbf y-\mathbf a)
=\mathbf f(\mathbf y-\mathbf a)
$$

Periodicity and zero mean also survive the shift. Uniqueness therefore gives $\mathbf G^0(\mathbf y+\mathbf a,\widehat{\mathbf y}+\mathbf a)=\mathbf G^0(\mathbf y,\widehat{\mathbf y})$, and the same holds for Γ₀. Choose $\mathbf a=-\widehat{\mathbf y}$: the kernel depends only on the separation $\mathbf y-\widehat{\mathbf y}$. ER-6d becomes

**Equation (ER-7) — Reference convolution and the original elastic equation**

$$
\begin{aligned}
\boldsymbol\varepsilon(\mathbf y)
&=\bar{\boldsymbol\varepsilon}
+\int_\Theta\boldsymbol\Gamma^0(\mathbf y-\widehat{\mathbf y})
\boldsymbol\tau(\widehat{\mathbf y})\,dV_{\widehat{\mathbf y}}\\
&=\bar{\boldsymbol\varepsilon}
+\boldsymbol\Gamma^0*[(\mathbf L-\mathbf C^0)\boldsymbol\varepsilon]
\end{aligned}
$$

The star abbreviates **convolution**: add the same response pattern shifted to each source location and weighted by that source. Substituting actual polarization closes the equation, but strain still occurs on both sides.

### The original Fourier-grid iteration

Store strain and stiffness at regularly spaced points $\mathbf y_\nu$. The subscript $\nu$ labels a grid point, not a TFA partition or a loading endpoint. An FFT expresses these sampled fields through spatial-wave coefficients; a hat denotes those coefficients, and $\boldsymbol\xi$ labels a frequency. For uniform reference elasticity, each wave has its own known response $\widehat{\boldsymbol\Gamma}^0(\boldsymbol\xi)$. It is obtained by solving the reference equilibrium equation for that wave.

Evaluate polarization from the current guess, transform it, apply the reference response, and transform back [2, algorithm (5)].

**Equation (ER-7a) — Classical Moulinec–Suquet update in this sign convention**

$$
\begin{aligned}
\boldsymbol\tau_\nu^j
&=(\mathbf L_\nu-\mathbf C^0)\boldsymbol\varepsilon_\nu^j\\
\widehat{\boldsymbol\tau}^{\,j}
&=\operatorname{FFT}(\boldsymbol\tau^j)\\
\boldsymbol\varepsilon^{j+1}
&=\bar{\boldsymbol\varepsilon}
+\operatorname{IFFT}
\left[\widehat{\boldsymbol\Gamma}^{\,0}
\widehat{\boldsymbol\tau}^{\,j}\right],
\qquad \widehat{\boldsymbol\Gamma}^{\,0}(\mathbf0)=\mathbf0
\end{aligned}
$$

The bracket is a matrix-vector product **at each frequency**. The inverse FFT returns a strain-like field on the same grid; adding it to the mean gives the next compatible strain guess. Since $\widehat{\boldsymbol\Gamma}^0=-\widehat{\boldsymbol\Gamma}^0_{\mathrm{MS}}$, this is the original update written with the present kernel sign. The zero block removes the fluctuation's mean, so adding $\bar{\boldsymbol\varepsilon}$ imposes the prescribed mean without a transform-normalization ambiguity.

This is the original Fourier-grid construction: **sample fields and apply the reference Fourier operator**. We have not replaced it by source-cell integration or partition averaging. A finite grid retains finitely many spatial frequencies and cannot resolve arbitrary fine variation. Consistent frequency and highest-frequency conventions remain part of the numerical discretization.

Repeat until actual stress equilibrium converges. At convergence, the polarization supplied to reference equilibrium agrees with the resulting strain, recovering the original heterogeneous problem. For plasticity, the paper's algorithm (8) instead evaluates stress using the local material update and forms $\boldsymbol\tau^j=\boldsymbol\sigma^j-\mathbf C^0\boldsymbol\varepsilon^j$ from the same accepted history.

**We now return to TFA.** The transferable idea is the exact reference-plus-difference split. The reduced construction will additionally distribute eigenstrains over partitions and average their reference responses; those are not steps of the original grid iteration.

## 4. Choose a reduced reference without inventing an eigenstrain law

For ER-4, the quantity to approximate is the induced strain $\mathbf P\boldsymbol\mu(\boldsymbol\varepsilon)$. A useful reference should map a strain change into an approximate induced-strain change.

Choose an anchor strain $\boldsymbol\varepsilon_*^A$ in each partition. Its actual eigenstrain and, when needed, its actual derivative are defined below with accepted history $\mathbf z_n^A$ fixed. The history argument is displayed here to clarify what is held fixed.

**Equation (ER-8) — Sensitivity and its correct interpretation**

$$
\begin{aligned}
\boldsymbol\mu_*^A
&:=\boldsymbol\mu^A(\boldsymbol\varepsilon_*^A;\mathbf z_n^A)\\
\mathbf H_{\mu,*}^A
&:=\left.
\frac{\partial\boldsymbol\mu^A(\boldsymbol\varepsilon^A;\mathbf z_n^A)}
{\partial\boldsymbol\varepsilon^A}
\right|_{\boldsymbol\varepsilon^A=\boldsymbol\varepsilon_*^A}\\
\boldsymbol\mu^A(\boldsymbol\varepsilon^A;\mathbf z_n^A)
&\approx\boldsymbol\mu_*^A
+\mathbf H_{\mu,*}^A
(\boldsymbol\varepsilon^A-\boldsymbol\varepsilon_*^A)
\end{aligned}
$$

Thus $\mathbf H_{\mu,*}^A$ is the actual sensitivity **at the anchor**, not at the unknown converged solution. This first-order approximation applies locally on a differentiable material branch. In general, total eigenstrain is **not** $\mathbf H_\mu^A\boldsymbol\varepsilon^A$: elastic unloading can give zero sensitivity while accumulated plastic strain remains nonzero.

Choose $\mathbf P_0$ from a homogeneous elastic reference on the same partitions. It maps eigenstrain into induced strain; we also need a map from strain into reference eigenstrain. Choose one constant $6\times6$ sensitivity $\mathbf H_{\mu,0}$, repeated in every partition. It may be selected from the anchor derivatives $\mathbf H_{\mu,*}^A$, or prescribed independently. The lecture does not assume a validated selection rule; using each partition's different derivative directly would generally lose the uniformity needed for the reference FFT.

**ER-8 approximates the actual law; ER-8a defines a different, numerical reference.** Therefore ER-8a uses $\mathbf H_{\mu,0}$, not $\mathbf H_{\mu,*}$. We match the actual eigenstrain at the anchor, but do not generally match its derivative there. With stacked vectors and the repeated reference matrix, define

**Equation (ER-8a) — Full affine reference eigenstrain**

$$
\begin{aligned}
\boldsymbol\mu_{\mathrm{ref}}(\boldsymbol\varepsilon)
&=\boldsymbol\mu_*+\mathbf H_{\mu,0}(\boldsymbol\varepsilon-\boldsymbol\varepsilon_*)
=\boldsymbol\mu_{\mathrm{off}}+\mathbf H_{\mu,0}\boldsymbol\varepsilon\\
\boldsymbol\mu_{\mathrm{off}}
&=\boldsymbol\mu_*-\mathbf H_{\mu,0}\boldsymbol\varepsilon_*
\end{aligned}
$$

“Affine” means a linear term plus a constant offset. The anchor values and offset may differ between partitions. Hold them and the reference sensitivity fixed during each update. The derivative of this **defined affine reference** is $\mathbf H_{\mu,0}$ everywhere, including at $\boldsymbol\varepsilon_*$. Evaluating the reference derivative there does not make it the actual derivative $\mathbf H_{\mu,*}$. They coincide only if we specifically choose them to match, consistently with the uniform-reference restriction. The actual nonlinear law remains in the residual; ER-8a does not replace it.

To see where the reference matrix comes from, subtract the **complete** reference response from both sides of ER-4, then expand it.

**Equation (ER-8b) — The offset appears on both sides**

$$
\begin{aligned}
\boldsymbol\varepsilon-\mathbf P_0\boldsymbol\mu_{\mathrm{ref}}(\boldsymbol\varepsilon)
&=\mathbf b+\mathbf P\boldsymbol\mu(\boldsymbol\varepsilon)
-\mathbf P_0\boldsymbol\mu_{\mathrm{ref}}(\boldsymbol\varepsilon)\\
(\mathbf I-\mathbf P_0\mathbf H_{\mu,0})\boldsymbol\varepsilon
-\mathbf P_0\boldsymbol\mu_{\mathrm{off}}
&=\mathbf b+\mathbf P\boldsymbol\mu(\boldsymbol\varepsilon)
-\mathbf P_0\boldsymbol\mu_{\mathrm{off}}
-\mathbf P_0\mathbf H_{\mu,0}\boldsymbol\varepsilon
\end{aligned}
$$

The identical offset terms cancel. The coefficient of strain on the left motivates the reference matrix:

**Equation (ER-9) — Candidate reduced reference**

$$
\mathbf A_0=\mathbf P_0\mathbf H_{\mu,0},
\qquad
\boxed{\mathbf M_0=\mathbf I-\mathbf A_0
=\mathbf I-\mathbf P_0\mathbf H_{\mu,0}}
$$

The product order matters: $\mathbf H_{\mu,0}$ converts a strain **change** into a reference eigenstrain change; $\mathbf P_0$ converts that change into an induced-strain change. Thus $\mathbf A_0\boldsymbol\varepsilon$ is only the strain-dependent part of the reference response, not its full value. ER-9 is a motivated **numerical choice**; an exact split does not guarantee a useful solver.

**When to choose the anchor.** Here $\boldsymbol\varepsilon^{(0)}$ denotes the initial guess for the current load step, and $j$ counts iterations within that step.

| Choice | Anchor and sensitivity policy | Tradeoff |
|---|---|---|
| Once per load step | Set $\boldsymbol\varepsilon_*=\boldsymbol\varepsilon^{(0)}$; select and freeze $\mathbf H_{\mu,0}$. The guess can be the previous accepted strain or a predictor for the new load. | Reuse the reference inverse; the sensitivity may become outdated as yielding develops. |
| Refresh during iteration | Set $\boldsymbol\varepsilon_*=\boldsymbol\varepsilon^j$ at a refresh; reselect $\mathbf H_{\mu,0}$ and freeze it for that update. | Track changing response, at the cost of rebuilding reference frequency inverses; improvement is not guaranteed. |

Moving the anchor **without changing the reference sensitivity** changes only the offset, which cancels in ER-8b. It therefore does not change the iteration. A simple starting implementation freezes the reference per load step; section 9 explains what controls convergence. Both policies use the same accepted history $\mathbf z_n$ throughout the step.

## 5. Strategy 1: reference fixed-point iteration on actual E/P

Turn ER-8b into an iteration: evaluate the actual response and the subtracted reference response at the current guess $j$, and solve the reference problem for the next guess $j+1$.

**Equation (ER-10) — Reference update before cancelling the offset**

$$
\begin{aligned}
\boldsymbol\varepsilon^{j+1}
-\mathbf P_0\boldsymbol\mu_{\mathrm{ref}}(\boldsymbol\varepsilon^{j+1})
&=\mathbf b+\mathbf P\boldsymbol\mu(\boldsymbol\varepsilon^j)
-\mathbf P_0\boldsymbol\mu_{\mathrm{ref}}(\boldsymbol\varepsilon^j)\\
\mathbf M_0\boldsymbol\varepsilon^{j+1}-\mathbf P_0\boldsymbol\mu_{\mathrm{off}}
&=\mathbf b+\mathbf P\boldsymbol\mu(\boldsymbol\varepsilon^j)
-\mathbf P_0\boldsymbol\mu_{\mathrm{off}}-\mathbf A_0\boldsymbol\varepsilon^j
\end{aligned}
$$

Although the two strain guesses differ, the **same reference offset** appears on both sides and cancels exactly. No assumption $\boldsymbol\mu=\mathbf H_{\mu,0}\boldsymbol\varepsilon$ is needed. As in the Moulinec–Suquet split, the right-hand side retains the difference between actual and reference responses.

**Equation (ER-11) — Reference fixed-point update**

$$
\mathbf M_0\boldsymbol\varepsilon^{j+1}
=\mathbf b+\mathbf P\boldsymbol\mu(\boldsymbol\varepsilon^j)
-\mathbf A_0\boldsymbol\varepsilon^j
$$

At a fixed point, the old and new strains coincide and the reference terms cancel, recovering ER-4. Subtracting $\mathbf M_0\boldsymbol\varepsilon^j$ gives a clearer implementation in terms of a correction.

**Equation (ER-12) — Residual correction form**

$$
\boxed{
\mathbf M_0\delta\boldsymbol\varepsilon^j=-\mathbf r(\boldsymbol\varepsilon^j),
\qquad
\boldsymbol\varepsilon^{j+1}
=\boldsymbol\varepsilon^j+\delta\boldsymbol\varepsilon^j
}
$$

Now $\mathbf H_{\mu,0}$ explicitly acts on a correction. This is nonlinear preconditioned Richardson iteration. It does not require the current exact material derivative: a chosen fixed reference is sufficient to define the algorithm.

Section 6 gives the implementation steps when the reference has the required FFT structure.

With stored P, there is one nonlinear loop. A direct FFT reference inverse adds no iterative linear loop. Relaxation can replace the update by $\boldsymbol\varepsilon^{j+1}=\boldsymbol\varepsilon^j+\omega\delta\boldsymbol\varepsilon^j$ with positive $\omega$; this changes convergence, not the fixed-point equation. Relaxation does not guarantee convergence.

## 6. When the reference solve really is an FFT solve

ER-12 asks us to solve $\mathbf M_0\delta\boldsymbol\varepsilon^j=-\mathbf r^j$, where $\mathbf M_0=\mathbf I-\mathbf P_0\mathbf H_{\mu,0}$ and $\mathbf r^j:=\mathbf r(\boldsymbol\varepsilon^j)$. We first construct P₀, then exploit its repeating spatial structure to solve this correction equation efficiently.

**Construct the reference elastic response.** Prescribe a test eigenstrain field $\boldsymbol\eta(\mathbf y)$ in the homogeneous material and solve for its total-strain response $\boldsymbol e_0(\mathbf y)$ at zero imposed average strain. This temporarily separates the elastic input–output map from the constitutive law; it does not assume that online eigenstrain is independent of strain. Actual P is defined by the same type of calculation in the heterogeneous material.

Stress enters only to enforce elastic equilibrium. With periodic compatibility, the Green solution from section 3 gives

**Equation (ER-13a) — Reference response to a prescribed eigenstrain input**

$$
\begin{aligned}
\boldsymbol\sigma_0&=\mathbf C^0(\boldsymbol e_0-\boldsymbol\eta),
&\nabla\cdot\boldsymbol\sigma_0&=\mathbf0,
&\langle\boldsymbol e_0\rangle&=\mathbf0\\
\boldsymbol\tau_0&=-\mathbf C^0\boldsymbol\eta,
&\boldsymbol e_0&=\boldsymbol\Gamma^0*\boldsymbol\tau_0
=-\boldsymbol\Gamma^0*(\mathbf C^0\boldsymbol\eta)
\end{aligned}
$$

Here $\boldsymbol\tau_0$ is an auxiliary **stress** polarization used to construct P₀, not the online strain-response difference $\mathbf P\boldsymbol\mu-\mathbf P_0\boldsymbol\mu_{\mathrm{ref}}$. The minus sign follows our convention $\mathbf f=-\nabla\cdot\boldsymbol\tau$ and $\boldsymbol\Gamma^0=-\boldsymbol\Gamma^0_{\mathrm{MS}}$.

Prescribe constant $\boldsymbol\eta^A$ inside source partition A and zero elsewhere. The convolution then integrates only over A. Average the resulting strain over observation partition B and take the constant input outside the integrals:

**Equation (ER-13) — Source input, averaged response, and influence block**

$$
\begin{aligned}
\frac{1}{|\Theta^B|}\int_{\Theta^B}\boldsymbol e_0(\mathbf y)\,dV_{\mathbf y}
&=\mathbf P_0^{BA}\boldsymbol\eta^A\\
\mathbf P_0^{BA}
&=-\frac{1}{|\Theta^B|}
\int_{\Theta^B}\int_{\Theta^A}
\boldsymbol\Gamma^0(\mathbf y-\widehat{\mathbf y})\mathbf C^0
\,dV_{\widehat{\mathbf y}}\,dV_{\mathbf y}
\end{aligned}
$$

A classical grid implementation distributes the test eigenstrain onto fine-grid points, applies $\operatorname{IFFT}[\widehat{\boldsymbol\Gamma}^0\operatorname{FFT}(-\mathbf C^0\boldsymbol\eta)]$, and averages the response into partitions. This discretizes ER-13, rather than evaluating an exact continuum cell integral; the reference grid can be finer than the partition lattice. A conventional positive-sign MS kernel gives the same result through $\boldsymbol\Gamma^0_{\mathrm{MS}}*(\mathbf C^0\boldsymbol\eta)$. An FE-based construction must instead use its matching discrete Green operator.

**Expose the convolution in the correction equation.** For congruent translated partitions on a periodic regular lattice, reference interactions depend only on separation. Write $\mathbf P_0^{BA}=\mathbf P_0[B-A]$: brackets label a periodic lattice offset, not a new operator. With the same $\mathbf H_{\mu,0}$ in every partition, ER-12 becomes

**Equation (ER-14) — The spatially coupled correction equation**

$$
\delta\boldsymbol\varepsilon^{j,B}
-\sum_A\mathbf P_0[B-A]\mathbf H_{\mu,0}
\delta\boldsymbol\varepsilon^{j,A}
=-\mathbf r^{j,B}
$$

Here $j$ labels the iteration and A or B the partition. The sum is a convolution: the same interaction pattern is shifted to each source. Before transformation, the correction in B depends on corrections throughout the sample.

**Change from locations to spatial patterns.** A Fourier transform describes the same partition values by their contributions to sinusoidal patterns. For a one-dimensional row of $M$ partitions, its definition is

**Equation (ER-14a) — Transform the residual and the reference kernel**

$$
\begin{aligned}
\widehat{\mathbf r}^{\,j}(\xi)
&=\sum_{B=0}^{M-1}\mathbf r^{j,B}e^{-i\xi B},
&\xi&=\frac{2\pi k}{M},\quad k=0,\ldots,M-1\\
\widehat{\mathbf P}_0(\xi)
&=\sum_{d=0}^{M-1}\mathbf P_0[d]e^{-i\xi d}
\end{aligned}
$$

The weights $e^{-i\xi B}=\cos(\xi B)-i\sin(\xi B)$ measure the field against each cosine/sine pattern. We sum over locations because each pattern spans the sample. The zero-frequency residual is the sum of partition residuals; dividing it by $M$ gives their average. The inverse transform recovers the original values. An **FFT** computes these transforms efficiently. In three dimensions, transform along all three lattice directions; $\boldsymbol\xi$ labels the resulting frequency vector. Transform spatial indices separately for each vector component or matrix entry.

In ER-14, set $d=B-A$. The Fourier weight separates as $e^{-i\xi(A+d)}=e^{-i\xi d}e^{-i\xi A}$, so the double sum factors into $\widehat{\mathbf P}_0\mathbf H_{\mu,0}\widehat{\delta\boldsymbol\varepsilon}^{\,j}$. This is why convolution becomes multiplication: each frequency couples only its own six components.

**Prepare once, then correct each guess.** Transform the reference kernel and form $\widehat{\mathbf M}_0$ below. Check every block is invertible and factorize it; reuse the factors while the reference stays fixed. For each load step, initialize strain and hold accepted history $\mathbf z_n$ fixed. The iteration is

**Equation (ER-15) — Actual residual, FFT reference solve, and strain update**

$$
\begin{aligned}
\text{Reference:}\qquad
\widehat{\mathbf M}_0(\boldsymbol\xi)
&=\mathbf I_6-\widehat{\mathbf P}_0(\boldsymbol\xi)\mathbf H_{\mu,0}\\[3pt]
\text{1. Actual residual:}\qquad
\mathbf r^j
&=\boldsymbol\varepsilon^j-\mathbf b
-\mathbf P\boldsymbol\mu(\boldsymbol\varepsilon^j;\mathbf z_n)\\
\text{2. Transform:}\qquad
\widehat{\mathbf r}^{\,j}
&=\operatorname{FFT}[\mathbf r^j]\\
\text{3. Solve at each frequency:}\qquad
\widehat{\mathbf M}_0(\boldsymbol\xi)
\widehat{\delta\boldsymbol\varepsilon}^{\,j}(\boldsymbol\xi)
&=-\widehat{\mathbf r}^{\,j}(\boldsymbol\xi)\\
\text{4. Transform back:}\qquad
\delta\boldsymbol\varepsilon^j
&=\operatorname{IFFT}[\widehat{\delta\boldsymbol\varepsilon}^{\,j}]\\
\text{5. Update:}\qquad
\boldsymbol\varepsilon^{j+1}
&=\boldsymbol\varepsilon^j+\delta\boldsymbol\varepsilon^j
\end{aligned}
$$

Check convergence after step 1; stop and accept the candidate history when the actual residual meets tolerance. Otherwise complete steps 2–5 and repeat. There is no need to transform an unknown correction beforehand: its Fourier coefficients are the unknowns in step 3. At zero frequency, $\widehat{\mathbf P}_0(\mathbf0)=\mathbf0$ and $\widehat{\mathbf M}_0(\mathbf0)=\mathbf I_6$.

**Why this helps.** One coupled $6M$-unknown reference problem becomes $M$ independent $6\times6$ systems. FFTs plus these small solves cost $O(M\log M)$ per correction, with $O(M)$ reference storage. This requires the convolution structure, not merely linearity. Irregular clusters or spatially varying reference sensitivities generally lose the direct frequency-by-frequency inverse.

**Why the heterogeneous residual is allowed.** Any known residual field can be transformed. The special structure is required of the reference operator being inverted, not of its right-hand side. Step 1 still computes actual $\mathbf P\boldsymbol\mu$ before transforming the residual; it does not assume actual P is a convolution. A dense actual P product still costs $O(M^2)$, so the complete iteration is not automatically $O(M\log M)$.

## 7. The Ladecký route: linearize first, then precondition

Ladecký et al. begin with periodic finite-element equilibrium [3, Sections 2–4]. Their unknowns are nodal displacement fluctuations. To connect their equation to familiar mechanics, let $\mathbf d_I$ be the displacement at node I and let $\mathbf B_I(\mathbf y)$ convert it into a strain contribution. Actual stress must balance at every independent node.

**Equation (ER-16) — Finite-element strains and nodal equilibrium**

$$
\boldsymbol\varepsilon(\mathbf y)
=\bar{\boldsymbol\varepsilon}
+\sum_I\mathbf B_I(\mathbf y)\mathbf d_I,
\qquad
\mathbf R_I=\int_\Theta\mathbf B_I^{\mathsf T}\boldsymbol\sigma\,d\Theta
=\mathbf0
$$

Newton differentiates the stress response. Define $\mathbf C_{\rm alg}$ as its derivative with respect to supplied strain at fixed accepted history. Assemble the nodal blocks below into $\mathbf K_{\rm T}$; $\mathbf R$ stacks nodal force residuals.

**Equation (ER-17) — Actual FE Newton correction and homogeneous reference**

$$
\begin{aligned}
\mathbf K_{\rm T}^{IJ}
&=\int_\Theta\mathbf B_I^{\mathsf T}\mathbf C_{\rm alg}(\mathbf y)\mathbf B_J\,d\Theta,
&\mathbf K_{\rm T}\delta\mathbf d&=-\mathbf R\\
\mathbf K_0^{IJ}
&=\int_\Theta\mathbf B_I^{\mathsf T}\mathbf C^0\mathbf B_J\,d\Theta,
&\mathbf K_0^{-1}\mathbf K_{\rm T}\delta\mathbf d&=-\mathbf K_0^{-1}\mathbf R
\end{aligned}
$$

These are the paper's Eqs. (9)–(12), translated from its discrete-gradient notation. Its reference stiffness is constant in space and its FE stencil repeats periodically. Its Eqs. (14)–(15) therefore apply $\mathbf K_0^{-1}$ through Fourier blocks. Rigid translations are removed, or handled using a pseudoinverse. Their algorithm uses Newton with preconditioned conjugate gradients for the suitable positive-definite symmetric problem.

**What transfers to E/P is the strategy: approximate the actual Jacobian by a cheap reference, while still solving the actual correction equation.** The FE stiffness itself is not the reduced TFA Jacobian.

## 8. Strategy 2: preconditioned Newton–Krylov on actual E/P

Differentiate ER-4. Local updates depend on their own partition strains, so $\mathbf H_\mu$ has local sensitivity matrices on its diagonal blocks. P supplies the spatial coupling.

**Equation (ER-18) — Actual reduced Jacobian and Newton correction**

$$
\delta\mathbf r
\approx(\mathbf I-\mathbf P\mathbf H_\mu)\delta\boldsymbol\varepsilon,
\qquad
\boxed{
\mathbf J_k\delta\boldsymbol\varepsilon=-\mathbf r_k,
\quad
\mathbf J_k=\mathbf I-\mathbf P\mathbf H_{\mu,k}
}
$$

The reference ER-9 approximates precisely this Jacobian. A material routine supplying $\mathbf C_{\rm alg}^A$ can supply its sensitivity without a separate differentiation of the plastic-strain formula: differentiate ER-3 and solve for $\mathbf H_\mu^A$.

**Equation (ER-19) — Material tangent to eigenstrain sensitivity**

$$
\mathbf C_{\rm alg}^A=\mathbf L^A(\mathbf I_6-\mathbf H_\mu^A),
\qquad
\mathbf H_\mu^A=\mathbf I_6-(\mathbf L^A)^{-1}\mathbf C_{\rm alg}^A
$$

This uses fixed elastic stiffness and the specified local update. At a yield transition, use the appropriate algorithmic derivative; smooth Newton convergence statements need additional care there.

Use the reference inverse to help an iterative linear solver obtain the actual Newton correction.

**Equation (ER-20) — Reference-preconditioned Newton system**

$$
\boxed{
\mathbf M_0^{-1}\mathbf J_k\delta\boldsymbol\varepsilon
=-\mathbf M_0^{-1}\mathbf r_k
}
$$

A Krylov solver builds corrections using repeated matrix-vector products. GMRES, the generalized minimal residual method, accommodates a generally nonsymmetric reduced Jacobian. Do not transfer conjugate gradients or Ladecký's mesh-independent bounds without establishing the required symmetry, positivity, and operator estimates for this reduced problem.

No dense Jacobian is necessary. Its product and the linear residual to monitor are

**Equation (ER-21) — Matrix-free Newton product and linear residual**

$$
\mathbf J_k\mathbf v=\mathbf v-\mathbf P(\mathbf H_{\mu,k}\mathbf v),
\qquad
\boldsymbol\rho_\ell=-\mathbf r_k-\mathbf J_k\delta\boldsymbol\varepsilon_\ell
$$

Here $\ell$ counts linear iterations. During this solve, $\mathbf r_k$, $\mathbf H_{\mu,k}$, and the chosen preconditioner remain fixed. When $\boldsymbol\rho_\ell$ is small, update the strain and recompute the *nonlinear* residual. Solving the linearization does not generally make that new nonlinear residual zero.

**One load step:** evaluate material response and the actual residual; evaluate sensitivities; solve ER-20 using GMRES and ER-15; update strains; repeat Newton until ER-4 converges; then accept histories. Newton may require a line search or a smaller load step. With stored P, there is one nonlinear loop with one iterative linear loop inside it, but no further iterative reference solve when ER-15 applies directly.

## 9. What the two strategies share, and where they differ

Both must evaluate the same residual and return the same converged model response on the same solution branch. Their different corrections are

**Equation (ER-22) — Approximate correction versus Newton correction**

$$
\begin{array}{ll}
\text{Strategy 1:}&\mathbf M_0\delta\boldsymbol\varepsilon=-\mathbf r\\
\text{Strategy 2:}&\mathbf J\delta\boldsymbol\varepsilon=-\mathbf r
\quad\text{using }\mathbf M_0\text{ to accelerate its solution}
\end{array}
$$

For one linear error component, suppose the actual Jacobian is $0.8$ and the reference is $0.1$. Strategy 1 multiplies its error by $1-0.8/0.1=-7$: it diverges despite both being invertible. Newton's scalar correction remains solvable. Fixed-point failure therefore does not prove that the same reference is unusable as a Krylov preconditioner.

To see how reference quality affects convergence, let $\boldsymbol\varepsilon_{\mathrm{sol}}$ be a solution and define $\mathbf e^j=\boldsymbol\varepsilon^j-\boldsymbol\varepsilon_{\mathrm{sol}}$. The label “sol” distinguishes the solution from the chosen anchor $\boldsymbol\varepsilon_*$. At a differentiable solution, the actual Jacobian is $\mathbf J_{\mathrm{sol}}=\mathbf I-\mathbf P\mathbf H_{\mu,\mathrm{sol}}$. Linearize the residual near this solution, then substitute into ER-12 with a fixed, invertible reference.

**Equation (ER-23) — Local error propagation and preconditioner quality**

$$
\begin{aligned}
\mathbf r(\boldsymbol\varepsilon^j)
&\approx\mathbf J_{\mathrm{sol}}\mathbf e^j
\qquad [\mathbf r(\boldsymbol\varepsilon_{\mathrm{sol}})=\mathbf0]\\
\mathbf e^{j+1}
&=\mathbf e^j-\mathbf M_0^{-1}\mathbf r(\boldsymbol\varepsilon^j)
\approx\underbrace{\left[\mathbf I-\mathbf M_0^{-1}\mathbf J_{\mathrm{sol}}\right]}_{\mathbf T}\mathbf e^j\\
\mathbf T
&=\mathbf M_0^{-1}
(\mathbf P\mathbf H_{\mu,\mathrm{sol}}-\mathbf P_0\mathbf H_{\mu,0})
\end{aligned}
$$

The matrix $\mathbf T$ propagates small strain errors from one iteration to the next. If all its eigenvalues have magnitude below one, the fixed-reference iteration converges locally; smaller magnitudes generally give faster eventual decay, though transient growth is possible. This is a near-solution result, not a guarantee from a distant guess or across a yield transition.

ER-23 shows why anchor selection can matter: a sensitivity chosen at $\boldsymbol\varepsilon_*$ may poorly represent the actual sensitivity at $\boldsymbol\varepsilon_{\mathrm{sol}}$. The target is the **coupled response** $\mathbf P\mathbf H_{\mu,\mathrm{sol}}$, so differences between P and P₀ also matter. A nearly singular $\mathbf M_0$ can amplify the mismatch. Refreshing the sensitivity may help, but a changing-reference iteration needs its own convergence assessment.

Newton–GMRES does not require this fixed-point condition. Fast Krylov convergence still depends on more than a nominal condition number, especially for nonsymmetric matrices.

| Consideration | Strategy 1 | Strategy 2 |
|---|---|---|
| Actual material derivative | Not required if a reference is prescribed | Required for the Newton product |
| Actual P products | One per residual evaluation | Residual evaluations and linear iterations |
| Storage beyond the model | A few strain-sized vectors | Additional Krylov vectors |
| Potential advantage | Simple loop and low workspace | More effective corrections for strongly nonlinear coupling |
| Main risk | Slow convergence or divergence | Tangent, Krylov, and preconditioner overhead |

Neither is automatically faster. Neither improves the underlying partition approximation merely by solving it differently.

## 10. The cost that the reference does not remove

With dense actual P, every partition can influence every other partition. Storage and one product cost $O(M^2)$, ignoring fixed six-component factors. FFT application of $\mathbf M_0^{-1}$ costs $O(M\log M)$, but **that does not make the complete iteration $O(M\log M)$**: the actual P product remains.

To understand the alternative, write the definition of P through the fine elastic solve. Let $\mathbf K_{\rm el}$ be the constrained heterogeneous elastic stiffness; $\mathbf F_\mu$ turns partition eigenstrains into equivalent nodal forces; and $\mathbf A_\varepsilon$ turns fine displacement DOFs into partition-average strain fluctuations. At zero imposed mean strain,

**Equation (ER-24) — The elastic inverse already contained in P**

$$
\mathbf K_{\rm el}\mathbf d=\mathbf F_\mu\boldsymbol\mu,
\qquad
\mathbf P\boldsymbol\mu=\mathbf A_\varepsilon\mathbf d,
\qquad
\boxed{\mathbf P=\mathbf A_\varepsilon\mathbf K_{\rm el}^{-1}\mathbf F_\mu}
$$

Thus storing P performs the elastic response work offline. Applying this definition without storing P requires that response online, through a reused factorization or an iterative heterogeneous solve. Compression offers other accuracy–cost tradeoffs. For fixed FE stencil size, Ladecký's actual *stiffness product* uses local operations of linear cost; our condensed P contains an *inverse*, which is a different task.

If that elastic inverse is evaluated iteratively, it introduces another nested elastic solve in either strategy. Its tolerance must be controlled so that P products remain accurate enough for the outer solve. The simpler loop counts above assume stored P; avoiding its storage is not a free implementation change.

This also explains a limitation of the analogy: standard full-field FFT evaluates material contrast locally; the reduced reference difference in ER-10 includes the nonlocal actual P. A homogeneous reference does not make that difference local.

Keeping fine equilibrium unknowns and reducing only material histories is another possible route, but it is not either condensed solver specified here.

## 11. Consequences worth testing before optimizing

The following are deductions from the equations, not claims of research novelty.

**Elastic steps already have an identity reduced Jacobian.** Within an elastic branch, the candidate plastic strain stays at its accepted value, so $\mathbf H_\mu=\mathbf0$ and $\mathbf J=\mathbf I$. Choosing $\mathbf H_{\mu,0}=\mathbf0$ then gives the exact correction in one update, provided the branch remains elastic. E/P has already eliminated the difficult elastic equilibrium problem. Ladecký's motivation of removing fine-mesh stiffness conditioning therefore cannot simply be copied to this reduced Jacobian.

**Scaling the reference stiffness alone cannot tune P₀.** Holding its shape, partitions, discretization, and $\mathbf H_{\mu,0}$ fixed, multiply every reference modulus by $\alpha>0$. The Green response scales inversely, so

**Equation (ER-25) — Cancellation of reference stiffness scale**

$$
\boldsymbol\Gamma^0(\alpha\mathbf C^0)
=\alpha^{-1}\boldsymbol\Gamma^0(\mathbf C^0),
\qquad
\mathbf P_0(\alpha\mathbf C^0)=\mathbf P_0(\mathbf C^0)
$$

Consequently $\mathbf M_0$ does not change. Changing Poisson ratio or anisotropy, or changing the reference sensitivity, can change it. If a sensitivity-selection rule itself depends on stiffness scale, that is a separate effect.

**An invertible reference is a design requirement.** One controlled starting family is $\mathbf H_{\mu,0}=\theta\mathbf I_6$, with $0\le\theta<1$. For a consistent stable Galerkin homogeneous elastic response with matching volume averaging, P₀ has eigenvalues in $[0,1]$. Hence the eigenvalues of $\mathbf M_0$ lie in $[1-\theta,1]$.

The bound follows by averaging the fine homogeneous elastic-energy projection; the partition energy weights are $c^A\mathbf C^0$. It requires consistent projection and averaging, and is not guaranteed for an arbitrary modified Green discretization.

The scalar sensitivity is an algebraic starting choice, not a faithful J2 law. Letting $\theta$ approach one can make the inverse large; at one, a represented compatible mode can make it singular. Invertibility does not guarantee fixed-point convergence. With $\theta=0$, strategy 1 reduces to ordinary E/P fixed-point iteration and no reference FFT is needed—a necessary baseline.

**The reference cannot repair lost physics.** Both strategies should agree with a trusted actual-E/P solve. Replacing actual P by a homogeneous or averaged-polarization interaction changes the approximation, rather than merely accelerating its solution.

## 12. Verification, implementation checks, and next question

A historical [algebraic audit](../numerical_tests/ep_reference_audit/README.md) checks the exact split, derivatives, and both solver structures on a two-segment bar with an explicitly artificial smooth nonlinear eigenstrain law. It also checks the homogeneous 3D reference on a $9^3$ fine grid averaged into $3^3$ partitions. Its scope remains separate from the newer actual-E/P plasticity implementation.

Both strategies reached the same two-segment solution within $10^{-13}$; the reference FFT inverse matched a dense solve within $6\times10^{-16}$ relative error. The split, derivative, shear conversion, and stiffness-scale checks passed. A poor but invertible reference gave local fixed-point amplification greater than one, confirming that invertibility alone is insufficient.

The [3D implementation and results](../numerical_tests/actual_ep_fft_benchmark/RESULTS_3D.md) now compare strategy 1 with dense Newton on six-component J2 loading histories. Actual E/P is built by matrix-free FFT **offline** linear elasticity, then stored. Independent displacement solves, the extruded 2D limit, rotations, actual derivatives and dense reference corrections challenge its correctness. Source, iteration and field-accuracy records are retained. Strategy 2's online Newton–Krylov algorithm is not yet implemented in that code; earlier averaged-D timings do not validate it.

This implementation uses odd periodic Cartesian grids, phase-aligned partitions and a prescribed $\mathbf H_{\mu,0}=0.4\mathbf I_6$. All six local strain components evolve, even when a prescribed mean component is zero. Its fine-grid FFT uses the classical spectral Green operator with a GMRES linear iteration, not the original basic fixed-point iteration. Fine source responses are checked before averaging into E/P. Refining that elastic grid tests the offline construction; it does not by itself enlarge the online partition system or establish nonlinear reduction accuracy.

For a full implementation, check the original nonlinear residual after every proposed update. A useful scale is a fixed positive strain scale or the larger of that scale and the imposed strain norm; do not divide only by an instantaneous loading value that can become zero. For Newton, also check the original linear residual ER-21. Use compatible tensor norms with engineering shear weights and volume weights. A small preconditioned residual alone is insufficient. Retain the previous accepted history if a step fails.

The remaining challenges are choosing a useful sensitivity, paying for actual P products, and demonstrating a benefit against unpreconditioned E/P and direct-solve baselines at identical accuracy.

**Checkpoint:** If the reference matrix changes but actual E/P and the material law stay fixed, which result should remain unchanged after convergence, and which computational quantities may change?

The next experiment should keep actual E/P and the loading history fixed, check solution agreement, and compare P products, material updates, reference setup, total time, and memory.

## Source anchors and attribution

1. **Fish, J.; Cui, J. (2026), “Eigenstate based homogenization,” CMAME 452, 118718.** [Local paper](../references/Fish_Cui_Eigenstate_Based_Homogenization.pdf), Section 2.2, Eqs. (7)–(10), PDF pp. 4–5. These equation passages were visually checked for the E/P starting relation and its actual heterogeneous influences. This lecture does not assess or reproduce the paper's complete EBH/CFP/TFP algorithms. ER-1 translates its tensor-index notation; the specific partition-strain solver construction is developed here.
2. **Moulinec, H.; Suquet, P. (1994), “A fast numerical method for computing the linear and nonlinear mechanical properties of composites,” C. R. Acad. Sci. Paris, Série II, 318, 1417–1423.** [Supplied paper](<C:/Users/camjo/Downloads/Fast_numerical_method_for_computing_the_linear_and.pdf>), read completely, including the French summary and examples. Printed p. 1419, Eqs. (3)–(6); p. 1420, algorithm (8). Supports the full-field reference split, convolution, and material-update iteration. It does not contain ER-9–ER-12 for reduced E/P.
3. **Ladecký, M. et al. (2023), “An optimal preconditioned FFT-accelerated finite element solver for homogenization,” Applied Mathematics and Computation 446, 127835.** [Supplied paper](<C:/Users/camjo/Downloads/An-optimal-preconditioned-FFT-accelerated-finite-element-solver-for-homogenization.pdf>), read completely, including numerical examples and the thermal appendix. Eqs. (9)–(15), Algorithm 1, Sections 4.3 and 5.1 support the FE Newton system, reference inverse, conditional spectral estimates, and local stiffness products. Section 6.1 connects its displacement and strain formulations. Its reduced E/P adaptation and the deductions in Section 11 are developed here, not results claimed by that paper.

The constitutive convention follows the existing [J2 update notes](../verified_notes/07_complete_j2_return_mapping_algorithm.md). The two new solver derivations are algebraically checked candidate methods. No literature-novelty claim is made.
