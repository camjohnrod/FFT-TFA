# Partition-averaged Lippmann-Schwinger homogenization with an online FFT

## Purpose and route

We want a nonlinear homogenization model with **one material state per partition**, interaction cost $O(M\log M)$ per operator application, and no stored $M^2$ interaction matrix. Here $M$ counts partitions; $N$ counts fine finite elements. Fine-grid work may occur offline, but the online solve should use only $M$ states.

The proposal replaces heterogeneous E/P TFA by a **different reduced model**: partitionwise constant polarization interacting through a homogeneous reference medium. Regular, translated partitions make this interaction a convolution, which an FFT applies efficiently. The reference response is derived from a fine FE discretization and averaged over partitions; it does not retain the original heterogeneous E/P operators.

Sections 1-7 state the assumptions, derive the model, explain its approximation, and describe its FFT implementation and nonlinear solve. Section 8 relates it to other frameworks. The document is self-contained; paper references provide background rather than missing derivation steps. Equation labels use `LS-`.

## 1. Assumptions, notation, and what is unknown

### 1.1 Mechanical and constitutive assumptions

- A connected periodic RVE $\Theta$, small strains, perfect bonding, no body forces, and prescribed mean strain $\bar{\boldsymbol\varepsilon}$.
- Periodic displacement fluctuation $\mathbf u$; fix its arbitrary translation by zero mean or by pinning one node. These choices give the same strains.
- Fixed, symmetric positive-definite elastic stiffness $\mathbf L^A$ in each partition $\Omega_A$. A partition may contain both phases; Section 4.4 gives the constitutive construction for it (added 2026-10-26).
- For the constitutive specialization developed here, use isotropic elasticity and associative, rate-independent $J_2$ plasticity with linear isotropic hardening. Require positive shear and bulk moduli, $H^A\geq0$, and initial yield stress $\sigma_{\mathrm y0}^A>0$. One local backward-Euler return map is evaluated at the **partition-average total strain**, with one plastic history per partition. Purely elastic phases are also allowed.
- During load step $n+1$, the committed history $\mathbf z_n^A=(\boldsymbol\mu_n^A,p_n^A)$ stays fixed while nonlinear trial strains change. Candidate histories are committed only after convergence. The scalar $p\geq0$ is accumulated equivalent plastic strain; $\boldsymbol\mu$ is plastic strain.
- Choose a fixed, homogeneous, symmetric positive-definite reference stiffness $\mathbf C_0$. Changing it changes the approximation defined below, not just the solver.

Section 1.4 specifies the local return map. An update at average strain need not equal the average of separate fine-point updates. We adopt the former deliberately.

### 1.2 Spatial discretization assumptions

For a concrete three-dimensional realization, take $N=n_e^3$ regular Hex8 elements with $2\times2\times2$ Gauss integration and $M=n_p^3$ congruent cubic partitions. Require $n_e/n_p$ to be an integer. Material interfaces need not lie on partition boundaries (Section 4.4). The fine reference FE mesh, quadrature, and partition averages must respect translations by one partition.

The partitions cover $\Theta$ without overlapping interiors. Their indicators satisfy $\chi_A(x)=1$ in $\Omega_A$, zero elsewhere, and $\sum_A\chi_A=1$ almost everywhere.

The averaged equations can be defined for arbitrary partitions, but the **single online partition-grid FFT** requires congruent translates on a periodic lattice. It also applies to other regular translated partition shapes; cubes are the example here.

### 1.3 Work-conjugate vectors and dimensions

Use engineering-strain and physical-stress Voigt vectors throughout:

$$
\boldsymbol\varepsilon
=(\varepsilon_{11},\varepsilon_{22},\varepsilon_{33},2\varepsilon_{12},2\varepsilon_{23},2\varepsilon_{13})^{\mathsf T},
\qquad
\boldsymbol\sigma
=(\sigma_{11},\sigma_{22},\sigma_{33},\sigma_{12},\sigma_{23},\sigma_{13})^{\mathsf T}.
$$

Plastic and equivalent eigenstrains use the strain convention. Thus $\boldsymbol\sigma^{\mathsf T}\boldsymbol\varepsilon$ is work density, without an extra shear weight. Differential equations use reconstructed symmetric tensors, indicated by the superscript $\mathrm{ten}$.

| Symbol | Meaning | Size / units |
|---|---|---|
| $A,B$ | Source and receiver partition | Indices |
| $c^A=\lvert\Omega_A\rvert/\lvert\Theta\rvert$, $\chi_A$ | Volume fraction and partition indicator | Dimensionless |
| $\boldsymbol\varepsilon^A=\langle\boldsymbol\varepsilon\rangle_A$ | Partition-average total strain | 6-vector / strain |
| $\boldsymbol\mu^A$, $\boldsymbol\mu^{*,A}$ | Actual plastic strain; equivalent reference eigenstrain | 6-vectors / strain |
| $\boldsymbol\sigma^A$, $\boldsymbol\tau^A$ | Returned partition stress; polarization | 6-vectors / stress |
| $\mathbf L^A$, $\mathbf C_0$, $\mathbf L_t^A$ | Actual elastic stiffness; reference stiffness; algorithmic stress tangent | $6\times6$ / stress |
| $\mathbf E^B$, $\mathbf P^{BA}$, $\mathbf P_0^{BA}$ | Heterogeneous loading/interaction blocks; reference eigenstrain interaction | $6\times6$ / dimensionless |
| $\mathbf D_0^{BA}$ | Reference polarization interaction | $6\times6$ / inverse stress |
| $\mathbf H_\mu^A=\partial\boldsymbol\mu^A/\partial\boldsymbol\varepsilon^A$ | Plastic-strain sensitivity at fixed committed history | $6\times6$ / dimensionless |
| $G^A$, $H^A$, $\sigma_{\mathrm y0}^A$ | Shear modulus, isotropic hardening modulus, and initial yield stress | Stress |

For any field $\mathbf q$, define $\langle\mathbf q\rangle_A=|\Omega_A|^{-1}\int_{\Omega_A}\mathbf q\,dV$. For stacked quantities use

$$
\mathbf e=\operatorname{col}(\boldsymbol\varepsilon^1,\ldots,\boldsymbol\varepsilon^M),
\quad
\mathbf s(\mathbf e)=\operatorname{col}(\boldsymbol\sigma^1,\ldots,\boldsymbol\sigma^M),
\quad
\mathbf b=\mathbf1_M\otimes\bar{\boldsymbol\varepsilon},
\quad
\mathbf C_{0,M}=\mathbf I_M\otimes\mathbf C_0.
$$

Here $\operatorname{col}$ stacks vectors vertically; $\boldsymbol\mu(\mathbf e)$ stacks the partition plastic strains in the same order. The online unknown is $\mathbf e\in\mathbb R^{6M}$. A superscript $(j)$ denotes a nonlinear iterate, $n$ a committed load-step endpoint, and a hat a Fourier transform. Numerical corrections are not physical load increments. Matrix inverses denote solve operators, not instructions to form dense inverses.

### 1.4 Specify the local material update

At a trial partition strain $\boldsymbol\varepsilon^A$, start from the same committed pair $(\boldsymbol\mu_n^A,p_n^A)$. Set $\mathbf v_{\rm vol}=(1,1,1,0,0,0)^{\mathsf T}$ and $\mathbf W=\operatorname{diag}(1,1,1,2,2,2)$. The elastic predictor and yield check are

$$
\begin{aligned}
\boldsymbol\sigma^{\mathrm{tr},A}
&=\mathbf L^A(\boldsymbol\varepsilon^A-\boldsymbol\mu_n^A),\\
\mathbf t^{\mathrm{tr},A}
&=\boldsymbol\sigma^{\mathrm{tr},A}
-\frac{\sigma_{11}^{\mathrm{tr},A}+\sigma_{22}^{\mathrm{tr},A}+\sigma_{33}^{\mathrm{tr},A}}{3}\mathbf v_{\rm vol},\\
q^{\mathrm{tr},A}
&=\sqrt{\frac32(\mathbf t^{\mathrm{tr},A})^{\mathsf T}\mathbf W\mathbf t^{\mathrm{tr},A}},\\
f^{\mathrm{tr},A}
&=q^{\mathrm{tr},A}-(\sigma_{\mathrm y0}^A+H^A p_n^A).
\end{aligned}
$$

Here $\mathbf t^{\mathrm{tr},A}$ is the deviatoric trial stress and $q^{\mathrm{tr},A}$ its von Mises equivalent stress. The matrix $\mathbf W$ accounts for the two tensor shear entries in stress contractions and converts the flow direction to engineering strain; it is not an extra factor in the work product $\boldsymbol\sigma^{\mathsf T}\boldsymbol\varepsilon$.

If $f^{\mathrm{tr},A}\leq0$, return $\boldsymbol\mu^A=\boldsymbol\mu_n^A$, $p^A=p_n^A$, and $\boldsymbol\sigma^A=\boldsymbol\sigma^{\mathrm{tr},A}$. Otherwise radial return gives

$$
\Delta\lambda^A=\frac{f^{\mathrm{tr},A}}{3G^A+H^A},
\qquad
\boldsymbol\mu^A
=\boldsymbol\mu_n^A+\frac{3\Delta\lambda^A}{2q^{\mathrm{tr},A}}\mathbf W\mathbf t^{\mathrm{tr},A},
\qquad
p^A=p_n^A+\Delta\lambda^A,
\qquad
\boldsymbol\sigma^A=\mathbf L^A(\boldsymbol\varepsilon^A-\boldsymbol\mu^A).
$$

The active branch has $q^{\mathrm{tr},A}>0$, so the normalization is defined. Do not evaluate it in an elastic hydrostatic state. In this Voigt ordering $G^A=L^A_{44}$. For a purely elastic phase with no initial eigenstrain, set $\boldsymbol\mu^A=\mathbf0$ and use $\boldsymbol\sigma^A=\mathbf L^A\boldsymbol\varepsilon^A$. All returned histories are candidates until the outer solve converges.

These equations define the local maps $\boldsymbol\mu^A(\boldsymbol\varepsilon^A;\mathbf z_n^A)$ and $\boldsymbol\sigma^A(\boldsymbol\varepsilon^A;\mathbf z_n^A)$ used below; they do not enforce interactions between partitions.

## 2. Why change the original E/P TFA model?

A partition-based E/P TFA representation [FC] retains the heterogeneous elastic response to partitionwise constant plastic strain:

**(LS-1) Heterogeneous E/P relation and local closure**

$$
\boldsymbol\varepsilon^B
=\mathbf E^B\bar{\boldsymbol\varepsilon}
+\sum_A\mathbf P^{BA}\boldsymbol\mu^A,
\qquad
\boldsymbol\mu^A=\boldsymbol\mu^A(\boldsymbol\varepsilon^A;\mathbf z_n^A),
\qquad
\boldsymbol\sigma^A=\mathbf L^A(\boldsymbol\varepsilon^A-\boldsymbol\mu^A).
$$

$\mathbf E$ and $\mathbf P$ are exact for the chosen heterogeneous **elastic FE problem and eigenstrain representation**, up to solve tolerance. This does not make the nonlinear model an exact continuum solution: partitionwise plastic strain and its average-strain update are reduced modeling choices.

A direct FE construction uses six macroscopic-strain and $6M$ partition-eigenstrain right-hand sides, with a reusable heterogeneous stiffness factorization. These are load cases, not $6+6M$ separate factorizations.

The residual is

**(LS-2) Original TFA residual**

$$
\mathbf r_{\rm TFA}(\mathbf e)
=\mathbf e-\mathbf E\bar{\boldsymbol\varepsilon}
-\mathbf P\boldsymbol\mu(\mathbf e)=\mathbf0.
$$

With a stored dense operator, each $\mathbf P\boldsymbol\mu$ costs $O(M^2)$ and storage is $O(M^2)$. Shifting source and receiver changes their material surroundings, so $\mathbf P^{BA}$ generally cannot be written as a function of $B-A$ alone. An FFT preconditioner can reduce iteration count but does not remove this dense residual product. Reusing unchanged induced-strain states can skip some products; it does not change the worst-case dense cost.

This is a limitation of direct FFT diagonalization of this operator, not a proof that all compression or matrix-free alternatives are impossible. The proposal below instead changes the reduced model so that its entire nonlocal operator has convolution structure.

## 3. Derive the exact reference-medium equation first

### 3.1 Equilibrium and compatible strain

Before making a partition approximation, start from the actual fields:

**(LS-3) Physical problem**

$$
\nabla\cdot\boldsymbol\sigma^{\mathrm{ten}}=\mathbf0,
\qquad
\boldsymbol\varepsilon^{\mathrm{ten}}
=\bar{\boldsymbol\varepsilon}^{\mathrm{ten}}+\nabla^s\mathbf u,
\qquad
\boldsymbol\sigma=\mathbf L(x)(\boldsymbol\varepsilon-\boldsymbol\mu).
$$

Define actual-minus-reference stress, called **polarization**:

**(LS-4) Exact stress split**

$$
\begin{aligned}
\boldsymbol\sigma&=\mathbf C_0\boldsymbol\varepsilon+\boldsymbol\tau,\\
\boldsymbol\tau
&=\boldsymbol\sigma-\mathbf C_0\boldsymbol\varepsilon
=(\mathbf L-\mathbf C_0)\boldsymbol\varepsilon-\mathbf L\boldsymbol\mu.
\end{aligned}
$$

This is an identity. Elastic contrast is in $(\mathbf L-\mathbf C_0)\boldsymbol\varepsilon$ and plasticity is in $-\mathbf L\boldsymbol\mu$; a homogeneous reference does not make the actual composite homogeneous.

Substitute the split into equilibrium. The constant reference stress associated with $\bar{\boldsymbol\varepsilon}$ has zero divergence, giving the homogeneous reference problem

$$
\nabla\cdot[\mathbf C_0(\nabla^s\mathbf u)^{\mathrm V}]^{\mathrm{ten}}
=-\nabla\cdot\boldsymbol\tau^{\mathrm{ten}},
$$

where $\mathrm V$ means engineering-Voigt conversion. If $\boldsymbol\tau$ were known, this would be a linear reference solve. Its displacement-to-strain response defines a strain Green operator:

**(LS-5) Exact Lippmann-Schwinger equation**

$$
\boxed{
\boldsymbol\varepsilon
=\bar{\boldsymbol\varepsilon}-\boldsymbol\Gamma_0*\boldsymbol\tau,
\qquad
\widehat{\boldsymbol\Gamma}_0(\mathbf0)=\mathbf0.
}
$$

We use the conventional Moulinec-Suquet minus sign [MS]. This fixes the sign of $\boldsymbol\Gamma_0$ throughout. Continuum kernels are understood weakly; the FE realization in Section 6 uses a specified discrete operator.

Homogeneous coefficients and periodic boundaries make the kernel depend only on $x-y$. That is why it is a convolution. The zero-frequency convention enforces $\langle\boldsymbol\varepsilon\rangle_\Theta=\bar{\boldsymbol\varepsilon}$. No reduced approximation has entered yet, and the exact physical solution is independent of the chosen $\mathbf C_0$.

### 3.2 Equivalent reference eigenstrain is not plastic strain

To express the same reference response using eigenstrain influence functions, write $\boldsymbol\tau=-\mathbf C_0\boldsymbol\mu^*$:

**(LS-6) Equivalent eigenstrain**

$$
\begin{aligned}
\boldsymbol\mu^*
&=\boldsymbol\varepsilon-\mathbf C_0^{-1}\boldsymbol\sigma\\
&=(\mathbf I_6-\mathbf C_0^{-1}\mathbf L)\boldsymbol\varepsilon
+\mathbf C_0^{-1}\mathbf L\boldsymbol\mu,\\
\boldsymbol\sigma&=\mathbf C_0(\boldsymbol\varepsilon-\boldsymbol\mu^*).
\end{aligned}
$$

$\boldsymbol\mu^*$ contains both elastic contrast and actual plastic strain. It is not a new plastic history variable or a material evolution law. With its exact spatial value, the reference body reproduces the actual fields; approximating it spatially changes that equivalence.

## 4. Reduce the equation to partition averages

### 4.1 State the two modeling choices

First, one material call per partition gives

$$
\boldsymbol\mu^A=\boldsymbol\mu^A(\boldsymbol\varepsilon^A;\mathbf z_n^A),
\qquad
\boldsymbol\sigma^A=\mathbf L^A(\boldsymbol\varepsilon^A-\boldsymbol\mu^A).
$$

Second, represent polarization by a constant on each partition:

$$
\boldsymbol\tau_M(x)=\sum_A\chi_A(x)\boldsymbol\tau^A,
\qquad
\boldsymbol\tau^A=\boldsymbol\sigma^A-\mathbf C_0\boldsymbol\varepsilon^A.
$$

This second choice is additional to original E/P TFA. It does **not** require the compatible strain reconstructed from the reference problem to be constant inside a partition.

### 4.2 Average the reference response over source and receiver

Insert $\boldsymbol\tau_M$ into LS-5 and average over $\Omega_B$. Linearity permits each constant $\boldsymbol\tau^A$ to be taken outside the integrals:

**(LS-7) Partition interaction and nonlinear equation**

$$
\begin{aligned}
\mathbf D_0^{BA}
&:=\frac{1}{|\Omega_B|}
\int_{\Omega_B}[\boldsymbol\Gamma_0*\chi_A](x)\,dV_x\\
&=\frac{1}{|\Omega_B|}
\int_{\Omega_B}\int_{\Omega_A}\boldsymbol\Gamma_0(x-y)\,dV_y\,dV_x.
\end{aligned}
$$

$$
\boxed{
\boldsymbol\varepsilon^B=\bar{\boldsymbol\varepsilon}
-\sum_A\mathbf D_0^{BA}
[\boldsymbol\sigma^A(\boldsymbol\varepsilon^A;\mathbf z_n^A)
-\mathbf C_0\boldsymbol\varepsilon^A].
}
$$

The double average accounts for finite source and receiver regions, not merely their centers. The continuum definition requires no fine mesh. Section 6 derives its FE-based discrete counterpart.

Define the eigenstrain interaction to obtain an equivalent form of the same equation:

**(LS-8) Equivalent $\mathbf P_0$ equation**

$$
\mathbf P_0^{BA}=\mathbf D_0^{BA}\mathbf C_0,
\qquad
\boxed{
\boldsymbol\varepsilon^B
=\bar{\boldsymbol\varepsilon}
+\sum_A\mathbf P_0^{BA}
[\boldsymbol\varepsilon^A-\mathbf C_0^{-1}\boldsymbol\sigma^A].
}
$$

Order matters: stiffness multiplies on the **source side**, so $\mathbf D_0^{BA}=\mathbf P_0^{BA}\mathbf C_0^{-1}$. $\mathbf P_0$ is dimensionless; $\mathbf D_0$ has compliance units.

### 4.3 Mean strain and macroscopic stress

Periodicity makes every reference fluctuation response have zero RVE mean:

**(LS-9) Mean consistency and output**

$$
\sum_B c^B\mathbf D_0^{BA}=\mathbf0,
\qquad
\sum_B c^B\boldsymbol\varepsilon^B=\bar{\boldsymbol\varepsilon},
\qquad
\bar{\boldsymbol\sigma}=\sum_A c^A\boldsymbol\sigma^A.
$$

For congruent partitions $c^A=1/M$. The computed mean is automatic at a converged root if the operator satisfies this identity. It does not imply that every partition strain equals the prescribed mean.

### 4.4 Partitions holding both phases (added 2026-10-26, revised 2026-10-27)

The equation above uses each partition's polarization only through its stress $\boldsymbol\sigma^A$ at the partition strain $\boldsymbol\varepsilon^A$. A partition that holds both phases needs a constitutive law for that pair. Call its part in each phase a **cluster**, with volume fractions $c_m^A+c_f^A=1$, one material state (strain, plastic strain, history) each. The law is the one the constant-polarization assumption itself implies (slides 11 and 14 of `references/10.20.2026 Progress.pptx`): the polarization is the same at every point of the partition, so both clusters carry it, against the same $\mathbf C_0$ as LS-7:

**(LS-22) Composite partition law**

$$
c_m^A\boldsymbol\varepsilon_m^A+c_f^A\boldsymbol\varepsilon_f^A=\boldsymbol\varepsilon^A,
\qquad
\boldsymbol\sigma_m(\boldsymbol\varepsilon_m^A;\mathbf z_{m,n}^A)-\mathbf C_0\boldsymbol\varepsilon_m^A=\boldsymbol\sigma_f(\boldsymbol\varepsilon_f^A;\mathbf z_{f,n}^A)-\mathbf C_0\boldsymbol\varepsilon_f^A,
\qquad
\boldsymbol\sigma^A=c_m^A\boldsymbol\sigma_m^A+c_f^A\boldsymbol\sigma_f^A .
$$

One $\mathbf C_0$ serves both the interaction and this law, so the polarization LS-7 receives, $\boldsymbol\tau^A=\boldsymbol\sigma^A-\mathbf C_0\boldsymbol\varepsilon^A$, is also the polarization of each cluster. In elasticity LS-22 has the closed form of slide 15, $\boldsymbol\sigma^A=\mathbf C_*^A\boldsymbol\varepsilon^A$ with $\mathbf C_*^A=\mathbf C_0+\big[c_m(\mathbf L_m-\mathbf C_0)^{-1}+c_f(\mathbf L_f-\mathbf C_0)^{-1}\big]^{-1}$, Brisard–Dormieux's (2012) consistent equivalent stiffness (their rule 23), which LS-25 gives with one piece per cluster. LS-22 uses no interface direction, so it holds for any number of phases or interfaces in a partition.

**Choosing $\mathbf C_0$.** As $\mathbf C_0\to\mathbf0$ LS-22 tends to equal stress (the Reuss average); with $\mathbf C_0$ equal to the matrix stiffness the fiber cluster takes no strain; with $\mathbf C_0$ between the phases the clusters strain in opposite directions and fiber-rich partitions lose stiffness; above every phase it tends to equal strain. The law has a solution only while $\mathbf C_0$ stays below every cluster's tangent (Brisard–Dormieux's Assumption 1 on the soft side): a yielding phase's shear tangent drops to about $GH/(3G+H)$, and to zero without hardening, so a fixed $\mathbf C_0>\mathbf 0$ eventually fails for a perfectly plastic or softening phase. The code sets $\mathbf C_0$ as a fraction of a named stiffness (`cp_reference_stiffness_scale`, default $0.05\times$ matrix), and section 7.4's preconditioner removes the cost of a small $\mathbf C_0$. Measured on `check.py`'s fiber (27² elements, 9² partitions, contrast 100, $H=10$ MPa), CP's largest macroscopic error against element-level TFA: 4.3 % at $0.01$, 4.5 % at $0.05$, 4.8 % at $0.07$ (just below the yielding matrix's shear tangent, $0.074\,G$), no solution at $0.1$; 64–66 % at $2$ and $10\times$ the fiber stiffness; 16 % with the matrix's bulk modulus and a shear modulus below the plastic tangent. The interaction alone prefers $\mathbf C_0$ near the matrix, where the polarization of matrix partitions vanishes: with LS-22 replaced by equal stress, the same scan gave 2.4 % at $1$, 3.4 % at $0.1$, 4.1 % at $0.01$.

**History (2026-10-26).** The law was first equal stress with the matrix stiffness in LS-7, i.e. two reference media ($\mathbf C_0=\mathbf0$ inside the partition only). Its cluster polarizations against the interaction's $\mathbf C_0$ differ, $\boldsymbol\tau_m-\boldsymbol\tau_f=\mathbf C_0(\boldsymbol\varepsilon_f-\boldsymbol\varepsilon_m)\neq\mathbf 0$ (in elasticity with $\mathbf C_0$ the matrix: $\boldsymbol\tau_m=\mathbf0$ and $\boldsymbol\tau_f\approx0.99\,\boldsymbol\sigma^A$), and LS-7 received only their average. It was more accurate (2.4 % above; the cut laminate exact) and as fast, but not constant polarization; the user chose the slides' single $\mathbf C_0$. Either way the online cost is $O(M\log M)$: LS-7 receives one polarization per partition. A Hashin–Shtrikman form of the split, $\boldsymbol\varepsilon+\mathbf P(\mathbf C_0)\boldsymbol\tau$ equal in both clusters with $\mathbf P$ the direction average of $\widehat{\boldsymbol\Gamma}_0(\mathbf n;\mathbf C_0)$, keeps one $\mathbf C_0$ but not constant polarization (LS-22 is $\mathbf P\to\infty$, equal stress $\mathbf P=\mathbf C_0^{-1}$); at $\mathbf C_0$ the matrix it gave 3.9 % and halved the worst cluster error. Not adopted.

LS-22 is the case of section 4.5's LS-25 and LS-26 in which every cluster is one partition's part in one phase; the code solves it in that general form (`ls_model.CPSolver`).

LS-22 is not exact for a laminate whose interfaces cut partitions: in transverse shear the exact state has equal stress, which LS-22 reaches only as $\mathbf C_0\to\mathbf0$ (at $0.01\times$ matrix the cut laminate of `check.py` is off by 1.4 %). Measured with the earlier equal-stress law on circular fibers against element-level TFA (one region per element): the macroscopic stress stayed within 1–5 %, about as close as one partition per element; stresses and plastic strains next to the fiber were as close with $2\times2$-element partitions and 2–5 times further off with $4\times4$. By default TFA uses the same clusters as its regions, so the two models share their material states.

### 4.5 Clusters spanning partitions (added 2026-10-27)

The material states need not follow the partitions. Let a **cluster** $g$ be any set of elements of one phase, with one material state (one uniform plastic eigenstrain $\boldsymbol\mu_g$, one history $\mathbf z_{n,g}$), and a **piece** $p=(A,g)$ the part of cluster $g$ inside partition $A$, with share $c_p$ of the partition's volume ($\sum_{p\in A}c_p=1$) and weight $w_p=c_p/v_g$ in its cluster's average ($v_g$ the cluster's volume in partition volumes, $\sum_{p\in g}w_p=1$). A partition still carries one polarization $\boldsymbol\tau^A$ (LS-7). Its "partition phases" clusters are LS-22's; `config.cp_clusters = "k-means"` takes k-means clusters of the elements' elastic strain concentration instead, found in each phase apart (Liu–Bessa–Liu 2016, section 2.2.1), which span many partitions.

Every piece carries its partition's polarization and its cluster's eigenstrain, with the cluster's elastic stiffness: $\boldsymbol\sigma_p=\mathbf L_g(\boldsymbol\varepsilon_p-\boldsymbol\mu_g)$ and $\boldsymbol\sigma_p-\mathbf C_0\boldsymbol\varepsilon_p=\boldsymbol\tau^A$. With the eigenstrains given, this is linear (slides 14 and 15 of `references/10.20.2026 Progress.pptx`, with $\boldsymbol\mu(x)=\boldsymbol\mu_g$ on each piece):

**(LS-25) Piece law**

$$
\boldsymbol\varepsilon_p=\mathbf K_g\boldsymbol\tau^A+\mathbf N_g\boldsymbol\mu_g,
\qquad
\boldsymbol\tau^A=\mathbf T_A\big(\boldsymbol\varepsilon^A-\boldsymbol\lambda_*^A\big),
\qquad
\boldsymbol\lambda_*^A=\sum_{p\in A}c_p\mathbf N_g\boldsymbol\mu_g,
$$

with $\mathbf K_g=(\mathbf L_g-\mathbf C_0)^{-1}$, $\mathbf N_g=\mathbf K_g\mathbf L_g$ and $\mathbf T_A=\big(\sum_{p\in A}c_p\mathbf K_g\big)^{-1}=\mathbf C_*^A-\mathbf C_0$; then $\boldsymbol\sigma^A=\mathbf C_0\boldsymbol\varepsilon^A+\boldsymbol\tau^A$ and the equivalent eigenstrain LS-18 receives is $\boldsymbol\mu_*^A=-\mathbf C_0^{-1}\boldsymbol\tau^A$. The pieces average to the partition strain by construction. A cluster's material update takes its **average** strain, $\boldsymbol\varepsilon_g=\sum_{p\in g}w_p\boldsymbol\varepsilon_p$, as TFA's regions do (`notes/3_tfa_baseline.md`, section 1), which closes the system:

**(LS-26) Cluster eigenstrains**

$$
\mathbf r_g=\boldsymbol\mu_g-\boldsymbol\mu\big(\boldsymbol\varepsilon_g(\boldsymbol\varepsilon,\boldsymbol\mu);\mathbf z_{n,g}\big)=\mathbf 0 .
$$

Since $\mathbf L_g$ is uniform over a single-phase cluster, $\mathbf L_g(\boldsymbol\varepsilon_g-\boldsymbol\mu_g)$ is the cluster's average stress, so that average lies on the yield surface; a single piece's stress need not. With one piece per cluster, $\boldsymbol\varepsilon_g=\boldsymbol\varepsilon_p$ and LS-25 with LS-26 is LS-22.

LS-25 is linear in the partition strains $\boldsymbol\varepsilon$ and eigenstrains $\boldsymbol\mu$, so the code keeps it as sparse block matrices, built once per lattice (`ls_model.PieceLaw`): $\boldsymbol\varepsilon_g=\mathbf E\boldsymbol\varepsilon+\mathbf F\boldsymbol\mu$ and $\boldsymbol\mu_*=\mathbf D_e(\boldsymbol\varepsilon-\boldsymbol\Lambda\boldsymbol\mu)$, with blocks $w_p\mathbf K_g\mathbf T_A$ in $\mathbf E$, $c_p\mathbf N_g$ in $\boldsymbol\Lambda$, $-\mathbf C_0^{-1}\mathbf T_A$ in $\mathbf D_e$, and

$$
\frac{\partial\boldsymbol\varepsilon_g}{\partial\boldsymbol\mu_h}=\delta_{gh}\mathbf N_g-\sum_{(p,q)}w_p\,c_q\,\mathbf K_g\mathbf T_A\mathbf N_h
$$

over the pairs of pieces $(p,q)$ of one partition $A$, $p$ of $g$ and $q$ of $h$ ($q=p$ included). A cluster all of whose pieces fill their partitions has $\boldsymbol\varepsilon_g=\langle\boldsymbol\varepsilon^A\rangle_g$ whatever $\boldsymbol\mu$, so its row of $\mathbf F$ vanishes; only the **coupled** clusters, those with a piece in a partition holding several, need $\mathbf F$. `check.py` computes LS-25 piece by piece, with no solver code, and checks these matrices against it. Like LS-22, LS-26 has a solution only while $\mathbf C_0$ stays below the tangent of every coupled cluster.

TFA can take k-means clusters too (`config.tfa_clusters`), independently of CP; both on k-means share one clustering.

## 5. Understand exactly what is approximated

### 5.1 The discarded polarization fluctuation

For any reconstructed field, write in partition $A$

$$
\boldsymbol\varepsilon(x)=\boldsymbol\varepsilon^A+\delta\boldsymbol\varepsilon^A(x),
\qquad
\langle\delta\boldsymbol\varepsilon^A\rangle_A=\mathbf0.
$$

This is an average-plus-fluctuation identity, not an extra physical assumption. With uniform $\mathbf L^A$ and plastic strain $\boldsymbol\mu^A$, actual-material stress and polarization are

**(LS-10) What constant polarization omits**

$$
\begin{aligned}
\boldsymbol\sigma_{\rm actual}(x)
&=\boldsymbol\sigma^A+\mathbf L^A\delta\boldsymbol\varepsilon^A(x),\\
\boldsymbol\tau_{\rm actual}(x)
&=\boldsymbol\tau^A+(\mathbf L^A-\mathbf C_0)\delta\boldsymbol\varepsilon^A(x).
\end{aligned}
$$

The new model omits the second term of the polarization. Original E/P TFA retains this elastic fluctuation through its heterogeneous solve. Thus uniform plastic strain is not equivalent to uniform polarization.

### 5.2 Which stress is equilibrated?

The reconstructed reference field has stress

**(LS-11) Model stress versus actual-material postprocessing**

$$
\begin{aligned}
\boldsymbol\sigma_{\rm model}(x)
&=\mathbf C_0\boldsymbol\varepsilon(x)+\boldsymbol\tau^A
=\boldsymbol\sigma^A+\mathbf C_0\delta\boldsymbol\varepsilon^A(x),\\
\boldsymbol\sigma_{\rm actual}(x)-\boldsymbol\sigma_{\rm model}(x)
&=(\mathbf L^A-\mathbf C_0)\delta\boldsymbol\varepsilon^A(x).
\end{aligned}
$$

The reference solve equilibrates $\boldsymbol\sigma_{\rm model}$ (in the FE weak sense for the discrete operator). Applying the actual stiffness afterward does not generally give an equilibrated fine stress. Both stresses have partition mean $\boldsymbol\sigma^A$, but their fluctuations differ.

Both methods can reconstruct nonuniform strain: TFA through the heterogeneous elastic solve, this model through the homogeneous reference solve. Fine-field reconstruction is optional **postprocessing with $N$-scale work**, not part of the proposed $M$-state online solve. The stored average kernel alone does not determine every fine strain value.

### 5.3 An energy interpretation for the stated elastic-plastic law

At fixed, uniform plastic strain in a phase-pure partition, expand the elastic energy. The cross term vanishes because the fluctuation has zero average:

**(LS-12) Actual versus reference fluctuation energy**

$$
\begin{aligned}
U_A^{\rm actual}
&=\frac{|\Omega_A|}{2}
(\boldsymbol\varepsilon^A-\boldsymbol\mu^A)^{\mathsf T}
\mathbf L^A(\boldsymbol\varepsilon^A-\boldsymbol\mu^A)
+\frac12\int_{\Omega_A}(\delta\boldsymbol\varepsilon^A)^{\mathsf T}
\mathbf L^A\delta\boldsymbol\varepsilon^A\,dV,\\
U_A^{\rm model}
&=\frac{|\Omega_A|}{2}
(\boldsymbol\varepsilon^A-\boldsymbol\mu^A)^{\mathsf T}
\mathbf L^A(\boldsymbol\varepsilon^A-\boldsymbol\mu^A)
+\frac12\int_{\Omega_A}(\delta\boldsymbol\varepsilon^A)^{\mathsf T}
\mathbf C_0\delta\boldsymbol\varepsilon^A\,dV.
\end{aligned}
$$

Varying $\sum_A U_A^{\rm model}$ over compatible displacement fields gives the stress in LS-11: variations of partition averages give $\boldsymbol\sigma^A$, and variations of within-partition fluctuations give $\mathbf C_0\delta\boldsymbol\varepsilon^A$. For the stated associative J2 model, retain the partition hardening energy and incremental plastic dissipation; since the fluctuation term is independent of plastic strain, its local update remains the return map at average strain. This interpretation is specific to the stated constitutive structure, not a claim for every nonlinear material law.

**Interpretation:** actual material stiffness governs average deformation; the chosen reference stiffness governs unresolved elastic fluctuations. This is a consistent modified model, not exact heterogeneous equilibrium. It explains reference dependence. It does not establish accuracy or guarantee convergence of a particular iterative solver.

### 5.4 Useful limiting cases

- If $\mathbf L^A=\mathbf C_0$ everywhere, the discarded term vanishes. With matched FE operators and the same partition history update, the model reduces to the same E/P TFA equation ($\mathbf E^B=\mathbf I_6$, $\mathbf P=\mathbf P_0$), even if yield or hardening properties vary.
- If within-partition strain fluctuations vanish, the elastic fluctuation error also vanishes. This is a special field condition, not something ensured by uniform eigenstrain.
- One partition per fully integrated Hex8 element does **not** imply zero fluctuation: strain varies between its Gauss points.
- Refining $N$ at fixed $M$ improves the reference FE approximation but does not remove the constant-polarization or single-history model errors. Refining $M$ reduces the spatial restriction; convergence and error rates require a separate approximation analysis.

## 6. Construct the reference kernel offline, apply it on $M$ states online

The preceding double integral defines a continuum operator. Sections 6.1–6.4 build its **FE-derived discrete counterpart**, using the specified mesh and quadrature, and section 6.5 the Moulinec–Suquet alternative; neither is asserted to equal the continuum operator at finite $N$.

### 6.1 FE derivation of $\mathbf P_0$

Let $\mathbf d$ be constrained periodic fluctuation DOFs and $\mathbf B(x)$ the engineering-strain/displacement matrix. Define

**(LS-13) Reference stiffness, source load, and strain averaging**

$$
\begin{aligned}
\mathbf K_0&=\int_\Theta\mathbf B^{\mathsf T}\mathbf C_0\mathbf B\,dV,\\
\mathbf F_{\mu,0}^A&=\int_{\Omega_A}\mathbf B^{\mathsf T}\mathbf C_0\,dV,\\
\mathbf A_\varepsilon^B&=\frac{1}{|\Omega_B|}\int_{\Omega_B}\mathbf B\,dV,\\
\mathbf K_0\mathbf d&=\sum_A\mathbf F_{\mu,0}^A\boldsymbol\mu^{*,A},
\qquad
\boldsymbol\varepsilon^B=\bar{\boldsymbol\varepsilon}+\mathbf A_\varepsilon^B\mathbf d.
\end{aligned}
$$

The homogeneous reference's macroscopic-strain load is zero by periodicity. With rigid translation removed, solve and substitute:

**(LS-14) FE-derived, partition-averaged reference interaction**

$$
\boxed{
\mathbf P_0^{BA}
=\mathbf A_\varepsilon^B\mathbf K_0^{-1}\mathbf F_{\mu,0}^A,
\qquad
\mathbf P_0=\mathbf A_\varepsilon\mathbf K_0^{-1}\mathbf F_{\mu,0},
\qquad
\mathbf D_0=\mathbf P_0\mathbf C_{0,M}^{-1}.
}
$$

All integrals use the chosen FE quadrature. If $N_d$ counts independent constrained displacement DOFs, $\mathbf K_0\in\mathbb R^{N_d\times N_d}$, $\mathbf F_{\mu,0}\in\mathbb R^{N_d\times6M}$, and $\mathbf A_\varepsilon\in\mathbb R^{6M\times N_d}$. On constrained DOFs, $\mathbf K_0^{-1}$ is an ordinary solve; on the full periodic nodal grid it means the inverse on the nonzero modes, with translation removed.

"FE-derived" means the operator reproduces **average responses of this reference FE discretization to partitionwise eigenstrains**. It does not mean equivalence to the actual nonlinear heterogeneous FE solution. It retains reference kinematics and averaging geometry, not unresolved phase contrasts inside a partition.

### 6.2 Why six source solves suffice

Let $\mathbf a_A$ be the partition-lattice position. Translation invariance of the reference mesh, material, quadrature, and averaging gives

**(LS-15) Coarse convolution kernel**

$$
\mathbf P_0^{BA}=\mathbf p_0[\mathbf a_B-\mathbf a_A]
\quad\text{(periodic offsets)}.
$$

Apply six unit engineering-eigenstrain components in one source partition, at zero mean strain. The responses in all receiver partitions give its $M$ blocks. Every other block column follows by translation. Store this offset kernel, not the dense matrix. This construction requires six load cases, possibly solved together as six right-hand sides.

### 6.3 Optional FFT construction on the fine FE mesh

For FFT construction, use the full periodic nodal stiffness before pinning; remove translation through its zero Fourier mode. This homogeneous regular FE stiffness is a nodal convolution, with $3\times3$ blocks $\mathbf k_0[\mathbf o]$ obtained by periodic assembly of the element stiffness. Therefore

**(LS-16) Fine-grid reference solve, offline only**

$$
(\mathbf K_0\mathbf d)_I=\sum_J\mathbf k_0[I-J]\mathbf d_J,
\qquad
\widehat{\mathbf d}(\boldsymbol\eta)
=\widehat{\mathbf k}_0(\boldsymbol\eta)^{-1}\widehat{\mathbf f}(\boldsymbol\eta)
\ (\boldsymbol\eta\ne\mathbf0),
\qquad
\widehat{\mathbf d}(\mathbf0)=\mathbf0.
$$

$\boldsymbol\eta$ labels fine nodal frequencies, not coarse partition frequencies. Fully integrated, stable Hex8 reference elasticity has only rigid translations as zero-energy periodic modes. Under these assumptions, six reference solves can be performed with $N$-scale FFT work and then averaged. Sparse-direct reference solves are an alternative. Neither is required online.

### 6.4 Online coarse-grid convolution

For a stacked partition vector $\mathbf v$, use the offset ordering in LS-15:

**(LS-17) Online $M$-grid application**

$$
\mathbf P_0\mathbf v
=\operatorname{IFFT}_M
\left[\widehat{\mathbf p}_0(\boldsymbol\xi)
\operatorname{FFT}_M(\mathbf v)(\boldsymbol\xi)\right].
$$

$\boldsymbol\xi$ labels coarse frequencies. Transform the six strain components over spatial lattice axes; multiply one $6\times6$ block per frequency. Use matched source/receiver ordering, FFT normalization, and engineering-shear conventions. Set/check $\widehat{\mathbf p}_0(\mathbf0)=\mathbf0$ to enforce LS-9. One **convolution application** includes forward and inverse transforms; it is not one individual FFT call.

Storage of the reference kernel is $O(M)$, and application costs $O(M\log M)$. These statements assume a fixed number of tensor components. Online constitutive work is $O(M)$; no fine-grid state is advanced.

### 6.5 Moulinec–Suquet kernel

The alternative to LS-14 (the only kernel in the code since 2026-10-26) evaluates the continuum Green operator at the frequencies $\boldsymbol\xi$ of the $N_3\times N\times N$ lattice of elements of the cubic cell ($N_3=N$ for a geometry that varies along $x_3$; any $N_3$, often 1, for one that does not):

**(LS-21) Moulinec–Suquet kernel**

$$
\widehat{\mathbf P}_0(\boldsymbol\xi)=\widehat{\boldsymbol\Gamma}_0(\mathbf n)\,\mathbf C_0,
\qquad
\widehat{\boldsymbol\Gamma}_0(\mathbf n)=\mathbf B(\mathbf n)\big[\mathbf B(\mathbf n)^{\mathsf T}\mathbf C_0\mathbf B(\mathbf n)\big]^{-1}\mathbf B(\mathbf n)^{\mathsf T},
\qquad
\mathbf n=\boldsymbol\xi/|\boldsymbol\xi|,
$$

where $\mathbf B(\mathbf n)$ ($6\times3$) is the engineering strain of a displacement varying along $\mathbf n$, per unit gradient. The cell is a cube, so frequencies in units of $2\pi/L$ compare across axes whatever each axis's element count. $\widehat{\mathbf P}_0(\mathbf 0)=\mathbf 0$ (LS-9). At every other frequency $\widehat{\mathbf P}_0$ is a projection, with eigenvalues 0 and 1, for any $\mathbf C_0$. Along an axis with an even count $N_i$, $\widehat{\boldsymbol\Gamma}_0$ is not even in a component $\xi_i=N_i/2$, so it has no real-valued meaning there; $\widehat{\mathbf P}_0=\mathbf I$ at those frequencies, Moulinec–Suquet's $\widehat{\boldsymbol\Gamma}_0=\mathbf C_0^{-1}$ (1998, §2.4.2), which makes the stress there zero.

The kernel of partitions of $k\times k\times k_3$ elements follows by linearity, exactly as the FE kernel of partitions follows from the FE kernel of elements: a unit eigenstrain in a partition is one in each of its elements, and a partition's strain is the mean over its elements, so, with $|d_1|,|d_2|<k$, $|d_3|<k_3$ and $\mathbf K=(k,k,k_3)$ acting componentwise,

$$
\mathbf P_0[\mathbf D]=\sum_{\mathbf d}\frac{(k-|d_1|)(k-|d_2|)(k_3-|d_3|)}{k^2k_3}\,\mathbf p_0^{\rm el}[\mathbf K\mathbf D+\mathbf d].
$$

A geometry that varies along $x_3$ uses cubes, $k_3=k$. One that does not uses columns through the thickness, $k_3=N_3$ (one partition along $x_3$): the average over every $x_3$ offset is a Fejér filter that vanishes at every $\xi_3\neq0$, so only the $\xi_3=0$ frequencies survive, where $n_3=0$. That is the in-plane kernel, the same to round-off for any $N_3$ (checked 2026-10-26, a fiber at $N_3=1$ and 3, and as cubes on a cubic mesh: every method agrees to $10^{-14}$ at a tight Newton tolerance).

With one element per partition (Spectral LS in the code) this is the standard Moulinec–Suquet scheme at that resolution; with $k>1$ it averages the fine operator over partitions as SCA does over clusters (Liu–Bessa–Liu 2016). The average is a congruence, so its eigenvalues stay in $[0,1]$. Like LS-14 it is non-consistent; as the element grid is refined it tends to the consistent operator of LS-7 (`1_research_context.md`, section 2).

## 7. Solve the nonlinear equation without forming its Jacobian

### 7.1 Residual and basic fixed-point iteration

Using the stacked notation from Section 1, the two residual forms are equivalent:

**(LS-18) Reduced residual**

$$
\boxed{
\begin{aligned}
\mathbf R(\mathbf e)
&=\mathbf e-\mathbf b+\mathbf D_0[\mathbf s(\mathbf e)-\mathbf C_{0,M}\mathbf e]\\
&=\mathbf e-\mathbf b-\mathbf P_0[\mathbf e-\mathbf C_{0,M}^{-1}\mathbf s(\mathbf e)].
\end{aligned}
}
$$

The basic scheme is

**(LS-19) Fixed-point update**

$$
\mathbf e^{(j+1)}
=\mathbf b+\mathbf P_0[\mathbf e^{(j)}-\mathbf C_{0,M}^{-1}\mathbf s(\mathbf e^{(j)})]
=\mathbf e^{(j)}-\mathbf R(\mathbf e^{(j)}).
$$

A midpoint reference, $\mathbf C_{0,\rm mid}=(\mathbf L_m+\mathbf L_f)/2$, corresponds to midpoint phase Lame constants for these isotropic phases. It is a familiar elastic basic-scheme choice [MS], not a guarantee of optimal convergence or accuracy for this reduced plastic model. A valid model can have a divergent basic iteration.

### 7.2 Derive the matrix-free Newton product

Differentiate the local stress update at fixed $\mathbf z_n^A$:

$$
\mathbf L_t^A
=\frac{\partial\boldsymbol\sigma^A}{\partial\boldsymbol\varepsilon^A}
=\mathbf L^A(\mathbf I_6-\mathbf H_\mu^A).
$$

Let $\mathbf L_{t,M}=\operatorname{diag}(\mathbf L_t^1,\ldots,\mathbf L_t^M)$ collect these blocks diagonally. Differentiating LS-18 gives

**(LS-20) Jacobian-vector product**

$$
\boxed{
\mathbf J\mathbf v
=\mathbf v+\mathbf D_0[(\mathbf L_{t,M}-\mathbf C_{0,M})\mathbf v]
=\mathbf v-\mathbf P_0[(\mathbf I_{6M}-\mathbf C_{0,M}^{-1}\mathbf L_{t,M})\mathbf v].
}
$$

The elastic branch has $\mathbf H_\mu^A=\mathbf0$, including elastic unloading from a nonzero plastic history. At a yield transition, use the selected branch derivative; a smooth two-sided derivative need not exist. Apply LS-20 with local block products and one coarse convolution, without storing $\mathbf J$.

### 7.3 One load step

1. Hold $\mathbf z_n$ fixed. Initialize $\mathbf e^{(0)}=\mathbf e_n+\mathbf1_M\otimes(\bar{\boldsymbol\varepsilon}_{n+1}-\bar{\boldsymbol\varepsilon}_n)$.
2. Evaluate candidate stress/history and $\mathbf R$. Check a scaled strain residual and verify that all values are finite.
3. If not converged, solve $\mathbf J\Delta\mathbf e=-\mathbf R$ by GMRES using LS-20, and update the strain guess through a backtracking line search: take $\alpha\Delta\mathbf e$ with $\alpha=1,\tfrac12,\tfrac14,\ldots$, the first that lowers the scaled residual by the fraction $10^{-4}\alpha$ (the code applies the same search to every method). Check the linear solver's status.
4. Repeat from the same committed history. On convergence, commit the final candidate history and return LS-9. On failure, do not commit it.

Use a convergence criterion such as $\|\mathbf R\|\le\epsilon_{\rm abs}+\epsilon_{\rm rel}\|\mathbf e\|$, where $\epsilon_{\rm abs}$ has a meaningful strain scale and $\epsilon_{\rm rel}$ is dimensionless. The absolute term avoids division by a vanishing strain norm at zero loading. Choose Krylov tolerances and restart length to control linear-solve error and storage; they are algorithm settings, not part of the reduced constitutive model.

Fixed-point and Newton methods solve the same reduced equation for fixed $\mathbf C_0$. If both converge to the same root at matched tolerances, their predictions agree; changing the solver does not repair model error. Total online cost is approximately $O(kM\log M)$, where $k$ counts residual and Krylov convolution applications over the chosen load path. Near-linearithmic **total** scaling needs iteration counts to remain controlled. GMRES basis storage is $O(rM)$ for restart length $r$, not free storage beyond the $O(M)$ kernel.

### 7.4 Fourier preconditioner (added 2026-10-27)

**Why a small $\mathbf C_0$ is slow.** LS-22 needs $\mathbf C_0$ below every cluster's tangent, so the code uses $\mathbf C_0=\alpha\,\mathbf C_n$, a small fraction $\alpha$ (default 0.05) of a named stiffness $\mathbf C_n$ (the matrix). The Moulinec–Suquet kernel does not depend on the scale of $\mathbf C_0$: $\widehat{\mathbf P}_0=\widehat{\boldsymbol\Gamma}_0(\mathbf C_0)\mathbf C_0$ and $\widehat{\boldsymbol\Gamma}_0(\alpha\mathbf C)=\widehat{\boldsymbol\Gamma}_0(\mathbf C)/\alpha$. Write $\mathbf A=\mathbf C_n^{-1}\mathbf L_{t,M}$; then LS-20 is

$$
\mathbf J=\mathbf I-\mathbf P_0+\tfrac1\alpha\,\mathbf P_0\mathbf A ,
$$

whose eigenvalues spread like $\max\mathbf A/\alpha$: about $10^4$ for a fiber 100 times stiffer than the matrix at $\alpha=0.01$, against about 100 at $\alpha=1$.

**The preconditioner.** Write LS-20 as $\mathbf J\mathbf v=\mathbf v-\mathbf P_0(\mathbf D\mathbf v)$ with $\mathbf D=\mathbf I-\mathbf C_0^{-1}\mathbf L_{t,M}$. $\mathbf P_0$ is diagonal in Fourier space (one $6\times6$ block per frequency); $\mathbf D$ is diagonal in real space (one block per partition), so in Fourier space it is a convolution whose diagonal is its mean $\langle\mathbf D\rangle$ over the lattice. The Fourier-diagonal operator closest to $\mathbf J$ in the Frobenius norm therefore keeps $\mathbf J$'s Fourier-space diagonal:

**(LS-23) Fourier preconditioner**

$$
\widehat{\mathbf M}(\boldsymbol\xi)=\mathbf I-\widehat{\mathbf P}_0(\boldsymbol\xi)\,\langle\mathbf D\rangle=\mathbf I-\widehat{\mathbf P}_0(\boldsymbol\xi)\,\big(\mathbf I-\mathbf C_0^{-1}\langle\mathbf L_t\rangle\big),
\qquad
\mathbf M^{-1}\mathbf v=\operatorname{IFFT}\big[\widehat{\mathbf M}(\boldsymbol\xi)^{-1}\,\operatorname{FFT}(\mathbf v)\big].
$$

It is the Jacobian of a homogeneous cell with the mean current tangent (in linear algebra, the optimal circulant preconditioner of T. Chan, 1988; the derivation above is ours, the paper not read). The same rule, from each method's own kernel and tangents, applies to CP and Spectral LS (`ls_model.get_fourier_preconditioner`). The code rebuilds it at every Newton correction ($6\times6$ inversions per frequency, a fraction of a millisecond) and applies it on the right: GMRES solves $\mathbf J\mathbf M^{-1}\mathbf y=-\mathbf R$ and $\Delta\mathbf e=\mathbf M^{-1}\mathbf y$, so it still measures the residual of $\mathbf J\Delta\mathbf e=-\mathbf R$ and the Newton correction is unchanged; each GMRES iteration costs one more FFT pair. If $\mathbf P_0$ were a projection $\mathbf Q$ and $\langle\mathbf L_t\rangle=\mathbf C_n$, $\mathbf M^{-1}=(\mathbf I-\mathbf Q)+\alpha\mathbf Q$ and the preconditioned operator would have the spectrum of the $\alpha=1$ Jacobian.

**Why it helps CP and not Spectral LS.** For Spectral LS (one partition per element) $\mathbf P_0$ is a projection $\mathbf Q$ onto compatible zero-mean strains. A strain that starts compatible keeps its residual in the range of $\mathbf Q$, where any preconditioner of this form turns the system into $\boldsymbol\Gamma_0(\mathbf C_p)\mathbf L_t$ with $\mathbf C_p=\langle\mathbf L_t\rangle$: exactly Spectral LS's own system with $\mathbf C_p$ as its reference medium. Spectral LS's solution does not depend on its reference medium (Schneider 2021, §2), which acts as the preconditioner of its balance equation (Schneider 2021, §3) and is chosen for convergence (Moulinec–Suquet 1998, §3.1); with the midpoint $(\mathbf L_m+\mathbf L_f)/2$ (`spectral_ls_reference_stiffness`) it already sits where LS-23 would put it. CP's partition-averaged kernel is not a projection (eigenvalues in $[0,1]$), and its solution depends on $\mathbf C_0$, as for any clustered LS equation (Schneider 2019, §2), so $\mathbf C_0$ is fixed by its model and the preconditioner must be applied explicitly. Both are switchable (`cp_fourier_preconditioner`, `spectral_ls_fourier_preconditioner`); the default applies it to CP only.

Measured on the default problem (64² elements, 32² partitions, 60 steps), GMRES iterations and online time:

| | CP | Spectral LS (midpoint reference) |
|---|---|---|
| no preconditioner | 40 705, 23.3 s | 11 487, 18.1 s |
| LS-23 | 7 146, 6.0–6.7 s | 11 484, 24.2 s |
| LS-23, then block-Jacobi | 5 538, 5.5 s | 11 598, 26.6 s |
| block-Jacobi alone | 9 951, 7.8 s | 11 717, 22.4 s |

Each at the default tolerance against its own solution at $10^{-11}$, CP with LS-23 reaches $2.0\times10^{-8}$ (macroscopic) and $8.7\times10^{-7}$ (worst cluster), Spectral LS without it $1.2\times10^{-8}$ and $5.8\times10^{-7}$, TFA $1.1\times10^{-8}$ and $2.2\times10^{-7}$: the same accuracy. Block-Jacobi on the partitions' diagonal blocks $\mathbf I-\mathbf P_0[\mathbf 0]\mathbf D_A$ after LS-23 stopped CP at $6.1\times10^{-6}$ and $1.2\times10^{-3}$ with 214 Newton corrections against 181, so its 0.5 s is not a saving at equal accuracy; it is not used. Conjugate gradients on Brisard–Dormieux's symmetric polarization form, definite exactly when LS-22 is well posed: not tried.

**Caveat, resolved.** On `check.py`'s cut laminate CP once stopped 30 times less accurately with LS-23 than without, because the earlier stopping test under-weighted part of CP's residual; LS-24 in section 7.5 fixed it.

### 7.5 A stopping test that does not depend on the reference medium (added 2026-10-27, revised the same day)

*The user has said this section is not yet clear to them; it is written step by step for that reason. Ask before
building on it.*

**Step 1: what Newton checks.** A load step stops when the relative residual is below `convergence_tolerance`, $\|\mathbf R\|/\|\bar{\boldsymbol\varepsilon}\|<10^{-6}$. That is a fair test across methods only if $\|\mathbf R\|$ measures the same thing for each. For TFA it does: its residual ER-4 is a plain strain error.

**Step 2: the LS residual depends on $\mathbf C_0$.** With $\mathbf C_0=\alpha\,\mathbf C_m$ and $\mathbf P_0$ independent of $\alpha$ (section 7.4), LS-18 reads

$$
\mathbf R=\underbrace{(\mathbf I-\mathbf P_0)\,\mathbf e-\bar{\boldsymbol\varepsilon}}_{\text{compatibility part}}
\;+\;\frac1\alpha\,\underbrace{\mathbf P_0\,\mathbf C_m^{-1}\boldsymbol\sigma}_{\text{equilibrium part}} .
$$

At $\alpha=0.01$ the same unbalanced stress gives a residual 100 times larger than at $\alpha=1$. So one tolerance would mean very different accuracies for different $\mathbf C_0$.

**Step 3: the obvious fix, and where it falls short.** Multiplying by $\alpha$ (the earlier LS-24, $\mathbf C_m^{-1}\mathbf C_0\mathbf R$) gives $\alpha\cdot(\text{compatibility part})+(\text{equilibrium part})$. For Spectral LS the compatibility part is exactly zero ($\mathbf P_0$ is a projection and the iterates stay compatible), so this is exact. For CP it is not zero, and multiplying by $\alpha$ shrinks it 100 times: CP could stop with a compatibility error 100 times larger than the tolerance. On the cut laminate it did (CP $3.3\times10^{-5}$ of its tight solution, every other method and problem below $10^{-6}$).

**Step 4: the fix.** Ask instead: *how large a strain correction would a cell made only of matrix need to cancel this residual?* That cell's Newton Jacobian is LS-20 with $\mathbf L_t=\mathbf C_m$ everywhere, a convolution (one $6\times6$ block per frequency):

**(LS-24) Residual in matrix units**

$$
\widehat{\mathbf M}_m(\boldsymbol\xi)=\mathbf I-\widehat{\mathbf P}_0(\boldsymbol\xi)\,(\mathbf I-\mathbf C_0^{-1}\mathbf C_m),
\qquad
\tilde{\mathbf R}=\mathbf M_m^{-1}\mathbf R ,
$$

and convergence and the line search judge $\tilde{\mathbf R}$; the Newton correction still solves $\mathbf J\Delta\mathbf e=-\mathbf R$. TFA keeps its own residual.

**Step 5: why it works.** Look at one Fourier mode where $\mathbf P_0$ has eigenvalue $p\in[0,1]$. There $\widehat{\mathbf M}_m=1+p\,(1/\alpha-1)$ and $\mathbf R=c+e/\alpha$ ($c$ compatibility, $e$ equilibrium part), so

$$
\tilde{\mathbf R}=\frac{c+e/\alpha}{1+p\,(1/\alpha-1)}\;\approx\;
\begin{cases}\alpha\,c+e & p\to1\ \text{(the earlier measure)}\\ c & p\to0\ \text{(the compatibility part, no longer shrunk)}\end{cases}
$$

Neither limit depends on $\alpha$. Spectral LS lives entirely at $p=1$, so for it LS-24 equals the earlier measure (identical results); CP's modes with small $p$ are now judged in full.

**Cost and effect.** One convolution per residual evaluation ($\widehat{\mathbf M}_m^{-1}$ is built once per lattice). Each method at the default tolerance against its own solution at $10^{-11}$ (macroscopic / worst cluster): cut laminate CP $1.1\times10^{-6}$ / $2.1\times10^{-6}$ (was $3.3\times10^{-5}$ / $4.4\times10^{-5}$); every other check problem and method unchanged to the bit. Default problem (64² elements, 32² partitions, 60 steps): CP $2.0\times10^{-8}$ / $7.8\times10^{-7}$ (182 Newton, 7 216 GMRES, 6.2 s), Spectral LS $1.2\times10^{-8}$ / $5.8\times10^{-7}$ (183, 11 487, 16.9 s), TFA $1.1\times10^{-8}$ / $2.2\times10^{-7}$ (112, 757, 7.5 s). All methods stop at the same accuracy. A first attempt that read the residual through the *mean* elastic stiffness instead of the matrix's loosened every method (to $10^{-5}$–$10^{-4}$ in the worst cluster): a stiff reading cell underestimates errors in the soft matrix, where they are largest.

### 7.6 Newton with cluster eigenstrains (added 2026-10-27)

With LS-25 and LS-26, Newton's unknowns stay the partition strains, and every residual evaluation solves the eigenstrains from them: Newton on LS-26 at fixed $\boldsymbol\varepsilon$, from the accepted history, with Jacobian

$$
\mathbf Z=\mathbf I-\mathbf H\,\mathbf F,
\qquad
\mathbf H=\operatorname{diag}(\mathbf H_{\mu,g})\ \text{at each cluster's average strain},
$$

the identity on uncoupled clusters (one step settles them: $\boldsymbol\mu_g=\boldsymbol\mu(\boldsymbol\varepsilon_g)$) and sparse on the coupled ones, factorized by sparse LU, each step through the outer Newton's backtracking line search on $\|\mathbf L_g\mathbf r_g\|$ (the return map's switch between elastic and plastic clusters can make full steps cycle), until $\|\mathbf L_g\mathbf r_g\|$ is below `cp_eigenstrain_tolerance` times the largest cluster stress. No FFT is involved. So every Newton iterate satisfies every cluster's return map, as LS-22's split did per mixed partition. Then LS-18's residual and its stopping test LS-24 are unchanged, and LS-20 holds with $\mathbf D$ replaced by the equivalent eigenstrain's response to a partition strain change $\mathbf v$, the eigenstrains following through LS-26 ($\mathbf Z\,\delta\boldsymbol\mu=\mathbf H\mathbf E\mathbf v$):

**(LS-27) CP's Newton product**

$$
\boxed{
\mathbf J\mathbf v=\mathbf v-\mathbf P_0(\mathbf D\mathbf v),
\qquad
\mathbf D\mathbf v=\mathbf D_e\big(\mathbf v-\boldsymbol\Lambda\,\mathbf Z^{-1}\mathbf H\mathbf E\,\mathbf v\big).
}
$$

When every cluster lies in one partition, $\mathbf D$ is one block per partition, $\mathbf I-\mathbf C_0^{-1}\mathbf L_t^A$ with LS-22's tangent; otherwise the eigenstrains couple the partitions a cluster spans. LS-23 takes $\langle\mathbf D\rangle$ as the lattice mean of $\mathbf D$'s response to the six uniform fields, which is the mean block in the first case. One product costs four sparse matrix–vector products and one sparse triangular solve on the coupled clusters besides the FFTs. `check.py`'s checks and finite differences of the residual (on clusters cut across partitions as well) confirm LS-27.

Measured on the default problem with LS-22's clusters (64² elements, 32² partitions, 60 steps, $\mathbf C_0=0.01\times$ matrix), against the previous code, which solved LS-22's split per mixed partition: 182 Newton corrections and 7 217 GMRES iterations (7 216 before), the same accuracy at the default tolerance against its own solution at $10^{-11}$ ($2.0\times10^{-8}$ macroscopic, $7.9\times10^{-7}$ worst cluster), the two solutions $9\times10^{-13}$ apart; online time 5–10 % longer (5.7–6.2 s against 5.4 s), the sparse products and solves. Without the preconditioner: 213 corrections, 55 449 GMRES iterations, 32.9 s, the same answer.

**Tried and not adopted.** Making the eigenstrains Newton unknowns beside the partition strains, with each correction solved for the polarization $\delta\boldsymbol\tau$ (on which every piece's response is local), gave the same solutions but 295 Newton corrections against 182, stopped 20–50 times less accurately at the same tolerance ($1.3\times10^{-6}$ macroscopic, $3.5\times10^{-5}$ worst cluster), since the outer Newton then has to find which clusters flow, and needed $\mathbf C_0$ below the tangent of every yielding cluster, not only the coupled ones.

## 8. Relationship to other frameworks

- **E/P TFA:** retains actual heterogeneous elastic fluctuations but assumes reduced plastic histories. Its dense implementation has quadratic interaction storage/application. The new model replaces those elastic fluctuations by a reference response; it is not simply a faster solver for LS-2.
- **SCA:** already uses partition/cluster-averaged reference interactions and local constitutive updates [SCA, Eqs. (16)-(18)]. Arbitrary irregular or disconnected clusters retain detailed geometry in their interaction tensors but generally do not form a convolution over cluster labels. Regular translated cells restore convolution. Matching this operator idea does not reproduce SCA's complete clustering or self-consistency algorithms; zero-mode and macroscopic constraints must also be matched.
- **Coarse FFT with averaged/filtered operators:** cell-averaged interactions can be defined directly from continuum reference elasticity without an $N$-grid. Filtering is an approximation to a specified averaged operator, not synonymous with exact cell averaging [BL, Section 3]. At matched material representation, such methods also admit $M$-grid online FFTs; using $N$ offline is not automatically an advantage.
- **FE/FFT methods with one history per element:** these exist. For Hookean/J2 materials, Gehrig-Schneider retain actual stiffness in the within-element stress fluctuation [GS, Eqs. (58)-(61)], unlike LS-11. Their fine displacement equilibrium and this partition-only online solve are therefore different constructions.

The distinction between $N$ and $M$ is computational, not a guarantee of retained physical detail. This framework averages an $N$-scale **reference** FE response offline and advances $M$ material states online. A direct $M$-cell FFT discretization may require no fine reference mesh. With aligned interfaces, both can retain the same phase geometry while approximating its strain fields and plastic histories differently.

**Summary:** the framework combines local actual-material updates with homogeneous, partition-averaged interactions. It uses actual stiffness for partition-average deformation and reference stiffness for within-partition fluctuations. Regular translated partitions permit an online $M$-grid FFT and linear-size kernel storage. These structural properties do not imply an accuracy advantage over other coarse discretizations.

## 9. Open points

These were found while implementing and testing the model, and are not yet resolved in the sections above.

- **k-means clusters (section 4.5).** A first look on `check.py`'s fiber and sphere with 8 matrix k-means clusters
  put CP's last macroscopic stress 1.7 % and 3.1 % from Spectral LS, against 2.4 % and 3.8 % with LS-22's 105 and 96
  clusters; TFA moved from 3.9 % and 12.5 % to 4.4 % and 13.7 %. Fewer material states coming closer is unexplained
  and measured on one quantity only; not yet studied.
- **Reference stiffness.** $\mathbf C_0$ changes CP's answer (Schneider 2019 notes the same for clustered
  Lippmann–Schwinger equations). With one $\mathbf C_0$ for LS-7 and LS-22, it must stay below every cluster's tangent
  in mixed partitions (section 4.4), so CP cannot run a perfectly plastic or softening phase at all, and on `check.py`'s
  fiber it costs accuracy (4.3 % against element-level TFA, against 2.4 % for the earlier equal-stress split with the
  matrix stiffness in LS-7). Candidates not adopted: a $\mathbf C_0$ that follows the tangent (as in SCA), and the
  Hashin–Shtrikman split of section 4.4, which keeps one $\mathbf C_0$ but not constant polarization.
- **Plastic branch of $\mathbf H_\mu$.** LS-20 needs it but this note does not state it
  (`notes/background_influence_functions_and_J2/` derives it; `material.get_eigenstrain_sensitivity`). Its
  finite-difference check went with the 2026-10-26 cut-back (git tag `pre-cleanup`).
- **Preconditioning Spectral LS.** No Fourier-diagonal preconditioner can do better than its reference medium
  (section 7.4); a gain would need a local preconditioner that keeps iterates compatible. Block-Jacobi did not.
- **Equal accuracy across methods** (section 7.5) was verified by hand on `check.py`'s problems and the default
  problem, not by `check.py` itself.
- **Convergence in $M$.** Section 5.4 leaves it open. Measured convergence is first order in partition size. Schneider
  (2019) proves convergence of clustered Lippmann–Schwinger equations as clusters are refined, for hardening
  materials only; softening is outside that theory.
- **The kernels of section 6 are not the operator of LS-7.** The FE-derived $\mathbf P_0$ (LS-14) averages the FE
  reference response, so it is non-consistent in Brisard–Dormieux's sense. The exact partition average of LS-7 can be
  computed from the partition lattice alone, as a weighted sum over aliases; see `1_research_context.md`, sections 2–3. Only
  the FE kernel makes LS reduce to E/P TFA in the matched-stiffness limit of section 5.4, since TFA's $\mathbf P$
  uses the same FE mesh.

## Paper references

- **[MS]** Moulinec, H.; Suquet, P. (1998), "A numerical method for computing the overall response of nonlinear composites with complex microstructure," *Computer Methods in Applied Mechanics and Engineering* 157, 69-94. [Paper](https://doi.org/10.1016/S0045-7825(97)00218-1). Basis for the reference split, periodic Lippmann-Schwinger equation, basic iteration, and elastoplastic local updates.
- **[FC]** Fish, J.; Cui, J. (2026), "Eigenstate based homogenization," *Computer Methods in Applied Mechanics and Engineering* 452, 118718. [Paper](https://doi.org/10.1016/j.cma.2025.118718). Background for the heterogeneous partition transformation in LS-1; the framework here does not reproduce the paper's full eigenstate reduction.
- **[SCA]** Liu, Z.; Bessa, M. A.; Liu, W. K. (2016), "Self-consistent clustering analysis: an efficient multi-scale scheme for inelastic heterogeneous materials." [Author preprint](https://zeliangliu.com/publication/liu-2016-self/preprint.pdf). Eqs. (16)-(18): cluster-averaged reference interactions and constitutive closure; Section 2.2.2: offline Fourier construction. Cited for the established averaging framework, not as evidence of an online cluster-grid FFT for arbitrary clusters.
- **[BL]** Brisard, S.; Legoll, F., "Periodic homogenization using the Lippmann-Schwinger formalism," *Parallel Computing* (2015); preprint (2014). [Full preprint](https://arxiv.org/pdf/1411.0330). Section 3.1, Eqs. (23)-(27), and Section 3.2 distinguish consistent cell-averaged interactions and approximate discrete Green operators. Its analysis is not a proof for this nonlinear FE-derived model.
- **[GS]** Gehrig, F.; Schneider, M. (2025), "Element-Based Internal Variable Formulations for Finite Element Discretizations in FFT-Based Homogenization Methods." [Full article](https://onlinelibrary.wiley.com/doi/full/10.1002/nme.70170). Section 3.2.1, Eqs. (58)-(61): average-strain updates with actual elastic stress fluctuations. This illustrates why one history per region does not imply constant total strain or polarization.

The derivation specifies the reduced equation, its modeling assumptions, and its reference-FE realization. Accuracy, general convergence, and superiority over other reduced models do not follow from the FFT representation alone.
