# The Lippmann–Schwinger equation: Spectral LS and FFT-TFA

## Purpose and route

We want a nonlinear homogenization model with a few material states (clusters), whose interaction costs $O(M\log M)$ per application with $M$ partitions, and stores no dense interaction matrix. TFA (`3_tfa_baseline.md`) has the material states but a dense $\mathbf P$: shifting a source and a receiver changes their heterogeneous surroundings, so $\mathbf P$ is no convolution. The Lippmann–Schwinger (LS) equation of a *homogeneous* reference medium is one, on any regular lattice.

Section 1 states the setting; section 2 derives the exact LS equation; section 3 builds its kernel on the elements and averages it over partitions; section 4 solves it on the elements, which is **Spectral LS**; section 5 derives **FFT-TFA**, the method under development: TFA's clusters interacting through that equation with the matrix as reference medium. Equation labels use `LS-` and, in section 5, `FT-`; the labels are not consecutive, since those of the earlier method CP (git tag `pre-cleanup-2`) are not used here.

## 1. Setting and notation

### 1.1 Assumptions

- A connected periodic cell $\Theta$, small strains, perfect bonding, no body forces, and a prescribed mean strain $\bar{\boldsymbol\varepsilon}$. The displacement fluctuation $\mathbf u$ is periodic.
- Isotropic elasticity and associative, rate-independent $J_2$ plasticity with linear isotropic hardening $H\geq0$, or a purely elastic phase.
- During load step $n+1$, the committed history $\mathbf z_n=(\boldsymbol\mu_n,p_n)$ stays fixed while nonlinear trial strains change; candidates are committed only after convergence. $p\geq0$ is the accumulated equivalent plastic strain and $\boldsymbol\mu$ the plastic strain.
- A fixed, homogeneous, symmetric positive-definite reference stiffness $\mathbf C_0$.

### 1.2 The three grids

$N$ regular elements resolve the geometry, $M$ congruent partitions (blocks of elements) form a periodic lattice, and clusters are regions of one phase inside one partition, each with one material state. A cluster $p$ has volume fraction $c_p$ of its partition $A_p$, so $\sum_{p\in A}c_p=1$, and $v_p=c_p/M$ of the cell. Regular translated partitions are what makes the online operator a convolution.

### 1.3 Work-conjugate vectors

Engineering-strain and physical-stress Voigt vectors throughout:

$$
\boldsymbol\varepsilon
=(\varepsilon_{11},\varepsilon_{22},\varepsilon_{33},2\varepsilon_{12},2\varepsilon_{23},2\varepsilon_{13})^{\mathsf T},
\qquad
\boldsymbol\sigma
=(\sigma_{11},\sigma_{22},\sigma_{33},\sigma_{12},\sigma_{23},\sigma_{13})^{\mathsf T},
$$

so $\boldsymbol\sigma^{\mathsf T}\boldsymbol\varepsilon$ is the work density, without an extra shear weight.

| Symbol | Meaning |
|---|---|
| $\mathbf L$, $\mathbf C_0$, $\mathbf L_t$ | Elastic stiffness; reference stiffness; algorithmic tangent |
| $\boldsymbol\mu$, $\boldsymbol\tau$ | Plastic strain; polarization |
| $\mathbf L_m$ | The matrix's elastic stiffness |
| $\mathbf P_0$, $\mathbf D_0=\mathbf P_0\mathbf C_0^{-1}$ | Reference eigenstrain and polarization interaction; dimensionless, compliance units |
| $\mathbf H_\mu=\partial\boldsymbol\mu/\partial\boldsymbol\varepsilon$ | Plastic-strain sensitivity at fixed committed history |

### 1.4 The local material update

At a trial strain $\boldsymbol\varepsilon$, from the committed pair $(\boldsymbol\mu_n,p_n)$, with $\mathbf v_{\rm vol}=(1,1,1,0,0,0)^{\mathsf T}$ and $\mathbf W=\operatorname{diag}(1,1,1,2,2,2)$:

$$
\boldsymbol\sigma^{\mathrm{tr}}=\mathbf L(\boldsymbol\varepsilon-\boldsymbol\mu_n),
\quad
\mathbf t^{\mathrm{tr}}=\boldsymbol\sigma^{\mathrm{tr}}-\tfrac13(\sigma^{\mathrm{tr}}_{11}+\sigma^{\mathrm{tr}}_{22}+\sigma^{\mathrm{tr}}_{33})\mathbf v_{\rm vol},
\quad
q^{\mathrm{tr}}=\sqrt{\tfrac32(\mathbf t^{\mathrm{tr}})^{\mathsf T}\mathbf W\mathbf t^{\mathrm{tr}}},
\quad
f^{\mathrm{tr}}=q^{\mathrm{tr}}-(\sigma_{\mathrm y0}+Hp_n).
$$

If $f^{\mathrm{tr}}\leq0$ the state stays elastic. Otherwise radial return gives

$$
\Delta\lambda=\frac{f^{\mathrm{tr}}}{3G+H},
\qquad
\boldsymbol\mu=\boldsymbol\mu_n+\frac{3\Delta\lambda}{2q^{\mathrm{tr}}}\mathbf W\mathbf t^{\mathrm{tr}},
\qquad
p=p_n+\Delta\lambda,
\qquad
\boldsymbol\sigma=\mathbf L(\boldsymbol\varepsilon-\boldsymbol\mu).
$$

Its derivative is $\mathbf L_t=\partial\boldsymbol\sigma/\partial\boldsymbol\varepsilon=\mathbf L(\mathbf I-\mathbf H_\mu)$ (`notes/background_influence_functions_and_J2/` derives the plastic branch of $\mathbf H_\mu$; `material.get_eigenstrain_sensitivity`). An update at a cluster's average strain need not equal the average of separate fine-point updates; we adopt the former deliberately.

## 2. The exact reference-medium equation

Before any approximation, start from the actual fields:

**(LS-3) Physical problem**

$$
\nabla\cdot\boldsymbol\sigma^{\mathrm{ten}}=\mathbf0,
\qquad
\boldsymbol\varepsilon^{\mathrm{ten}}=\bar{\boldsymbol\varepsilon}^{\mathrm{ten}}+\nabla^s\mathbf u,
\qquad
\boldsymbol\sigma=\mathbf L(x)(\boldsymbol\varepsilon-\boldsymbol\mu).
$$

Define actual-minus-reference stress, the **polarization**:

**(LS-4) Exact stress split**

$$
\boldsymbol\sigma=\mathbf C_0\boldsymbol\varepsilon+\boldsymbol\tau,
\qquad
\boldsymbol\tau=\boldsymbol\sigma-\mathbf C_0\boldsymbol\varepsilon=(\mathbf L-\mathbf C_0)\boldsymbol\varepsilon-\mathbf L\boldsymbol\mu.
$$

This is an identity: elastic contrast is in $(\mathbf L-\mathbf C_0)\boldsymbol\varepsilon$ and plasticity in $-\mathbf L\boldsymbol\mu$. Substituting it into equilibrium gives a homogeneous reference problem with $\boldsymbol\tau$ as its source. If $\boldsymbol\tau$ were known, this would be a linear reference solve, whose strain response defines a Green operator:

**(LS-5) Exact Lippmann–Schwinger equation**

$$
\boxed{
\boldsymbol\varepsilon=\bar{\boldsymbol\varepsilon}-\boldsymbol\Gamma_0*\boldsymbol\tau,
\qquad
\widehat{\boldsymbol\Gamma}_0(\mathbf0)=\mathbf0.
}
$$

The minus sign is Moulinec–Suquet's [MS]. Homogeneous coefficients and periodic boundaries make the kernel depend only on $x-y$, which is why it is a convolution. The zero-frequency convention enforces $\langle\boldsymbol\varepsilon\rangle=\bar{\boldsymbol\varepsilon}$. No approximation has entered yet, and the exact solution does not depend on $\mathbf C_0$.

The same response in eigenstrain form, $\boldsymbol\tau=-\mathbf C_0\boldsymbol\mu^*$:

**(LS-6) Equivalent eigenstrain**

$$
\boldsymbol\mu^*=\boldsymbol\varepsilon-\mathbf C_0^{-1}\boldsymbol\sigma,
\qquad
\boldsymbol\sigma=\mathbf C_0(\boldsymbol\varepsilon-\boldsymbol\mu^*).
$$

$\boldsymbol\mu^*$ contains both elastic contrast and plastic strain; it is not a plastic history variable.

Periodicity makes every reference fluctuation response have zero cell mean:

**(LS-9) Mean consistency and output**

$$
\sum_B c^B\mathbf D_0^{BA}=\mathbf0,
\qquad
\langle\boldsymbol\varepsilon\rangle=\bar{\boldsymbol\varepsilon},
\qquad
\bar{\boldsymbol\sigma}=\langle\boldsymbol\sigma\rangle.
$$

## 3. The kernel

### 3.1 On the elements

The code evaluates the continuum Green operator at the frequencies $\boldsymbol\xi$ of the lattice of elements:

**(LS-21) Moulinec–Suquet kernel**

$$
\widehat{\mathbf P}_0(\boldsymbol\xi)=\widehat{\boldsymbol\Gamma}_0(\mathbf n)\,\mathbf C_0,
\qquad
\widehat{\boldsymbol\Gamma}_0(\mathbf n)=\mathbf B(\mathbf n)\big[\mathbf B(\mathbf n)^{\mathsf T}\mathbf C_0\mathbf B(\mathbf n)\big]^{-1}\mathbf B(\mathbf n)^{\mathsf T},
\qquad
\mathbf n=\boldsymbol\xi/|\boldsymbol\xi|,
$$

where $\mathbf B(\mathbf n)$ ($6\times3$) is the engineering strain of a displacement varying along $\mathbf n$, per unit gradient. The cell is a cube, so frequencies in units of $2\pi/L$ compare across axes whatever each axis's element count. $\widehat{\mathbf P}_0(\mathbf 0)=\mathbf 0$ (LS-9). At every other frequency $\widehat{\mathbf P}_0$ is a projection, with eigenvalues 0 and 1, for any $\mathbf C_0$. Along an axis with an even count $N_i$, $\widehat{\boldsymbol\Gamma}_0$ is not even in a component $\xi_i=N_i/2$, so it has no real-valued meaning there; $\widehat{\mathbf P}_0=\mathbf I$ at those frequencies, Moulinec–Suquet's $\widehat{\boldsymbol\Gamma}_0=\mathbf C_0^{-1}$ (1998, §2.4.2), which makes the stress there zero.

### 3.2 Averaged over partitions

A polarization constant on each partition, inserted in LS-5 and averaged over a partition $B$, gives an interaction between partitions that accounts for their finite size, not merely their centres:

**(LS-7) Partition interaction**

$$
\boldsymbol\varepsilon^B=\bar{\boldsymbol\varepsilon}-\sum_A\mathbf D_0^{BA}\boldsymbol\tau^A,
\qquad
\mathbf D_0^{BA}=\frac{1}{|\Omega_B|}\int_{\Omega_B}\int_{\Omega_A}\boldsymbol\Gamma_0(x-y)\,dV_y\,dV_x .
$$

**(LS-8) The same in eigenstrain form:** $\mathbf P_0^{BA}=\mathbf D_0^{BA}\mathbf C_0$, acting on $\boldsymbol\mu^*$ of LS-6.

The code does not evaluate this double integral. It follows the elements' kernel by linearity: a unit eigenstrain in a partition is one in each of its elements, and a partition's strain is the mean over its elements, so, for partitions of $k\times k\times k_3$ elements, with $|d_1|,|d_2|<k$, $|d_3|<k_3$ and $\mathbf K=(k,k,k_3)$ acting componentwise,

$$
\mathbf P_0[\mathbf D]=\sum_{\mathbf d}\frac{(k-|d_1|)(k-|d_2|)(k_3-|d_3|)}{k^2k_3}\,\mathbf p_0^{\rm el}[\mathbf K\mathbf D+\mathbf d].
$$

A geometry that varies along $x_3$ uses cubes, $k_3=k$. One that does not has one element layer and columns through the thickness; averaging over every $x_3$ offset would keep only the $\xi_3=0$ frequencies, where $n_3=0$: the in-plane kernel. This averages the fine operator over partitions as SCA does over clusters [SCA]. The average is a congruence, so its eigenvalues stay in $[0,1]$. It is non-consistent in Brisard–Dormieux's sense; as the element grid is refined it tends to the exact partition average (`1_research_context.md`, section 2).

### 3.3 Applying it

Translation invariance gives $\mathbf P_0^{BA}=\mathbf p_0[\mathbf a_B-\mathbf a_A]$, one $6\times6$ block per periodic offset, so for a field $\mathbf v$ on a lattice:

**(LS-17) Application by FFT**

$$
\mathbf P_0\mathbf v=\operatorname{IFFT}\left[\widehat{\mathbf p}_0(\boldsymbol\xi)\operatorname{FFT}(\mathbf v)(\boldsymbol\xi)\right],
$$

$O(M)$ storage and $O(M\log M)$ work on $M$ lattice points. Order matters: stiffness multiplies on the source side, $\mathbf D_0=\mathbf P_0\mathbf C_0^{-1}$.

## 4. Spectral LS: the equation on the elements

With one partition and one cluster per element, the material state lives on the same lattice as the kernel, and LS-5 with LS-6 is the Moulinec–Suquet scheme [MS]. Stack the element strains as $\mathbf e$ and stresses as $\mathbf s(\mathbf e)$, with $\mathbf b$ the uniform mean strain and $\mathbf C_{0,M}$, $\mathbf L_{t,M}$ the diagonal blocks.

**(LS-18) Residual**

$$
\boxed{
\mathbf R(\mathbf e)=\mathbf e-\mathbf b-\mathbf P_0\big[\mathbf e-\mathbf C_{0,M}^{-1}\mathbf s(\mathbf e)\big].
}
$$

**(LS-20) Newton product**

$$
\boxed{
\mathbf J\mathbf v=\mathbf v-\mathbf P_0\big[(\mathbf I-\mathbf C_{0,M}^{-1}\mathbf L_{t,M})\mathbf v\big].
}
$$

The elastic branch has $\mathbf H_\mu=\mathbf0$, including elastic unloading from a nonzero plastic history. At a yield transition, use the selected branch's derivative. The product needs local blocks and one convolution, never a stored $\mathbf J$.

### 4.1 One load step

1. Hold $\mathbf z_n$ fixed. Start from $\mathbf e_n$ plus the uniform macrostrain increment.
2. Evaluate the candidate stress, history and residual. Stop when the relative residual (`newton.get_relative_residual`) is below `convergence_tolerance`, its denominator never below `residual_strain_scale_floor`, so that it stays defined at zero imposed strain.
3. Solve $\mathbf J\Delta\mathbf e=-\mathbf R$, and update through a backtracking line search: $\alpha\Delta\mathbf e$ with $\alpha=1,\tfrac12,\tfrac14,\ldots$, the first that lowers the relative residual by the fraction $10^{-4}\alpha$ (the same search for every method, `newton.Solver`).
4. Repeat from the same committed history; on convergence, commit the final candidate.

### 4.2 Conjugate gradients

GMRES keeps every Krylov vector since its last restart and orthogonalizes each new one against all of them, so an iteration costs more the more iterations a correction needs. Conjugate gradients (CG) keep only the latest search direction, at a fixed cost per iteration, but need a system symmetric and positive definite in some inner product. Spectral LS's is, in the inner product of its reference medium. This is the strain-based CG of Zeman et al., as presented in Schneider's review (2021, §3; Zeman et al. not read).

**The inner product.** $\langle\mathbf u,\mathbf v\rangle_0=\sum_A\mathbf u_A^{\mathsf T}\mathbf C_0\mathbf v_A$: the work of the stress $\mathbf C_0\mathbf u$ on $\mathbf v$.

**$\mathbf P_0$ is a projection, orthogonal in it.** At every frequency $\widehat{\mathbf P}_0^2=\widehat{\mathbf P}_0$, and $\mathbf C_0\widehat{\mathbf P}_0=\mathbf C_0\widehat{\boldsymbol\Gamma}_0\mathbf C_0$ is symmetric, so $\mathbf P_0=\mathbf Q$ is the $\langle\cdot,\cdot\rangle_0$-orthogonal projection onto its range, the compatible zero-mean strains (with the highest frequencies).

**The residual and every correction stay in that range.** A step starts from a compatible strain plus the uniform increment, so $\mathbf e-\mathbf b$ lies in the range of $\mathbf Q$ and $\mathbf Q\mathbf b=\mathbf0$; LS-18 then gives $\mathbf R=\mathbf Q\,\mathbf C_{0,M}^{-1}\mathbf s$, in the range too, and CG's iterates stay there.

**On that range $\mathbf J$ is symmetric and positive definite.** For $\mathbf v=\mathbf Q\mathbf v$, LS-20 gives $\mathbf J\mathbf v=\mathbf Q\,\mathbf C_{0,M}^{-1}\mathbf L_{t,M}\mathbf v$, and with $\mathbf C_0\mathbf Q=\mathbf Q^{\mathsf T}\mathbf C_0$:

**(LS-28) Spectral LS's Newton operator in the reference medium's inner product**

$$
\boxed{
\langle\mathbf J\mathbf u,\mathbf v\rangle_0=\mathbf u^{\mathsf T}\mathbf L_{t,M}\,\mathbf v
\qquad\text{for } \mathbf u=\mathbf Q\mathbf u,\ \mathbf v=\mathbf Q\mathbf v .
}
$$

J2's algorithmic tangent with linear isotropic hardening is symmetric, and positive definite when every yielding phase hardens ($H>0$), so CG applies. The code stops at a direction of non-positive curvature. The answer does not depend on the Krylov method: CG stops, as GMRES does, once $\|\mathbf R+\mathbf J\Delta\mathbf e\|$ is below `krylov_tolerance` times $\|\mathbf R\|$.

### 4.3 A stopping test that does not depend on the reference medium

*The user has said this section is not yet clear to them; it is written step by step for that reason. Ask before building on it.*

**Step 1: what Newton checks.** A load step stops when the relative residual is below `convergence_tolerance`. That is fair across methods only if $\|\mathbf R\|$ measures the same thing for each. For TFA and FFT-TFA it does: their residuals are plain strain errors, and FFT-TFA's reference medium is the matrix.

**Step 2: Spectral LS's residual depends on $\mathbf C_0$.** Section 4.2 gave $\mathbf R=\mathbf Q\,\mathbf C_{0,M}^{-1}\mathbf s$: the unbalanced stress, read through $\mathbf C_0$. Spectral LS's default $\mathbf C_0$ is the midpoint of the phases, much stiffer than the matrix, so the same unbalanced stress reads as a much smaller residual than with $\mathbf C_0=\mathbf L_m$. One tolerance would then mean different accuracies for different $\mathbf C_0$.

**Step 3: the fix.** Ask instead: *how large a strain correction would a cell made only of matrix need to cancel this residual?* That cell's Newton Jacobian is LS-20 with $\mathbf L_t=\mathbf L_m$ everywhere, a convolution, one $6\times6$ block per frequency:

**(LS-24) Residual in matrix units**

$$
\widehat{\mathbf M}_m(\boldsymbol\xi)=\mathbf I-\widehat{\mathbf P}_0(\boldsymbol\xi)\,(\mathbf I-\mathbf C_0^{-1}\mathbf L_m),
\qquad
\tilde{\mathbf R}=\mathbf M_m^{-1}\mathbf R .
$$

Convergence and the line search judge $\tilde{\mathbf R}$; the Newton correction still solves $\mathbf J\Delta\mathbf e=-\mathbf R$.

**Step 4: why it works.** Take $\mathbf C_0=\alpha\mathbf L_m$ for intuition. $\widehat{\mathbf P}_0$ does not depend on $\alpha$, and on the range of $\mathbf Q$ it acts as the identity, so $\mathbf M_m$ acts as $1/\alpha$ there, while $\mathbf R$ carries a factor $1/\alpha$. They cancel: $\tilde{\mathbf R}$ does not depend on $\alpha$. With $\mathbf C_0=\mathbf L_m$, $\mathbf M_m$ is the identity and $\tilde{\mathbf R}=\mathbf R$: the same test as FFT-TFA's.

**Cost and effect.** One convolution per residual evaluation, and one more stored kernel-sized array. Spectral LS may use either stiffness (`C0` in `config.spectral_ls`). The two stop equally accurately and agree to about $10^{-10}$ in every cluster (100 random fibers on 128²; a 32³ sphere); Newton iterations are identical. Online, the matrix costs about 3 % more at contrast 34 and none on the sphere, and 17 % more at contrast 100, from 20 % more CG iterations.

## 5. FFT-TFA: TFA's clusters with the matrix as reference medium

FFT-TFA (`code/fft_tfa_model.py`) keeps TFA's material model, one uniform eigenstrain per cluster updated at the cluster's average strain (`3_tfa_baseline.md`, section 1), and replaces TFA's FE influence functions by the exact LS equation LS-5 of a homogeneous medium $\mathbf C_0=\mathbf L_m$, averaged over the clusters. Its interaction is applied by an FFT on the partitions plus exact corrections between the clusters of nearby partitions.

### 5.1 Clusters and polarization

With $\mathbf C_0=\mathbf L_m$, LS-4 gives a matrix element the polarization $\boldsymbol\tau=-\mathbf L_m\boldsymbol\mu_g$: uniform on a matrix cluster with uniform eigenstrain, and exact whatever the cluster's shape. An inclusion element's polarization $(\mathbf L_i-\mathbf L_m)\boldsymbol\varepsilon-\mathbf L_i\boldsymbol\mu_g$ follows its strain, and is taken uniform on each cluster. So each cluster $g$ has one strain $\boldsymbol\varepsilon_g$, the unknowns of the method, and

**(FT-1) Cluster polarization and update**

$$
\boldsymbol\tau_g=(\mathbf L_g-\mathbf C_0)\,\boldsymbol\varepsilon_g-\mathbf L_g\boldsymbol\mu_g,
\qquad
\boldsymbol\mu_g=\boldsymbol\mu\big(\boldsymbol\varepsilon_g;\mathbf z_{n,g}\big),
\qquad
\boldsymbol\sigma_g=\mathbf L_g(\boldsymbol\varepsilon_g-\boldsymbol\mu_g).
$$

Every cluster's stress is $\mathbf C_0\boldsymbol\varepsilon_g+\boldsymbol\tau_g$, so the cell's stress is $\mathbf L_m\langle\boldsymbol\varepsilon\rangle+\langle\boldsymbol\tau\rangle$, the volume average of the cluster stresses. For a matrix cluster the first term of $\boldsymbol\tau_g$ vanishes. No inverse of $\mathbf L_i-\mathbf L_m$ is needed, so the inclusion may have any stiffness, the matrix's included. Clusters are found by k-means inside each partition's part in each phase, on the elements' elastic strain concentration (Liu–Bessa–Liu 2016, §2.2.1), or are those parts themselves.

### 5.2 Interaction

Inserting the piecewise-uniform polarization into LS-5 and averaging over a cluster $p$ of partition $A$ gives the exact strain $\boldsymbol\varepsilon_p=\bar{\boldsymbol\varepsilon}-\sum_t\mathbf D_{pt}\boldsymbol\tau_t$, with $\mathbf D_{pt}$ the element kernel $\mathbf D_0=\mathbf P_0\mathbf C_0^{-1}$ (LS-21) averaged over the elements of $p$ and summed over those of $t$. Applied as it stands, that is $O(n^2)$ in the clusters. FFT-TFA splits it into what the partition lattice carries, each partition's mean polarization $\bar{\boldsymbol\tau}_B=\sum_{t\in B}c_t\boldsymbol\tau_t$ through the partition-averaged kernel $\bar{\mathbf D}_0=\mathbf P_0\mathbf C_0^{-1}$ of section 3.2, and corrections kept only between nearby partitions:

**(FT-2) FFT-TFA's interaction**

$$
\boldsymbol\varepsilon_p=\bar{\boldsymbol\varepsilon}-\big(\boldsymbol\Pi\,\mathbf m\big)_p,
\qquad
\mathbf m_p=\sum_B\bar{\mathbf D}_0[A-B]\,\bar{\boldsymbol\tau}_B+\sum_{t\in\mathcal N(A)}\mathbf C_{pt}\boldsymbol\tau_t,
$$

where $\mathcal N(A)$ holds the clusters of the partitions within `reach` of $A$: 0 is $A$ alone, 1 adds its neighbours (8 in-plane for columns, 26 in 3D; offsets taken modulo the lattice, so a lattice two partitions wide does not count a neighbour twice), `"full"` every partition. The first sum is one FFT convolution on the partitions (LS-17).

**(FT-3) Correction blocks**

$$
\mathbf C_{pt}=\frac1{n_p}\sum_{\boldsymbol\delta}n_{pt}(\boldsymbol\delta)\,\mathbf D_0[\boldsymbol\delta-\mathbf K\mathbf d]
-c_t\,\bar{\mathbf D}_0[-\mathbf d],
\qquad
\mathbf d=B-A,
$$

with $n_p$ the elements of $p$, $\mathbf K$ the partition's size in elements along each axis, and $n_{pt}(\boldsymbol\delta)$ the number of element pairs $(a\in p,\,b\in t)$ at offset $\boldsymbol\delta=a-b$ within their partitions: the cross-correlation of the two clusters' indicators, by FFT on twice the partition. So $\mathbf C_{pt}$ replaces the partition FFT's share of the interaction between $p$ and $t$ by the exact one. At full reach the interaction is the exact LS-5 average (`check.py` checks this to round-off on every problem). With one element per partition every $\mathbf C_{pt}$ vanishes and FFT-TFA is Spectral LS with $\mathbf C_0=\mathbf L_m$.

**(FT-4) Mean strain**

$$
(\boldsymbol\Pi\mathbf m)_p=\mathbf m_p-\frac1M\sum_q c_q\,\mathbf m_q .
$$

The exact interaction has zero cell mean (LS-9); truncated, it does not (measured on `check.py`'s fiber: $10^{-3}$ of the macroscopic stress with reach 0, $10^{-4}$ with reach 1). $\boldsymbol\Pi$ removes that mean, so $\langle\boldsymbol\varepsilon\rangle=\bar{\boldsymbol\varepsilon}$ exactly and the cluster stresses average to the cell's stress. It is our choice; it moves the macroscopic stress by the same $10^{-3}$–$10^{-4}$.

### 5.3 Newton solve

**(FT-5) Residual**

$$
\mathbf r_p=\boldsymbol\varepsilon_p-\bar{\boldsymbol\varepsilon}+\big(\boldsymbol\Pi\,\mathbf m(\boldsymbol\tau(\boldsymbol\varepsilon))\big)_p ,
$$

a strain error like TFA's ER-4. Its norm is the volume average over the clusters. With $\mathbf C_0=\mathbf L_m$ the all-matrix Jacobian of LS-24 is the identity, so FT-5 needs no such measure and stops at the same accuracy as the others (section 5.6). Each step starts from the previous one's cluster strains plus the macrostrain increment.

**(FT-6) Newton product**

$$
\mathbf J\mathbf v=\mathbf v+\boldsymbol\Pi\,\mathbf m(\mathbf T\mathbf v),
\qquad
\mathbf T_g=\mathbf L_{t,g}-\mathbf C_0 ,
$$

one block per cluster, with $\mathbf L_{t,g}=\mathbf L_g(\mathbf I-\mathbf H_{\mu,g})$. GMRES solves each correction with $\mathbf J$'s $6\times6$ diagonal blocks inverted (FT-7) on the right, so it still measures the residual of $\mathbf J\Delta\boldsymbol\varepsilon=-\mathbf r$ and the preconditioner changes only the iteration count.

**(FT-7) Block-Jacobi preconditioner**

Write FT-6 as $\mathbf J=\mathbf I+\mathbf G\mathbf T$, with $\mathbf G$ the strain change $\boldsymbol\Pi\mathbf m$ of FT-2 with FT-4, from cluster polarization to cluster strain. Since $\mathbf T$ is one block per cluster,

$$
\mathbf J_{pp}=\mathbf I+\mathbf G_{pp}\mathbf T_p,
\qquad
\mathbf G_{pp}=c_p\,\bar{\mathbf D}_0[\mathbf 0]+\mathbf C_{pp}-\frac1M\sum_s c_s\,\mathbf C_{sp},
$$

the partition FFT's part adding nothing to FT-4's cell mean ($\hat{\mathbf P}_0(\mathbf 0)=\mathbf 0$). $\mathbf G_{pp}$ depends on the geometry only and is built once; each Newton correction inverts one $6\times6$ block per cluster.

CG does not apply here: the truncated interaction, weighted by volume, is not symmetric and its symmetric part not semidefinite (section 5.5's patch-test failure), and even at full reach $\mathbf J=\mathbf I+\mathbf G\mathbf T$ is self-adjoint only in inner products built from $\mathbf T$ or $\mathbf G^{-1}$, where $\mathbf T$ is zero in elastic matrix clusters and negative in yielding ones and $\mathbf G$ is singular on uniform fields. Block-Jacobi holds it to 10–20 GMRES iterations per correction, well inside one restart, so GMRES's orthogonalization costs little.

### 5.4 Relation to TFA and SCA

FFT-TFA and TFA share the clusters, the eigenstrains and the update at the cluster's average strain; they differ in the operator by which the clusters interact, which is where SCA and TFA differ (Schneider 2019, CMAME 354, §1 and §4: the two schemes are identical up to one Eshelby–Green-type operator, clustered for SCA). Here that operator is SCA's cluster-averaged LS operator [SCA, Eqs. (16)–(18)] at a fixed reference medium, the matrix, applied by a partition FFT with near corrections instead of a stored dense matrix.

Precisely, the LS equation $\boldsymbol\varepsilon=\bar{\boldsymbol\varepsilon}-\boldsymbol\Gamma_0*\boldsymbol\tau$, with $\boldsymbol\tau=(\mathbf L-\mathbf C_0)\boldsymbol\varepsilon-\mathbf L\boldsymbol\mu$, is solved at three resolutions. TFA resolves the strain at the fine scale (FE), so $\boldsymbol\tau$ varies inside every cluster. SCA and FFT-TFA take the strain, hence $\boldsymbol\tau$, uniform on each cluster and keep only each cluster's average response, $\boldsymbol\Gamma_0$ averaged over the clusters (Schneider's clustered operator). Spectral LS is the case where every voxel is its own cluster. Averaging costs nothing where $\boldsymbol\tau$ is uniform anyway: with $\mathbf C_0=\mathbf L_m$ a matrix cluster's $\boldsymbol\tau$ is $-\mathbf L_m\boldsymbol\mu_g$, independent of the strain, so FFT-TFA and TFA agree exactly in the matrix; an inclusion's $\boldsymbol\tau$ follows its strain, which varies, so there the averaging is an approximation. With any other $\mathbf C_0$ the term $(\mathbf L-\mathbf C_0)\boldsymbol\varepsilon$ is present in every cluster, which is why SCA's answer depends on $\mathbf C_0$ and FFT-TFA's choice confines the averaging error to the inclusion.

At full reach FFT-TFA's interaction is the exact cluster-averaged operator (section 5.6), so FFT-TFA is SCA's equation with $\mathbf C_0=\mathbf L_m$. Its difference from TFA on the same clusters therefore splits into three parts, each measurable: the inclusion's $\boldsymbol\tau$ uniform per cluster (FFT-TFA at full reach against TFA), the truncation (FFT-TFA against itself at full reach), and the Moulinec–Suquet kernel against TFA's FE operators. On a laminate the inclusion's exact strain is uniform, so the first part vanishes; this is why FFT-TFA at full reach must match the laminates' exact solutions.

Zhang et al. (2019, Comput. Mech. 64, §2.3) approximate cluster interactions on a uniform coarse grid as well, by volume-fraction-weighted cell–cell integrals with the cross terms inside a cell set to zero; they assemble a dense cluster matrix from them, with neither FFT nor corrections.

### 5.5 Measured limits

Measured on `check.py`'s problems (27² elements, 9² partitions; 12³ elements, 4³ partitions) unless stated.

- **Uniform polarization (patch test).** A polarization uniform over the whole cell gives no strain change in LS-5; FT-2 gives one, up to 16 % of the eigenstrain at reach 0 and 1.7 % at reach 1 on the fiber, exact only where each partition holds one cluster. FT-3 corrects, near each partition, also how the response to the partitions' *mean* polarization varies over its clusters; the far partitions' share of that variation, which cancels it for a uniform field, is left out. A consistent variant corrects only each partition's deviation from its mean, $\mathbf C_{pt}=\mathbf D_{pt}-c_t\sum_{t'\in B}\mathbf D_{pt'}$: it passes the patch test exactly and leaves the cut laminate as it is, but drops the variation of the mean field's strain over a partition's clusters, which is real. At the same cost and stopping accuracy (126 fibers on 180², 540² Spectral LS; sphere on 48³, 96³ Spectral LS): at reach 0 the two agree within 0.3 points; at reach 1 the variant is worse everywhere (126 fibers, 45² partitions, 4 matrix clusters each, macroscopic: 1.95 % against 1.17 %; sphere 12³: 1.83 % against 1.17 %). Not adopted.
- **Cut laminate.** The cut partitions line up along the layer, so their far-field error adds up instead of decaying: 25 % in the worst cluster at reach 0, 15 % at reach 1, 9 % at reach 2, exact at full reach (macroscopic 3.1 %).
- **Neighbour blocks need not help.** The truncated operator's eigenvalues leave $[0,1]$ (the exact one's range) further at reach 1: by up to 0.04 against 0.01 at reach 0.
- **k-means clusters of a few elements.** With clusters of 4–17 elements, a cluster on the verge of yielding converges elastic at one tolerance and plastic at another (9 % in one cluster), with TFA's exact operator too (14 %): the clustered model has several solutions there, not the interaction.
- **σ33 along z-invariant fibers** runs low: one polarization per inclusion cluster. Split by the decomposition of section 5.4 on a 30-fiber cell (128² elements, 8² partitions, 8 matrix and 1 fiber cluster per partition, $\varepsilon_{33}$ + $\gamma_{13}$ to peak, against 512² Spectral LS): σ33 is −8.75 % at reach 0, −5.47 % at reach 1, −5.97 % at full reach (SCA's equation, $\mathbf C_0=\mathbf L_m$), −0.10 % for TFA, and −0.32 % for Spectral LS on the same 128² elements. So the truncation beyond the neighbours does not cause it, the kernel accounts for at most 0.3 points, and the inclusion's polarization uniform per cluster accounts for the rest. In σ13 the same averaging offsets TFA's stiff material-state error (TFA +16.2 %, FFT-TFA at full reach +7.1 %), an error cancellation.

### 5.6 Checks

`check.py` runs FFT-TFA on every problem: on the aligned laminates it must match the exact solution (the cut laminate is reported, section 5.5), and at full reach it must match it on every laminate, which separates the truncation, a model error, from the law and the Newton solve. Also at full reach it must match it with each partition's matrix part cut into two clusters. On every problem its interaction at full reach must equal LS-5 averaged over its clusters from the elements (`interaction_check_tolerance`). On the fiber and the sphere every method must stop within `stopping_accuracy_tolerance` of its own solution at $10^{-11}$.

## 6. Open points

- **Convergence in $M$.** Schneider (2019) proves convergence of clustered Lippmann–Schwinger equations as clusters are refined, for hardening materials only. FFT-TFA's own open points are in the README's roadmap.
- **The partition kernel is not the exact partition average.** It can be computed from the partition lattice alone, as a weighted sum over aliases: `1_research_context.md`, sections 2–3.

## Paper references

- **[MS]** Moulinec, H.; Suquet, P. (1998), "A numerical method for computing the overall response of nonlinear composites with complex microstructure," *Computer Methods in Applied Mechanics and Engineering* 157, 69-94. [Paper](https://doi.org/10.1016/S0045-7825(97)00218-1). The reference split, the periodic Lippmann–Schwinger equation, and elastoplastic local updates.
- **[SCA]** Liu, Z.; Bessa, M. A.; Liu, W. K. (2016), "Self-consistent clustering analysis: an efficient multi-scale scheme for inelastic heterogeneous materials." [Author preprint](https://zeliangliu.com/publication/liu-2016-self/preprint.pdf). Eqs. (16)-(18): cluster-averaged reference interactions and constitutive closure; Section 2.2.2: offline Fourier construction.

The reading notes in `references/reading_notes/` hold the others cited here.
