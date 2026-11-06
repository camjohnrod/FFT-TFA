# The Lippmann–Schwinger equation and Spectral LS

## Purpose and route

We want a nonlinear homogenization model with a few material states (clusters), whose interaction costs $O(M\log M)$ per application with $M$ partitions, and stores no dense interaction matrix. TFA (`3_tfa_baseline.md`) has the material states but a dense $\mathbf P$: shifting a source and a receiver changes their heterogeneous surroundings, so $\mathbf P$ is no convolution. The Lippmann–Schwinger (LS) equation of a *homogeneous* reference medium is one, on any regular lattice.

Section 1 states the setting; section 2 derives the exact LS equation; section 3 builds its kernel on the elements and averages it over partitions; section 4 solves it on the elements, which is **Spectral LS**. **FFT-TFA**, the method under development, uses the same equation and kernel; its derivation is `0_fft_tfa_lecture.md` (equations `FT-`). Equation labels here use `LS-`; the labels are not consecutive, since those of the earlier method CP (git tag `pre-cleanup-2`) are not used here.

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

## 5. FFT-TFA

FFT-TFA, the method under development, is derived in `0_fft_tfa_lecture.md`, which holds equations FT-1 to FT-7: TFA's
clusters interacting through LS-5 with $\mathbf C_0=\mathbf L_m$, averaged over the clusters, applied by an FFT on the
partitions (sections 3.2–3.3) plus exact corrections between the clusters of nearby partitions.

## 6. Open points

- **Convergence in $M$.** Schneider (2019) proves convergence of clustered Lippmann–Schwinger equations as clusters are refined, for hardening materials only. FFT-TFA's own open points are in the README's roadmap.
- **The partition kernel is not the exact partition average.** It can be computed from the partition lattice alone, as a weighted sum over aliases: `1_research_context.md`, sections 2–3.

## Paper references

- **[MS]** Moulinec, H.; Suquet, P. (1998), "A numerical method for computing the overall response of nonlinear composites with complex microstructure," *Computer Methods in Applied Mechanics and Engineering* 157, 69-94. [Paper](https://doi.org/10.1016/S0045-7825(97)00218-1). The reference split, the periodic Lippmann–Schwinger equation, and elastoplastic local updates.
- **[SCA]** Liu, Z.; Bessa, M. A.; Liu, W. K. (2016), "Self-consistent clustering analysis: an efficient multi-scale scheme for inelastic heterogeneous materials." [Author preprint](https://zeliangliu.com/publication/liu-2016-self/preprint.pdf). Eqs. (16)-(18): cluster-averaged reference interactions and constitutive closure; Section 2.2.2: offline Fourier construction.

The reading notes in `references/reading_notes/` hold the others cited here.
