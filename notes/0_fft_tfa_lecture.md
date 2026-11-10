# FFT-TFA: the theory behind the code

**Goal.** Predict the stress–strain response of an elastoplastic composite with a handful of material states, as TFA
does, but compute how those states interact with an FFT on a coarse grid instead of with a dense matrix.

**Reason.** TFA's dense matrix makes memory and cost per iteration grow with the *square* of the number of material
states. An FFT on a grid of $N_{\text{part}}$ blocks costs $O(N_{\text{part}}\log N_{\text{part}})$ and stores
$O(N_{\text{part}})$, times a factor set by how finely each block is described (section 7), whatever the number of
material states inside the blocks. That is what lets the model be refined.

The lecture derives the method step by step and points to the code for each step. It assumes small-strain elasticity
and knowing what an FFT does. The labels LS-\*, FT-\*, ER-\* are the ones the code's comments cite. Every figure is
computed by the code (`notes/figures/make_figures.py`).

**Contents.** 1 The problem · 2 Three grids · 3 The Lippmann–Schwinger equation · 4 TFA and its cost · 5 The matrix as
reference medium · 6 The clustered equation · 7 A fast interaction · 8 Solving a load step · 9 The methods side by side
· 10 Known limits · Appendix

### Notation

**Where things live.** One letter at three scales: element $\alpha$ lies in cluster $a$, which lies in partition $A$
(and $\beta$ in $b$ in $B$). So Greek $\alpha, \beta$ are **elements**, lowercase $a, b$ are **clusters**, and capital
$A, B$ are **partitions**. Every field is written as a function of where it lives: $\boldsymbol\varepsilon(\alpha)$ is
the strain of element $\alpha$, $\boldsymbol\varepsilon(a)$ the strain of cluster $a$, $\boldsymbol\tau(B)$ the mean
polarization of partition $B$; in section 7 also of which polynomial, $\boldsymbol\tau(B,\psi)$. Plain $x, y$ are
coordinates: points of the cell in the continuous equation LS-5, positions inside a partition in section 7.

**One interaction symbol.** $\boldsymbol\Gamma(\text{target}\leftarrow\text{source})$ is the strain at the target per
unit polarization at the source, averaged over the target and summed over the source. The same symbol serves between
elements, $\boldsymbol\Gamma(\alpha\leftarrow\beta)$; between clusters, $\boldsymbol\Gamma(a\leftarrow b)$; and between
partitions, $\boldsymbol\Gamma(A\leftarrow B)$.

**Labels.** A capital subscript is a name, not an index: $M$ matrix, $I$ inclusion, $0$ reference, $T$ tangent, $P$
plastic. A word as a subscript is a name too: $\boldsymbol\Gamma_{\text{far}}$, $\boldsymbol\Gamma_{\text{corr}}$.

| Symbol | Meaning |
|---|---|
| $\boldsymbol\varepsilon,\ \boldsymbol\sigma$ | strain, stress (Voigt vectors of 6, engineering shear $\gamma_{12}=2\varepsilon_{12}$) |
| $\boldsymbol\varepsilon_P$ | plastic strain |
| $\boldsymbol\tau$ | polarization: the stress the reference medium gets wrong (section 3) |
| $\bar{\boldsymbol\varepsilon},\ \bar{\boldsymbol\sigma}$ | macro strain and macro stress: cell averages, $\langle\cdot\rangle$ |
| $\mathbf L,\ \mathbf L_M,\ \mathbf L_I$ | elastic stiffness: local, of the matrix, of the inclusion |
| $\mathbf L_0,\ \mathbf L_T$ | stiffness of the reference medium; tangent stiffness $d\boldsymbol\sigma/d\boldsymbol\varepsilon$ |
| $\lvert a\rvert,\ \lvert A\rvert$ | number of elements of cluster $a$, of partition $A$ (the same for every partition) |
| $v(a)$ | volume fraction of cluster $a$ in its partition, $\lvert a\rvert/\lvert A\rvert$ |
| $\varphi,\ \psi$ | polynomials on a partition (section 7): $\varphi$ at the target, $\psi$ at the source |
| $N_{\text{elem}},\ N_{\text{part}},\ N_{\text{clust}},\ N_{\text{poly}}$ | numbers of elements, partitions, clusters, polynomials |

---

## 1. The problem

A composite: stiff elastic inclusions (fibers, a sphere, a layer) in a softer matrix that yields plastically. We study
one periodic cubic cell under a **macro strain** $\bar{\boldsymbol\varepsilon}$ that follows a load path, and want
the **macro stress** $\bar{\boldsymbol\sigma}=\langle\boldsymbol\sigma\rangle$ along it. In the cell:

$$
\large
\begin{aligned}
\text{equilibrium:}\quad & \nabla\cdot\boldsymbol\sigma=\mathbf 0\\[2pt]
\text{compatibility:}\quad & \boldsymbol\varepsilon=\bar{\boldsymbol\varepsilon}+\nabla^s\mathbf u,\qquad \mathbf u\ \text{periodic}\\[2pt]
\text{material law:}\quad & \boldsymbol\sigma=\mathbf L\,(\boldsymbol\varepsilon-\boldsymbol\varepsilon_P)
\end{aligned}
$$

$\mathbf L$ is $\mathbf L_M$ in the matrix and $\mathbf L_I$ in the inclusion. The plastic strain
$\boldsymbol\varepsilon_P$ is zero in the elastic inclusion.

**The material law is a black box** (`material.py`; derivation in `notes/background_influence_functions_and_J2/`).
At each load step, given a strain and the plastic history of the previous step, the radial return gives the plastic
strain and the stress, and its derivative gives the tangent:

$$
\large
\boldsymbol\varepsilon\ \longmapsto\ \boldsymbol\varepsilon_P(\boldsymbol\varepsilon),\ \ \boldsymbol\sigma(\boldsymbol\varepsilon),
\qquad\qquad
\mathbf L_T=\frac{d\boldsymbol\sigma}{d\boldsymbol\varepsilon}=\mathbf L\Big(\mathbf I-\frac{d\boldsymbol\varepsilon_P}{d\boldsymbol\varepsilon}\Big)
$$

The rest of the lecture is about the other half: how the material points interact through equilibrium.

> **Why a reduced model.** A full-field solve has unknowns at every element, $10^4$–$10^6$ of them, at every Newton
> iteration of every load step. A reduced model keeps a few material states instead.

---

## 2. Three grids

![The three grids](figures/grids.png)

| Grid | What it is | What it decides |
|---|---|---|
| **Elements** $\alpha$ | a regular mesh that resolves the geometry; each element is matrix or inclusion (its **phase**) | geometry |
| **Partitions** $A$ | a regular lattice of blocks of $k\times k\times k$ elements ($k\times k$ columns if the geometry is constant through the thickness) | speed: they carry the FFT |
| **Clusters** $a$ | elements of one phase inside one partition, sharing **one material state**: one strain, stress and plastic strain | accuracy |

A partition holds one cluster per phase, or more, found by k-means on each element's elastic strain concentration
(`cell.cluster_by_k_means`, as in SCA: Liu–Bessa–Liu 2016, §2.2.1). Each cluster lies inside one partition: that is
what connects the two coarse grids. Spectral LS, the comparison method, is the case of one element per partition and
per cluster.

---

## 3. The Lippmann–Schwinger equation

> **Why.** In a heterogeneous body, how one point responds to another depends on everything around them. In a
> *homogeneous* periodic body, it depends only on the vector between them: the response is a convolution, which an FFT
> turns into a product. So we move the heterogeneity into a source term.

#### Step 1 · Pick a homogeneous reference stiffness $\mathbf L_0$, and call the stress it gets wrong the polarization

$$
\large
\boldsymbol\tau=\boldsymbol\sigma-\mathbf L_0\boldsymbol\varepsilon
=\underbrace{(\mathbf L-\mathbf L_0)\,\boldsymbol\varepsilon}_{\text{elastic contrast}}\ \underbrace{-\ \mathbf L\,\boldsymbol\varepsilon_P}_{\text{plasticity}}
\tag{LS-4}
$$

This is an identity, not an approximation.

#### Step 2 · Substitute into equilibrium

$$
\large
\nabla\cdot\big(\mathbf L_0\,\nabla^s\mathbf u\big)=-\nabla\cdot\boldsymbol\tau
$$

This is the homogeneous medium $\mathbf L_0$, loaded by the polarization.

#### Step 3 · Solve the homogeneous problem with its Green operator

$\boldsymbol\Gamma(x\leftarrow y)$ is the strain at $x$ per unit polarization at $y$, in the reference medium. In a
homogeneous periodic medium it depends only on $x-y$. Then

$$
\large
\boxed{\ \boldsymbol\varepsilon(x)=\bar{\boldsymbol\varepsilon}-\int\boldsymbol\Gamma(x\leftarrow y)\,\boldsymbol\tau(y)\,dy\ }
\tag{LS-5}
$$

This is the **Lippmann–Schwinger equation** (Moulinec–Suquet 1998, eq. 7).

- It is **exact**, for any $\mathbf L_0$.
- It is an **equation**, not a formula: $\boldsymbol\tau$ depends on $\boldsymbol\varepsilon$. We solve it by Newton.
- The integral is a **convolution**. In Fourier space it becomes one $6\times6$ matrix per frequency, known in closed
  form (Moulinec–Suquet 1998, eq. 4; `ls_model.get_moulinec_suquet_kernel`). At zero frequency it is zero, which keeps
  $\langle\boldsymbol\varepsilon\rangle=\bar{\boldsymbol\varepsilon}$.

#### On the elements

With $\boldsymbol\tau$ uniform on each element, the integral becomes a sum over elements:

$$
\large
\boldsymbol\varepsilon(\alpha)=\bar{\boldsymbol\varepsilon}-\sum_\beta\boldsymbol\Gamma(\alpha\leftarrow\beta)\,\boldsymbol\tau(\beta)
$$

Applying it costs one FFT, a $6\times6$ product per frequency and one inverse FFT:
$O(N_{\text{elem}}\log N_{\text{elem}})$ (LS-17, `ls_model.apply_on_lattice`). Solving it with one material state
per element is **Spectral LS**, Moulinec–Suquet's method (`ls_model.SpectralLSSolver`; `notes/2_ls_model.md`
section 4).

---

## 4. Fewer material states: TFA, and its cost

> **Why.** Spectral LS keeps one material state per element. TFA keeps one per cluster: one plastic strain
> $\boldsymbol\varepsilon_P(a)$, updated at the cluster's average strain $\boldsymbol\varepsilon(a)$. The strain itself
> still varies inside a cluster.

From a finite-element model of the composite, TFA precomputes exact influence functions $\mathbf E$ and $\mathbf P$
(`notes/3_tfa_baseline.md`):

$$
\large
\boldsymbol\varepsilon(a)=\mathbf E(a)\,\bar{\boldsymbol\varepsilon}+\sum_b\mathbf P(a\leftarrow b)\,\boldsymbol\varepsilon_P(b)
\tag{ER-4}
$$

This is exact in elasticity. But $\mathbf P$ is **dense**: $(6N_{\text{clust}})^2$ numbers, every product touches all
of them, and building it takes $6N_{\text{clust}}$ FE solves. It cannot be a convolution: moving a pair of clusters
changes what lies around them.

**FFT-TFA's plan:** keep TFA's material model (clusters, one plastic strain each, updated at the cluster's average
strain), and build the interaction from LS-5, whose kernel is a convolution whatever the geometry.

---

## 5. The key choice: the matrix as reference medium

> **Why.** To average LS-5 over clusters, we want $\boldsymbol\tau$ uniform on each cluster. The choice of
> $\mathbf L_0$ decides how true that is.

Take $\mathbf L_0=\mathbf L_M$ in LS-4:

$$
\large
\begin{aligned}
\text{matrix:}\quad & \boldsymbol\tau=-\mathbf L_M\,\boldsymbol\varepsilon_P
&& \text{one plastic strain per cluster}\ \Rightarrow\ \text{uniform, exactly}\\[4pt]
\text{inclusion:}\quad & \boldsymbol\tau=(\mathbf L_I-\mathbf L_M)\,\boldsymbol\varepsilon
&& \text{follows the strain}\ \Rightarrow\ \text{approximated as uniform}
\end{aligned}
$$

The matrix's polarization is exact, whatever the cluster's shape. The inclusion's is taken uniform on each cluster,
at the cluster's average strain: the method's one approximation on the material side. A yielding inclusion would add
$-\mathbf L_I\boldsymbol\varepsilon_P$, uniform like the matrix's.

![Polarization with the matrix as reference medium](figures/polarization.png)

*A fiber under $\bar\varepsilon_{11}$ and $\bar\gamma_{12}$ (Spectral LS on 64² elements,
$\boldsymbol\tau=\boldsymbol\sigma-\mathbf L_M\boldsymbol\varepsilon$). While elastic, the matrix has no polarization
at all; once it yields, its polarization is its plastic strain. The inclusion's polarization varies with its strain.
The speckle at the interface is the known ringing of the Moulinec–Suquet discretization.*

So each cluster has one unknown, its strain $\boldsymbol\varepsilon(a)$. With $\mathbf L(a)$ the stiffness of its
phase:

$$
\large
\boxed{\
\boldsymbol\tau(a)=\big(\mathbf L(a)-\mathbf L_M\big)\,\boldsymbol\varepsilon(a)-\mathbf L(a)\,\boldsymbol\varepsilon_P(a),
\qquad
\boldsymbol\sigma(a)=\mathbf L(a)\big(\boldsymbol\varepsilon(a)-\boldsymbol\varepsilon_P(a)\big)\ }
\tag{FT-1}
$$

with $\boldsymbol\varepsilon_P(a)$ from the radial return at $\boldsymbol\varepsilon(a)$. With any other
$\mathbf L_0$ the strain term would appear in the matrix too, and the matrix would be approximated as well.

---

## 6. Averaging over clusters: the clustered equation

Insert a polarization uniform on each cluster into LS-5, and average over cluster $a$:

$$
\large
\boldsymbol\varepsilon(a)=\bar{\boldsymbol\varepsilon}-\sum_b\boldsymbol\Gamma(a\leftarrow b)\,\boldsymbol\tau(b),
\qquad
\boldsymbol\Gamma(a\leftarrow b)=\text{mean over }\alpha\text{ in }a\text{ of }\sum_{\beta\text{ in }b}\boldsymbol\Gamma(\alpha\leftarrow\beta)
$$

This has the same form as LS-5, one level up: clusters instead of elements. Given FT-1, it is exact. It is SCA's
clustered equation (Liu–Bessa–Liu 2016, eqs. 16–17) with $\mathbf L_0=\mathbf L_M$. Schneider (2019, §4) compares it
with TFA: the two differ only in whether the Green operator is averaged over the clusters. TFA leaves the strain free
inside a cluster; SCA and FFT-TFA take it uniform.

> **The problem.** $\boldsymbol\Gamma(a\leftarrow b)$ is a dense table of $6\times6$ blocks, one per pair of clusters:
> no cheaper than TFA's $\mathbf P$. Section 7 is the actual idea of FFT-TFA.

---

## 7. A fast interaction: polynomial moments by FFT, exact inside each partition

**Goal.** Replace the dense table $\boldsymbol\Gamma(a\leftarrow b)$ of section 6 by something an FFT can apply, and
that becomes exact as one setting, the **degree**, is raised.

**Reason.** Partitions are translated copies of one block. Anything that depends only on the offset between two
partitions is a convolution on the partitions, which one FFT applies. $\boldsymbol\Gamma(a\leftarrow b)$ is not of that
form, since clusters have different shapes in different partitions. So we describe every partition's polarization with
the same few functions.

#### Step 1 · Choose the same functions on every partition: polynomials

Take the polynomials in the element positions up to a total degree (1, $x$, $y$, $xy$, $x^2$, …), made orthonormal over
a partition's elements:

$$
\large
\sum_{\alpha\text{ in }A}\varphi(\alpha)\,\psi(\alpha)=
\begin{cases}1 & \varphi=\psi\\ 0 & \varphi\neq\psi\end{cases}
$$

$\varphi(\alpha)$ is the polynomial's value at $\alpha$'s position *inside its own partition*, so every partition has
the same polynomials. The first is the constant, $1/\sqrt{\lvert A\rvert}$ (`fft_tfa_model.get_polynomial_basis`: the
elements lie on a grid, so products of orthonormal polynomials along each axis are orthonormal). Their number
$N_{\text{poly}}$ is 1, 3, 6, 10 for degrees 0–3 in 2D, and 1, 4, 10, 20 in 3D.

#### Step 2 · Summarize each partition's polarization by its moments

Define the **moment** of partition $B$'s polarization along $\psi$:

$$
\large
\boldsymbol\tau(B,\psi)=\sum_{\beta\text{ in }B}\psi(\beta)\,\boldsymbol\tau(\beta)
$$

Since the polynomials are orthonormal, the polynomial that fits $\boldsymbol\tau$ best on $B$ (least squares) is
$\sum_\psi\psi(\beta)\,\boldsymbol\tau(B,\psi)$: the **summary** of $B$'s polarization. With $\boldsymbol\tau$ uniform
on each cluster, the moment is a sum over $B$'s clusters:

$$
\large
\boldsymbol\tau(B,\psi)=\sum_{b\text{ in }B}m(b,\psi)\,\boldsymbol\tau(b),
\qquad
m(b,\psi)=\sum_{\beta\text{ in }b}\psi(\beta)
$$

$m(b,\psi)$ is the moment of cluster $b$ itself: a number fixed by the geometry, computed once
(`fft_tfa_model.get_cluster_moments`).

![A partition's polarization and its summaries](figures/summary.png)

*FFT-TFA's polarization, its shear component, at the end of the load of figure 2 (4² partitions, two matrix clusters
per partition's matrix part, solved at degree 1), and each partition's summary. Degree 0 keeps each partition's mean,
higher degrees follow the field across it. A step at the fiber's edge needs many polynomials: at degree 3 the summary
still overshoots there.*

#### Step 3 · Let the summaries interact

Put every partition's summary into LS-5 on the elements. The strain change it causes at the elements of partition $A$,
$\Delta\boldsymbol\varepsilon(\alpha)$, has moments defined like the polarization's,
$\Delta\boldsymbol\varepsilon(A,\varphi)=\sum_{\alpha\text{ in }A}\varphi(\alpha)\,\Delta\boldsymbol\varepsilon(\alpha)$:

$$
\large
\Delta\boldsymbol\varepsilon(A,\varphi)=\sum_B\sum_\psi\boldsymbol\Gamma(A,\varphi\leftarrow B,\psi)\,\boldsymbol\tau(B,\psi),
\qquad
\boldsymbol\Gamma(A,\varphi\leftarrow B,\psi)=\sum_{\alpha\text{ in }A}\ \sum_{\beta\text{ in }B}\varphi(\alpha)\,\boldsymbol\Gamma(\alpha\leftarrow\beta)\,\psi(\beta)
$$

$\boldsymbol\Gamma(A,\varphi\leftarrow B,\psi)$ is the strain in $A$ along $\varphi$ per unit polarization in $B$
along $\psi$: the **moment kernel**.

Why summarize the strain too: the strain at every element would need the FFT on the elements, which is what we avoid.
A cluster needs only its average strain. Using the same functions for the polarization and for reading the strain is a
Galerkin approximation, as in Brisard–Dormieux (2012, §4.1–4.2), whose functions are constants on each cell.

#### Step 4 · Read each cluster's strain from its partition's summary

$A$'s strain change is summarized as $\sum_\varphi\varphi(\alpha)\,\Delta\boldsymbol\varepsilon(A,\varphi)$, and
cluster $a$ takes its average over $a$'s elements:

$$
\large
\Delta\boldsymbol\varepsilon_{\text{far}}(a)=\sum_\varphi\frac{m(a,\varphi)}{\lvert a\rvert}\,\Delta\boldsymbol\varepsilon(A,\varphi)
$$

#### Step 5 · It is a convolution

$\boldsymbol\Gamma(\alpha\leftarrow\beta)$ depends only on the offset between the elements, and the polynomials are the
same in every partition, so $\boldsymbol\Gamma(A,\varphi\leftarrow B,\psi)$ depends only on the offset between $A$ and
$B$. The sum over $B$ in step 3 is then a convolution on the partitions: one FFT of every partition's moments, one
$6N_{\text{poly}}\times6N_{\text{poly}}$ block product per frequency, one inverse FFT (`ls_model.apply_on_lattice`). Steps 2–4 together give the **far field**:

$$
\large
\boxed{\
\Delta\boldsymbol\varepsilon_{\text{far}}(a)=\sum_b\boldsymbol\Gamma_{\text{far}}(a\leftarrow b)\,\boldsymbol\tau(b),
\qquad
\boldsymbol\Gamma_{\text{far}}(a\leftarrow b)=\sum_\varphi\sum_\psi\frac{m(a,\varphi)}{\lvert a\rvert}\,\boldsymbol\Gamma(A,\varphi\leftarrow B,\psi)\,m(b,\psi)\ }
\tag{FT-2}
$$

$\boldsymbol\Gamma_{\text{far}}(a\leftarrow b)$ is never formed. It names, pair by pair, what the FFT computes.

#### Step 6 · Keep the exact interaction inside each partition

How good is the far field? Compare it with the exact $\boldsymbol\Gamma(a\leftarrow b)$, pair by pair:

![The far field's error](figures/far_field.png)

*A fiber on 64² elements, 8² partitions, two matrix clusters per partition's matrix part. The error is a volume-weighted
rms over all cluster pairs at each distance, as a share of the exact interaction within a partition.*

Between partitions, the far field converges with the degree: one partition apart it is 8 % off at degree 0 and 0.3 % at
degree 4. Within a partition it is not good enough: 54 % off at degree 0, still 9 % at degree 4. There the kernel is
at its sharpest and the polarization steps at every interface, and polynomials follow neither (the overshoot in the
summary figure).

So, inside each partition, replace the far field's share by the exact interaction. The **correction** is what the far
field misses:

$$
\large
\boldsymbol\Gamma_{\text{corr}}(a\leftarrow b)=\boldsymbol\Gamma(a\leftarrow b)-\boldsymbol\Gamma_{\text{far}}(a\leftarrow b),
\qquad a\text{ and }b\text{ in the same partition}
\tag{FT-3}
$$

The split $\boldsymbol\Gamma=\boldsymbol\Gamma_{\text{far}}+\boldsymbol\Gamma_{\text{corr}}$ is exact for every pair.
The approximation is to drop the corrections between different partitions, and the degree controls how much that
drops.

#### Step 7 · Make a uniform polarization cause no strain

A polarization that is the same everywhere must cause no strain, since $\boldsymbol\Gamma$ is zero at zero frequency
(section 3). The far field passes this test exactly: the constant is one of the polynomials, so the summary of a uniform
polarization is exact, and $\boldsymbol\Gamma$ turns it into zero. The corrections of one partition do not: only their
sum over *all* partitions vanishes. So they act on the polarization's deviation from its cell mean,
$\boldsymbol\tau(b)-\langle\boldsymbol\tau\rangle$, with

$$
\large
\langle\boldsymbol\tau\rangle=\frac1{N_{\text{part}}}\sum_b v(b)\,\boldsymbol\tau(b)
$$

#### Step 8 · Keep the mean strain

The exact interaction changes no cell average, so $\langle\boldsymbol\varepsilon\rangle=\bar{\boldsymbol\varepsilon}$.
The far field keeps this; the corrections do not quite. So subtract the cell mean of the strain change.

#### Result · FFT-TFA's interaction (`fft_tfa_model.get_strain_change`)

$$
\large
\boxed{\
\begin{aligned}
\boldsymbol\varepsilon(a)&=\bar{\boldsymbol\varepsilon}-\big(\Delta\boldsymbol\varepsilon(a)-\langle\Delta\boldsymbol\varepsilon\rangle\big)\\[6pt]
\Delta\boldsymbol\varepsilon(a)&=\underbrace{\sum_b\boldsymbol\Gamma_{\text{far}}(a\leftarrow b)\,\boldsymbol\tau(b)}_{\text{far field, by FFT}}
\ +\underbrace{\sum_{b\text{ in }A}\boldsymbol\Gamma_{\text{corr}}(a\leftarrow b)\,\big(\boldsymbol\tau(b)-\langle\boldsymbol\tau\rangle\big)}_{\text{corrections inside }a\text{'s partition}}
\end{aligned}\ }
\tag{FT-4}
$$

We write this as
$\boldsymbol\varepsilon=\bar{\boldsymbol\varepsilon}-\boldsymbol\Gamma_{\text{FFT-TFA}}\,\boldsymbol\tau$:
$\boldsymbol\Gamma_{\text{FFT-TFA}}$ is FFT-TFA's approximation of $\boldsymbol\Gamma(a\leftarrow b)$.

#### Computing it offline (`fft_tfa_model.get_interaction`, once)

Both tables are sums of the element kernel, which depends only on the offset between two elements, so write it
$\boldsymbol\Gamma(\text{offset})$. For $\alpha$ in $A$ and $\beta$ in $B$, with $k$ elements per partition side:

$$
\large
\text{offset from }\beta\text{ to }\alpha\;=\;k\,\mathbf D+\boldsymbol\delta,
\qquad
\begin{aligned}
\mathbf D&=\text{offset from }B\text{ to }A,\ \text{in partitions}\\
\boldsymbol\delta&=\text{offset of }\alpha\text{ inside }A\text{ minus that of }\beta\text{ inside }B,\ \text{in elements}
\end{aligned}
$$

Grouping the element pairs by $\boldsymbol\delta$:

$$
\large
\begin{aligned}
\boldsymbol\Gamma(A,\varphi\leftarrow B,\psi)&=\sum_{\boldsymbol\delta}w(\varphi,\psi,\boldsymbol\delta)\ \boldsymbol\Gamma\big(k\mathbf D+\boldsymbol\delta\big),
&w(\varphi,\psi,\boldsymbol\delta)&=\sum_{\text{positions }\alpha,\beta\text{ with offset }\boldsymbol\delta}\varphi(\alpha)\,\psi(\beta)\\[6pt]
\boldsymbol\Gamma(a\leftarrow b)&=\frac{1}{\lvert a\rvert}\sum_{\boldsymbol\delta}\text{pairs}(a,b,\boldsymbol\delta)\ \boldsymbol\Gamma(\boldsymbol\delta),
&&a\text{ and }b\text{ in the same partition}
\end{aligned}
$$

The first gives the moment kernels at every offset $\mathbf D$ (`get_moment_kernels`). In the second,
$\text{pairs}(a,b,\boldsymbol\delta)$ counts the element pairs of $a$ and $b$ with offset $\boldsymbol\delta$; for every
offset at once, that is the cross-correlation of the two clusters' indicator arrays, one FFT padded to twice the
partition (`get_correction_blocks`, which subtracts FT-2's share to give FT-3).

**Memory.** The moment kernels hold one $6N_{\text{poly}}\times6N_{\text{poly}}$ block per frequency of the partition
grid, the corrections one $6\times6$ block per pair of clusters in the same partition. The kernels grow with
$N_{\text{poly}}^2$, not with the clusters: in 3D, degree 3 stores 25 times as much as degree 1.

#### Three checks of the derivation

- **Degree 0.** The one polynomial is the constant, and FT-2 becomes
  $\boldsymbol\Gamma_{\text{far}}(a\leftarrow b)=v(b)\,\boldsymbol\Gamma(A\leftarrow B)$, with
  $\boldsymbol\Gamma(A\leftarrow B)$ section 6's average one level up: each partition acts through its mean
  polarization, like Brisard and Dormieux's cells (2010, §3.2).
- **Full degree.** With $k$ elements per partition side, polynomials up to degree $k-1$ in each coordinate span every
  function on a partition's elements; their total degree is $2(k-1)$ in 2D, $3(k-1)$ in 3D (`degree = "full"`).
  The summaries are exact, $\boldsymbol\Gamma_{\text{far}}=\boldsymbol\Gamma(a\leftarrow b)$, the corrections vanish, and FT-4 is the exact
  clustered equation of section 6. `check.py` confirms this to round-off on every problem (`get_interaction_error`),
  and that a uniform polarization causes no strain at the problem's degree (`get_patch_test_error`).
- **One element per partition.** Every degree is full, and FFT-TFA is Spectral LS with $\mathbf L_0=\mathbf L_M$.

---

## 8. Solving a load step

Newton's method needs a residual, the linear system for a correction $\delta\boldsymbol\varepsilon$, and a
preconditioner for that system:

$$
\large
\begin{aligned}
\text{residual:}\quad & \mathbf r=\boldsymbol\varepsilon-\bar{\boldsymbol\varepsilon}+\boldsymbol\Gamma_{\text{FFT-TFA}}\,\boldsymbol\tau(\boldsymbol\varepsilon) && \text{(FT-5)}\\[6pt]
\text{Newton system:}\quad & \delta\boldsymbol\varepsilon+\boldsymbol\Gamma_{\text{FFT-TFA}}\,(\mathbf L_T-\mathbf L_M)\,\delta\boldsymbol\varepsilon=-\mathbf r && \text{(FT-6)}\\[6pt]
\text{preconditioner:}\quad & \text{invert each cluster's own }6\times6\text{ block}\quad \mathbf I+\boldsymbol\Gamma_{\text{FFT-TFA}}(a\leftarrow a)\,\big(\mathbf L_T(a)-\mathbf L_M\big) && \text{(FT-7)}
\end{aligned}
$$

- **FT-5** is a strain error, like TFA's. Convergence is judged on it directly: with $\mathbf L_0=\mathbf L_M$ it is
  already the "matrix units" measure that Spectral LS has to compute (LS-24, `notes/2_ls_model.md` section 4.3).
- **FT-6**: $\mathbf L_T-\mathbf L_M$ is how each cluster's polarization changes with its strain. The system is never
  assembled: each product is one FFT on the partitions and one sparse product for the corrections. It is not
  symmetric, so GMRES solves it.
- **FT-7** (block-Jacobi) cuts GMRES's iterations. Its blocks, $\boldsymbol\Gamma_{\text{FFT-TFA}}(a\leftarrow a)$, are
  each cluster's effect on itself: the far field's self-term, the cluster's own correction, less what the two cell
  means of steps 7 and 8 take back. They are built once (`fft_tfa_model.get_own_strain_change`).

**One load step** (`newton.Solver.solve_step`):
1. Start from the previous strains plus the macro strain increment.
2. Evaluate FT-1 and FT-5.
3. Solve FT-6 for $\delta\boldsymbol\varepsilon$ by GMRES, and step with a backtracking line search.
4. Repeat until the relative residual is below `convergence_tolerance`, then accept the plastic history.

---

## 9. The methods side by side

| | Material states | Uniform per state | Interaction | Stored | Cost per product |
|---|---|---|---|---|---|
| **FFT-TFA** | clusters | $\boldsymbol\varepsilon_P$, $\boldsymbol\varepsilon$ | polynomial moments by partition FFT + exact inside each partition | moment kernels on the partitions + blocks inside each partition | $O(N_{\text{poly}}N_{\text{part}}\log N_{\text{part}}+N_{\text{poly}}^2N_{\text{part}})$ + blocks |
| **Spectral LS** | elements | everything | element FFT | kernel and fields on the elements | $O(N_{\text{elem}}\log N_{\text{elem}})$ |
| **TFA** | clusters | $\boldsymbol\varepsilon_P$ | exact FE influence functions | dense, $(6N_{\text{clust}})^2$ | $O(N_{\text{clust}}^2)$ |
| SCA (not in the code) | clusters | $\boldsymbol\varepsilon_P$, $\boldsymbol\varepsilon$ | clustered LS operator | dense, $(6N_{\text{clust}})^2$ | $O(N_{\text{clust}}^2)$ |

**Where FFT-TFA's error against TFA on the same clusters comes from.** It has three parts, each measurable with the
code:

1. the inclusion's polarization taken uniform per cluster (FFT-TFA at full degree against TFA);
2. the far field's polynomial summary (FFT-TFA against itself at full degree);
3. the Moulinec–Suquet kernel against TFA's finite elements.

**Relation to the literature.**
- **Moulinec–Suquet (1998):** the reference-medium equation and its FFT solution.
- **SCA (Liu–Bessa–Liu 2016):** the clustered equation, with $\boldsymbol\Gamma(a\leftarrow b)$ built as a dense matrix
  on the fine grid (their eq. 27). They found a reference medium equal to the matrix too stiff once the matrix yields
  (§3.2, Fig. 5), and update it self-consistently at every increment. FFT-TFA keeps $\mathbf L_M$ fixed, which makes
  the matrix's polarization exact (section 5) and leaves one kernel, built once.
- **Brisard–Dormieux (2010, 2012):** the same regular cells, in linear elasticity, with a polarization constant on each
  cell, used both to approximate the polarization and to test the equation (2012, §4.1–4.2). FFT-TFA's far field uses
  polynomials up to a degree in place of constants, and adds clusters inside the cells, with the exact interaction
  among a cell's own clusters.
- **Zhang et al. (2019, §2.3):** cluster interactions on a uniform coarse grid too, by cell-to-cell integrals with the
  cross terms inside a cell set to zero, assembled into a dense matrix, with neither FFT nor corrections.
- Whether polynomial moments on an FFT grid appear elsewhere, for instance in fast methods for integral equations, is
  not yet checked.

---

## 10. Known limits

#### The far field converges with the degree, not with the partitions alone

At degree 0 each partition acts through its mean, and refining the partitions can stall: the far field's error between
neighbours stays near 8 % whatever their size (figure above). Raising the degree is what converges.

**Structures that cut through partitions in a line** show it most. A layer whose interfaces cut partitions 6 elements
across (`check.py`'s cut laminate, on 54² elements) has its worst cluster 24 % off at degree 0, 8.5 % at degree 1,
3.0 % at degree 3, 0.6 % at degree 4, and exact from degree 5. The polarization steps across the cut partitions, which
line up along the layer, so their errors add up instead of decaying. Degree 5 follows any profile across 6 elements.

#### Partitions should be smaller than the inclusions

With partitions about an inclusion across, the field around an inclusion reaches past its partition, and only a high
degree follows it. Partitions at most about half an inclusion across work best (README, "Status and roadmap").

#### A floor far away

Beyond two partitions, from degree 2 on, the far field's error levels off at a few hundredths of a percent instead of
falling with the degree (figure above). Our
reading, not verified: the element kernel varies from element to element even far from its source, the discrete
kernel's high-frequency content, and polynomials smooth that over.

#### Other limits

- **σ33 along z-invariant fibers runs low,** at full degree too. Of the three parts in section 9, this comes from
  part 1: one polarization per inclusion cluster.
- **Clusters of a few elements** can give the clustered problem several solutions (TFA's too).

---

## Appendix

#### The equations in the code

| Equation | Code |
|---|---|
| Material law, $\mathbf L_T$ | `material.radial_return`, `material.get_algorithmic_stiffness` |
| LS-5 on the elements: kernel and FFT product (LS-21, LS-17) | `ls_model.get_moulinec_suquet_kernel`, `ls_model.apply_on_lattice` |
| Polynomials $\varphi$, cluster moments $m(a,\varphi)$ | `fft_tfa_model.get_polynomial_basis`, `get_cluster_moments` |
| FT-1, FT-5: polarization, residual | `fft_tfa_model.FFTTFASolver.evaluate` |
| FT-2: moment kernels, far field | `fft_tfa_model.get_moment_kernels`, `get_strain_change` |
| FT-3: corrections | `fft_tfa_model.get_correction_blocks` |
| FT-4: interaction | `fft_tfa_model.get_strain_change`, built by `get_interaction` |
| FT-6, FT-7: Newton system, preconditioner | `fft_tfa_model.FFTTFASolver.get_newton_correction`, `get_own_strain_change` |
| Newton loop, line search, GMRES | `newton.Solver`, `newton.solve_by_gmres` |
| Clusters | `cell.cluster_by_phase`, `cell.cluster_by_k_means`, `cell.get_clusters` |

#### Names in the code and in the other notes

| Here | Code | `notes/2_ls_model.md`, `notes/3_tfa_baseline.md` |
|---|---|---|
| $\mathbf L_0$, $\mathbf L_M$ | `C0`, `matrix_L` | $\mathbf C_0$, $\mathbf L_m$ |
| $\boldsymbol\varepsilon_P$ | `plastic_strain` | $\boldsymbol\mu$, the eigenstrain |
| $v(a)$ | `Clusters.share` | $c_p$ |
| $\boldsymbol\Gamma(\alpha\leftarrow\beta)$, $\boldsymbol\Gamma(A\leftarrow B)$ | `element_Gamma`; the degree-0 moment kernel | $\mathbf D_0$, $\mathbf D_0^{BA}$ (LS-7) |
| $\varphi$, $\psi$; $N_{\text{poly}}$ | columns of `basis`; `p` | |
| $m(a,\varphi)$ | `moments` | |
| $\boldsymbol\Gamma(A,\varphi\leftarrow B,\psi)$ | `kernels`; in Fourier space `moment_kernels` | |
| $\boldsymbol\Gamma_{\text{corr}}(a\leftarrow b)$ | `correction_blocks` | |
| $\boldsymbol\Gamma_{\text{FFT-TFA}}$, its own blocks | `get_strain_change`, `own_strain_change` | |
