# Displacement influence functions

This note derives the fine-scale displacement representation used in Fish and Cui's Eq. (6). The starting point is the finite-element equilibrium problem for a periodic elastic RVE subjected to a prescribed macroscopic strain and a prescribed current eigenstrain field. The result expresses the displacement fluctuation as a superposition of the responses to those two inputs. This is the elastic foundation of the course's TFA plasticity framework; the current eigenstrain is treated as given only for this influence-function construction.

## Purpose and route

How can one elastic RVE solve be reused for many loading and eigenstrain states? We exploit linearity: derive constrained FE equilibrium, separate its two imposed inputs, solve for nodal fluctuations, and interpolate the response at an observation point. The influence functions will let Lecture 2 replace distributed eigenstrain by partition amplitudes; Lecture 3 will differentiate and average those responses to obtain ST-14.

## Setting and dimensions

Let $n_{\mathrm{dim}}\in\{2,3\}$ be the spatial dimension. Counting the independent components of a symmetric tensor determines the length of our strain and stress vectors:

**Equation (IF-0) - Number of independent strain components**

$$
n_\varepsilon
=\frac{n_{\mathrm{dim}}(n_{\mathrm{dim}}+1)}{2}
$$

Let $\Theta\subset\mathbb R^{n_{\mathrm{dim}}}$ be the RVE domain and let $d\Theta$ denote its volume element. The overbar in $\bar{\boldsymbol\varepsilon}$ identifies the prescribed macroscopic strain. Here $\mathbf y$ is an observation coordinate and $\widehat{\mathbf y}$ an eigenstrain source coordinate used for integration. The fields $\boldsymbol\varepsilon(\mathbf y)$, $\boldsymbol\sigma(\mathbf y)$, and $\boldsymbol\mu(\mathbf y)$ are the fine-scale strain, stress, and eigenstrain.

The strain vectors use engineering shear: in 3D their order is $(\varepsilon_{11},\varepsilon_{22},\varepsilon_{33},2\varepsilon_{12},2\varepsilon_{23},2\varepsilon_{13})$; stress vectors use $(\sigma_{11},\sigma_{22},\sigma_{33},\sigma_{12},\sigma_{23},\sigma_{13})$. In 2D use $(11,22,12)$ with the same shear convention and specify the elastic plane-stress or plane-strain reduction. The subsequent plasticity model is 3D. Strain and eigenstrain are dimensionless, stress and $\mathbf L$ have stress units, displacement has length units, and $\mathbf B$ has inverse-length units. The matrix $\mathbf L$ maps elastic strain to stress and $\mathbf B$ maps nodal displacements to strain; both are constructed below. For a virtual strain $\delta\boldsymbol\varepsilon$, these work-conjugate vectors make $\delta\boldsymbol\varepsilon^{\mathsf T}\boldsymbol\sigma$ the internal virtual-work density. A superscript $\mathrm{ten}$ identifies a tensor reconstructed from its vector; $\mathrm V$ identifies engineering-Voigt conversion when both representations appear together.

To distinguish imposed deformation from the unknown periodic fluctuation, write the local total displacement $\mathbf u^{\mathrm{tot}}$ using the tensor form $\bar{\boldsymbol\varepsilon}^{\mathrm{ten}}$ of the prescribed macroscopic strain.

**Equation (IF-K1) - Affine displacement and periodic fluctuation**

$$
\mathbf u^{\mathrm{tot}}(\mathbf y)
=\bar{\boldsymbol\varepsilon}^{\mathrm{ten}}\mathbf y+\mathbf u(\mathbf y)
$$

An irrelevant macroscopic translation is omitted. The unknown $\mathbf u$ is periodic; the total displacement need not be periodic.

To construct the discrete fluctuation, let $N_N$ be the number of nodes, $N_I$ a scalar nodal shape function, and $\mathbf d_I$ the nodal fluctuation vector. Nodal values are related by the periodic and normalization constraints.

**Equation (IF-K2) - Explicit nodal interpolation**

$$
\mathbf u(\mathbf y)=\sum_{I=1}^{N_N}N_I(\mathbf y)\mathbf d_I
$$

Let $N_d$ be the number of independent periodic displacement-fluctuation degrees of freedom after the zero-mean constraint is imposed. At an observation point $\mathbf y\in\Theta$,

**Equation (IF-4) - Constrained FE interpolation**

$$
\mathbf u(\mathbf y)
=\mathbf N(\mathbf y)\mathbf d
$$

The interpolation reconstructs a spatial displacement from independent DOFs, which fixes the following matrix sizes.

**Equation (IF-D1) - Displacement and interpolation sizes**

$$
\mathbf u(\mathbf y)\in\mathbb R^{n_{\mathrm{dim}}},
\qquad
\mathbf N(\mathbf y)\in\mathbb R^{n_{\mathrm{dim}}\times N_d},
\qquad
\mathbf d\in\mathbb R^{N_d}
$$

The stress and strain vectors require one entry per independent tensor component:

**Equation (IF-D2) - Stress and strain vector sizes**

$$
\bar{\boldsymbol\varepsilon},
\boldsymbol\varepsilon(\mathbf y),
\boldsymbol\sigma(\mathbf y),
\boldsymbol\mu(\widehat{\mathbf y})
\in\mathbb R^{n_\varepsilon}
$$

The stiffness acts on displacement DOFs, while the coarse-strain load operator has one column per imposed strain component.

**Equation (IF-D3) - Global stiffness and load-operator sizes**

$$
\mathbf K\in\mathbb R^{N_d\times N_d},
\qquad
\mathbf F_\varepsilon\in\mathbb R^{N_d\times n_\varepsilon}
$$

The strain-displacement matrix converts DOFs to strain; the elastic matrix converts that strain to stress.

**Equation (IF-D4) - Strain and elastic-operator sizes**

$$
\mathbf B(\widehat{\mathbf y})
\in\mathbb R^{n_\varepsilon\times N_d},
\qquad
\mathbf L(\widehat{\mathbf y})
\in\mathbb R^{n_\varepsilon\times n_\varepsilon}
$$

The operators have the following meanings:

- $\mathbf N(\mathbf y)$ maps the constrained global displacement DOFs to displacement at $\mathbf y$
- $\mathbf B(\mathbf y)$ maps those displacement DOFs to the symmetric strain fluctuation at $\mathbf y$
- $\mathbf L(\mathbf y)$ maps elastic strain to stress at $\mathbf y$
- $\mathbf K$ is the assembled constrained elastic stiffness
- $\mathbf F_\varepsilon$ maps macroscopic strain to its equivalent nodal load

The independent vector $\mathbf d$ contains the constrained fluctuation DOFs; $N_d$ counts those independent scalar entries.

## How the influence functions arise

At a fixed coarse-scale point, assume small strain, a periodic RVE, and a fixed, symmetric positive-definite linear-elastic stiffness field $\mathbf L(\mathbf y)$, with no body force. The macroscopic strain $\bar{\boldsymbol\varepsilon}$ and current eigenstrain field $\boldsymbol\mu(\mathbf y)$ are prescribed to the elastic RVE problem. Its unknown is the periodic displacement-fluctuation DOF vector $\mathbf d$.

Differentiation converts the nodal interpolation to strain. Let $\mathbf D_{\mathbf y}$ be the engineering-Voigt form of the symmetric derivative: it returns normal displacement derivatives and the sums of cross derivatives for engineering shear. Applying it to each column of $\mathbf N$ constructs $\mathbf B$.

**Equation (IF-K3) - Construction of the strain-displacement matrix**

$$
\mathbf B(\mathbf y):=\mathbf D_{\mathbf y}\mathbf N(\mathbf y)
$$

Differentiating the affine displacement gives the imposed macroscopic strain. Adding the fluctuation derivative therefore gives total fine-scale strain:

**Equation (IF-A1) - Macroscopic and fluctuation strain**

$$
\boldsymbol\varepsilon(\mathbf y)
=
\bar{\boldsymbol\varepsilon}
+
\mathbf B(\mathbf y)\mathbf d
$$

### Periodic normalization and average-strain consistency

To relate fine-scale fields to macroscopic quantities, define the volume average of any field $\mathbf q$ on the RVE. Here $|\Theta|$ denotes its volume:

**Equation (IF-M1) - RVE volume average**

$$
\langle\mathbf q\rangle_\Theta
:=
\frac{1}{|\Theta|}
\int_\Theta \mathbf q(\mathbf y)\,d\Theta
$$

Periodicity alone does not fix the constant translation of the fluctuation displacement: $\mathbf u(\mathbf y)$ and $\mathbf u(\mathbf y)+\mathbf c$ have the same strain for every constant vector $\mathbf c$. To fix this arbitrary translation, impose the zero-mean normalization.

**Equation (IF-M2) - Removal of rigid translation**

$$
\boxed{
\langle\mathbf u\rangle_\Theta=\mathbf0
}
$$

This makes the displacement fluctuation unique. It does not imply that $\mathbf u$ vanishes pointwise.

The fluctuation strain is

**Equation (IF-M3) - Fluctuation strain**

$$
\boldsymbol\varepsilon^{\mathrm{fluc}}(\mathbf y)
=
\left(\nabla_{\mathbf y}^{s}\mathbf u(\mathbf y)\right)^{\mathrm V}
=
\mathbf B(\mathbf y)\mathbf d
$$

Here $\nabla^s\mathbf u=(\nabla\mathbf u+(\nabla\mathbf u)^{\mathsf T})/2$ is a symmetric tensor gradient; Eq. (IF-M3) uses its engineering-Voigt representation on the vector side. Its zero average is derived from periodicity rather than imposed. Let $u_{i,j}=\partial u_i/\partial y_j$, let $\mathbf n$ be the outward unit normal, and let $dS$ be surface area. The dyadic product has components $(\mathbf a\otimes\mathbf b)_{ij}=a_i b_j$. Componentwise, the divergence theorem gives

**Equation (IF-M4) - Divergence theorem for the displacement gradient**

$$
\int_\Theta u_{i,j}\,d\Theta
=
\int_{\partial\Theta}u_i n_j\,dS
$$

Symmetrizing and dividing by the RVE volume expresses the tensor average through boundary values.

**Equation (IF-M5) - Boundary representation of average fluctuation strain**

$$
\left\langle\nabla_{\mathbf y}^{s}\mathbf u\right\rangle_\Theta
=
\frac{1}{2|\Theta|}
\int_{\partial\Theta}
\left(
\mathbf u\otimes\mathbf n
+
\mathbf n\otimes\mathbf u
\right)dS
$$

On each pair of opposite RVE faces, $\mathbf u$ has the same periodic value while the outward unit normals have opposite signs. The two surface contributions therefore cancel, giving

**Equation (IF-M6) - Periodic cancellation**

$$
\boxed{
\left\langle
\boldsymbol\varepsilon^{\mathrm{fluc}}
\right\rangle_\Theta
=
\left\langle
\left(\nabla_{\mathbf y}^{s}\mathbf u\right)^{\mathrm V}
\right\rangle_\Theta
=
\mathbf0
}
$$

Averaging Eq. (IF-A1) over the full RVE now yields

**Equation (IF-M7) - Macroscopic average-strain consistency**

$$
\boxed{
\left\langle\boldsymbol\varepsilon\right\rangle_\Theta
=
\bar{\boldsymbol\varepsilon}
}
$$

Thus, $\langle\mathbf u\rangle_\Theta=\mathbf0$ is an imposed uniqueness condition, whereas $\langle\boldsymbol\varepsilon^{\mathrm{fluc}}\rangle_\Theta=\mathbf0$ is a consequence of periodicity. Equation (IF-M7) confirms that the prescribed macroscopic strain equals the full-RVE average of the fine-scale total strain.

The full-RVE result does not require the fluctuation strain to average to zero inside each partition $\Theta^B$:

**Equation (IF-M8) - RVE cancellation does not imply partition cancellation**

$$
\left\langle
\boldsymbol\varepsilon^{\mathrm{fluc}}
\right\rangle_\Theta
=\mathbf0
\quad\not\Longrightarrow\quad
\left\langle
\boldsymbol\varepsilon^{\mathrm{fluc}}
\right\rangle_{\Theta^B}
=\mathbf0
$$

These generally nonzero partition averages motivate the partition-level strain transformation that follows the partitioned displacement representation in `02_partitioned_eigenstrain_displacement.md`.

Eigenstrain creates no local elastic stress when it equals total strain. Subtract it from the total strain before applying elasticity:

**Equation (IF-A2) - Elastic stress with prescribed eigenstrain**

$$
\boldsymbol\sigma(\mathbf y)
=
\mathbf L(\mathbf y)
\left[
\bar{\boldsymbol\varepsilon}
+
\mathbf B(\mathbf y)\mathbf d
-
\boldsymbol\mu(\mathbf y)
\right]
$$

Dimensionally,

**Equation (IF-A3) - Constitutive product sizes**

$$
\underbrace{\mathbf L}_{n_\varepsilon\times n_\varepsilon}
\left[
\underbrace{\bar{\boldsymbol\varepsilon}}_{n_\varepsilon\times1}
+
\underbrace{\mathbf B}_{n_\varepsilon\times N_d}
\underbrace{\mathbf d}_{N_d\times1}
-
\underbrace{\boldsymbol\mu}_{n_\varepsilon\times1}
\right]
\in\mathbb R^{n_\varepsilon}
$$

To obtain an equation for FE DOFs, start from force balance with no body force, then test it against an admissible displacement. The divergence acts on the stress tensor, reconstructed from the stress vector:

**Equation (IF-W1) - Fine-scale force equilibrium**

$$
\nabla_{\mathbf y}\cdot\boldsymbol\sigma^{\mathrm{ten}}(\mathbf y)
=
\mathbf0
\qquad \mathbf y\in\Theta
$$

Choose an admissible periodic virtual displacement $\delta\mathbf u=\mathbf N\delta\mathbf d$, where $\delta\mathbf d\in\mathbb R^{N_d}$ is arbitrary and $\delta\boldsymbol\varepsilon=\mathbf B\delta\mathbf d$. Multiplication of Eq. (IF-W1) by $\delta\mathbf u$ and integration by parts gives

**Equation (IF-W2) - Integration by parts**

$$
0
=
\int_{\partial\Theta}
\delta\mathbf u\cdot
\left(\boldsymbol\sigma^{\mathrm{ten}}\mathbf n\right)
\,dS
-
\int_\Theta
\delta\boldsymbol\varepsilon^{\mathsf T}
\boldsymbol\sigma
\,d\Theta
$$

The boundary integral cancels between opposite RVE faces because $\delta\mathbf u$ is periodic and the corresponding tractions are anti-periodic. Therefore,

**Equation (IF-W3) - Discrete virtual work**

$$
\delta\mathbf d^{\mathsf T}
\left[
\int_\Theta
\mathbf B^{\mathsf T}(\mathbf y)
\boldsymbol\sigma(\mathbf y)
\,d\Theta
\right]
=
0
\qquad
\text{for every }\delta\mathbf d\in\mathbb R^{N_d}
$$

Because $\delta\mathbf d$ is arbitrary, the finite-element weak equilibrium equation is

**Equation (IF-A4) - Constrained FE equilibrium**

$$
\int_\Theta
\mathbf B^{\mathsf T}(\mathbf y)
\boldsymbol\sigma(\mathbf y)
\,d\Theta
=
\mathbf0_{N_d}
$$

Substitution of Eq. (IF-A2) into Eq. (IF-A4) gives

**Equation (IF-A5) - Separation of displacement and imposed inputs**

$$
\left[
\int_\Theta
\mathbf B^{\mathsf T}\mathbf L\mathbf B
\,d\Theta
\right]\mathbf d
+
\left[
\int_\Theta
\mathbf B^{\mathsf T}\mathbf L
\,d\Theta
\right]\bar{\boldsymbol\varepsilon}
-
\int_\Theta
\mathbf B^{\mathsf T}\mathbf L\boldsymbol\mu
\,d\Theta
=
\mathbf0_{N_d}
$$

To reuse the displacement-dependent term, define its coefficient as the constrained stiffness.

**Equation (IF-A6) - Constrained elastic stiffness**

$$
\mathbf K
:=
\int_\Theta
\mathbf B^{\mathsf T}(\mathbf y)
\mathbf L(\mathbf y)
\mathbf B(\mathbf y)
\,d\Theta
\in\mathbb R^{N_d\times N_d}
$$

Move the macroscopic-strain term to the right-hand side. Its negative sign comes from this rearrangement.

**Equation (IF-A7) - Macroscopic-strain load operator**

$$
\mathbf F_\varepsilon
:=
-
\int_\Theta
\mathbf B^{\mathsf T}(\mathbf y)
\mathbf L(\mathbf y)
\,d\Theta
\in\mathbb R^{N_d\times n_\varepsilon}
$$

The eigenstrain term already has the opposite sign in stress, so its equivalent load enters the right-hand side positively.

**Equation (IF-A8) - Distributed eigenstrain load**

$$
\mathbf f_\mu[\boldsymbol\mu]
:=
\int_\Theta
\mathbf B^{\mathsf T}(\widehat{\mathbf y})
\mathbf L(\widehat{\mathbf y})
\boldsymbol\mu(\widehat{\mathbf y})
\,d\widehat\Theta
\in\mathbb R^{N_d}
$$

The vector $\mathbf f_\mu[\boldsymbol\mu]$ is the equivalent nodal load generated by the complete eigenstrain field. The square brackets emphasize that $\mathbf f_\mu$ acts on a spatial field. It is not multiplication by a single finite-dimensional eigenstrain vector.

Equilibrium becomes

**Equation (IF-A9) - Elastic equilibrium with two inputs**

$$
\mathbf K\mathbf d
=
\mathbf F_\varepsilon\bar{\boldsymbol\varepsilon}
+
\mathbf f_\mu[\boldsymbol\mu]
$$

For a connected, mechanically stable RVE with the stated positive-definite elasticity, the periodic and zero-mean constraints remove rigid translation and $\mathbf K$ is invertible on the independent displacement space. Solving Eq. (IF-A9) gives

**Equation (IF-A10) - Solution on the independent displacement space**

$$
\mathbf d
=
\mathbf K^{-1}\mathbf F_\varepsilon
\bar{\boldsymbol\varepsilon}
+
\int_\Theta
\mathbf K^{-1}
\mathbf B^{\mathsf T}(\widehat{\mathbf y})
\mathbf L(\widehat{\mathbf y})
\boldsymbol\mu(\widehat{\mathbf y})
\,d\widehat\Theta
$$

The inverse denotes the solution operator; in computation one factors the constrained stiffness and solves systems rather than forming a dense inverse. Equation (IF-A10) gives global nodal displacement DOFs. Fish and Cui require the displacement field at an arbitrary observation point $\mathbf y$, so Eq. (IF-4) must now be applied to both terms. The influence functions package these repeated source-to-solution-to-observation operations.

## Field-valued influence functions

Define the coarse-strain displacement influence function

**Equation (IF-5) - Macroscopic-strain displacement influence**

$$
\boxed{
\mathbf H(\mathbf y)
:=
\mathbf N(\mathbf y)\mathbf K^{-1}\mathbf F_\varepsilon
\in\mathbb R^{n_{\mathrm{dim}}\times n_\varepsilon}
}
$$

Each column of $\mathbf H(\mathbf y)$ is the displacement fluctuation at observation point $\mathbf y$ produced by one unit component of macroscopic strain, with the eigenstrain contribution set to zero.

Define the eigenstrain displacement influence kernel

**Equation (IF-6) - Distributed eigenstrain influence kernel**

$$
\boxed{
\mathbf h(\mathbf y,\widehat{\mathbf y})
:=
\mathbf N(\mathbf y)
\mathbf K^{-1}
\mathbf B^{\mathsf T}(\widehat{\mathbf y})
\mathbf L(\widehat{\mathbf y})
\in\mathbb R^{n_{\mathrm{dim}}\times n_\varepsilon}
}
$$

For fixed $\widehat{\mathbf y}$, each column of $\mathbf h(\mathbf y,\widehat{\mathbf y})$ is a displacement-response density per source volume. A source volume element contributes $\mathbf h\boldsymbol\mu\,d\widehat\Theta$; a finite eigenstrain value at a single point of zero volume does not by itself generate a finite load. The kernel has units of length per RVE volume, while $\mathbf H$ has units of length.

Substituting Eqs. (IF-5)-(IF-6) into Eq. (IF-A10), then applying Eq. (IF-4), gives the displacement fluctuation

**Equation (IF-7) - Displacement superposition**

$$
\boxed{
\mathbf u(\mathbf y)
=
\mathbf H(\mathbf y)\bar{\boldsymbol\varepsilon}
+
\int_\Theta
\mathbf h(\mathbf y,\widehat{\mathbf y})
\boldsymbol\mu(\widehat{\mathbf y})
\,d\widehat\Theta
}
$$

Equation (IF-7) is the matrix-form counterpart of Fish and Cui's displacement transformation relation.

The purpose of Eqs. (IF-5)-(IF-7) is to separate:

- fixed elastic information carried by $\mathbf K$, $\mathbf F_\varepsilon$, $\mathbf H$, and $\mathbf h$
- current loading and history information carried by $\bar{\boldsymbol\varepsilon}$ and $\boldsymbol\mu$

When the RVE geometry, boundary conditions, mesh, and elastic stiffness remain fixed, $\mathbf H$ and $\mathbf h$ can be computed offline and reused as the macroscopic strain and eigenstrain evolve.

## Source-to-observation interpretation

The kernel combines three operations whose order is fixed by the mechanics:

1. Convert source eigenstrain to a nodal-load density using $\mathbf B^{\mathsf T}\mathbf L$, then integrate over source volume.
2. Solve constrained elastic equilibrium with $\mathbf K$ to obtain displacement DOFs.
3. Apply $\mathbf N(\mathbf y)$ to observe the displacement at the requested point.

The source integrand has $N_d$ entries; source-volume integration is still required to obtain a nodal force:

**Equation (IF-9) - Source integrand size**

$$
\underbrace{\mathbf B^{\mathsf T}(\widehat{\mathbf y})}_{N_d\times n_\varepsilon}
\underbrace{\mathbf L(\widehat{\mathbf y})}_{n_\varepsilon\times n_\varepsilon}
\underbrace{\boldsymbol\mu(\widehat{\mathbf y})}_{n_\varepsilon\times1}
\in\mathbb R^{N_d}
$$

The observation-side multiplication is

**Equation (IF-10) - Observation reconstruction size**

$$
\underbrace{\mathbf N(\mathbf y)}_{n_{\mathrm{dim}}\times N_d}
\underbrace{\mathbf d}_{N_d\times1}
=
\underbrace{\mathbf u(\mathbf y)}_{n_{\mathrm{dim}}\times1}
$$

Therefore, $\mathbf N$ is evaluated at $\mathbf y$, because it reconstructs the displacement at the observation point. The source dependence is carried by $\mathbf B^{\mathsf T}(\widehat{\mathbf y})\mathbf L(\widehat{\mathbf y})$. The integral in Eq. (IF-7) sums at $\mathbf y$ the displacement responses generated by eigenstrain at every source point $\widehat{\mathbf y}\in\Theta$.

## Result and next question

The main result is Eq. (IF-7): fixed elastic response operators act on current macroscopic strain and eigenstrain. Periodicity recovers the macroscopic average strain, while normalization fixes displacement uniqueness. In three dimensions $\mathbf K$ has force/length units and the integrated equivalent loads have force units; 2D integrals are understood per unit thickness unless a thickness is included.

**Checkpoint:** Why can the influence functions remain fixed when eigenstrain changes, and what information would prevent a finite list of amplitudes from representing an arbitrary eigenstrain field exactly?

Lecture 2 answers the second question by introducing an explicit spatial approximation.

## Source anchor

J. Fish and J. Cui, transformation-field preliminaries, Section 2.1 and Section 2.2, Eqs. (1)-(6), PDF page 4. The FE realization is a pedagogical derivation of that representation, with supporting FE context in J. Fish, *Practical Multiscaling*, Section 4.2.2. All equalities involving the FE solution hold in the chosen discrete space; finite mesh resolution remains an approximation to the continuum problem.
