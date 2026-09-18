# Partition-average strain transformation

This note derives Fish and Cui Section 2.2, Eqs. (8)-(10), PDF pages 4-5. The purpose is to convert the displacement influence representation into a reduced relation for the average strain in every observation partition. These partition strains are the inputs to the later partition-level constitutive updates.

## Purpose and route

The constitutive model needs strain, whereas Lecture 2 produces displacement. We therefore apply small-strain kinematics and average over each observation partition. Linearity allows the derivatives and integrals to be stored as transformation operators. The target is ST-12 and its component form ST-14, completing the elastic backbone before we ask how the unknown eigenstrains evolve.

## Starting point and dimensions

Assume the fixed, stable, periodic elastic RVE of the displacement construction and a fixed piecewise-constant eigenstrain partitioning. $\Theta$ is the RVE, $\Theta^A$ a source partition, $\Theta^B$ an observation partition, $M$ the number of partitions, and $\mathbf y$ an observation coordinate. $\mathbf u$ is displacement fluctuation, $\bar{\boldsymbol\varepsilon}$ imposed macroscopic strain, and $\boldsymbol\mu^A$ the eigenstrain amplitude assigned to partition $A$. The imposed strain and partition amplitudes are independent of observation position when differentiating influence functions; the displacement field generally varies with $\mathbf y$.

Let $n_{\mathrm{dim}}\in\{2,3\}$ be the spatial dimension. Counting independent strain components fixes the vector length:

**Equation (ST-1) - Number of strain components**

$$
n_\varepsilon=
\begin{cases}
3, & n_{\mathrm{dim}}=2\\
6, & n_{\mathrm{dim}}=3
\end{cases}
$$

Use $(11,22,33,12,23,13)$ with doubled tensor shear components for 3D strain vectors, or $(11,22,12)$ for a specified 2D elastic reduction. Stress vectors use undoubled shear stresses. The macroscopic strain and partition eigenstrains are dimensionless and satisfy

**Equation (ST-2) - Input vector sizes**

$$
\bar{\boldsymbol\varepsilon},\boldsymbol\mu^A
\in\mathbb R^{n_\varepsilon}
$$

The partitioned displacement-fluctuation representation is

**Equation (ST-3) - Partitioned displacement representation**

$$
\mathbf u(\mathbf y)
=\mathbf H(\mathbf y)\bar{\boldsymbol\varepsilon}
+\sum_{A=1}^{M}\mathbf h^A(\mathbf y)\boldsymbol\mu^A
$$

The columns of $\mathbf H$ and $\mathbf h^A$ are the displacement responses to unit macroscopic-strain and unit partition-eigenstrain components, respectively. Both have length units and the following sizes.

**Equation (ST-4) - Displacement influence sizes**

$$
\mathbf H(\mathbf y),\mathbf h^A(\mathbf y)
\in\mathbb R^{n_{\mathrm{dim}}\times n_\varepsilon}
$$

The operator $\mathbf D_{\mathbf y}$ is the engineering-Voigt representation of the symmetric gradient. With $u_{i,j}=\partial u_i/\partial y_j$, its explicit 3D action explains the shear convention and its inverse-length units.

**Equation (ST-5a) - Engineering strain-displacement derivative**

$$
\mathbf D_{\mathbf y}\mathbf u
=(u_{1,1},u_{2,2},u_{3,3},u_{1,2}+u_{2,1},u_{2,3}+u_{3,2},u_{1,3}+u_{3,1})^{\mathsf T}
$$

Apply this derivative column by column to a displacement influence matrix. It turns each unit-input displacement response into a unit-input strain response, so

**Equation (ST-5) - Strain influence sizes**

$$
\mathbf D_{\mathbf y}\mathbf H(\mathbf y),
\mathbf D_{\mathbf y}\mathbf h^A(\mathbf y)
\in\mathbb R^{n_\varepsilon\times n_\varepsilon}
$$

## Pointwise strain

The total fine-scale strain is the macroscopic strain plus the symmetric gradient of the displacement fluctuation:

**Equation (ST-6) - Total fine-scale strain**

$$
\boldsymbol\varepsilon(\mathbf y)
=\bar{\boldsymbol\varepsilon}
+\mathbf D_{\mathbf y}\mathbf u(\mathbf y)
$$

Substitute Eq. (ST-3). The input amplitudes have no observation-coordinate dependence, so the derivative acts only on the influence functions. Let $\mathbf I_{n_\varepsilon}$ be the $n_\varepsilon\times n_\varepsilon$ identity matrix. Collecting the macroscopic-strain terms gives

**Equation (ST-7) - Pointwise strain transformation**

$$
\boxed{
\boldsymbol\varepsilon(\mathbf y)
=\left[\mathbf I_{n_\varepsilon}+\mathbf D_{\mathbf y}\mathbf H(\mathbf y)\right]
\bar{\boldsymbol\varepsilon}
+\sum_{A=1}^{M}
\mathbf D_{\mathbf y}\mathbf h^A(\mathbf y)\boldsymbol\mu^A
}
$$

The identity contribution preserves the imposed macroscopic strain before adding its fluctuation response.

## Average over observation partition B

The material model will use the average strain seen in observation partition $B$. With $|\Theta^B|$ its positive volume, define that input by

**Equation (ST-8) - Observation-partition average**

$$
\boldsymbol\varepsilon^B
:=
\frac{1}{|\Theta^B|}
\int_{\Theta^B}\boldsymbol\varepsilon(\mathbf y)\,d\Theta
\in\mathbb R^{n_\varepsilon}
$$

Insert Eq. (ST-7), use linearity of integration, and take the partition-independent vectors $\bar{\boldsymbol\varepsilon}$ and $\boldsymbol\mu^A$ outside the integral:

**Equation (ST-9) - Average of the pointwise transformation**

$$
\begin{aligned}
\boldsymbol\varepsilon^B
={}&
\left[
\mathbf I_{n_\varepsilon}+
\frac{1}{|\Theta^B|}
\int_{\Theta^B}
\mathbf D_{\mathbf y}\mathbf H(\mathbf y)\,d\Theta
\right]\bar{\boldsymbol\varepsilon}
\\
&+
\sum_{A=1}^{M}
\left[
\frac{1}{|\Theta^B|}
\int_{\Theta^B}
\mathbf D_{\mathbf y}\mathbf h^A(\mathbf y)\,d\Theta
\right]\boldsymbol\mu^A
\end{aligned}
$$

This motivates the macroscopic-strain transformation operator

**Equation (ST-10) - Macroscopic-strain transformation operator**

$$
\boxed{
\mathbf E^B
:=
\mathbf I_{n_\varepsilon}+
\frac{1}{|\Theta^B|}
\int_{\Theta^B}
\mathbf D_{\mathbf y}\mathbf H(\mathbf y)\,d\Theta
}
\in\mathbb R^{n_\varepsilon\times n_\varepsilon}
$$

To store the average response in $B$ to a unit eigenstrain component imposed in $A$, define the eigenstrain interaction operator

**Equation (ST-11) - Source-to-observation strain interaction**

$$
\boxed{
\mathbf P^{BA}
:=
\frac{1}{|\Theta^B|}
\int_{\Theta^B}
\mathbf D_{\mathbf y}\mathbf h^A(\mathbf y)\,d\Theta
}
\in\mathbb R^{n_\varepsilon\times n_\varepsilon}
$$

Equation (ST-9) becomes Fish and Cui Eq. (8):

**Equation (ST-12) - Partition-average strain relation**

$$
\boxed{
\boldsymbol\varepsilon^B
=\mathbf E^B\bar{\boldsymbol\varepsilon}
+\sum_{A=1}^{M}\mathbf P^{BA}\boldsymbol\mu^A
}
$$

Each product has sizes $(n_\varepsilon\times n_\varepsilon)(n_\varepsilon\times1)$, so it returns a partition-strain vector:

**Equation (ST-13) - Transformation product sizes**

$$
\mathbf E^B\bar{\boldsymbol\varepsilon}\in\mathbb R^{n_\varepsilon},
\qquad
\mathbf P^{BA}\boldsymbol\mu^A\in\mathbb R^{n_\varepsilon}
$$

## Row, column, and partition meanings

Using Voigt-component indices $r,s\in\{1,\ldots,n_\varepsilon\}$,

**Equation (ST-14) - Component form of the partition strain relation**

$$
\varepsilon_r^B
=\sum_{s=1}^{n_\varepsilon}E_{rs}^B\bar\varepsilon_s
+\sum_{A=1}^{M}\sum_{s=1}^{n_\varepsilon}P_{rs}^{BA}\mu_s^A
$$

- Row $r$ identifies the average output strain component observed in partition $B$
- Column $s$ identifies the applied macroscopic-strain component in $\mathbf E^B$ or source eigenstrain component in $\mathbf P^{BA}$
- $A$ identifies the source eigenstrain partition
- $B$ identifies the observation-strain partition
- $\mathbf P^{AA}$ is a self-interaction block; $\mathbf P^{BA}$ with $B\ne A$ is a cross-partition interaction block

The column $\mathbf P_{:s}^{BA}$ is the complete average-strain vector produced in partition $B$ by a unit value of eigenstrain component $s$ in partition $A$. For an actual amplitude $\mu_s^A$, denote its contribution by $\boldsymbol\varepsilon^{B;A,s}$; the semicolon separates the observation partition from the source partition and component. This is a contribution, not a load-step increment.

**Equation (ST-15) - Contribution of one source component**

$$
\boldsymbol\varepsilon^{B;A,s}
=\mathbf P_{:s}^{BA}\mu_s^A
$$

## Averaging is not a constant-strain assumption

Equation (ST-8) defines an average of the varying pointwise strain in partition $B$. It does not impose

**Equation (ST-16) - Constant total strain is not imposed**

$$
\boldsymbol\varepsilon(\mathbf y)=\boldsymbol\varepsilon^B
\quad\text{for every }\mathbf y\in\Theta^B
$$

The operators $\mathbf E^B$ and $\mathbf P^{BA}$ are computed offline from fine-scale elastic RVE solutions. Online, Eq. (ST-12) evaluates a reduced partition-average strain used in the later constitutive updates; it is not a new fine-scale equilibrium solve. Partition averages alone cannot recover distinct pointwise fields having the same average. Optional pointwise reconstruction requires $\mathbf H(\mathbf y)$ and $\mathbf h^A(\mathbf y)$, or an equivalent recovery representation.

The customary piecewise-constant approximation in Fish and Cui Eq. (7) applies to the eigenstrain field. The total pointwise strain in Eq. (ST-7) can still vary inside a partition.

## Checks and the missing constitutive relation

Both $\mathbf E^B$ and $\mathbf P^{BA}$ are dimensionless: differentiation removes the length unit of the displacement influence, and averaging preserves strain units. The order $BA$ always means observation first, source second. Setting every eigenstrain amplitude to zero recovers the elastic macroscopic-strain response. Full-RVE averaging must recover the imposed macroscopic strain, even though individual partition averages differ.

ST-14 completes the strain transformation, but it does not determine $\boldsymbol\mu^A$ during nonlinear loading. With prescribed eigenstrain it is a linear evaluation. With evolving material history, partition strains and eigenstrains must satisfy both this elastic interaction relation and constitutive evolution. This is why the next group of lectures needs a nonlinear solution technique and a material model.

**Checkpoint:** If the elastic influence matrices stay fixed, why can the overall response still be nonlinear once eigenstrain is allowed to evolve?

Lecture 4 develops residual linearization and Newton's method in the familiar full-FEM setting. Lectures 5-7 then construct the plasticity update required inside that architecture. Lecture 8 will connect that material response to this TFA partition relation, after clarifying why prescribed average strain does not generally specify each local strain. Lectures 9-11 will develop the coupled solve, its derivatives and the implementation.

## Source anchor

J. Fish and J. Cui, transformation-field preliminaries, Section 2.2, Eqs. (8)-(10), PDF pages 4-5. The matrix notation above is a Voigt-form reorganization of the paper's tensor-index equations.
