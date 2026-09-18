# Partitioned eigenstrain and displacement influence functions

This note derives the eigenstrain model reduction used immediately after the distributed displacement influence relation. The purpose is to replace a spatially varying eigenstrain field and its repeated source integral by a finite set of partition eigenstrains multiplying precomputed displacement-response operators.

## Purpose and route

Lecture 1 separated fixed elastic response from changing eigenstrain, but its source integral still requires a spatial field. We now use a piecewise-constant basis to reduce that field to finitely many amplitudes, then use linearity to precompute each partition's displacement response. Lecture 3 will turn these displacement responses into the partition strain inputs needed by a material law. The partition plastic evolution will be supplied by Lectures 5-7 and connected to this representation in Lecture 8.

## Setting and dimensions

Assume a fixed, mechanically stable periodic elastic RVE with fixed elastic stiffness, mesh, and normalization. Let $n_{\mathrm{dim}}\in\{2,3\}$ be the spatial dimension. The number of independent strain components determines the size of each partition amplitude:

**Equation (P-0) - Number of strain components**

$$
n_\varepsilon=
\begin{cases}
3, & n_{\mathrm{dim}}=2\\
6, & n_{\mathrm{dim}}=3
\end{cases}
$$

In 3D, strain-like vectors use $(11,22,33,12,23,13)$ with doubled tensor shear components; in 2D use $(11,22,12)$ with the same engineering-shear convention and a specified elastic reduction. Stress-like vectors use undoubled shear stress. All strain-like amplitudes are dimensionless. Let $\Theta\subset\mathbb R^{n_{\mathrm{dim}}}$ be the RVE, $\mathbf y\in\Theta$ an observation point, and $\widehat{\mathbf y}\in\Theta$ an eigenstrain source point.

The prescribed macroscopic strain and fine-scale eigenstrain are

**Equation (P-D1) - Macroscopic strain and eigenstrain sizes**

$$
\bar{\boldsymbol\varepsilon}\in\mathbb R^{n_\varepsilon},
\qquad
\boldsymbol\mu(\widehat{\mathbf y})\in\mathbb R^{n_\varepsilon}
$$

The previously derived displacement representation is

**Equation (P-D2) - Distributed displacement representation**

$$
\mathbf u(\mathbf y)
=
\mathbf H(\mathbf y)\bar{\boldsymbol\varepsilon}
+
\int_\Theta
\mathbf h(\mathbf y,\widehat{\mathbf y})
\boldsymbol\mu(\widehat{\mathbf y})
\,d\widehat\Theta
$$

Here $\mathbf u$ is the periodic displacement fluctuation, $\mathbf H$ maps macroscopic strain to that fluctuation, and $\mathbf h$ is a response density integrated over source volume. Their output and input roles fix their sizes.

**Equation (P-D3) - Displacement-response sizes**

$$
\mathbf u(\mathbf y)\in\mathbb R^{n_{\mathrm{dim}}},
\qquad
\mathbf H(\mathbf y)\in\mathbb R^{n_{\mathrm{dim}}\times n_\varepsilon},
\qquad
\mathbf h(\mathbf y,\widehat{\mathbf y})\in
\mathbb R^{n_{\mathrm{dim}}\times n_\varepsilon}
$$

If strain and eigenstrain are dimensionless, $\mathbf H$ has units of length, while the distributed kernel $\mathbf h$ has units of length per RVE volume because it is integrated over $d\widehat\Theta$.

## Piecewise-constant eigenstrain approximation

The eigenstrain field contains many fine-scale values. To obtain a reduced description, partition the RVE into $M$ nonoverlapping subdomains:

**Equation (P-1) - RVE partitioning**

$$
\Theta=\bigcup_{A=1}^{M}\Theta^A,
\qquad
\Theta^A\cap\Theta^B=\varnothing
\quad(A\ne B)
$$

Partition interiors are disjoint; shared boundaries have zero volume and may be assigned to either neighbor. Statements about membership and unity below hold almost everywhere. Define the scalar indicator function of partition $A$:

**Equation (P-2) - Partition indicator**

$$
N^A(\widehat{\mathbf y})=
\begin{cases}
1, & \widehat{\mathbf y}\in\Theta^A\\
0, & \widehat{\mathbf y}\notin\Theta^A
\end{cases}
$$

Every source point belongs to exactly one partition, so the indicators form a partition of unity:

**Equation (P-3) - Partition of unity**

$$
\sum_{A=1}^{M}N^A(\widehat{\mathbf y})=1
$$

Let $\boldsymbol\mu^A\in\mathbb R^{n_\varepsilon}$ be the eigenstrain vector assigned to partition $A$. The piecewise-constant approximation is

**Equation (P-4) - Piecewise-constant eigenstrain approximation**

$$
\boxed{
\boldsymbol\mu(\widehat{\mathbf y})
\approx
\sum_{A=1}^{M}
\underbrace{N^A(\widehat{\mathbf y})}_{1\times1}
\underbrace{\boldsymbol\mu^A}_{n_\varepsilon\times1}
}
\in\mathbb R^{n_\varepsilon}
$$

If $\widehat{\mathbf y}\in\Theta^B$, then $N^B(\widehat{\mathbf y})=1$ and all other indicators vanish. Hence

**Equation (P-5) - Eigenstrain inside one partition**

$$
\boldsymbol\mu(\widehat{\mathbf y})\approx\boldsymbol\mu^B
$$

Thus, the field is represented by $M n_\varepsilon$ partition amplitudes instead of an independently varying eigenstrain vector at every fine-scale integration point.

The scalar $N^A$ is a partition indicator, whereas $N_I$ denotes an FE nodal shape function and $\mathbf N$ the assembled FE interpolation matrix. The indicator describes where eigenstrain is imposed; it does not restrict where the resulting displacement can be nonzero.

## Partition-integrated displacement response

The next goal is to remove the repeated source integration from the online calculation. From this point, $\boldsymbol\mu$ and $\mathbf u$ denote the reduced eigenstrain field and its corresponding elastic solution. The approximation is Eq. (P-4); subsequent equalities are exact within that FE and partition representation. Substitute the represented field into Eq. (P-D2):

**Equation (P-6) - Insertion of the reduced eigenstrain field**

$$
\mathbf u(\mathbf y)
=
\mathbf H(\mathbf y)\bar{\boldsymbol\varepsilon}
+
\int_\Theta
\mathbf h(\mathbf y,\widehat{\mathbf y})
\sum_{A=1}^{M}N^A(\widehat{\mathbf y})\boldsymbol\mu^A
\,d\widehat\Theta
$$

Exchange the finite sum and the integral:

**Equation (P-7) - Exchange of finite summation and integration**

$$
\mathbf u(\mathbf y)
=
\mathbf H(\mathbf y)\bar{\boldsymbol\varepsilon}
+
\sum_{A=1}^{M}
\int_\Theta
\mathbf h(\mathbf y,\widehat{\mathbf y})
N^A(\widehat{\mathbf y})\boldsymbol\mu^A
\,d\widehat\Theta
$$

The vector $\boldsymbol\mu^A$ is a partition amplitude and is independent of the integration coordinate $\widehat{\mathbf y}$. It can therefore be taken outside the integral:

**Equation (P-8) - Extraction of partition amplitudes**

$$
\mathbf u(\mathbf y)
=
\mathbf H(\mathbf y)\bar{\boldsymbol\varepsilon}
+
\sum_{A=1}^{M}
\left[
\int_\Theta
\mathbf h(\mathbf y,\widehat{\mathbf y})
N^A(\widehat{\mathbf y})
\,d\widehat\Theta
\right]
\boldsymbol\mu^A
$$

This motivates defining the partition-integrated displacement influence function

**Equation (P-9) - Partition-integrated displacement influence**

$$
\boxed{
\mathbf h^A(\mathbf y)
:=
\int_\Theta
\mathbf h(\mathbf y,\widehat{\mathbf y})
N^A(\widehat{\mathbf y})
\,d\widehat\Theta
}
\in\mathbb R^{n_{\mathrm{dim}}\times n_\varepsilon}
$$

Because $N^A$ is an indicator, the same operator can be evaluated by integrating only over its source partition.

**Equation (P-9a) - Restriction to source partition**

$$
\mathbf h^A(\mathbf y)=\int_{\Theta^A}\mathbf h(\mathbf y,\widehat{\mathbf y})\,d\widehat\Theta
$$

After integration, $\widehat{\mathbf y}$ is no longer a free variable. Therefore, $\mathbf h^A$ depends on the observation point $\mathbf y$ and the partition index $A$, but not on the source coordinate $\widehat{\mathbf y}$. If eigenstrain is dimensionless, $\mathbf h^A$ has units of length.

Using Eq. (P-9), the displacement fluctuation becomes the finite sum

**Equation (P-10) - Partition displacement superposition**

$$
\boxed{
\mathbf u(\mathbf y)
=
\mathbf H(\mathbf y)\bar{\boldsymbol\varepsilon}
+
\sum_{A=1}^{M}
\mathbf h^A(\mathbf y)\boldsymbol\mu^A
}
$$

Let $\mathbf u^A(\mathbf y)$ denote the displacement contribution generated by partition $A$. Its product dimensions are consistent:

**Equation (P-11) - One partition contribution**

$$
\underbrace{\mathbf h^A(\mathbf y)}_{n_{\mathrm{dim}}\times n_\varepsilon}
\underbrace{\boldsymbol\mu^A}_{n_\varepsilon\times1}
=
\underbrace{\mathbf u^A(\mathbf y)}_{n_{\mathrm{dim}}\times1}
$$

For eigenstrain component $s\in\{1,\ldots,n_\varepsilon\}$, select a column to isolate one source response. The notation $(:,s)$ means all displacement rows in column $s$. The column

**Equation (P-12) - One influence-function column**

$$
\mathbf h^A_{(:,s)}(\mathbf y)
\in\mathbb R^{n_{\mathrm{dim}}}
$$

This column is the displacement fluctuation at $\mathbf y$ produced by a unit value of eigenstrain component $s$ throughout partition $A$, with the other components and partitions set to zero. Weighting each unit-component response by its actual amplitude reconstructs the partition contribution.

**Equation (P-13) - Component expansion of a partition response**

$$
\mathbf h^A(\mathbf y)\boldsymbol\mu^A
=
\sum_{s=1}^{n_\varepsilon}
\mathbf h^A_{(:,s)}(\mathbf y)\mu_s^A
$$

This is the displacement contribution from partition $A$. Equation (P-10) superposes the macroscopic-strain response with the contributions from all partitions.

## Offline-online meaning

For a fixed elastic RVE and a fixed partitioning, $\mathbf H(\mathbf y)$ and every $\mathbf h^A(\mathbf y)$ can be computed during the offline stage. During the online nonlinear calculation, the current values of $\bar{\boldsymbol\varepsilon}$ and $\boldsymbol\mu^A$ are inserted into Eq. (P-10); the source integral in Eq. (P-D2) does not need to be recomputed at every update.

## Result and next question

Equation (P-10) reduces the distributed input to $M n_\varepsilon$ amplitudes. Check the reduction by setting all amplitudes to zero, then by activating one component in one partition: the results must be the macroscopic response and that partition's corresponding influence column, respectively. The spatial approximation is on eigenstrain; displacement and total strain can still vary within a partition.

**Checkpoint:** Why can a locally supported eigenstrain produce displacement throughout the RVE, and what operation will extract the average strain seen by a different partition?

Lecture 3 uses differentiation and observation-partition averaging to answer this question.

## Source anchor

J. Fish and J. Cui, transformation-field preliminaries, Section 2.2, Eqs. (6)-(7), PDF page 4. The notation here uses bold vectors and matrices and writes the partition indicator as $N^A$.
