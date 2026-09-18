# Three-dimensional plasticity: yield, flow, and hardening

## Purpose and route

Lecture 4 needs a rule that converts strain and committed material history into stress. Linear elasticity has no permanent deformation: removing stress removes elastic strain. Plasticity introduces a changing stress-free strain and a memory of prior plastic flow. It supplies the physical origin of the evolving eigenstrain left undetermined by ST-14.

We first split elastic and plastic strain, then identify the stress measure that drives yielding. A yield criterion decides which stresses are admissible; an associated flow rule gives the direction of plastic change; accumulated history controls hardening; and loading conditions determine when that change is active. These are distinct ingredients. Lecture 6 will turn their continuous evolution laws into a discrete material update.

## Assumptions and local notation

Assume small strain, three-dimensional rate-independent von Mises plasticity, isotropic linear elasticity, associated flow, and linear isotropic hardening. Use positive bulk and shear moduli, positive initial yield stress, and nonnegative hardening modulus. This includes perfect plasticity when hardening is zero, but not softening.

| Symbol | Meaning | Units |
|---|---|---|
| $\boldsymbol\varepsilon$, $\boldsymbol\varepsilon^{\mathrm e}$, $\boldsymbol\varepsilon^{\mathrm p}$ | Total, elastic, and plastic strain tensors | Dimensionless |
| $\boldsymbol\sigma$, $\mathbf s$ | Cauchy and deviatoric stress tensors | Stress |
| $\mathbf I_3$ | Second-order identity tensor | Dimensionless |
| $\mathbb C$, $K_{\mathrm b}$, $G$ | Isotropic elasticity tensor, bulk modulus, shear modulus | Stress |
| $\sigma_{\mathrm{mean}}$, $\sigma_{\mathrm{eq}}$ | Mean and von Mises equivalent stress | Stress |
| $p$ | Accumulated equivalent plastic strain | Dimensionless, nonnegative |
| $\lambda$ | Plastic multiplier for the chosen yield-function normalization | Dimensionless, nondecreasing |
| $\sigma_{\mathrm y0}$, $H$ | Initial yield stress and hardening modulus | Stress |
| $f$, $\sigma_{\mathrm y}(p)$ | Yield function and current yield strength | Stress |

Bold stress and strain symbols now denote symmetric $3\times3$ tensors, not the Voigt vectors of Lecture 4. Their six independent components are unchanged. A colon is the tensor double contraction, $\mathbf a:\mathbf b=\sum_{i,j=1}^3a_{ij}b_{ij}$. The trace is $\operatorname{tr}(\mathbf a)=\sum_{i=1}^3a_{ii}$, and $\mathbb C:\mathbf a$ is a second-order tensor obtained by contracting the last two indices of $\mathbb C$ with $\mathbf a$.

For conversion to FEM, use strain order $(11,22,33,12,23,13)$ with doubled tensor shear strains and undoubled shear stresses. For example, $\mathbf s:\mathbf s=s_{11}^2+s_{22}^2+s_{33}^2+2(s_{12}^2+s_{23}^2+s_{13}^2)$; a plain dot product of undoubled stress components would miss the shear weights. $p$ denotes accumulated strain, never pressure; bold $\mathbf H$ in the TFA influence-function construction is unrelated to scalar $H$ here.

## 1. Plastic strain explains permanent deformation

To distinguish recoverable deformation from permanent deformation, adopt the small-strain additive split. This is a kinematic assumption of the model.

**Equation (EN-1) - Elastic-plastic strain split**

$$
\boldsymbol\varepsilon=\boldsymbol\varepsilon^{\mathrm e}+\boldsymbol\varepsilon^{\mathrm p}
$$

Only elastic strain creates elastic stress, so inserting the split into the elastic law gives

**Equation (EN-2) - Stress from elastic strain**

$$
\boldsymbol\sigma=\mathbb C:(\boldsymbol\varepsilon-\boldsymbol\varepsilon^{\mathrm p})
$$

If total strain equals plastic strain locally, elastic stress vanishes. Thus plastic strain is a stress-free eigenstrain in the constitutive sense. A spatially varying eigenstrain can still generate stress when compatibility and boundary conditions prevent that stress-free deformation. This is the connection to the eigenstrain field in Lectures 1-3. Lecture 8 will identify the partition eigenstrain with partition plastic strain, retaining the mechanics multiplier convention used here.

## 2. Separate volumetric and shape-changing stress

The present metal-plasticity model responds to shape-changing stress. First extract mean stress to identify the part that acts equally in all directions.

**Equation (J2-1) - Mean stress**

$$
\sigma_{\mathrm{mean}}:=\tfrac13\operatorname{tr}(\boldsymbol\sigma)
$$

Use this scalar to separate the hydrostatic and deviatoric parts.

**Equation (J2-2) - Stress decomposition**

$$
\boldsymbol\sigma=\sigma_{\mathrm{mean}}\mathbf I_3+\mathbf s
$$

Solving the decomposition for its remaining part defines the deviatoric operation, which will also be used on strain.

**Equation (J2-3) - Deviatoric stress and operator**

$$
\mathbf s=\operatorname{dev}(\boldsymbol\sigma),
\qquad
\operatorname{dev}(\mathbf a):=\mathbf a-\tfrac13\operatorname{tr}(\mathbf a)\mathbf I_3
$$

Since $\operatorname{tr}(\mathbf I_3)=3$, subtracting the hydrostatic part removes the trace exactly.

**Equation (J2-4) - Zero deviatoric trace**

$$
\operatorname{tr}(\mathbf s)
=\operatorname{tr}(\boldsymbol\sigma)-\tfrac13\operatorname{tr}(\boldsymbol\sigma)\,3=0
$$

Isotropic elasticity separately resists volume change through $K_{\mathrm b}$ and shape change through $G$. Its tensor form will explain the return-mapping stress correction.

**Equation (J2-E1) - Volumetric and deviatoric elasticity**

$$
\mathbb C:\mathbf a
=K_{\mathrm b}\operatorname{tr}(\mathbf a)\mathbf I_3+2G\operatorname{dev}(\mathbf a)
$$

Here $\mathbf a$ is any symmetric elastic-strain tensor. Plastic incompressibility means the plastic strain rate changes shape without changing volume. Let a dot denote differentiation with respect to the loading parameter $t$; it does not introduce inertia.

**Equation (J2-5) - Plastic incompressibility**

$$
\operatorname{tr}(\dot{\boldsymbol\varepsilon}^{\mathrm p})=0,
\qquad
\operatorname{tr}(\Delta\boldsymbol\varepsilon^{\mathrm p})=0
$$

The increment statement follows by integration over a load step. Elastic volume change remains possible. The associated von Mises flow rule below also yields this zero-trace property, so incompressibility is consistent with that rule rather than an additional correction to it.

## 3. A scalar stress measure defines admissibility

To compare a multiaxial stress with a scalar yield strength, measure the magnitude of deviatoric stress. Its second invariant is defined by

**Equation (J2-6) - Second deviatoric invariant**

$$
J_2:=\tfrac12\mathbf s:\mathbf s
$$

Multiply by the conventional normalization and take the square root to obtain a quantity with stress units.

**Equation (J2-7) - Von Mises equivalent stress**

$$
\sigma_{\mathrm{eq}}:=\sqrt{3J_2}=\sqrt{\tfrac32\mathbf s:\mathbf s}
$$

The normalization makes equivalent stress equal the magnitude of axial stress in uniaxial loading. Hydrostatic stress has $\mathbf s=\mathbf0$ and hence zero equivalent stress. This pressure independence is a constitutive choice appropriate to the present von Mises model, not a universal property of all materials.

Compare the equivalent stress with the current strength $\sigma_{\mathrm y}(p)$ to define admissibility.

**Equation (J2-8) - Yield function**

$$
f(\boldsymbol\sigma,p):=\sigma_{\mathrm{eq}}-\sigma_{\mathrm y}(p)
$$

An accepted state requires $f\leq0$. The region $f<0$ is the elastic interior; $f=0$ is the yield surface. A trial state with $f>0$ violates this model's admissibility condition and requires correction. The function $\sigma_{\mathrm y}(p)$ is specified in Section 6.

## 4. The yield normal gives a flow direction only after a constitutive postulate

The yield surface describes allowed stresses; it does not by itself specify plastic deformation. First establish a geometric fact. At fixed $p$, a tangent stress perturbation $d\boldsymbol\sigma^{\mathrm{tan}}$ stays on a smooth level surface, so its first-order change in $f$ vanishes.

**Equation (J2-9) - Tangent perturbation of a level surface**

$$
df=\frac{\partial f}{\partial\boldsymbol\sigma}:d\boldsymbol\sigma^{\mathrm{tan}}=0
$$

This is the tensor analogue of a gradient having zero dot product with tangent vectors. The gradient is therefore normal to the surface where it is nonzero. Associated plasticity then adds the physical postulate that the plastic strain rate follows that outward normal.

**Equation (J2-10) - Continuous associated flow rule**

$$
\dot{\boldsymbol\varepsilon}^{\mathrm p}
=\dot\lambda\frac{\partial f}{\partial\boldsymbol\sigma},
\qquad \dot\lambda\geq0
$$

The gradient sets direction; $\dot\lambda$ sets the nonnegative amount of flow. To evaluate the gradient, differentiate $\sigma_{\mathrm{eq}}^2=\tfrac32\mathbf s:\mathbf s$ at fixed history and at $\sigma_{\mathrm{eq}}>0$.

**Equation (J2-11a) - Differential of equivalent stress**

$$
\begin{aligned}
2\sigma_{\mathrm{eq}}\,d\sigma_{\mathrm{eq}}&=3\mathbf s:d\mathbf s\\
d\mathbf s&=d\boldsymbol\sigma-\tfrac13\operatorname{tr}(d\boldsymbol\sigma)\mathbf I_3\\
\mathbf s:d\mathbf s&=\mathbf s:d\boldsymbol\sigma
\end{aligned}
$$

The last equality uses $\mathbf s:\mathbf I_3=\operatorname{tr}(\mathbf s)=0$. Divide the first line by $2\sigma_{\mathrm{eq}}$ and identify the coefficient of $d\boldsymbol\sigma$. Hardening contributes no stress derivative at fixed $p$.

**Equation (J2-11) - Von Mises yield normal**

$$
\frac{\partial f}{\partial\boldsymbol\sigma}
=\frac32\frac{\mathbf s}{\sigma_{\mathrm{eq}}}
$$

Substitution into the flow postulate supplies the explicit evolution direction.

**Equation (J2-12) - Continuous von Mises flow**

$$
\dot{\boldsymbol\varepsilon}^{\mathrm p}
=\dot\lambda\frac32\frac{\mathbf s}{\sigma_{\mathrm{eq}}}
$$

Taking the trace recovers plastic incompressibility. The normal is dimensionless but is not a unit tensor: its squared norm is $3/2$. At zero equivalent stress the normalized direction is undefined, but the positive yield strength places that state inside the elastic domain, where no flow direction needs evaluation.

## 5. Accumulated history measures path length

The components of plastic strain can increase or decrease as the loading direction changes. Their final values cannot measure total plastic activity. Define a nonnegative rate from the instantaneous plastic-strain rate, then integrate it along the path.

**Equation (J2-13) - Equivalent plastic-strain rate**

$$
\dot p:=\sqrt{\tfrac23\dot{\boldsymbol\varepsilon}^{\mathrm p}:\dot{\boldsymbol\varepsilon}^{\mathrm p}}
$$

The factor $2/3$ makes $\dot p$ equal the magnitude of the axial plastic-strain rate under incompressible uniaxial plastic flow. With $t_n,t_{n+1}$ the load-step endpoints, accumulated history adds the entire path contribution.

**Equation (J2-14) - Continuous history accumulation**

$$
p_{n+1}=p_n+\Delta p,
\qquad
\Delta p=\int_{t_n}^{t_{n+1}}\dot p\,dt
$$

To relate this history measure to the multiplier, insert Eq. (J2-12) and use $\mathbf s:\mathbf s=\tfrac23\sigma_{\mathrm{eq}}^2$.

**Equation (J2-15a) - Multiplier normalization calculation**

$$
\dot p^{\,2}
=\tfrac23\dot\lambda^{\,2}\tfrac94
\frac{\mathbf s:\mathbf s}{\sigma_{\mathrm{eq}}^2}
=\dot\lambda^{\,2}
$$

Both rates are nonnegative, so taking the square root and integrating gives the selected normalization. Define $\Delta\lambda=\int_{t_n}^{t_{n+1}}\dot\lambda\,dt$.

**Equation (J2-15) - History and multiplier increments**

$$
\dot p=\dot\lambda,
\qquad \Delta p=\Delta\lambda
$$

Their roles remain different: $p$ is stored accumulated history, while the multiplier determines the amount of flow. Opposite plastic increments can cancel in the tensor but both add to $p$. Therefore the norm of a net finite tensor increment is not the general continuous path accumulation. Lecture 6's backward-Euler update does satisfy $\Delta p=\sqrt{\tfrac23\Delta\boldsymbol\varepsilon^{\mathrm p}:\Delta\boldsymbol\varepsilon^{\mathrm p}}$ for its single endpoint-direction increment; that is a property of the discrete update.

## 6. Hardening and loading conditions complete the model

To describe increasing resistance to plastic deformation without changing the surface's shape or center, let the yield strength grow linearly with accumulated history.

**Equation (J2-16) - Linear isotropic hardening**

$$
\sigma_{\mathrm y}(p)=\sigma_{\mathrm y0}+Hp,
\qquad \sigma_{\mathrm y0}>0,\quad H\geq0
$$

Here $H$ is the slope of strength versus accumulated plastic strain, not the slope of stress versus total strain. Substitution makes the full yield function explicit.

**Equation (J2-17) - Hardened yield function**

$$
f(\boldsymbol\sigma,p)=\sigma_{\mathrm{eq}}-(\sigma_{\mathrm y0}+Hp)
$$

Admissibility, nonnegative flow, and flow only on the surface are encoded together by the continuous Kuhn-Tucker conditions.

**Equation (J2-18) - Continuous loading conditions**

$$
f\leq0,\qquad \dot\lambda\geq0,\qquad \dot\lambda f=0
$$

Thus $f<0$ requires no plastic flow, while positive flow requires $f=0$. Being on the surface alone does not prove flow: unloading or neutral loading can have $f=0$ and $\dot\lambda=0$.

During sustained active flow the changing stress and hardening state must stay on the moving surface. Differentiating its equation along the loading path gives consistency.

**Equation (J2-19) - Continuous active-flow consistency**

$$
\dot f
=\frac{\partial f}{\partial\boldsymbol\sigma}:\dot{\boldsymbol\sigma}-H\dot p=0
\qquad\text{when }\dot\lambda>0
$$

The gradient normal in Section 4 held $p$ fixed to identify a direction. Consistency instead allows $p$ to evolve. It supplies the scalar restriction that will determine the plastic amount; it is not another direction rule. In the discrete algorithm, active consistency will be imposed as $f_{n+1}=0$ at the endpoint.

## Result and next question

The model now contains kinematics, elasticity, a yield criterion, a flow law, a history measure, hardening, and loading conditions. It predicts no yielding under purely hydrostatic stress and no plastic volume increment. These are immediate model checks, while complete uniaxial and reversed-loading examples remain later exercises.

**Checkpoint:** Why do the yield criterion and associated flow direction still leave the amount of plastic deformation undetermined at a new strain?

Lecture 6 answers by choosing backward Euler and using endpoint consistency to solve for that amount. The distinction matters: a continuous constitutive law and a numerical time-step algorithm are not the same object.

## Source anchors

Dunne and Petrinic, *Introduction to Computational Plasticity*, Sections 2.2.2-2.2.3 for incompressibility and equivalent measures, Section 2.3.1 for normality, and Section 2.4.1 for linear isotropic hardening. The tensor-first ordering and the explicit level-set explanation reorganize the teaching without changing the constitutive assumptions. The local mechanics yield function has stress units. This convention and the accumulated-plastic-strain normalization are retained when the local model is used in the TFA partition framework.
