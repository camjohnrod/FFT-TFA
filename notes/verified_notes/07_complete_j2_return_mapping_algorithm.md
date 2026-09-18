# Complete three-dimensional $J_2$ return-mapping algorithm

This note assembles the elastic trial state, radial-return direction, and closed-form plastic multiplier into a complete strain-controlled update for small-strain von Mises plasticity with linear isotropic hardening. It then verifies that an active plastic correction ends on the updated yield surface.

## Purpose and route

Lecture 6 derived the plastic amount. This lecture turns that derivation into an executable local update: form a trial state, choose a branch, compute stress and history, and verify the endpoint. We then reconnect it to Lecture 4's Newton method. The local technique enforces constitutive admissibility at fixed supplied strain; the outer technique adjusts displacement to enforce force equilibrium.

## Assumptions, inputs, and stored state

Assume small strain, isotropic linear elasticity, associative von Mises plasticity, plastic incompressibility, rate-independent response, linear isotropic hardening, and backward-Euler integration. Require $G>0$, bulk modulus $K_{\mathrm b}>0$, $H\geq0$, and $\sigma_{\mathrm y0}>0$.

At the start of load step $n+1$:

- the supplied total strain $\boldsymbol\varepsilon_{n+1}$ is dimensionless
- the committed plastic strain $\boldsymbol\varepsilon_n^{\mathrm p}$ is a dimensionless symmetric tensor
- the committed accumulated equivalent plastic strain $p_n$ is a nonnegative dimensionless scalar
- $\mathbb C$ is the elastic stiffness tensor and $G$ is the shear modulus
- $\sigma_{\mathrm y0}$ is the initial yield stress and $H$ is the linear isotropic hardening modulus

The quantities $G$, $H$, $\sigma_{\mathrm y0}$, and all stresses have units of stress. The accumulated history $p$ and the plastic multiplier increment $\Delta\lambda$ are dimensionless under the normalization $\Delta p=\Delta\lambda$.

Bold stress and strain symbols denote symmetric $3\times3$ tensors. $\mathbf I_3$ is the identity tensor, $\operatorname{tr}(\mathbf a)=\sum_i a_{ii}$, $\operatorname{dev}(\mathbf a)=\mathbf a-\tfrac13\operatorname{tr}(\mathbf a)\mathbf I_3$, and $\mathbf a:\mathbf b=\sum_{i,j}a_{ij}b_{ij}$. The stiffness acts as $\mathbb C:\mathbf a=K_{\mathrm b}\operatorname{tr}(\mathbf a)\mathbf I_3+2G\operatorname{dev}(\mathbf a)$ for symmetric elastic strain $\mathbf a$.

The superscript $\mathrm{tr}$ refers to the predictor for step $n+1$; that subscript is omitted on trial quantities. $\Delta$ denotes a load-step increment. The yield function is $f(\boldsymbol\sigma,p)=\sigma_{\mathrm{eq}}-(\sigma_{\mathrm y0}+Hp)$, with $f\leq0$ required for admissibility. Backward Euler uses the endpoint flow direction and enforces $\Delta\lambda\geq0$ and $\Delta\lambda f_{n+1}=0$. The constitutive history consists of both $\boldsymbol\varepsilon_n^{\mathrm p}$ and $p_n$.

The supplied strain is an input to this local calculation. Inside global Newton it is the current provisional strain, not yet the converged load-step strain. Returned stress and history are therefore candidates until the outer equilibrium solve converges.

## Elastic trial state

The trial calculation freezes the committed plastic state and predicts the stress that would result if the new strain were accommodated elastically.

**Equation (RM-1) - Elastic trial stress**

$$
\boldsymbol\sigma^{\mathrm{tr}}
=
\mathbb C:
\left(
\boldsymbol\varepsilon_{n+1}
-
\boldsymbol\varepsilon_n^{\mathrm p}
\right)
$$

Separate the trial stress into its shape-changing magnitude, which tests yielding, and its mean part, which will be retained during return. These measures are

**Equation (RM-2) - Trial stress measures**

$$
\mathbf s^{\mathrm{tr}}
=
\operatorname{dev}(\boldsymbol\sigma^{\mathrm{tr}}),
\qquad
\sigma_{\mathrm{mean}}^{\mathrm{tr}}
=
\frac{1}{3}\operatorname{tr}(\boldsymbol\sigma^{\mathrm{tr}}),
\qquad
\sigma_{\mathrm{eq}}^{\mathrm{tr}}
=
\sqrt{\frac{3}{2}\mathbf s^{\mathrm{tr}}:\mathbf s^{\mathrm{tr}}}
$$

The trial yield function tests this elastic prediction against the yield strength stored at step $n$.

**Equation (RM-3) - Trial yield check**

$$
f^{\mathrm{tr}}
=
\sigma_{\mathrm{eq}}^{\mathrm{tr}}
-
\left(
\sigma_{\mathrm y0}+Hp_n
\right)
$$

## Elastic branch

If $f^{\mathrm{tr}}\leq0$, the elastic prediction is admissible. No new plastic flow occurs, so the stress is accepted and both internal variables remain unchanged.

**Equation (RM-4) - Elastic update**

$$
\boldsymbol\sigma_{n+1}
=
\boldsymbol\sigma^{\mathrm{tr}},
\qquad
\boldsymbol\varepsilon_{n+1}^{\mathrm p}
=
\boldsymbol\varepsilon_n^{\mathrm p},
\qquad
p_{n+1}=p_n,
\qquad
\Delta\lambda=0
$$

## Plastic branch

If $f^{\mathrm{tr}}>0$, the trial state is inadmissible and requires a plastic correction. For this backward-Euler model, the trial overstress is removed by two effects: plastic flow reduces equivalent stress by $3G\Delta\lambda$ and raises strength by $H\Delta\lambda$. Equating the corrected stress to the updated strength gives $f^{\mathrm{tr}}-(3G+H)\Delta\lambda=0$ (derived in Lecture 6). Solving it gives the closed-form multiplier.

**Equation (RM-5) - Plastic multiplier**

$$
\Delta\lambda
=
\frac{f^{\mathrm{tr}}}{3G+H}
$$

The associated von Mises flow rule supplies the plastic-strain increment. Radial return makes the updated and trial deviatoric stresses parallel, so the direction can be evaluated from the trial state.

**Equation (RM-6) - Plastic-strain increment**

$$
\Delta\boldsymbol\varepsilon^{\mathrm p}
=
\Delta\lambda
\frac{3}{2}
\frac{\mathbf s^{\mathrm{tr}}}
{\sigma_{\mathrm{eq}}^{\mathrm{tr}}}
$$

The active branch guarantees $\sigma_{\mathrm{eq}}^{\mathrm{tr}}>\sigma_{\mathrm y0}+Hp_n>0$, so the normalization is defined. Do not evaluate it in the elastic branch, where a hydrostatic trial state may have zero equivalent stress.

The plastic-strain tensor stores the directional permanent deformation, while the scalar accumulated history adds the nonnegative multiplier increment.

**Equation (RM-7) - Internal-variable updates**

$$
\boldsymbol\varepsilon_{n+1}^{\mathrm p}
=
\boldsymbol\varepsilon_n^{\mathrm p}
+
\Delta\boldsymbol\varepsilon^{\mathrm p},
\qquad
p_{n+1}
=
p_n+\Delta\lambda
$$

Because the plastic increment is deviatoric, it does not change hydrostatic stress. The correction shortens only the trial deviatoric stress.

**Equation (RM-8) - Corrected deviatoric stress**

$$
\mathbf s_{n+1}
=
\mathbf s^{\mathrm{tr}}
-
2G\Delta\boldsymbol\varepsilon^{\mathrm p}
=
\left(
1-
\frac{3G\Delta\lambda}
{\sigma_{\mathrm{eq}}^{\mathrm{tr}}}
\right)
\mathbf s^{\mathrm{tr}}
$$

The complete corrected stress combines the unchanged trial mean stress with the corrected deviatoric stress.

**Equation (RM-9) - Complete corrected stress**

$$
\boldsymbol\sigma_{n+1}
=
\sigma_{\mathrm{mean}}^{\mathrm{tr}}\mathbf I_3
+
\mathbf s_{n+1}
$$

## Verification of the corrected state

The radial correction reduces the equivalent stress by $3G\Delta\lambda$.

**Equation (RM-10) - Corrected equivalent stress**

$$
\sigma_{\mathrm{eq},n+1}
=
\sigma_{\mathrm{eq}}^{\mathrm{tr}}
-
3G\Delta\lambda
$$

The updated hardened yield function is therefore

**Equation (RM-11) - Corrected yield residual**

$$
\begin{aligned}
f_{n+1}
&=
\sigma_{\mathrm{eq},n+1}
-
\left(
\sigma_{\mathrm y0}+Hp_{n+1}
\right)\\
&=
\left(
\sigma_{\mathrm{eq}}^{\mathrm{tr}}
-3G\Delta\lambda
\right)
-
\left[
\sigma_{\mathrm y0}
+H\left(p_n+\Delta\lambda\right)
\right]\\
&=
f^{\mathrm{tr}}
-
(3G+H)\Delta\lambda
\end{aligned}
$$

Substitution of the closed-form multiplier verifies consistency at the end of the plastic correction.

**Equation (RM-12) - Yield-surface consistency**

$$
f_{n+1}
=
f^{\mathrm{tr}}
-(3G+H)
\frac{f^{\mathrm{tr}}}{3G+H}
=0
$$

Thus an active plastic update satisfies

**Equation (RM-13) - Plastic endpoint**

$$
\sigma_{\mathrm{eq},n+1}
=
\sigma_{\mathrm y0}+Hp_{n+1}
$$

A corrected value $f_{n+1}>0$ would remain inadmissible. Although $f_{n+1}<0$ is admissible for an elastic state, it is inconsistent with this active branch because $\Delta\lambda>0$ requires $f_{n+1}=0$ through the complementarity condition $\Delta\lambda f_{n+1}=0$.

## Use the returned state inside global Newton

At a Gauss point, Lecture 4 supplies $\boldsymbol\varepsilon_{n+1}^{\mathrm V,(k)}=\mathbf B\mathbf d_{n+1}^{(k)}$ in engineering Voigt form. Here $(k)$ is a global iterate, $\mathbf d$ the free displacement vector, and $\mathbf B$ the strain-displacement matrix. Convert that vector to a tensor before applying this lecture's contractions: the shear tensor entries are half the engineering shear entries. Convert the returned tensor stress to the vector order $(11,22,33,12,23,13)$ without doubling its shear entries.

To specify the complete local call, define the committed history collection $\mathbf z_n=(\boldsymbol\varepsilon_n^{\mathrm p},p_n)$. The tensor update above realizes Lecture 4's stress map $\mathcal S$ and history map $\mathcal Z$ for this particular material model.

**Equation (RM-14) - Local call at a global iterate**

$$
\boldsymbol\sigma_{n+1}^{\mathrm V,(k)}
=\mathcal S(\boldsymbol\varepsilon_{n+1}^{\mathrm V,(k)};\mathbf z_n),
\qquad
\mathbf z_{n+1}^{\mathrm{cand},(k)}
=\mathcal Z(\boldsymbol\varepsilon_{n+1}^{\mathrm V,(k)};\mathbf z_n)
$$

The superscript $\mathrm V$ distinguishes vectors from the tensors in the local derivation. Assemble internal force from the returned stress, not from an inadmissible trial stress. If global equilibrium has not converged, Newton changes displacement and the local call starts again from the same $\mathbf z_n$. It must not start from the previous iterate's candidate history.

The generic constitutive tangent can now be motivated explicitly. In engineering Voigt form, $\mathbf C$ is the $6\times6$ representation of $\mathbb C$, and the discrete stress update satisfies the elastic law with the candidate plastic strain.

**Equation (EN-22) - Returned stress in FEM representation**

$$
\boldsymbol\sigma_{n+1}^{\mathrm V,(k)}
=\mathbf C\left(\boldsymbol\varepsilon_{n+1}^{\mathrm V,(k)}
-\boldsymbol\varepsilon_{n+1}^{\mathrm p,V,(k)}\right)
$$

Differentiate this complete update with respect to supplied strain while holding $\boldsymbol\varepsilon_n^{\mathrm p}$ and $p_n$ fixed. The plastic response changes with strain, so its derivative must be included.

**Equation (EN-24) - Generic plasticity tangent structure**

$$
\mathbf C_{\mathrm{alg}}^{(k)}
=\mathbf C\left[\mathbf I_6
-\left.\frac{\partial\boldsymbol\varepsilon_{n+1}^{\mathrm p,V}}
{\partial\boldsymbol\varepsilon_{n+1}^{\mathrm V}}\right|_{\boldsymbol\varepsilon_{n+1}^{\mathrm V,(k)},\,\mathbf z_n\,\mathrm{fixed}}\right]
$$

Here $\mathbf I_6$ is the identity matrix on engineering-strain vectors, and $\mathbf C_{\mathrm{alg}}$ has stress units. On the strictly elastic branch the plastic-strain derivative vanishes, giving $\mathbf C_{\mathrm{alg}}=\mathbf C$. At a yield-branch transition the applicable derivative must follow the selected branch; there need not be a single smooth two-sided derivative.

The global tangent is then assembled using Lecture 4's Eq. (EN-26). During active return the multiplier and normalized trial direction both depend on strain. Differentiating both is the remaining task needed for the closed-form consistent tangent; Eq. (EN-24) establishes its structure but does not perform that derivation.

## Compact algorithm and checks

For each supplied total strain:

1. Read the fixed committed pair $(\boldsymbol\varepsilon_n^{\mathrm p},p_n)$ and form Eqs. (RM-1)-(RM-3).
2. If $f^{\mathrm{tr}}\leq0$, return Eq. (RM-4) without evaluating a normalized flow direction.
3. Otherwise compute the multiplier, plastic increment, candidate history, and corrected stress using Eqs. (RM-5)-(RM-9).
4. Check the active endpoint using Eqs. (RM-11)-(RM-13), and check zero plastic trace, unchanged mean stress, and nondecreasing $p$.
5. Return the stress and candidate history to the outer solve. The elastic tangent is known; the plastic consistent tangent will be derived in Lecture 10 after the TFA connection and reduced solver are established.
6. Commit the candidate history only after global equilibrium converges. Discard it if the load step fails.

The branch equations above describe exact arithmetic. In computation, yield and force residuals require separate, dimensionally consistent tolerances. A yield residual has stress units, whereas the global residual has force units. Small numerical residuals should be assessed against those tolerances, not literal floating-point equality.

**Checkpoint:** This update requires a supplied local total strain. If only the average strain of a heterogeneous RVE is prescribed, what information is still missing before the partition material updates can be evaluated?

This completes the stress-and-history update and its place in Newton's architecture. Lecture 8 next connects this local update to the TFA partition relation, beginning with a concrete explanation of supplied local strain versus imposed average strain. Lecture 9 develops the coupled solve with symbolic material derivatives, Lecture 10 evaluates those derivatives, and Lecture 11 assembles and checks the implementation.

## Source anchors and scope

Primary teaching source: F. Dunne and N. Petrinic, *Introduction to Computational Plasticity*, Chapter 5, especially the implicit radial-return development in Section 5.2.1. The tensor update follows printed pp. 146-149 (PDF pp. 161-164), Eqs. (5.7)-(5.20); the tangent construction is motivated by Section 5.3. The formulation here uses the standard normalization $\Delta p=\Delta\lambda$. This normalization will also be used in the TFA partition routines; no paper-specific multiplier conversion is required. Fish's *Practical Multiscaling*, Section 4.2.5, supplies the later reduced-solver connection; see [REFERENCE_GUIDE.md](../REFERENCE_GUIDE.md).

The coupled TFA formulation, closed-form consistent tangent and complete RVE loading checks remain to be taught and approved according to [SYLLABUS.md](../SYLLABUS.md). This local update is analytical for the stated model; an outer spatial solve determines the strains supplied to it when those strains are not individually prescribed.
