# Nonlinear finite-element equilibrium and Newton's method

## Purpose and route

Lectures 1-3 derived an elastic response to prescribed macroscopic strain and eigenstrain, ending with the partition strain relation ST-12 and its component form ST-14. Why is a nonlinear solver needed if those elastic operators are fixed? Because the current eigenstrain is no longer prescribed when a material evolves: its value depends on the strain and stored history that the solution itself must determine.

We first learn the solution architecture in conventional full FEM. Starting from virtual work, we recover linear-elastic equilibrium, replace the elastic law by a general history-dependent stress update, and derive Newton's correction by a first-order Taylor expansion. Finally, the chain rule identifies the material tangent that the local update must supply. Lectures 5-7 construct that update for plasticity.

This is a teaching bridge, not yet the reduced TFA Newton system. In the RVE backbone, $\mathbf d$ represented periodic fluctuations and strain included an imposed macroscopic part. Here $\mathbf d$ represents free displacement DOFs of a body $\Omega$. The common technique is enforcing equilibrium while evaluating a local material response; the unknowns and boundary conditions of the two problems differ.

## Assumptions and notation

Assume three-dimensional small strain, quasi-static equilibrium, fixed geometry and FE shape functions, and prescribed external forces independent of displacement. Homogeneous essential displacement constraints are eliminated. For nonzero prescribed displacements, a known displacement lifting contributes to strain and the reduced right-hand side; it cannot simply be omitted from $\mathbf B\mathbf d$.

| Quantity | Role | Size and units |
|---|---|---|
| $\Omega$, $dV$ | Body and volume element | Volume |
| $N_N$, $N_d$ | Number of nodes and independent free DOFs | Counts |
| $N_I$, $\mathbf d_I$ | Nodal shape function and displacement vector | Scalar, dimensionless; $3$-vector, length |
| $\mathbf d$, $\mathbf B$ | Free displacement vector and strain-displacement matrix | $N_d$-vector, length; $6\times N_d$, inverse length |
| $\mathbf C$, $\mathbf C_{\mathrm{alg}}$ | Elastic matrix and discrete material tangent | $6\times6$, stress |
| $\mathbf F^{\mathrm{int}}$, $\mathbf F^{\mathrm{ext}}$, $\mathbf R$ | Internal force, external force, force residual | $N_d$-vectors, force |
| $\mathbf K$, $\mathbf K_{\mathrm T}$ | Elastic stiffness and global tangent | $N_d\times N_d$, force/length |
| $\mathbf z_n$ | Complete committed material history at a Gauss point | Model-dependent state collection |

The elastic matrix $\mathbf C$ plays the same constitutive role here as $\mathbf L(\mathbf y)$ in the heterogeneous RVE. Bold symbols in this lecture denote vectors and matrices. To preserve internal work in matrix form, strain uses engineering shear.

**Equation (EN-D1) - Engineering-strain vector**

$$
\boldsymbol\varepsilon
=(\varepsilon_{11},\varepsilon_{22},\varepsilon_{33},\gamma_{12},\gamma_{23},\gamma_{13})^{\mathsf T},
\qquad \gamma_{ij}=2\varepsilon_{ij}
$$

Its work-conjugate stress vector uses physical shear stress in the same order.

**Equation (EN-D2) - Stress vector**

$$
\boldsymbol\sigma
=(\sigma_{11},\sigma_{22},\sigma_{33},\sigma_{12},\sigma_{23},\sigma_{13})^{\mathsf T}
$$

Strain is dimensionless. Superscripts $\mathrm{int}$ and $\mathrm{ext}$ indicate force type; $n,n+1$ are load-step endpoints and $(k)$ is a Newton iterate within step $n+1$. A load step changes the prescribed loading. A Newton iteration adjusts a provisional solution at that same loading.

## 1. Kinematics supplies strain to the material law

The material response is evaluated at integration points, so we need a map from nodal displacement to pointwise strain. Begin with the explicit nodal interpolation of body displacement $\mathbf v$; $I$ is a node index.

**Equation (EN-3a) - Nodal displacement interpolation**

$$
\mathbf v(\mathbf x)=\sum_{I=1}^{N_N}N_I(\mathbf x)\mathbf d_I
$$

Here $\mathbf x$ locates a point in the conventional body. Taking the symmetric derivative, mapping to engineering strain, and eliminating constrained DOFs gives the strain-displacement relation at each point.

**Equation (EN-3) - Strain supplied by displacement**

$$
\boldsymbol\varepsilon=\mathbf B\mathbf d
$$

Each row of $\mathbf B$ selects an output strain component; each column corresponds to a displacement DOF. Under the fixed-geometry assumptions this map is linear, so its derivative is the same matrix. This derivative will enter the Newton tangent.

**Equation (EN-4) - Kinematic derivative**

$$
\frac{\partial\boldsymbol\varepsilon}{\partial\mathbf d}=\mathbf B
$$

## 2. Virtual work produces the equilibrium residual

Why integrate stress against $\mathbf B^{\mathsf T}$? A stress field produces nodal forces through its work on admissible virtual displacements. Let $\delta\mathbf d$ be arbitrary virtual free DOFs. Then $\delta\boldsymbol\varepsilon=\mathbf B\delta\mathbf d$, and weak equilibrium states that internal and external virtual work agree.

**Equation (EN-5a) - Discrete virtual-work balance**

$$
\int_\Omega(\mathbf B\delta\mathbf d)^{\mathsf T}\boldsymbol\sigma\,dV
=\delta\mathbf d^{\mathsf T}\mathbf F^{\mathrm{ext}}
$$

Factor out $\delta\mathbf d^{\mathsf T}$. Since the virtual DOFs are arbitrary, their coefficient must vanish. This identifies the internal force and the equilibrium condition.

**Equation (EN-5) - Internal nodal force and equilibrium**

$$
\mathbf F^{\mathrm{int}}:=\int_\Omega\mathbf B^{\mathsf T}\boldsymbol\sigma\,dV,
\qquad \mathbf F^{\mathrm{int}}=\mathbf F^{\mathrm{ext}}
$$

For a linear-elastic material with no eigenstrain, the stress is $\boldsymbol\sigma=\mathbf C\boldsymbol\varepsilon$. Substitute the kinematics to expose the displacement dependence of internal force.

**Equation (EN-6) - Linear-elastic internal force**

$$
\mathbf F^{\mathrm{int}}
=\left[\int_\Omega\mathbf B^{\mathsf T}\mathbf C\mathbf B\,dV\right]\mathbf d
$$

The bracket is fixed for fixed elasticity, geometry, and discretization. Define it once to reuse the linear force-displacement map.

**Equation (EN-7) - Elastic stiffness matrix**

$$
\mathbf K:=\int_\Omega\mathbf B^{\mathsf T}\mathbf C\mathbf B\,dV
$$

At the new prescribed load, equilibrium therefore reduces to one linear solve on a mechanically stable constrained space.

**Equation (EN-8) - Linear-elastic equilibrium**

$$
\mathbf K\mathbf d_{n+1}=\mathbf F^{\mathrm{ext}}_{n+1}
$$

This is the same linearity exploited by the influence-function construction: a fixed operator acts on changing inputs. Gauss quadrature evaluates the volume integrals by weighted material-point evaluations within each element, followed by assembly into the global force and stiffness.

## 3. A changing material state makes internal force nonlinear

For a history-dependent material, the stress at the new step must be computed from a supplied strain and the complete state stored at step $n$. Introduce $\mathcal S$ for the discrete stress-update map and $\mathcal Z$ for its candidate history update. Their purpose is to define what the global solver asks the local material calculation to return, before selecting a particular material model.

**Equation (EN-9) - Local material-update interface**

$$
\boldsymbol\sigma_{n+1}=\mathcal S(\boldsymbol\varepsilon_{n+1};\mathbf z_n),
\qquad
\mathbf z_{n+1}^{\mathrm{cand}}=\mathcal Z(\boldsymbol\varepsilon_{n+1};\mathbf z_n)
$$

The semicolon separates the changing strain input from fixed committed history. Material properties and the selected load increment are implicit fixed parameters of these maps. The history may contain several variables: the plasticity model will need both plastic strain and accumulated hardening history.

Substitute the strain supplied by displacement into the stress map. This is the source of the global nonlinear dependence.

**Equation (EN-10) - Displacement-dependent constitutive response**

$$
\boldsymbol\sigma_{n+1}(\mathbf d_{n+1})
=\mathcal S(\mathbf B\mathbf d_{n+1};\mathbf z_n)
$$

The governing force balance has the same physical meaning as before, but it is now an equation for a nonlinear function of displacement.

**Equation (EN-11) - Nonlinear equilibrium**

$$
\mathbf F^{\mathrm{int}}_{n+1}(\mathbf d_{n+1})
=\mathbf F^{\mathrm{ext}}_{n+1}
$$

The local update makes this force function concrete at every Gauss point.

**Equation (EN-12) - Internal force from the material update**

$$
\mathbf F^{\mathrm{int}}_{n+1}(\mathbf d_{n+1})
=\int_\Omega\mathbf B^{\mathsf T}
\mathcal S(\mathbf B\mathbf d_{n+1};\mathbf z_n)\,dV
$$

Plasticity will provide a specific instance: elastic properties can remain fixed while the new stress-free strain evolves with the solution. Fixed elastic matrices therefore do not imply a linear overall problem.

## 4. Newton's method corrects force imbalance

We generally cannot isolate displacement algebraically in Eq. (EN-11). Newton's method instead solves a sequence of local linear approximations. At the current guess $\mathbf d_{n+1}^{(k)}$, quantify the unmet force balance using external minus internal force.

**Equation (EN-13) - Force residual**

$$
\mathbf R_{n+1}^{(k)}
:=\mathbf F^{\mathrm{ext}}_{n+1}-\mathbf F^{\mathrm{int},(k)}_{n+1}
$$

Let $\delta\mathbf d_{n+1}^{(k)}$ denote a correction to this guess, not a physical load-step increment. A first-order Taylor expansion predicts how that correction changes internal force.

**Equation (EN-14) - Local linearization of internal force**

$$
\mathbf F^{\mathrm{int}}_{n+1}(\mathbf d_{n+1}^{(k)}+\delta\mathbf d_{n+1}^{(k)})
\approx\mathbf F^{\mathrm{int},(k)}_{n+1}
+\mathbf K_{\mathrm T}^{(k)}\delta\mathbf d_{n+1}^{(k)}
$$

The coefficient is the derivative evaluated at the current guess, with loading and committed history fixed.

**Equation (EN-15) - Global tangent stiffness**

$$
\mathbf K_{\mathrm T}^{(k)}
:=\left.\frac{\partial\mathbf F^{\mathrm{int}}_{n+1}}{\partial\mathbf d_{n+1}}
\right|_{\mathbf d_{n+1}^{(k)},\,\mathbf z_n\,\mathrm{fixed}}
$$

Require the linearized internal force to equal the prescribed external force, then subtract the current internal force. This gives the correction equation and fixes its sign consistently with Eq. (EN-13).

**Equation (EN-16) - Newton correction equation**

$$
\mathbf K_{\mathrm T}^{(k)}\delta\mathbf d_{n+1}^{(k)}=\mathbf R_{n+1}^{(k)}
$$

Apply the correction and reevaluate the nonlinear force; the Taylor approximation itself is not the final equilibrium test.

**Equation (EN-17) - Updated displacement guess**

$$
\mathbf d_{n+1}^{(k+1)}=\mathbf d_{n+1}^{(k)}+\delta\mathbf d_{n+1}^{(k)}
$$

The tangent is a derivative, not a secant matrix satisfying $\mathbf F^{\mathrm{int}}=\mathbf K_{\mathrm T}\mathbf d$. Newton's linearization is local; convergence is not guaranteed from an arbitrary initial guess. Load subdivision or globalization may be needed when a full correction fails. Those choices do not change the equilibrium equation.

## 5. The chain rule determines the material tangent

At every Newton iterate, first evaluate the strain demanded by the current displacement guess.

**Equation (EN-18) - Strain at a global iterate**

$$
\boldsymbol\varepsilon_{n+1}^{(k)}=\mathbf B\mathbf d_{n+1}^{(k)}
$$

Call the local update from Eq. (EN-9) using this strain and the unchanged history $\mathbf z_n$. To linearize the resulting stress consistently, differentiate the discrete stress-update algorithm itself.

**Equation (EN-23) - Algorithmic material tangent**

$$
\mathbf C_{\mathrm{alg}}^{(k)}
:=\left.\frac{\partial\mathcal S(\boldsymbol\varepsilon;\mathbf z_n)}
{\partial\boldsymbol\varepsilon}\right|_{\boldsymbol\varepsilon_{n+1}^{(k)}}
$$

The tangent has stress units and maps an infinitesimal strain perturbation to the corresponding stress perturbation. For a linear-elastic update the derivative is simply the elastic matrix.

**Equation (EN-25) - Elastic material tangent**

$$
\mathbf C_{\mathrm{alg}}=\mathbf C
$$

To obtain the global derivative, differentiate Eq. (EN-12). The fixed $\mathbf B$ can stay outside the material derivative, and Eq. (EN-4) supplies the inner derivative.

**Equation (EN-26) - Assembly of the consistent global tangent**

$$
\begin{aligned}
\mathbf K_{\mathrm T}^{(k)}
&=\int_\Omega\mathbf B^{\mathsf T}
\frac{\partial\boldsymbol\sigma_{n+1}^{(k)}}{\partial\boldsymbol\varepsilon_{n+1}^{(k)}}
\frac{\partial\boldsymbol\varepsilon_{n+1}^{(k)}}{\partial\mathbf d_{n+1}^{(k)}}\,dV\\
&=\int_\Omega\mathbf B^{\mathsf T}\mathbf C_{\mathrm{alg}}^{(k)}\mathbf B\,dV
\end{aligned}
$$

The product sizes are $(N_d\times6)(6\times6)(6\times N_d)$, giving the required $N_d\times N_d$ stiffness. There is no geometric or displacement-dependent external-load tangent under the stated assumptions. An exact discrete material tangent supports rapid local Newton convergence on a smooth branch; at branch transitions the derivative can change.

## 6. One complete global load step

Given the converged displacement $\mathbf d_n$, all committed local histories $\mathbf z_n$, and the new external force, start with a displacement guess. Reusing the old displacement is a simple choice for the present fixed homogeneous constraints.

**Equation (EN-27) - Initial displacement guess**

$$
\mathbf d_{n+1}^{(0)}=\mathbf d_n
$$

For $k=0,1,2,\ldots$:

1. Evaluate Eq. (EN-18) at every Gauss point.
2. Compute its stress and candidate history using Eq. (EN-9), always starting from $\mathbf z_n$.
3. Assemble internal force and evaluate the residual using Eqs. (EN-5) and (EN-13).
4. If equilibrium meets the specified tolerance, accept the displacement and commit all candidate histories.
5. Otherwise obtain the material tangents, assemble Eq. (EN-26), solve Eq. (EN-16), and update Eq. (EN-17).
6. Reevaluate every local update from the same committed history at the new strain. Never accumulate history from unconverged guesses. If the global solve fails, keep the old committed state and retry with an appropriate solver or load-step adjustment.

A residual tolerance must have a stated scale. For example, let $R_{\mathrm{abs}}$ be an absolute force tolerance, $\eta_R$ a dimensionless relative tolerance, and $F_{\mathrm{ref}}>0$ a fixed characteristic force for this step. With $\|\cdot\|_2$ the Euclidean vector norm, a possible stopping test is

**Equation (EN-28) - Scaled equilibrium check**

$$
\|\mathbf R_{n+1}^{(k)}\|_2
\leq R_{\mathrm{abs}}+\eta_R F_{\mathrm{ref}}
$$

This is a numerical choice, not a constitutive law. Force equilibrium does not certify time-step accuracy or the correctness of a material update.

## Result, checks, and next question

If every material update is linear elastic, Eq. (EN-26) recovers $\mathbf K$, and a Newton correction solves the linear problem. In the nonlinear case, the outer solver needs a local stress update and its derivative. That is the precise reason to study plasticity next.

**Checkpoint:** Why must a local material update be recomputed from step-$n$ history after each Newton correction, even if its previous candidate stress was constitutively admissible?

Lecture 5 specifies the local plasticity model, Lecture 6 integrates it, and Lecture 7 places the resulting return algorithm into this global loop. The plasticity-specific tangent structure is explained there; its closed-form evaluation is deferred to Lecture 10, after Lectures 8-9 connect the material routine to the TFA partition problem and establish the reduced solver.

## Source anchors

Dunne and Petrinic, *Introduction to Computational Plasticity*, Section 4.5.2 for nonlinear FE equilibrium and tangent methods; Section 5.2.1, printed pp. 146-149 (PDF pp. 161-164), for implicit material integration; and Section 5.3, printed pp. 150-152, for the material Jacobian. The generic update interface and outside-in lecture organization are pedagogical notation choices. Fish-Cui Section 2.2, Eqs. (6)-(10), supplies the preceding TFA backbone, not the conventional body boundary conditions used here.
