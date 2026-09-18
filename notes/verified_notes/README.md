# Verified lecture notes

These seven lectures contain the student-approved derivation sequence, professionally reorganized under the authorized September 2026 editorial review. Each lecture states its own assumptions, repeats needed notation, motivates its techniques and equations, and ends with a conceptual checkpoint and source anchors. New course material still requires explicit approval under `TEACHING_PROTOCOL.md`.

## Reading route

| Lecture | File | Question and technique |
|---|---|---|
| 1 | [Displacement influence functions](01_displacement_influence_functions.md) | How can fixed elastic response be reused? Derive weak equilibrium and exploit superposition |
| 2 | [Partitioned eigenstrain and displacement](02_partitioned_eigenstrain_displacement.md) | How can a distributed eigenstrain field be reduced? Use partition indicators and preintegrated responses |
| 3 | [Partition-average strain transformation](03_partition_strain_transformation.md) | What strain does each partition experience? Differentiate and average to derive ST-12 and ST-14 |
| 4 | [Nonlinear FEM and Newton](04_elastoplastic_fem_newton_structure.md) | How do we solve when material response depends on unknown strain and history? Linearize the residual and assemble its tangent |
| 5 | [Plasticity: yield, flow, and hardening](05_j2_yield_flow_and_hardening.md) | What local model supplies that history-dependent stress? Introduce the plastic split and continuous constitutive laws |
| 6 | [Elastic trial and radial return](06_j2_trial_state_and_radial_return.md) | How are those laws integrated over a step? Use backward Euler, an elastic predictor, and scalar consistency |
| 7 | [Complete return update inside Newton](07_complete_j2_return_mapping_algorithm.md) | How are stress and history returned and checked? Assemble branches, verify admissibility, and reconnect to global equilibrium |

Lectures 1-3 complete the elastic relation but leave evolving eigenstrain undetermined. Lecture 4 explains the numerical architecture needed for a changing constitutive response. Lectures 5-7 supply its plasticity model and stress/history update. Each file stands alone through a compact prerequisite recap; the full derivations build across the sequence.

## Shared conventions and equation locations

See [NOTATION.md](../NOTATION.md) for the shared dictionary and [TEACHING_PROTOCOL.md](../TEACHING_PROTOCOL.md) for lecture formatting. Tensors and engineering-Voigt vectors are explicitly distinguished. $A$ is always the source partition and $B$ the observation partition in $\mathbf P^{BA}$.

- IF labels belong to Lecture 1; P labels to Lecture 2; ST labels to Lecture 3
- ST-12 is the matrix partition-strain equation; ST-14 is its component expansion
- EN labels primarily belong to Lecture 4; EN-1 and EN-2 (plastic split and elastic law) are in Lecture 5; EN-22 and EN-24 (returned stress and generic plasticity tangent) are in Lecture 7
- J2 labels belong to Lecture 5, RR labels to Lecture 6, and RM labels to Lecture 7
- Labels are stable identifiers, not a claim of consecutive numbering. Removed redundant displays leave gaps; inserted derivation steps use suffixes

## Completion boundary

The complete local stress-and-history update and active yield-surface verification are established. Lecture 7 contains the generic tangent structure, not its closed-form evaluation. The conceptual connection between supplied local strain and the coupled spatial problem has been taught live as part of Lecture 8 but is not yet approved as notes; restart where CURRENT_STATE specifies, not with the tangent derivation. Lectures 8-11 are plans only in the syllabus: TFA closure, reduced Newton with symbolic derivative, consistent derivatives, then implementation and verification. No notes for those lectures have been authored.

[CURRENT_STATE.md](../CURRENT_STATE.md) owns the exact restart point. [SYLLABUS.md](../SYLLABUS.md) owns the complete course roadmap. [EDITORIAL_REVIEW.md](../EDITORIAL_REVIEW.md) records the editorial corrections and verification scope. [REFERENCE_GUIDE.md](../REFERENCE_GUIDE.md) defines how to use the four original source PDFs. The destination is a strain-driven periodic RVE implementation of TFA with small-strain J2 plasticity and linear isotropic hardening.
