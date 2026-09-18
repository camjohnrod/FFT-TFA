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