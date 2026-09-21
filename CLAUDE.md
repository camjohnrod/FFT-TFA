# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

FFT-TFA is a research-notes repository developing FFT-accelerated Transformation Field Analysis (TFA) for elastoplastic composite homogenization. `code/` holds a single Python/numpy file (`MAIN.py`) that is being built up step by step: only `get_L`, `get_grid_id` and `get_periodic_mesh` are implemented so far, and every other function is still a `pass` stub or a short wish-list of calls to stubs. There is no dependency manifest, linter, or test suite. Do not assume a build/lint/test toolchain beyond "Python with numpy" (the local environment has Python 3.12 and numpy 2.x; scipy is not installed); check with the user before adding commands or dependencies here.

Run it with `python3 code/MAIN.py`. While the later stubs return `None`, `main()` runs to the end without error and `E`, `P`, `P0` are all `None`; that is expected, not a bug to chase.

Indexing convention (shared by nodes, elements and partitions): x-fastest, `id = i + count_per_side*j + count_per_side**2*k`, via `get_grid_id`. Hex8 corners use the same rule with count 2 (corner = a + 2b + 4c for offsets a, b, c in {0, 1}), not the textbook counter-clockwise order; `get_B` must use this ordering. Prefer x-fastest ordering wherever an ordering choice comes up.

## Layout

- `notes/Formulation.md` — the main synthesis document: derives the actual eigenstrain/plasticity (E/P) partition-strain relation (Fish–Cui) and two solution strategies — a Moulinec–Suquet-style FFT fixed-point iteration, and a Ladecký-style reference-preconditioned Newton–Krylov (GMRES) solve.
- `notes/verified_notes/` — a sequential derivation chain building up to `Formulation.md`. Read via its own `README.md`, which gives the intended reading order (influence functions → partitioned eigenstrain → partition-average strain transformation → nonlinear FEM/Newton structure → J2 yield/flow/hardening → elastic trial and radial return → complete J2 return-mapping algorithm).
- `references/` — source PDFs cited by the notes: Fish & Cui on eigenstate-based homogenization, Dvorak 1992, a computational plasticity text, Fish's practical multiscaling, the original Moulinec–Suquet paper, and the Ladecký et al. FFT-preconditioned FE solver paper. All citation links in the notes point at these local copies.
- `code/MAIN.py` — the sole implementation file: a config block of RVE/material/loading parameters (domain and inclusion size, element and partition counts per side, elastic moduli, J2 yield stress and hardening modulus, strain steps, fixed-point tolerance), then the functions, then `main`. The later stubs (`get_plastic_eigenstrain`, `standard_richardson_iteration`, `fft_preconditioned_richardson_iteration`) map onto `Formulation.md`'s strategy 1 (the local J2 return map ER-3 following `verified_notes/07`; the un-preconditioned baseline ER-4/ER-11 with θ=0, Section 11; the FFT reference solve ER-15). There is no code yet for strategy 2 (Newton–Krylov, Section 8).

## How MAIN.py is organized

The offline stage, which produces the influence operators, is the part being built now. Its design choices are settled and are not obvious from reading stubs:

- **Influence operators come from a finite-element solve, not from a Fourier kernel.** `main` builds a periodic 3D hexahedral mesh, the strain-displacement matrix `B`, a per-material stiffness (`get_L`, 6×6) and then a per-element stiffness array (one 6×6 per element, from element material ids: 0 = matrix, 1 = inclusion). `get_influence_functions(B, L_per_element, element_nodes, element_partition_ids)` builds `K` (`get_K`) and two load sets, `get_F_macrostrain` (unit macro-strain) and `get_F_eigenstrain` (unit eigenstrain in each partition), then calls `solve_influence_function(K, F, ...)` once per load set to get `E` and `P`. That helper chains `solve_for_displacements` (`K u = F`) and `get_partition_average_strain` (average the resulting strain over each partition). These are ST-10/ST-11 and ER-24. `solve_influence_function` does not yet return its result.
- **`P₀` is the same function called a second time** with one uniform reference stiffness in every element (currently a placeholder using the matrix stiffness). This keeps `P₀` consistent with the discretization and avoids implementing the Fourier Green operator or its sign convention.
- **`E` and `P` are the partition strain-influence matrices `E^B`, `P^BA` of the notes**, not Young's modulus or pressure; Young's modulus is `elastic_modulus_*` in the config.
- **Geometry**: 9 elements and 3 partitions per side, 3D, with the inclusion equal to the centre partition (`inclusion_side_length` = domain / 3, volume fraction 1/27). This is deliberate: each partition must hold a single material, so inclusion edges must fall on partition edges (and element edges). Only the matrix yields; the inclusion stays elastic.
- `K` is singular (rigid translations), so the displacement solve must pin one node; a dense `np.linalg.solve` is the intended approach at this mesh size.

## Working on MAIN.py

The user is building this deliberately, to understand the motivation for every function. Follow their process:

- Work top-down and demand-driven: write the caller as a short wish-list of calls, stub what it needs with `pass`, then implement leaf functions first, one small step at a time. Don't implement ahead of what was asked, and don't write code when told not to.
- Readability comes first. Use descriptive names; avoid opaque single-letter operators beyond the notation already in use (`E, P, K, F, B, L`), and avoid clever abstractions such as returning factorization objects or callables.
- **Do not add comments or docstrings** to `MAIN.py`; the user writes them. Explain shapes and reasoning in chat instead.
- Everything stays in `MAIN.py` for now; don't split it into modules.
- Equations in chat are shown as unicode in code blocks (LaTeX does not render in the user's view).

## Conventions

- **Equation labeling is per-file and load-bearing.** Each note file has its own prefix: `IF-` (01, displacement influence functions), `P-` (02, partitioned eigenstrain), `ST-` (03, partition-average strain transformation), `EN-` (04–05, elastoplastic FEM/Newton structure and J2 yield/flow/hardening — this prefix spans two files), `RR-` (06, elastic trial and radial return), `RM-` (07, complete J2 return-mapping algorithm), and `ER-` in `Formulation.md` for the actual E/P relation and its two solvers. Cross-references between files cite these labels directly (e.g. `Formulation.md` builds on `ST-` results from `03_partition_strain_transformation.md`). When editing derivations, preserve existing labels and continue the numbering scheme rather than renumbering.
- **Engineering strain/stress vector convention** (defined in `Formulation.md`, Equation ER-2): strain uses engineering shear (`2ε₁₂` etc.), stress uses physical shear, so their dot product is mechanical work. Every operator across the notes must respect this convention.
- **Sign convention for the reference Green operator**: this repo's `Γ⁰` is the *negative* of the standard Moulinec–Suquet strain Green operator (`Γ⁰_MS = -Γ⁰`). This is called out explicitly wherever Fourier formulas are quoted from the literature — preserve the stated sign when adding or checking equations.
- **Notes must be self-contained.** Every link/citation in `notes/` must resolve to a file actually present in this repo (no local machine paths, no links to not-yet-created companion code or planning docs like a syllabus/reference guide). If you cite external literature, add the PDF to `references/` and link to it there, or cite it as plain text without a link.
- `.gitattributes` sets `* text=auto eol=lf`, so files are normalized to LF line endings, including when working from WSL on a Windows checkout.

## Updating this file

Once `code/` gets a dependency manifest, linter config, or test suite, add the actual build, lint, and single-test commands here — do not guess at them from the `MAIN.py` skeleton. Also revise "How MAIN.py is organized" as the skeleton fills in, since it describes design decisions rather than the current function list.
