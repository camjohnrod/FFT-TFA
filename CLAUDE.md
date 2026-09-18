# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

FFT-TFA is a research-notes repository developing FFT-accelerated Transformation Field Analysis (TFA) for elastoplastic composite homogenization. `code/` now holds a single Python/numpy skeleton (`MAIN.py`) with its function bodies unimplemented (`pass`) — there is still no dependency manifest, linter, or test suite, and no way to run anything meaningful yet. Do not assume a build/lint/test toolchain beyond "Python with numpy"; check with the user before adding commands here.

## Layout

- `notes/Formulation.md` — the main synthesis document: derives the actual eigenstrain/plasticity (E/P) partition-strain relation (Fish–Cui) and two solution strategies — a Moulinec–Suquet-style FFT fixed-point iteration, and a Ladecký-style reference-preconditioned Newton–Krylov (GMRES) solve.
- `notes/verified_notes/` — a sequential derivation chain building up to `Formulation.md`. Read via its own `README.md`, which gives the intended reading order (influence functions → partitioned eigenstrain → partition-average strain transformation → nonlinear FEM/Newton structure → J2 yield/flow/hardening → elastic trial and radial return → complete J2 return-mapping algorithm).
- `references/` — source PDFs cited by the notes: Fish & Cui on eigenstate-based homogenization, Dvorak 1992, a computational plasticity text, Fish's practical multiscaling, the original Moulinec–Suquet paper, and the Ladecký et al. FFT-preconditioned FE solver paper. All citation links in the notes point at these local copies.
- `code/MAIN.py` — the sole implementation file: a config block of RVE/material/loading parameters (domain and inclusion size, element and partition counts per side, elastic moduli, J2 yield stress and hardening modulus, strain steps, fixed-point tolerance) followed by four unimplemented stub functions that map directly onto `Formulation.md`'s strategy 1: `get_influence_functions` (builds P, P₀ — Sections 1 and 6), `get_plastic_eigenstrain` (the local J2 return-mapping update, ER-3, following `verified_notes/07`), `standard_richardson_iteration` (the un-preconditioned baseline, ER-4/ER-11 with θ=0, Section 11), and `fft_preconditioned_richardson_iteration` (the FFT reference solve, ER-15). `main` is also unimplemented. There is no code yet for strategy 2 (Newton–Krylov, Section 8).

## Conventions

- **Equation labeling is per-file and load-bearing.** Each note file has its own prefix: `IF-` (01, displacement influence functions), `P-` (02, partitioned eigenstrain), `ST-` (03, partition-average strain transformation), `EN-` (04–05, elastoplastic FEM/Newton structure and J2 yield/flow/hardening — this prefix spans two files), `RR-` (06, elastic trial and radial return), `RM-` (07, complete J2 return-mapping algorithm), and `ER-` in `Formulation.md` for the actual E/P relation and its two solvers. Cross-references between files cite these labels directly (e.g. `Formulation.md` builds on `ST-` results from `03_partition_strain_transformation.md`). When editing derivations, preserve existing labels and continue the numbering scheme rather than renumbering.
- **Engineering strain/stress vector convention** (defined in `Formulation.md`, Equation ER-2): strain uses engineering shear (`2ε₁₂` etc.), stress uses physical shear, so their dot product is mechanical work. Every operator across the notes must respect this convention.
- **Sign convention for the reference Green operator**: this repo's `Γ⁰` is the *negative* of the standard Moulinec–Suquet strain Green operator (`Γ⁰_MS = -Γ⁰`). This is called out explicitly wherever Fourier formulas are quoted from the literature — preserve the stated sign when adding or checking equations.
- **Notes must be self-contained.** Every link/citation in `notes/` must resolve to a file actually present in this repo (no local machine paths, no links to not-yet-created companion code or planning docs like a syllabus/reference guide). If you cite external literature, add the PDF to `references/` and link to it there, or cite it as plain text without a link.
- `.gitattributes` sets `* text=auto eol=lf`, so files are normalized to LF line endings, including when working from WSL on a Windows checkout.

## Updating this file

Once `code/` gets a dependency manifest, linter config, or test suite, add the actual build, lint, and single-test commands here — do not guess at them from the bare `MAIN.py` skeleton.
