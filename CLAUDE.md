# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

FFT-TFA is at the scaffold stage. The README contains only the title, and there is no source code, build system, dependency manifest, linter, or test suite yet. Do not assume a language or toolchain; check with the user or inspect `code/` before adding build/test commands here.

## Layout

- `code/` — implementation (currently empty)
- `notes/` — working notes (currently empty)
- `references/` — reference material (currently empty)

Each directory holds only a `.gitkeep` placeholder; remove it once real files are added.

## Conventions

- `.gitattributes` sets `* text=auto eol=lf`, so files are normalized to LF line endings, including when working from WSL on a Windows checkout.

## Updating this file

Once code lands in `code/`, add the build, lint, and single-test commands and a short architecture overview here.
