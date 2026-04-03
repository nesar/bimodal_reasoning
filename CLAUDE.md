# CLAUDE.md — Project-level instructions for Claude Code

## Code standards

### No placeholder or fabricated details
- Never use fake, made-up, or approximate values as stand-ins for real data
- If a data field is unavailable (e.g., no RA/Dec in this HDF5), label it honestly
  using what IS known (e.g., row index) — do not invent coordinate-style names
- Physical estimates (e.g., SFR from M★/age) are acceptable if the derivation is
  documented inline — do not silently pass off approximations as measured quantities
- Stub files for planned future work are acceptable if explicitly named `*_stub.py`
  and clearly labeled in their docstrings

### Minimal, modular code
- Each function does one thing; short focused functions over long procedural blocks
- No comments on self-evident code; no docstrings on trivial getters/setters
- No helper abstractions for single-use operations
- No defensive validation of internal state — validate only at system boundaries
  (user input, file I/O, external APIs)
- No backwards-compat shims, feature flags, or speculative future-proofing

### Physical quantities
- Always include units in variable names: `age_gyr`, `log_mass`, `sfr_msun_yr`
- Never store dimensionless normalized values without noting they are normalized

## Plotting
- All plots use the publication style defined in `analysis/plots.py:setup_style()`
- Every plot function accepts an optional `ax` argument so panels can be composed
- Save at 300 dpi, tight bbox, no title unless the figure stands alone
- Use the project color palette (`COLORS` dict in `analysis/plots.py`)

## Tokenization
- Every new tokenization strategy must be registered in `STRATEGY_REGISTRY`
  in `spec_tokenizer.py` — no hard-coded special cases scattered elsewhere

## Git hygiene
- Never commit model weights, HDF5 data, or experiment outputs
- See `.gitignore` for the full exclusion list
