# rextio-numpy

**Rextio plugin that lowers eligible NumPy code to native Rust.**

This is the first-party [Rextio](https://github.com/rextio/rextio) plugin for NumPy: it defines the rules for translating supported NumPy usage in typed Python functions into Rust (via the `ndarray` crate and friends), while everything else stays on the Python fallback — following Rextio's core contract (CPython-equivalent semantics or fall back).

## Status

**Pre-development.** This repo is scaffolded ahead of the Rextio core plugin contract (route taxonomy, capability manifest, plugin `describe()` protocol). Implementation starts once that contract lands in core.

Planned initial rule set:

- Typed `float64` 1D/2D element-wise operations
- `dot`
- `sum` / `mean` reductions

Per the plugin protocol, machine-readable **rule records** (id, pattern, constraint, diagnostic code, guidance) ship first; **fix templates** are added incrementally.

Functions explicitly decorated with `@numba.*` are respected as the user's opt-in to Numba's contract and stay on the fallback route — this plugin only lowers undecorated eligible code.

## License

MIT
