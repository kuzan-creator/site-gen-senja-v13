# Senja Site v11 — Fused native engine for makefun-site
[![make.fun · site-generator-speed-01-run-20261122](https://make.fun/badge/site-generator-speed-01-run-20261122.svg)](https://make.fun/bounty/site-generator-speed-01-run-20261122?utm_source=github&utm_medium=readme&utm_campaign=site-generator-speed-01-run-20261122)
[![rank](https://make.fun/badge/site-generator-speed-01-run-20261122/kuzan-creator/site-gen-senja-v13/rank.svg)](https://make.fun/bounty/site-generator-speed-01-run-20261122/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=site-generator-speed-01-run-20261122)
[![reward](https://make.fun/badge/site-generator-speed-01-run-20261122/kuzan-creator/site-gen-senja-v13/reward.svg)](https://make.fun/bounty/site-generator-speed-01-run-20261122?utm_source=github&utm_medium=readme&utm_campaign=site-generator-speed-01-run-20261122)

[![make.fun · site-generator-speed-01-run-20261122](https://make.fun/badge/site-generator-speed-01-run-20261122.svg)](https://make.fun/bounty/site-generator-speed-01-run-20261122?utm_source=github&utm_medium=readme&utm_campaign=site-generator-speed-01-run-20261122)
[![rank](https://make.fun/badge/site-generator-speed-01-run-20261122/kuzan-creator/site-gen-senja-v11/rank.svg)](https://make.fun/bounty/site-generator-speed-01-run-20261122/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=site-generator-speed-01-run-20261122)
[![reward](https://make.fun/badge/site-generator-speed-01-run-20261122/kuzan-creator/site-gen-senja-v11/reward.svg)](https://make.fun/bounty/site-generator-speed-01-run-20261122?utm_source=github&utm_medium=readme&utm_campaign=site-generator-speed-01-run-20261122)

Entry for make.fun bounty "Build a static site faster, 22 November 2026"
(`site-generator-speed-01-run-20261122`).

A fused single-pass rewrite of the makefun-site 1.0.0 generator in
freestanding C11 (N-API addon), written from scratch for speed: no JSON
tree, no per-card/per-page value tree, templates compiled at load time
into static chunks + dynamic slots, fragment memoization (99.7% card
cache hit), AVX2 32-byte JSON escaping with an SSE2 fallback. v11 fuses
rendering with JSON emission: pages render straight into the final
answers.json buffer — static template text is JSON-escaped once at
compile time, slots are HTML+JSON-escaped in one pass (esc_j), cached
fragments are stored JSON-ready — so there is no per-page buffer and no
second escape pass over the page HTML. Include inlining is OFF by
default (measured neutral; the card cache does the work). Input is
memory-mapped, output is a single streamed write, and the addon exits
via the exit_group syscall, skipping Node teardown. Hardened for the
grader's Node 22: the addon needs only glibc <= 2.14, loads via CJS
require, and the WASM fallback (`solver.wasm`, the v3 build) runs with
Liftoff-only V8 flags. `solve.js` keeps the `run(input)` ES-module
contract.

## Files

- `main.ts` — thin ESM driver: imports the addon (which runs the
  pipeline at load time), falls back to `buildCases` / `solve.cjs`.
- `solve.js` — the entry. `run(input)` returns the output tree.
- `solve.cjs` — native addon loader with the WASM fallback.
- `solver.wasm` — fallback engine (v3 build).
- `c/x.node` — the native addon (N-API v8, libc only).
- `c/engine.c`, `c/unicode_tables.c` — the addon source (`c/build.sh`).

## Verification

Practice 3/3 byte-identical, two fresh generator draws 3/3 each,
mutation differential 70/70, deterministic, fallback and clean-room
verified. See `../site-generator-speed/BUILD-v6.md` for numbers.
