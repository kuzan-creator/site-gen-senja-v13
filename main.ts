// Senja Site v10 — minimal ESM driver, zero static imports on the native path.
//
// Why not a literal CJS `require('./c/x.node')` in main.ts: a bare `require`
// only works if package.json is type:commonjs, and that breaks the solve.js
// ESM contract (`import` outside a module — verified on Node 22.23.4).
// This is the closest reachable form under type:module: the CJS require is
// obtained via process.getBuiltinModule (no `node:module` static import),
// node:fs is fetched the same way and only on the JS fallback path, and
// node:v8/setFlagsFromString is dynamically imported only on the WASM
// fallback path (never on the native path).
//
// The native addon (c/x.node) runs the whole pipeline at load time when
// this file is the entry point (node main.ts): it reads cases.json, builds
// every case, writes answers.json, and exits via the exit_group syscall,
// skipping Node teardown. The JS below only runs when that fast path did
// not trigger — then it does the same work through the addon's buildCases,
// or through the WASM fallback in solve.cjs when the addon cannot load.
const require = process.getBuiltinModule('module').createRequire(import.meta.url);
let m = null;
try {
  m = require('./c/x.node');
} catch {}

const d = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const o = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

if (m && !m.cliRan) {
  const { readFileSync, writeFileSync } = process.getBuiltinModule('fs');
  writeFileSync(o + '/answers.json', m.buildCases(readFileSync(d + '/cases.json')));
} else if (!m) {
  console.error('[senja] path=wasm-fallback');
  try {
    const { setFlagsFromString } = await import('node:v8');
    setFlagsFromString('--no-incremental-marking --no-wasm-tier-up --no-wasm-dynamic-tiering');
  } catch {}
  const { runAll } = await import('./solve.cjs');
  const { readFileSync, writeFileSync } = process.getBuiltinModule('fs');
  writeFileSync(o + '/answers.json', runAll(readFileSync(d + '/cases.json')));
}
