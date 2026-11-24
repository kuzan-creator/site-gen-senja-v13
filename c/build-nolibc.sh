#!/bin/sh
# Build the Senja v13 TRUE zero-libc native addon (x.node).
#
# No libc.so.6: the engine + nolibc.c are linked with -nostdlib and the
# only undefined dynamic symbols must be the N-API entry points (resolved
# by Node at dlopen). nolibc.c provides malloc/free/memcpy/printf/strtod/
# getenv/syscalls itself (see nolibc.c for the full list and the
# differential test that validates the float core vs glibc).
#
# Flags that matter:
#   -nostdlib            no crt files, no -lc, no libgcc runtime libs
#   -fno-stack-protector no __stack_chk_guard (a libc TLS symbol)
#   -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0
#                        no __*_chk fortified redirects (they are libc)
#   -fno-builtin-memcpy -fno-builtin-memset -fno-builtin-memmove
#   -fno-builtin-strlen  keep the compiler from replacing our memory
#                        functions with its own builtin expansions that
#                        would bypass nolibc.c (and from folding strlen
#                        calls into constants the linker can't see)
#   -flto                NOT used here: LTO + -nostdlib is fragile (the
#                        standalone harness fails to link under -flto with
#                        undefined main/memcpy/memset from LTO's artificial
#                        sections, and the .node miscompiles and segfaults).
#                        The glibc build keeps -flto; the nolibc build
#                        trades a little cross-TU inlining for a binary
#                        that actually runs.
set -e
cd "$(dirname "$0")"
CC="${CC:-cc}"
CFLAGS="-O3 -march=x86-64-v2 -std=c11 -shared -fPIC -nostdlib -static-libgcc -fno-stack-protector -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 -fno-builtin-memcpy -fno-builtin-memset -fno-builtin-memmove -fno-builtin-strlen -I/usr/include/node -DNAPI_VERSION=8 -DSENJA_NOLIBC -Wl,--version-script=exports.map -Wall -Wextra -Wno-unused-parameter"
# shellcheck disable=SC2086
# NOTE: must come AFTER the sources: the linker only pulls objects
# from a static archive for symbols that are already undefined, and with
# -nostdlib libgcc is not linked by default (it provides __cpu_model for
# __builtin_cpu_supports).
"$CC" $CFLAGS -o x-nolibc.node engine.c unicode_tables.c nolibc.c 
strip x-nolibc.node
ls -la x-nolibc.node
echo "--- readelf -d (NEEDED entries; must NOT list libc.so.6) ---"
readelf -d x-nolibc.node | grep -i needed || echo "(no NEEDED entries)"
echo "--- undefined dynamic symbols (must be napi_* only) ---"
nm -D --undefined-only x-nolibc.node
echo "--- ldd ---"
ldd x-nolibc.node || true
