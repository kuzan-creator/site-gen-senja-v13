// nolibc.c — zero-libc runtime for the Senja v13 .node (and its harness).
//
// Provides the handful of libc/libm symbols engine.c uses, implemented with
// direct Linux syscalls and exact arithmetic. The .node is linked with
// -nostdlib, so readelf -d shows NO NEEDED libc.so.6 and `nm -D
// --undefined-only` shows ONLY napi_* symbols (resolved from Node itself).
//
// What engine.c actually needs (audited 2026-11-24, v13):
//   syscalls: open/read/write/close/lseek/mmap/munmap, exit_group
//   memory:   malloc/free/realloc/calloc (+ compiler-emitted memcpy/memset)
//   string:   memcpy/memmove/memset/memcmp/strlen/strchr/strcmp/strncmp/memchr
//   stdio:    snprintf (incl. %.0f / %.*e / %.16e floats), fprintf(stderr,...)
//   misc:     strtod, getenv, abort, ceil/trunc/isnan/isinf/fmax/fmin, clock
//   link:     __cxa_finalize, _ITM_*, __gmon_start__, __stack_chk_fail, stderr
//
// Float formatting/conversion is EXACT (small bignum core): the engine's
// js_num() relies on correctly-rounded %e (prec 14/15/16) and %.0f, and on
// strtod() round-tripping exactly like glibc's. Differential-tested against
// glibc on millions of random doubles/decimal strings (see scratch-v13).
//
// Single-threaded only (the engine runs on Node's main thread; the harness
// is single-threaded too). No locks anywhere.
//
// Compile with -fno-builtin-memcpy -fno-builtin-memset -fno-builtin-memmove
// -fno-builtin-strlen (so GCC never turns our loops into calls to the very
// functions being defined) and -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0 (so the
// headers never emit __*_chk calls, which we do not provide).

#define _GNU_SOURCE
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdarg.h>
#include <time.h> // struct timespec / clockid_t for clock_gettime
#include <cpuid.h> // __get_cpuid_count for nl_have_avx2
#include <emmintrin.h> // SSE2 intrinsics for the memcpy/memset/memcmp paths
#include <stdarg.h>
#include <sys/syscall.h> // syscall numbers only (no libc link)
#include <sys/mman.h>    // PROT_*/MAP_* constants only
#include <fcntl.h>       // O_* constants only

#ifdef SENJA_COUNTERS
// Allocator + string/memory call counters (defined here so every function
// below can bump them; printed by the harness).
long cnt_malloc_calls, cnt_malloc_bytes, cnt_free_calls, cnt_realloc_calls;
long cnt_realloc_copy_bytes, cnt_calloc_calls, cnt_mmap_calls;
long cnt_strlen_calls, cnt_strcmp_calls, cnt_strncmp_calls, cnt_memcmp_calls;
long cnt_memchr_calls, cnt_strchr_calls, cnt_memcpy_calls, cnt_memcpy_bytes;
long cnt_memset_calls, cnt_memset_bytes, cnt_memmove_calls;
long cnt_strlen_bytes, cnt_memcmp_bytes, cnt_memmove_bytes;
#endif

// ---------------------------------------------------------------------------
// Direct syscalls (x86-64). The kernel ABI: rax=nr, rdi/rsi/rdx/r10/r8/r9.
// ---------------------------------------------------------------------------
static long nl_sys6(long n, long a1, long a2, long a3, long a4, long a5, long a6) {
  register long r10 __asm__("r10") = a4;
  register long r8 __asm__("r8") = a5;
  register long r9 __asm__("r9") = a6;
  long r;
  __asm__ volatile("syscall"
                   : "=a"(r)
                   : "a"(n), "D"(a1), "S"(a2), "d"(a3), "r"(r10), "r"(r8), "r"(r9)
                   : "rcx", "r11", "memory");
  return r;
}
static long nl_sys3(long n, long a1, long a2, long a3) {
  return nl_sys6(n, a1, a2, a3, 0, 0, 0);
}
static long nl_sys1(long n, long a1) { return nl_sys6(n, a1, 0, 0, 0, 0, 0); }

// Syscall numbers (x86-64), spelled out so this file does not depend on
// any libc header defining them.
#define NL_SYS_read 0
#define NL_SYS_write 1
#define NL_SYS_open 2
#define NL_SYS_close 3
#define NL_SYS_lseek 8
#define NL_SYS_mmap 9
#define NL_SYS_munmap 11
#define NL_SYS_exit_group 231
#define NL_SYS_clock_gettime 228
#define NL_SYS_openat 257
#define NL_AT_FDCWD (-100)

long nl_read(int fd, void *buf, unsigned long n) {
  return nl_sys3(NL_SYS_read, fd, (long)buf, (long)n);
}
long nl_write(int fd, const void *buf, unsigned long n) {
  return nl_sys3(NL_SYS_write, fd, (long)buf, (long)n);
}
long nl_close(int fd) { return nl_sys1(NL_SYS_close, fd); }
long nl_lseek(int fd, long off, int whence) {
  return nl_sys3(NL_SYS_lseek, fd, off, whence);
}
void *nl_mmap(void *addr, unsigned long len, int prot, int flags, int fd, long off) {
  return (void *)nl_sys6(NL_SYS_mmap, (long)addr, (long)len, prot, flags, fd, off);
}
long nl_munmap(void *addr, unsigned long len) {
  return nl_sys3(NL_SYS_munmap, (long)addr, (long)len, 0);
}
void nl_exit_group(int code) {
  nl_sys1(NL_SYS_exit_group, code);
  __builtin_unreachable();
}
struct nl_timespec { long tv_sec; long tv_nsec; };
long nl_clock_gettime(int clk, struct nl_timespec *ts) {
  return nl_sys3(NL_SYS_clock_gettime, clk, (long)ts, 0);
}

// libc-compatible names, implemented on the syscalls above. The engine
// calls open/read/write/close/lseek/mmap/munmap directly.
int open(const char *path, int flags, ...) {
  // openat(AT_FDCWD, ...) == open() semantics, relative paths included.
  // The mode argument is only read when O_CREAT is set; fetch it anyway.
  int mode = 0;
  if (flags & 0100 /* O_CREAT */) {
    va_list ap; va_start(ap, flags); mode = va_arg(ap, int); va_end(ap);
  }
  return (int)nl_sys6(NL_SYS_openat, NL_AT_FDCWD, (long)path, flags, mode, 0, 0);
}
long read(int fd, void *buf, unsigned long n) { return nl_read(fd, buf, n); }
long write(int fd, const void *buf, unsigned long n) { return nl_write(fd, buf, n); }
int close(int fd) { return (int)nl_close(fd); }
long lseek(int fd, long off, int whence) { return nl_lseek(fd, off, whence); }
void *mmap(void *addr, unsigned long len, int prot, int flags, int fd, long off) {
  return nl_mmap(addr, len, prot, flags, fd, off);
}
int munmap(void *addr, unsigned long len) { return (int)nl_munmap(addr, len); }

// clock(): process CPU time in CLOCKS_PER_SEC (1e6) units, like libc.
long clock(void) {
  struct nl_timespec ts;
  if (nl_clock_gettime(2 /* CLOCK_PROCESS_CPUTIME_ID */, &ts) != 0) return -1;
  return (long)(ts.tv_sec * 1000000L + ts.tv_nsec / 1000L);
}

// clock_gettime (used by the engine's SENJA_PROFILE timers).
int clock_gettime(clockid_t clk, struct timespec *ts) {
  return (int)nl_clock_gettime((int)clk, (struct nl_timespec *)ts);
}

// CPUID-based AVX2 detection. __builtin_cpu_supports("avx2") returns 0
// under static libgcc + -nostdlib (its __cpu_model is never initialized),
// which silently disables every AVX2 path in the engine. Check directly:
// CPUID.1:ECX.OSXSAVE + AVX, XCR0 XMM+YMM state, CPUID.7:EBX.AVX2.
int nl_have_avx2(void) {
  unsigned int eax, ebx, ecx, edx;
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return 0;
  if (!(ecx & bit_OSXSAVE) || !(ecx & bit_AVX)) return 0;
  unsigned int xcr0_lo, xcr0_hi;
  __asm__ volatile("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
  if ((xcr0_lo & 0x6u) != 0x6u) return 0; // XMM + YMM state must be enabled
  if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return 0;
  return (ebx & bit_AVX2) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Memory / string primitives. SSE2 16-byte bulk ops (x86-64-v2 baseline).
// No calls to libc: these ARE the definitions the engine resolves to.
// ---------------------------------------------------------------------------
#ifndef NL_NO_STRING
void *memcpy(void *dst, const void *src, unsigned long n) {
#ifdef SENJA_COUNTERS
  cnt_memcpy_calls++;
  cnt_memcpy_bytes += (long)n;
#endif
  unsigned char *d = (unsigned char *)dst;
  const unsigned char *s = (const unsigned char *)src;
  if (n == 0 || d == s) return dst;
  // Head bytes to 16-align d (unaligned SSE2 loads are legal, but aligned
  // stores are faster; correctness never depends on alignment).
  while (n && ((uintptr_t)d & 15)) { *d++ = *s++; n--; }
  while (n >= 64) {
    __m128i a0 = _mm_loadu_si128((const __m128i *)(s + 0));
    __m128i a1 = _mm_loadu_si128((const __m128i *)(s + 16));
    __m128i a2 = _mm_loadu_si128((const __m128i *)(s + 32));
    __m128i a3 = _mm_loadu_si128((const __m128i *)(s + 48));
    _mm_storeu_si128((__m128i *)(d + 0), a0);
    _mm_storeu_si128((__m128i *)(d + 16), a1);
    _mm_storeu_si128((__m128i *)(d + 32), a2);
    _mm_storeu_si128((__m128i *)(d + 48), a3);
    d += 64; s += 64; n -= 64;
  }
  while (n >= 16) {
    _mm_storeu_si128((__m128i *)d, _mm_loadu_si128((const __m128i *)s));
    d += 16; s += 16; n -= 16;
  }
  while (n--) *d++ = *s++;
  return dst;
}

void *memmove(void *dst, const void *src, unsigned long n) {
#ifdef SENJA_COUNTERS
  cnt_memmove_calls++;
  cnt_memmove_bytes += (long)n;
#endif
  unsigned char *d = (unsigned char *)dst;
  const unsigned char *s = (const unsigned char *)src;
  if (n == 0 || d == s) return dst;
  if (d < s) return memcpy(dst, src, n);
  // Backwards copy.
  d += n; s += n;
  while (n && ((uintptr_t)d & 15)) { d--; s--; n--; *d = *s; }
  while (n >= 64) {
    d -= 64; s -= 64; n -= 64;
    __m128i a0 = _mm_loadu_si128((const __m128i *)(s + 0));
    __m128i a1 = _mm_loadu_si128((const __m128i *)(s + 16));
    __m128i a2 = _mm_loadu_si128((const __m128i *)(s + 32));
    __m128i a3 = _mm_loadu_si128((const __m128i *)(s + 48));
    _mm_storeu_si128((__m128i *)(d + 0), a0);
    _mm_storeu_si128((__m128i *)(d + 16), a1);
    _mm_storeu_si128((__m128i *)(d + 32), a2);
    _mm_storeu_si128((__m128i *)(d + 48), a3);
  }
  while (n >= 16) {
    d -= 16; s -= 16; n -= 16;
    _mm_storeu_si128((__m128i *)d, _mm_loadu_si128((const __m128i *)s));
  }
  while (n--) { d--; s--; *d = *s; }
  return dst;
}

void *memset(void *dst, int c, unsigned long n) {
#ifdef SENJA_COUNTERS
  cnt_memset_calls++;
  cnt_memset_bytes += (long)n;
#endif
  unsigned char *d = (unsigned char *)dst;
  if (n == 0) return dst;
  unsigned char v = (unsigned char)c;
  __m128i vv = _mm_set1_epi8((char)v);
  while (n && ((uintptr_t)d & 15)) { *d++ = v; n--; }
  while (n >= 64) {
    _mm_storeu_si128((__m128i *)(d + 0), vv);
    _mm_storeu_si128((__m128i *)(d + 16), vv);
    _mm_storeu_si128((__m128i *)(d + 32), vv);
    _mm_storeu_si128((__m128i *)(d + 48), vv);
    d += 64; n -= 64;
  }
  while (n >= 16) { _mm_storeu_si128((__m128i *)d, vv); d += 16; n -= 16; }
  while (n--) *d++ = v;
  return dst;
}

int memcmp(const void *a, const void *b, unsigned long n) {
#ifdef SENJA_COUNTERS
  cnt_memcmp_calls++;
  cnt_memcmp_bytes += (long)n;
#endif
  const unsigned char *x = (const unsigned char *)a, *y = (const unsigned char *)b;
  while (n >= 16) {
    __m128i va = _mm_loadu_si128((const __m128i *)x);
    __m128i vb = _mm_loadu_si128((const __m128i *)y);
    unsigned eq = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(va, vb));
    if (eq != 0xFFFFu) {
      unsigned i = (unsigned)__builtin_ctz(~eq);
      return (int)x[i] - (int)y[i];
    }
    x += 16; y += 16; n -= 16;
  }
  while (n--) { if (*x != *y) return (int)*x - (int)*y; x++; y++; }
  return 0;
}

void *memchr(const void *s, int c, unsigned long n) {
#ifdef SENJA_COUNTERS
  cnt_memchr_calls++;
#endif
  const unsigned char *p = (const unsigned char *)s;
  unsigned char v = (unsigned char)c;
  while (n--) { if (*p == v) return (void *)p; p++; }
  return 0;
}

unsigned long strlen(const char *s) {
#ifdef SENJA_COUNTERS
  cnt_strlen_calls++;
#endif
  const char *p = s;
  while (*p) p++;
#ifdef SENJA_COUNTERS
  cnt_strlen_bytes += (long)(p - s);
#endif
  return (unsigned long)(p - s);
}

char *strchr(const char *s, int c) {
#ifdef SENJA_COUNTERS
  cnt_strchr_calls++;
#endif
  char v = (char)c;
  for (;;) { if (*s == v) return (char *)s; if (!*s) return 0; s++; }
}

int strcmp(const char *a, const char *b) {
#ifdef SENJA_COUNTERS
  cnt_strcmp_calls++;
#endif
  while (*a && *a == *b) { a++; b++; }
  return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

int strncmp(const char *a, const char *b, unsigned long n) {
#ifdef SENJA_COUNTERS
  cnt_strncmp_calls++;
#endif
  while (n--) {
    if (*a != *b) return (int)(unsigned char)*a - (int)(unsigned char)*b;
    if (!*a) return 0;
    a++; b++;
  }
  return 0;
}
#endif /* NL_NO_STRING */

// ---------------------------------------------------------------------------
// libm bits used by the engine (js_num / js_slice_bounds / page count).
// ---------------------------------------------------------------------------
double trunc(double x) {
  union { double d; uint64_t u; } v = { x };
  uint64_t a = v.u & 0x7FFFFFFFFFFFFFFFu;
  if (a < 0x4330000000000000u) { // |x| < 2^52: clear fraction bits
    if (a == 0) return x; // keep -0
    int e = (int)(a >> 52); // biased exponent
    int fbits = 52 - (e - 1023); // low bits of the significand = fraction
    if (fbits >= 53) return (v.u >> 63) ? -0.0 : 0.0; // |x| < 1
    if (fbits <= 0) return x; // already integral
    v.u &= ~((1uLL << fbits) - 1);
    return v.d;
  }
  return x; // already integral (|x| >= 2^52)
}

double ceil(double x) {
  double t = trunc(x);
  if (x > 0 && t != x) return t + 1.0;
  return t;
}

int isnan(double x) {
  union { double d; uint64_t u; } v = { x };
  return ((v.u >> 52) & 0x7FFu) == 0x7FFu && (v.u & 0xFFFFFFFFFFFFFu) != 0;
}

int isinf(double x) {
  union { double d; uint64_t u; } v = { x };
  return ((v.u >> 52) & 0x7FFu) == 0x7FFu && (v.u & 0xFFFFFFFFFFFFFu) == 0;
}

double fmax(double a, double b) {
  if (isnan(a)) return b;
  if (isnan(b)) return a;
  if (a == b) { // +0 vs -0: fmax returns +0
    union { double d; uint64_t u; } va = { a }, vb = { b };
    if ((va.u >> 63) != (vb.u >> 63)) return (va.u >> 63) ? b : a;
    return a;
  }
  return a > b ? a : b;
}

double fmin(double a, double b) {
  if (isnan(a)) return b;
  if (isnan(b)) return a;
  if (a == b) {
    union { double d; uint64_t u; } va = { a }, vb = { b };
    if ((va.u >> 63) != (vb.u >> 63)) return (va.u >> 63) ? a : b;
    return a;
  }
  return a < b ? a : b;
}

// syscall(): variadic wrapper so engine.c's syscall(SYS_exit_group, 0)
// works unchanged. Only used for exit_group in-engine (the other
// syscalls go through the typed wrappers above).
long syscall(long n, ...) {
  va_list ap;
  va_start(ap, n);
  long a1 = va_arg(ap, long);
  long a2 = va_arg(ap, long);
  long a3 = va_arg(ap, long);
  long a4 = va_arg(ap, long);
  long a5 = va_arg(ap, long);
  long a6 = va_arg(ap, long);
  va_end(ap);
  return nl_sys6(n, a1, a2, a3, a4, a5, a6);
}

void abort(void) { nl_exit_group(134); __builtin_unreachable(); }

// Linker/runtime dummies: referenced by compiler-generated init/fini code
// in a shared object. Defining them as no-ops keeps dlopen happy without
// libc. (_ITM_* are weak in practice; defining them is harmless.)
void __cxa_finalize(void *d) { (void)d; }
void _ITM_deregisterTMCloneTable(void) {}
void _ITM_registerTMCloneTable(void) {}
void __gmon_start__(void) {}
void __stack_chk_fail(void) { nl_exit_group(134); }

// stderr: the engine only ever passes it to our fprintf, which ignores the
// pointer and writes to fd 2. A dummy non-NULL pointer keeps the reference
// resolvable without libc's FILE machinery.
typedef struct __nl_FILE { char opaque[64]; } __nl_FILE;
static __nl_FILE nl_stderr_storage;
__nl_FILE *stderr = &nl_stderr_storage;

// ---------------------------------------------------------------------------
// Allocator: segregated-fit over mmap. No libc malloc/free.
//   - 17 size classes: 16, 32, ..., 2^20. A block is 16-byte aligned;
//     its 16-byte header is [next-ptr | class+size]; the class word
//     survives on the free list (it lives in the header's second word).
//   - Free blocks are pushed onto their class list and reused for the
//     same class, so the engine's repetitive pattern (Buf doubling, tmp
//     vectors) reuses memory instead of growing the heap.
//   - Requests > 2^20 - 16 bytes get a dedicated mmap; freed with munmap.
//   - Bump carving comes from 64 MB mmap chunks (untouched pages cost
//     nothing: the kernel only commits what we touch).
//   - Single-threaded: no locks.
// ---------------------------------------------------------------------------
#define NL_NCLASS 17
#define NL_CHUNK (1u << 26) // 64 MB
static void *nl_free_list[NL_NCLASS];
static unsigned char *nl_bump, *nl_bump_end;

static int nl_class_for(unsigned long need) {
  // smallest i with (16u << i) >= need; need >= 16 always (header incl.)
  unsigned long s = 16;
  int i = 0;
  while (s < need && i < NL_NCLASS - 1) { s <<= 1; i++; }
  return i;
}
static unsigned long nl_class_size(int i) { return 16uL << i; }

// header: word0 = free-list next (when free), word1 = (size<<8)|cls
static void *nl_block_to_ptr(unsigned char *blk) { return blk + 16; }
static unsigned char *nl_ptr_to_block(void *p) { return (unsigned char *)p - 16; }
static unsigned long nl_hdr_cls(unsigned char *blk) {
  return *(unsigned long *)(blk + 8) & 0xFFuL;
}
static unsigned long nl_hdr_size(unsigned char *blk) {
  return *(unsigned long *)(blk + 8) >> 8;
}

static unsigned char *nl_carve(unsigned long size) {
  if ((unsigned long)(nl_bump_end - nl_bump) < size) {
    unsigned long chunk = NL_CHUNK > size ? NL_CHUNK : size;
    void *m = nl_mmap(0, chunk, 3 /* PROT_READ|PROT_WRITE */,
                      0x22 /* MAP_PRIVATE|MAP_ANONYMOUS */, -1, 0);
    if (m == (void *)-1) return 0;
#ifdef SENJA_COUNTERS
    cnt_mmap_calls++;
#endif
    nl_bump = (unsigned char *)m;
    nl_bump_end = nl_bump + chunk;
  }
  unsigned char *b = nl_bump;
  nl_bump += size;
  return b;
}

#ifndef NL_NO_ALLOC
void *malloc(unsigned long size) {
#ifdef NL_BUMP_ONLY
  // Ablation: pure bump allocator (free is a no-op). Used only to test
  // whether the segregated-fit allocator is the nolibc slowdown.
  if (size == 0) size = 1;
  unsigned long need = (size + 16 + 15) & ~15uL;
  unsigned char *blk = nl_carve(need);
  if (!blk) return 0;
  *(unsigned long *)(blk + 8) = (need << 8) | 0xFFuL; // dedicated marker
  return nl_block_to_ptr(blk);
#else
#ifdef SENJA_COUNTERS
  cnt_malloc_calls++;
  cnt_malloc_bytes += (long)size;
#endif
  if (size == 0) size = 1;
  unsigned long need = size + 16;
  if (need > nl_class_size(NL_NCLASS - 1)) {
    // Large: dedicated mmap, page-rounded.
    unsigned long total = (need + 4095) & ~4095uL;
    unsigned char *blk = (unsigned char *)nl_mmap(
        0, total, 3, 0x22, -1, 0);
    if (blk == (unsigned char *)-1) return 0;
    *(unsigned long *)(blk + 8) = (total << 8) | 0xFFuL;
    return nl_block_to_ptr(blk);
  }
  int cls = nl_class_for(need);
  unsigned long csize = nl_class_size(cls);
  void *blk = nl_free_list[cls];
  if (blk) {
    nl_free_list[cls] = *(void **)blk;
  } else {
    blk = nl_carve(csize);
    if (!blk) return 0;
  }
  *(unsigned long *)((unsigned char *)blk + 8) = (csize << 8) | (unsigned long)cls;
  return nl_block_to_ptr(blk);
#endif
}

void free(void *p) {
#ifdef NL_BUMP_ONLY
  (void)p; // ablation: free is a no-op; the bump chunk is never reclaimed
  return;
#else
  if (!p) return;
#ifdef SENJA_COUNTERS
  cnt_free_calls++;
#endif
  unsigned char *blk = nl_ptr_to_block(p);
  unsigned long cls = nl_hdr_cls(blk);
  if (cls == 0xFFuL) {
    nl_munmap(blk, nl_hdr_size(blk));
    return;
  }
  *(void **)blk = nl_free_list[cls];
  nl_free_list[cls] = blk;
#endif
}

void *realloc(void *p, unsigned long size) {
#ifdef SENJA_COUNTERS
  cnt_realloc_calls++;
#endif
  if (!p) return malloc(size);
  if (size == 0) { free(p); return 0; }
  unsigned char *blk = nl_ptr_to_block(p);
  unsigned long cls = nl_hdr_cls(blk);
  unsigned long old_cap;
  if (cls == 0xFFuL) old_cap = nl_hdr_size(blk) - 16;
  else old_cap = nl_class_size((int)cls) - 16;
  if (size <= old_cap) return p; // fits in place (class blocks never shrink)
  void *np = malloc(size);
  if (!np) return 0;
  unsigned long copy = size < old_cap ? size : old_cap;
#ifdef SENJA_COUNTERS
  cnt_realloc_copy_bytes += (long)copy;
#endif
  memcpy(np, p, copy);
  free(p);
  return np;
}

void *calloc(unsigned long n, unsigned long size) {
#ifdef SENJA_COUNTERS
  cnt_calloc_calls++;
#endif
  if (size && n > (~0uL) / size) return 0;
  void *p = malloc(n * size);
  if (p) memset(p, 0, n * size);
  return p;
}
#endif /* NL_NO_ALLOC */

// ---------------------------------------------------------------------------
// Exact decimal core (bignum). The engine's js_num() needs correctly-rounded
// %e (prec 14/15/16) and %.0f, and strtod() must round-trip exactly like
// glibc's — a 1-ulp slip would change the chosen digit count and corrupt
// output bytes. So all float<->decimal conversion here is EXACT (integer
// arithmetic on the double's true binary value), never long-double
// approximation. Differential-tested vs glibc (scratch-v13/test_float).
// ---------------------------------------------------------------------------
#define NL_BIG_LIMBS 192
typedef struct { uint32_t w[NL_BIG_LIMBS]; int n; } Big; // base 2^32, LE

static void big_zero(Big *b) { b->n = 0; }
static int big_is_zero(const Big *b) { return b->n == 0; }
static void big_from_u64(Big *b, uint64_t v) {
  b->n = 0;
  while (v) { b->w[b->n++] = (uint32_t)v; v >>= 32; }
}
static int big_bitlen(const Big *b) {
  if (b->n == 0) return 0;
  uint32_t top = b->w[b->n - 1];
  int bits = (b->n - 1) * 32;
  while (top) { bits++; top >>= 1; }
  return bits;
}
// b *= m (m small, <= 10). Returns 0 on overflow (should not happen).
static int big_mul_small(Big *b, uint32_t m) {
  uint64_t carry = 0;
  for (int i = 0; i < b->n; i++) {
    uint64_t cur = (uint64_t)b->w[i] * m + carry;
    b->w[i] = (uint32_t)cur;
    carry = cur >> 32;
  }
  while (carry) {
    if (b->n >= NL_BIG_LIMBS) return 0;
    b->w[b->n++] = (uint32_t)carry;
    carry >>= 32;
  }
  return 1;
}
// b /= d (d small, <= 10); returns remainder. b becomes the quotient.
static uint32_t big_div_small(Big *b, uint32_t d) {
  uint64_t rem = 0;
  for (int i = b->n - 1; i >= 0; i--) {
    uint64_t cur = (rem << 32) | b->w[i];
    b->w[i] = (uint32_t)(cur / d);
    rem = cur % d;
  }
  while (b->n > 0 && b->w[b->n - 1] == 0) b->n--;
  return (uint32_t)rem;
}
// b <<= bits (bits >= 0).
static int big_shl(Big *b, int bits) {
  if (bits == 0 || b->n == 0) return 1;
  int word_shift = bits / 32, bit_shift = bits % 32;
  if (b->n + word_shift >= NL_BIG_LIMBS) return 0;
  for (int i = b->n - 1; i >= 0; i--) b->w[i + word_shift] = b->w[i];
  for (int i = 0; i < word_shift; i++) b->w[i] = 0;
  b->n += word_shift;
  if (bit_shift) {
    uint32_t carry = 0;
    for (int i = word_shift; i < b->n; i++) {
      uint32_t nc = b->w[i] >> (32 - bit_shift);
      b->w[i] = (b->w[i] << bit_shift) | carry;
      carry = nc;
    }
    if (carry) {
      if (b->n >= NL_BIG_LIMBS) return 0;
      b->w[b->n++] = carry;
    }
  }
  return 1;
}
// b >>= bits; *round = bit (bits-1) of the original (the highest dropped
// bit), *sticky = whether any lower dropped bit is set. Together they are
// exactly the round-to-nearest-even decision for the shift.
static void big_shr_round(Big *b, int bits, int *round, int *sticky) {
  *round = 0; *sticky = 0;
  if (bits <= 0 || b->n == 0) return;
  int bl = big_bitlen(b);
  if (bits - 1 < bl) *round = (b->w[(bits - 1) / 32] >> ((bits - 1) % 32)) & 1;
  int lim = bits - 1; // sticky covers bits [0, lim)
  for (int i = 0; i < b->n && i * 32 < lim; i++) {
    uint32_t w = b->w[i];
    if (i == lim / 32) {
      int keep = lim % 32;
      w &= keep == 0 ? 0 : ((1u << keep) - 1);
    }
    if (w) { *sticky = 1; break; }
  }
  int word_shift = bits / 32, bit_shift = bits % 32;
  if (word_shift >= b->n) { b->n = 0; return; }
  for (int i = 0; i < b->n - word_shift; i++) b->w[i] = b->w[i + word_shift];
  b->n -= word_shift;
  if (bit_shift) {
    uint32_t carry = 0;
    for (int i = b->n - 1; i >= 0; i--) {
      uint32_t nc = b->w[i] << (32 - bit_shift);
      b->w[i] = (b->w[i] >> bit_shift) | carry;
      carry = nc;
    }
  }
  while (b->n > 0 && b->w[b->n - 1] == 0) b->n--;
}

// value = N * 2^exp2, rounded to the nearest double (ties to even).
// Handles normals, subnormals, overflow (+-inf) and underflow (+-0).
// value = q * 2^shift with q a 53-bit integer (normal) or the denormal
// significand (subnormal); the shift applied to N is L-53 (or, for
// subnormals, the shift that lands the ulp on 2^-1074).
static double big_to_double(const Big *N, int exp2, int neg) {
  union { double d; uint64_t u; } out;
  uint64_t sign = neg ? 0x8000000000000000uLL : 0;
  if (big_is_zero(N)) {
    out.u = sign;
    return out.d;
  }
  Big q = *N;
  int L = big_bitlen(&q);
  int top_exp = exp2 + L - 1; // value in [2^top_exp, 2^(top_exp+1))
  int shift; // value = q * 2^shift after the code below
  int round = 0, sticky = 0;
  int is_sub = 0;
  if (top_exp >= -1022) {
    // Normal: q = round(N / 2^(L-53)), value = q * 2^(exp2+L-53).
    shift = exp2 + L - 53;
    int drobits = L - 53;
    if (drobits > 0) {
      big_shr_round(&q, drobits, &round, &sticky);
      if (round && (sticky || (q.n > 0 && (q.w[0] & 1)))) {
        uint64_t carry = 1;
        for (int i = 0; i < q.n && carry; i++) {
          uint64_t cur = (uint64_t)q.w[i] + carry;
          q.w[i] = (uint32_t)cur;
          carry = cur >> 32;
        }
        if (carry) q.w[q.n++] = 1;
        if (big_bitlen(&q) == 54) { // was 2^53-1 -> 2^53: renormalise
          int r2 = 0, s2 = 0;
          big_shr_round(&q, 1, &r2, &s2); // dropped bit is 0 here
          shift++;
          (void)r2; (void)s2;
        }
      }
    } else if (drobits < 0) {
      if (!big_shl(&q, -drobits)) {
        out.u = sign | 0x7FF0000000000000uLL;
        return out.d;
      }
    }
  } else {
    // Subnormal: q = round(N * 2^(exp2+1074)) = round(value / 2^-1074).
    is_sub = 1;
    shift = -1074;
    int up = exp2 + 1074;
    if (up < 0) {
      big_shr_round(&q, -up, &round, &sticky);
      if (round && (sticky || (q.n > 0 && (q.w[0] & 1)))) {
        uint64_t carry = 1;
        for (int i = 0; i < q.n && carry; i++) {
          uint64_t cur = (uint64_t)q.w[i] + carry;
          q.w[i] = (uint32_t)cur;
          carry = cur >> 32;
        }
        if (carry) q.w[q.n++] = 1;
      }
    } else if (up > 0) {
      if (!big_shl(&q, up)) {
        out.u = sign | 0x7FF0000000000000uLL;
        return out.d;
      }
    }
  }
  if (big_is_zero(&q)) {
    out.u = sign;
    return out.d;
  }
  if (is_sub) {
    // q is the denormal significand. Rounding may have carried it to
    // 2^52: that is the smallest normal (exponent field 1, significand 0).
    if (big_bitlen(&q) >= 53) {
      out.u = sign | (1uLL << 52);
      return out.d;
    }
    uint64_t sig = ((uint64_t)(q.n > 1 ? q.w[1] : 0) << 32) | q.w[0];
    out.u = sign | sig;
    return out.d;
  }
  // Normal: q has exactly 53 bits (top bit set).
  int e = shift + 52; // unbiased exponent
  if (e > 1023) {
    out.u = sign | 0x7FF0000000000000uLL;
    return out.d;
  }
  uint64_t sig = (((uint64_t)q.w[1] << 32) | q.w[0]) & 0xFFFFFFFFFFFFFuLL;
  out.u = sign | ((uint64_t)(e + 1023) << 52) | sig;
  return out.d;
}

// Decompose a double into sign, 53-bit significand M and binary exponent
// e2 with value = M * 2^e2 (M < 2^53, e2 in [-1074, 971]).
static void dbl_decompose(double v, int *neg, uint64_t *M, int *e2) {
  union { double d; uint64_t u; } x = { v };
  *neg = (int)(x.u >> 63);
  uint64_t bits = x.u & 0x7FFFFFFFFFFFFFFFuLL;
  int exp = (int)((bits >> 52) & 0x7FFu);
  uint64_t frac = bits & 0xFFFFFFFFFFFFFuLL;
  if (exp == 0) { // zero or subnormal
    *M = frac;
    *e2 = -1074;
  } else {
    *M = frac | 0x10000000000000uLL;
    *e2 = exp - 1023 - 52;
  }
}

// Generate P significant decimal digits of |v| (v != 0, finite).
// TRUNCATED, not rounded: digits[] gets the first P digits '0'..'9' (no
// point); *exp10 gets the decimal exponent (value = 0.digits * 10^(exp10+1));
// *sticky_out is 1 iff any nonzero digit exists beyond position P in the
// exact value. Truncation + sticky is deliberate: callers round ONCE from
// these, which avoids double-rounding errors (e.g. 0.1499999 rounded via
// 2 digits then to 1 digit would wrongly give 0.2 instead of 0.1).
// Method: exact integer N = v * 10^t with t = F + P + 2 (F = -e2 when
// e2 < 0), computed as M * 5^t * 2^(t+e2) in the bignum; then drop to P
// digits by exact division.
static void dbl_digits(double v, int P, char *digits, int *exp10, int *sticky_out) {
  int neg;
  uint64_t M;
  int e2;
  dbl_decompose(v, &neg, &M, &e2);
  (void)neg;
  if (M == 0) { // v == 0 (callers only pass v != 0, but be safe)
    for (int i = 0; i < P; i++) digits[i] = '0';
    digits[P] = 0;
    *exp10 = 0;
    if (sticky_out) *sticky_out = 0;
    return;
  }
  int F = e2 < 0 ? -e2 : 0;
  int t = F + P + 2;
  Big N;
  big_from_u64(&N, M);
  for (int i = 0; i < t; i++) big_mul_small(&N, 5);
  big_shl(&N, t + e2); // t + e2 >= 0 by construction
  // N has D digits; D >= P+3 by construction (N >= 10^(P+2)).
  Big tmp = N;
  int D = 0;
  while (!big_is_zero(&tmp)) { big_div_small(&tmp, 10); D++; }
  int drop = D - P; // >= 3
  int r_last = 0, sticky = 0;
  for (int i = 0; i < drop; i++) {
    uint32_t r = big_div_small(&N, 10);
    if (i == drop - 1) r_last = (int)r;
    else if (r) sticky = 1;
  }
  if (sticky_out) *sticky_out = (r_last != 0) || sticky;
  *exp10 = (D - 1) - t;
  // Extract P digits, most significant first.
  for (int i = P - 1; i >= 0; i--) {
    digits[i] = (char)('0' + big_div_small(&N, 10));
  }
  digits[P] = 0;
}

// Round the kept decimal digits (length `kept`, digits[0..kept-1]) once,
// using the next digit `rd` and whether any further nonzero digit exists
// (`sticky`). Ties go to even. Returns 1 if the rounding carried past the
// most significant digit (caller then prepends '1'), else 0. digits[] is
// modified in place.
static int round_digits(char *digs, int kept, char rd, int sticky) {
  int up = 0;
  if (rd > '5') up = 1;
  else if (rd == '5' && (sticky || (digs[kept - 1] & 1))) up = 1;
  if (!up) return 0;
  for (int i = kept - 1; i >= 0; i--) {
    if (digs[i] < '9') { digs[i]++; return 0; }
    digs[i] = '0';
  }
  return 1; // 999.. -> 1000..
}

// Format |v| (finite, nonzero) in %e style with `prec` digits after the
// point: "d[.ddd]e(+|-)XX" (exponent: sign + at least two digits, glibc
// style). Returns length (excluding NUL). buf must hold >= 32 bytes.
static int fmt_e(double v, int prec, char *buf) {
  char digs[64];
  int exp10, sticky = 0;
  int P = prec + 2; // prec+1 kept + 1 rounding digit
  if (P > 63) P = 63;
  if (prec > 61) prec = 61; // cap: engine uses 15/16/17; digits beyond are 0
  dbl_digits(v, P, digs, &exp10, &sticky);
  int kept = prec + 1;
  char rd = digs[kept];
  int carry = round_digits(digs, kept, rd, sticky);
  if (carry) exp10++;
  int n = 0;
  buf[n++] = carry ? '1' : digs[0];
  if (prec > 0) {
    buf[n++] = '.';
    for (int i = 1; i < kept; i++) buf[n++] = carry ? '0' : digs[i];
  }
  buf[n++] = 'e';
  buf[n++] = exp10 < 0 ? '-' : '+';
  int ae = exp10 < 0 ? -exp10 : exp10;
  if (ae < 10) buf[n++] = '0';
  char eb[8];
  int el = 0;
  if (ae == 0) eb[el++] = '0';
  else { while (ae) { eb[el++] = (char)('0' + ae % 10); ae /= 10; } }
  for (int i = el - 1; i >= 0; i--) buf[n++] = eb[i];
  buf[n] = 0;
  return n;
}

// Format |v| in %f style with `prec` digits after the point. Used by
// fprintf %.1f (PROFILE, compiled out) and %f generally.
static int fmt_f(double v, int prec, char *buf) {
  char d0[8];
  int exp10, sticky0 = 0;
  dbl_digits(v, 2, d0, &exp10, &sticky0);
  int intdig = exp10 >= 0 ? exp10 + 1 : 0;
  int total_kept = intdig + prec; // digits to emit (excl. the point)
  int need = exp10 + prec + 2; // significant digits incl. rounding digit
  int n = 0;
  if (need <= 0) {
    // So small that every emitted digit and the rounding digit are 0.
    buf[n++] = '0';
    if (prec > 0) {
      buf[n++] = '.';
      for (int i = 0; i < prec; i++) buf[n++] = '0';
    }
    buf[n] = 0;
    return n;
  }
  if (need > 400) need = 400;
  char digs[420];
  int sticky = 0;
  dbl_digits(v, need, digs, &exp10, &sticky);
  // Build the kept digit string (length total_kept). Emitted digit i is at
  // decimal position exp10-i when exp10 >= 0, or -1-i when exp10 < 0; its
  // significant index is i, resp. exp10+1+i (negative => leading '0').
  char kept[420];
  for (int i = 0; i < total_kept; i++) {
    int idx = (exp10 >= 0) ? i : exp10 + 1 + i;
    kept[i] = (idx >= 0 && idx < need - 1) ? digs[idx] : '0';
  }
  char rd = digs[need - 1];
  int carry;
  if (total_kept == 0) {
    // %.0f of a value < 1: no kept digits; the result is "1" iff the
    // first significant digit rounds up (exact 0.5 ties to even -> "0").
    carry = (rd > '5' || (rd == '5' && sticky)) ? 1 : 0;
  } else {
    carry = round_digits(kept, total_kept, rd, sticky);
  }
  int idig = intdig;
  if (carry) idig++; // 9.99 -> 10.0: the point moves right one digit
  // Emit: integer part (idig digits), '.', fractional part (prec digits).
  // After a carry the digit sequence is '1' + zeros.
  if (idig == 0) {
    buf[n++] = '0'; // no integer digits (value < 1, no carry): leading zero
  } else {
    for (int i = 0; i < idig; i++) {
      if (carry) buf[n++] = (i == 0) ? '1' : '0';
      else buf[n++] = (i < total_kept) ? kept[i] : '0';
    }
  }
  if (prec > 0) {
    buf[n++] = '.';
    for (int i = 0; i < prec; i++) {
      int idx = idig + i;
      if (carry) buf[n++] = '0';
      else buf[n++] = (idx < total_kept) ? kept[idx] : '0';
    }
  }
  buf[n] = 0;
  return n;
}

// ---------------------------------------------------------------------------
// strtod: exact decimal -> double (correctly rounded, ties to even).
// Only decimal notation is supported (no hex floats: the engine never feeds
// them — fp_number accepts digits/./e/E only, js_num's round-trip strings
// are decimal). Differential-tested vs glibc (scratch-v13/test_float).
// ---------------------------------------------------------------------------
double strtod(const char *s, char **endptr) {
  const char *p = s;
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\v' || *p == '\f' || *p == '\r') p++;
  int neg = 0;
  if (*p == '+' || *p == '-') { neg = (*p == '-'); p++; }
  // inf / nan (case-insensitive)
  if ((((unsigned char)*p) | 32) == 'i') {
    if ((((unsigned char)p[1]) | 32) == 'n' && (((unsigned char)p[2]) | 32) == 'f') {
      if (endptr) *endptr = (char *)(p + 3);
      union { double d; uint64_t u; } v;
      v.u = neg ? 0xFFF0000000000000uLL : 0x7FF0000000000000uLL;
      return v.d;
    }
  }
  if ((((unsigned char)*p) | 32) == 'n') {
    if ((((unsigned char)p[1]) | 32) == 'a' && (((unsigned char)p[2]) | 32) == 'n') {
      if (endptr) *endptr = (char *)(p + 3);
      union { double d; uint64_t u; } v;
      v.u = neg ? 0xFFF8000000000000uLL : 0x7FF8000000000000uLL;
      return v.d;
    }
  }
  Big D;
  big_zero(&D);
  int ndig = 0;       // significant digits collected
  int frac_digits = 0; // digits after the point (all, incl. leading zeros)
  int seen_dot = 0, seen_digit = 0, leading = 1;
  // Integer part.
  while (*p >= '0' && *p <= '9') {
    seen_digit = 1;
    if (leading && *p == '0') { p++; continue; } // skip leading zeros
    leading = 0;
    big_mul_small(&D, 10);
    // add digit
    uint64_t carry = (uint64_t)(*p - '0');
    for (int i = 0; i < D.n && carry; i++) {
      uint64_t cur = (uint64_t)D.w[i] + carry;
      D.w[i] = (uint32_t)cur;
      carry = cur >> 32;
    }
    if (carry) D.w[D.n++] = (uint32_t)carry;
    ndig++;
    p++;
  }
  if (*p == '.') {
    seen_dot = 1;
    p++;
    while (*p >= '0' && *p <= '9') {
      seen_digit = 1;
      frac_digits++;
      if (leading && *p == '0') { p++; continue; }
      leading = 0;
      big_mul_small(&D, 10);
      uint64_t carry = (uint64_t)(*p - '0');
      for (int i = 0; i < D.n && carry; i++) {
        uint64_t cur = (uint64_t)D.w[i] + carry;
        D.w[i] = (uint32_t)cur;
        carry = cur >> 32;
      }
      if (carry) D.w[D.n++] = (uint32_t)carry;
      ndig++;
      p++;
    }
  }
  (void)seen_dot;
  if (!seen_digit) {
    if (endptr) *endptr = (char *)s;
    return 0.0;
  }
  // Exponent.
  long exp_part = 0;
  if (*p == 'e' || *p == 'E') {
    const char *e = p + 1;
    int eneg = 0;
    if (*e == '+' || *e == '-') { eneg = (*e == '-'); e++; }
    if (*e >= '0' && *e <= '9') {
      long ev = 0;
      while (*e >= '0' && *e <= '9') {
        if (ev < 1000000) ev = ev * 10 + (*e - '0');
        e++;
      }
      exp_part = eneg ? -ev : ev;
      p = e;
    }
  }
  if (endptr) *endptr = (char *)p;
  if (big_is_zero(&D)) {
    union { double d; uint64_t u; } v;
    v.u = neg ? 0x8000000000000000uLL : 0;
    return v.d;
  }
  // value = D * 10^E10. D < 10^64 (<= 63 digits from the engine's numbuf).
  long E10 = exp_part - frac_digits;
  if (E10 > 340) { // >= 10^341 > max double
    union { double d; uint64_t u; } v;
    v.u = neg ? 0xFFF0000000000000uLL : 0x7FF0000000000000uLL;
    return v.d;
  }
  if (E10 <= -390) { // < 10^-326 < half of min denormal
    union { double d; uint64_t u; } v;
    v.u = neg ? 0x8000000000000000uLL : 0;
    return v.d;
  }
  if (E10 >= 0) {
    for (long i = 0; i < E10; i++) big_mul_small(&D, 10);
    return big_to_double(&D, 0, neg);
  }
  // E10 < 0: N = round(D * 2^sh / 10^-E10), value = N * 2^-sh.
  long ae = -E10;
  int sh = 80 - big_bitlen(&D) + (int)(ae * 10 / 3);
  if (sh < 64) sh = 64;
  Big N = D;
  big_shl(&N, sh);
  int r_last = 0, sticky = 0;
  for (long i = 0; i < ae; i++) {
    uint32_t r = big_div_small(&N, 10);
    if (i == ae - 1) r_last = (int)r;
    else if (r) sticky = 1;
  }
  if (r_last > 5 || (r_last == 5 && (sticky || (N.n > 0 && (N.w[0] & 1))))) {
    uint64_t carry = 1;
    for (int i = 0; i < N.n && carry; i++) {
      uint64_t cur = (uint64_t)N.w[i] + carry;
      N.w[i] = (uint32_t)cur;
      carry = cur >> 32;
    }
    if (carry) N.w[N.n++] = 1;
  }
  return big_to_double(&N, -sh, neg);
}

// ---------------------------------------------------------------------------
// getenv via /proc/self/environ (read once, cached). The engine queries
// MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR (CLI path) and SENJA_* debug flags.
// ---------------------------------------------------------------------------
static char nl_environ[65536];
static int nl_environ_loaded = 0;
static void nl_load_environ(void) {
  if (nl_environ_loaded) return;
  nl_environ_loaded = 1;
  int fd = open("/proc/self/environ", 0 /* O_RDONLY */);
  if (fd < 0) return;
  long off = 0;
  for (;;) {
    long r = nl_read(fd, nl_environ + off, sizeof(nl_environ) - 1 - (unsigned long)off);
    if (r <= 0) break;
    off += r;
    if (off >= (long)sizeof(nl_environ) - 1) break;
  }
  nl_close(fd);
  nl_environ[off] = 0;
  nl_environ[sizeof(nl_environ) - 1] = 0;
}
char *getenv(const char *name) {
  nl_load_environ();
  unsigned long nl = strlen(name);
  for (char *p = nl_environ; *p; p += strlen(p) + 1) {
    if (strncmp(p, name, nl) == 0 && p[nl] == '=') return p + nl + 1;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// strtol / strtoll / atoi / atol: GCC rewrites atoi() calls into strtol,
// so these must exist even though engine.c hand-rolls its own parsing.
// Base 0 auto-detects (0x hex, 0 octal, else decimal); overflow clamps
// to LONG_MAX / LONG_MIN like glibc.
long strtol(const char *s, char **endptr, int base) {
  const char *p = s;
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\v' || *p == '\f' || *p == '\r') p++;
  int neg = 0;
  if (*p == '+' || *p == '-') { neg = (*p == '-'); p++; }
  if (base == 0) {
    if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) { base = 16; p += 2; }
    else if (p[0] == '0') base = 8;
    else base = 10;
  } else if (base == 16 && p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
    p += 2;
  }
  unsigned long v = 0;
  int any = 0, overflow = 0;
  unsigned long limit = neg ? 0x8000000000000000uL : 0x7FFFFFFFFFFFFFFFuL;
  while (1) {
    int d;
    char c = *p;
    if (c >= '0' && c <= '9') d = c - '0';
    else if (c >= 'a' && c <= 'z') d = c - 'a' + 10;
    else if (c >= 'A' && c <= 'Z') d = c - 'A' + 10;
    else break;
    if (d >= base) break;
    any = 1;
    if (!overflow) {
      if (v > (limit - (unsigned long)d) / (unsigned long)base) overflow = 1;
      else v = v * (unsigned long)base + (unsigned long)d;
    }
    p++;
  }
  if (endptr) *endptr = (char *)(any ? p : s);
  if (overflow) return neg ? (-9223372036854775807L - 1) : 9223372036854775807L;
  if (neg) return (long)(0uL - v);
  return (long)v;
}
long long strtoll(const char *s, char **endptr, int base) {
  return (long long)strtol(s, endptr, base);
}
int atoi(const char *s) { return (int)strtol(s, 0, 10); }
long atol(const char *s) { return strtol(s, 0, 10); }

// vsnprintf / snprintf / fprintf. Supports the formats the engine uses:
//   ints: %d %i %u %x %X %o %ld %lu %lld %llu %zu %zd %c %s %p %% %.*s
//   floats: %e %E %f %F %g %G with precision (incl. %.*e, %.0f, %.16e)
// Floats go through the exact decimal core above. %g follows the C rule
// (shortest of %e/%f at the given precision, trailing zeros stripped).
// ---------------------------------------------------------------------------
typedef struct { char *buf; unsigned long size, pos, total; } Out;
static void out_c(Out *o, char c) {
  if (o->pos + 1 < o->size) o->buf[o->pos] = c;
  o->pos++;
  o->total++;
}
static void out_s(Out *o, const char *s, unsigned long n) {
  for (unsigned long i = 0; i < n; i++) out_c(o, s[i]);
}
static void out_pad(Out *o, char c, long n) {
  while (n-- > 0) out_c(o, c);
}

// Format an unsigned integer into tmp (reversed build), returns length.
static int u64_to_str(char *tmp, uint64_t v, int base, int upper) {
  static const char dig_lo[] = "0123456789abcdef";
  static const char dig_up[] = "0123456789ABCDEF";
  const char *dig = upper ? dig_up : dig_lo;
  int n = 0;
  if (v == 0) tmp[n++] = '0';
  else {
    char rev[24];
    int r = 0;
    while (v) { rev[r++] = dig[v % (uint64_t)base]; v /= (uint64_t)base; }
    while (r > 0) tmp[n++] = rev[--r];
  }
  return n;
}

static int nl_vsnprintf(char *buf, unsigned long size, const char *fmt, va_list ap) {
  Out o = { buf, size, 0, 0 };
  if (size > 0) buf[0] = 0;
  for (const char *f = fmt; *f; f++) {
    if (*f != '%') { out_c(&o, *f); continue; }
    f++;
    if (*f == '%') { out_c(&o, '%'); continue; }
    // flags
    int left = 0, plus = 0, space = 0, alt = 0, zero = 0;
    int done = 0;
    while (!done) {
      switch (*f) {
        case '-': left = 1; f++; break;
        case '+': plus = 1; f++; break;
        case ' ': space = 1; f++; break;
        case '#': alt = 1; f++; break;
        case '0': zero = 1; f++; break;
        default: done = 1; break;
      }
    }
    // width
    int width = -1;
    if (*f == '*') { width = va_arg(ap, int); f++; if (width < 0) { left = 1; width = -width; } }
    else if (*f >= '0' && *f <= '9') {
      width = 0;
      while (*f >= '0' && *f <= '9') width = width * 10 + (*f++ - '0');
    }
    // precision
    int prec = -1;
    if (*f == '.') {
      f++;
      if (*f == '*') { prec = va_arg(ap, int); f++; }
      else {
        prec = 0;
        while (*f >= '0' && *f <= '9') prec = prec * 10 + (*f++ - '0');
      }
      if (prec < 0) prec = 0;
    }
    // length
    int len = 0; // 0=int, 1=l, 2=ll, 3=z, 4=h, 5=hh
    if (*f == 'l') { f++; if (*f == 'l') { len = 2; f++; } else len = 1; }
    else if (*f == 'z') { len = 3; f++; }
    else if (*f == 'h') { f++; if (*f == 'h') { len = 5; f++; } else len = 4; }
    else if (*f == 't' || *f == 'j') { len = 3; f++; }
    char conv = *f;
    if (conv == 'd' || conv == 'i' || conv == 'u' || conv == 'x' || conv == 'X' || conv == 'o') {
      int is_signed = (conv == 'd' || conv == 'i');
      int base = (conv == 'x' || conv == 'X') ? 16 : (conv == 'o' ? 8 : 10);
      uint64_t uv;
      int neg = 0;
      if (is_signed) {
        int64_t sv;
        if (len == 2) sv = va_arg(ap, long long);
        else if (len == 1 || len == 3) sv = va_arg(ap, long);
        else if (len == 4) sv = (short)va_arg(ap, int);
        else if (len == 5) sv = (signed char)va_arg(ap, int);
        else sv = va_arg(ap, int);
        if (sv < 0) { neg = 1; uv = (uint64_t)(-(sv + 1)) + 1; } else uv = (uint64_t)sv;
      } else {
        if (len == 2) uv = va_arg(ap, unsigned long long);
        else if (len == 1 || len == 3) uv = va_arg(ap, unsigned long);
        else if (len == 4) uv = (unsigned short)va_arg(ap, unsigned int);
        else if (len == 5) uv = (unsigned char)va_arg(ap, unsigned int);
        else uv = va_arg(ap, unsigned int);
      }
      char num[24];
      int nlen = 0;
      if (prec == 0 && uv == 0) nlen = 0;
      else nlen = u64_to_str(num, uv, base, conv == 'X');
      // prefix / sign
      char sign = 0;
      if (neg) sign = '-';
      else if (is_signed && plus) sign = '+';
      else if (is_signed && space) sign = ' ';
      int prefix = 0;
      if (alt && base == 16 && uv != 0) prefix = 2;
      else if (alt && base == 8 && (nlen == 0 || num[0] != '0')) prefix = 1;
      int zeros = 0;
      if (prec >= 0) {
        if (prec > nlen) zeros = prec - nlen;
      } else if (zero && !left && width > 0) {
        int w = width - nlen - (sign ? 1 : 0) - prefix;
        if (w > 0) zeros = w;
      }
      int field = (sign ? 1 : 0) + prefix + zeros + nlen;
      int pad = width > field ? width - field : 0;
      if (!left) out_pad(&o, ' ', pad);
      if (sign) out_c(&o, sign);
      if (prefix == 2) out_s(&o, conv == 'X' ? "0X" : "0x", 2);
      else if (prefix == 1) out_c(&o, '0');
      out_pad(&o, '0', zeros);
      out_s(&o, num, (unsigned long)nlen);
      if (left) out_pad(&o, ' ', pad);
    } else if (conv == 'c') {
      int ch = va_arg(ap, int);
      int pad = width > 1 ? width - 1 : 0;
      if (!left) out_pad(&o, ' ', pad);
      out_c(&o, (char)ch);
      if (left) out_pad(&o, ' ', pad);
    } else if (conv == 's') {
      const char *s = va_arg(ap, const char *);
      if (!s) s = "(null)";
      unsigned long sl = strlen(s);
      if (prec >= 0 && (unsigned long)prec < sl) sl = (unsigned long)prec;
      int pad = width > (int)sl ? width - (int)sl : 0;
      if (!left) out_pad(&o, ' ', pad);
      out_s(&o, s, sl);
      if (left) out_pad(&o, ' ', pad);
    } else if (conv == 'p') {
      void *ptr = va_arg(ap, void *);
      char num[24];
      int nlen = u64_to_str(num, (uint64_t)(uintptr_t)ptr, 16, 0);
      int field = 2 + nlen;
      int pad = width > field ? width - field : 0;
      if (!left) out_pad(&o, ' ', pad);
      out_s(&o, "0x", 2);
      out_s(&o, num, (unsigned long)nlen);
      if (left) out_pad(&o, ' ', pad);
    } else if (conv == 'n') {
      int *ip = va_arg(ap, int *);
      if (ip) *ip = (int)o.total;
    } else if (conv == 'e' || conv == 'E' || conv == 'f' || conv == 'F' || conv == 'g' || conv == 'G') {
      double v = va_arg(ap, double);
      int pr = prec < 0 ? 6 : prec;
      if (pr > 60) pr = 60; // cap: engine uses <= 17; extra digits are 0
      char tmp[512];
      int tlen = 0;
      int isneg = 0;
      if (isnan(v)) {
        const char *s = conv >= 'a' ? "nan" : "NAN";
        // glibc prints "-nan" for negative NaN; keep the sign bit.
        union { double d; uint64_t u; } u = { v };
        if (u.u >> 63) { tmp[tlen++] = '-'; }
        while (*s) tmp[tlen++] = *s++;
        tmp[tlen] = 0;
      } else if (isinf(v)) {
        if (v < 0) tmp[tlen++] = '-';
        const char *s = conv >= 'a' ? "inf" : "INF";
        while (*s) tmp[tlen++] = *s++;
        tmp[tlen] = 0;
      } else {
        if (v < 0 || (v == 0.0 && (1.0 / v) < 0)) { isneg = 1; v = -v; }
        if (v == 0.0) {
          if (conv == 'e' || conv == 'E') {
            tlen += fmt_e(0.0, pr, tmp + tlen); // "0.000e+00" via fmt_e(0)
          } else if (conv == 'f' || conv == 'F') {
            tmp[tlen++] = '0';
            if (pr > 0) { tmp[tlen++] = '.'; for (int i = 0; i < pr; i++) tmp[tlen++] = '0'; }
          } else { tmp[tlen++] = '0'; }
          tmp[tlen] = 0;
        } else if (conv == 'e' || conv == 'E') {
          tlen += fmt_e(v, pr, tmp + tlen);
          tmp[tlen] = 0;
        } else if (conv == 'f' || conv == 'F') {
          tlen += fmt_f(v, pr, tmp + tlen);
          tmp[tlen] = 0;
        } else { // %g / %G
          int P = pr == 0 ? 1 : pr;
          char digs[64];
          int exp10, sticky = 0;
          // P+1 truncated digits, then round ONCE to P.
          dbl_digits(v, P + 1, digs, &exp10, &sticky);
          int carry = round_digits(digs, P, digs[P], sticky);
          if (carry) {
            exp10++;
            digs[0] = '1';
            for (int i = 1; i < P; i++) digs[i] = '0';
          }
          // strip trailing zeros (but keep at least one digit)
          int k = P;
          while (k > 1 && digs[k - 1] == '0') k--;
          if (exp10 < -4 || exp10 >= P) {
            // %e style with k-1 digits after the point
            tmp[tlen++] = digs[0];
            if (k > 1) {
              tmp[tlen++] = '.';
              for (int i = 1; i < k; i++) tmp[tlen++] = digs[i];
            }
            tmp[tlen++] = conv == 'G' ? 'E' : 'e';
            tmp[tlen++] = exp10 < 0 ? '-' : '+';
            int ae = exp10 < 0 ? -exp10 : exp10;
            if (ae < 10) tmp[tlen++] = '0';
            char eb[8];
            int el = 0;
            if (ae == 0) eb[el++] = '0';
            else { while (ae) { eb[el++] = (char)('0' + ae % 10); ae /= 10; } }
            for (int i = el - 1; i >= 0; i--) tmp[tlen++] = eb[i];
          } else {
            // %f style
            if (exp10 < 0) {
              tmp[tlen++] = '0';
              tmp[tlen++] = '.';
              for (int i = 0; i < -exp10 - 1; i++) tmp[tlen++] = '0';
              for (int i = 0; i < k; i++) tmp[tlen++] = digs[i];
            } else {
              int intdig = exp10 + 1;
              for (int i = 0; i < intdig; i++) tmp[tlen++] = i < k ? digs[i] : '0';
              if (k > intdig) {
                tmp[tlen++] = '.';
                for (int i = intdig; i < k; i++) tmp[tlen++] = digs[i];
              }
            }
          }
          tmp[tlen] = 0;
        }
        // Uppercase %E: the 'e' in the exponent (all %e paths above emit
        // lowercase; %G already emits 'E' itself).
        if (conv == 'E') for (int i = 0; i < tlen; i++) if (tmp[i] == 'e') tmp[i] = 'E';
        if (isneg) {
          // shift right by one and prepend '-'
          for (int i = tlen; i >= 0; i--) tmp[i + 1] = tmp[i];
          tmp[0] = '-';
          tlen++;
        }
      }
      int pad = width > tlen ? width - tlen : 0;
      if (!left) out_pad(&o, zero ? '0' : ' ', pad);
      out_s(&o, tmp, (unsigned long)tlen);
      if (left) out_pad(&o, ' ', pad);
    } else {
      // Unknown conversion: echo it literally (never happens in-engine).
      out_c(&o, '%');
      if (conv) out_c(&o, conv);
    }
  }
  if (o.size > 0) {
    if (o.pos < o.size) o.buf[o.pos] = 0;
    else o.buf[o.size - 1] = 0;
  }
  return (int)o.total;
}

int snprintf(char *buf, unsigned long size, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int r = nl_vsnprintf(buf, size, fmt, ap);
  va_end(ap);
  return r;
}

int vsnprintf(char *buf, unsigned long size, const char *fmt, va_list ap) {
  return nl_vsnprintf(buf, size, fmt, ap);
}

// fwrite / fputs / fputc: gcc optimizes fprintf(stderr, "literal") into
// fwrite, and fprintf(stderr, "%s", s) into fputs. The engine only ever
// writes to stderr; route those to fd 2. (Return values are discarded
// by the engine's fprintf call sites.)
unsigned long fwrite(const void *ptr, unsigned long size, unsigned long nmemb, void *stream) {
  (void)stream;
  unsigned long total = size * nmemb;
  long off = 0;
  while (off < (long)total) {
    long w = nl_write(2, (const char *)ptr + off, total - (unsigned long)off);
    if (w <= 0) break;
    off += w;
  }
  return size ? (unsigned long)off / size : 0;
}
int fputs(const char *s, void *stream) {
  (void)stream;
  unsigned long n = strlen(s);
  long off = 0;
  while (off < (long)n) {
    long w = nl_write(2, s + off, n - (unsigned long)off);
    if (w <= 0) return -1;
    off += w;
  }
  return 0;
}
int fputc(int c, void *stream) {
  (void)stream;
  char ch = (char)c;
  if (nl_write(2, &ch, 1) != 1) return -1;
  return c;
}

// fprintf: the engine only ever writes to stderr (PROFILE dumps, CHUNK-BAD
// dumps, the "[senja] path=native" marker). Format into a stack buffer and
// write(2) it. (A 4 KB buffer covers every engine message by 10x.)
int fprintf(void *stream, const char *fmt, ...) {
  (void)stream; // always stderr in-engine; anything else is a bug
  char tmp[4096];
  va_list ap;
  va_start(ap, fmt);
  int r = nl_vsnprintf(tmp, sizeof(tmp), fmt, ap);
  va_end(ap);
  unsigned long n = r < 0 ? 0 : (unsigned long)r;
  if (n > sizeof(tmp) - 1) n = sizeof(tmp) - 1;
  long off = 0;
  while (off < (long)n) {
    long w = nl_write(2, tmp + off, n - (unsigned long)off);
    if (w <= 0) break;
    off += w;
  }
  return r;
}
