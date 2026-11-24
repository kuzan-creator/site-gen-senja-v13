// Senja Site v6 — flat-architecture rewrite (see BUILD-v6.md).
// Leaf functions below are carried over verbatim from v5 (our own code,
// already verified byte-identical); the architecture above them — flat
// JSON parse, no JSON tree, no Val tree, constant-folded template
// slots, AVX2 scans — is new.
#define _GNU_SOURCE
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdbool.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <limits.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <cpuid.h>
#include <xmmintrin.h>
#include <emmintrin.h>  // SSE2
#include <tmmintrin.h>  // SSSE3 _mm_shuffle_epi8 (x86-64-v2)
#include <smmintrin.h>  // SSE4.1 (kept for parity with v5's helpers)
#include <immintrin.h>  // AVX2
#ifndef SENJA_STANDALONE
#include <node_api.h>
#endif
#include "unicode_tables.h"

// ---- v7 profiling (compiled out unless SENJA_PROFILE is defined) ----
#ifdef SENJA_PROFILE
#include <time.h>
static double prof_parse, prof_compile, prof_md, prof_render, prof_jw, prof_esc, prof_mdin, prof_fm, prof_slugw, prof_other;
static long prof_ninstr, prof_neval, prof_nesc, prof_njw, prof_nrender, prof_nmdin, prof_nfast, prof_ngen, prof_nslot;
/* v10 instrumentation: CPU-time (clock()) coarse phases + counters. */
static double c_parse, c_compile, c_fm, c_md, c_render, c_other, c_words;
static long c_hit, c_miss, c_hit_sh, c_hit_it, c_hit_bytes, c_miss_bytes, c_inc, c_inc_c, c_pages, c_tpls, c_cases;
static inline double senja_cpu_now(void) { return (double)clock() / (double)CLOCKS_PER_SEC; }
static inline double prof_now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9; }
#define PROF_T0 double _pt0 = prof_now()
#define PROF_ADD(x) do { (x) += prof_now() - _pt0; } while (0)
#else
#define PROF_T0
#define PROF_ADD(x)
#endif

// Runtime CPU dispatch (set once in senja_init).
static int have_avx2 = 0;
#ifdef SENJA_NOLIBC
// True zero-libc build: __builtin_cpu_supports() returns 0 under static
// libgcc + -nostdlib (its __cpu_model is never initialized), which silently
// disables every AVX2 path. Use a direct CPUID check instead (nolibc.c).
extern int nl_have_avx2(void);
static inline int senja_have_avx2(void) { return nl_have_avx2(); }
#else
static inline int senja_have_avx2(void) { return __builtin_cpu_supports("avx2") ? 1 : 0; }
#endif
#ifdef SENJA_COUNTERS
// AVX2-liveness counters (harness only): prove the AVX2 dispatch is
// actually taken at runtime (v6 lesson: a guard once made AVX2 dead code
// while the binary still contained AVX2 instructions). Compiled out of
// the shipped .node (SENJA_COUNTERS is never defined there).
long cnt_avx2_nl = 0, cnt_sse2_nl = 0, cnt_avx2_jw = 0, cnt_sse2_jw = 0;
long cnt_avx2_jwr = 0, cnt_sse2_jwr = 0, cnt_avx2_fb = 0, cnt_sse2_fb = 0;
long cnt_avx2_nje = 0, cnt_sse2_nje = 0;
#endif
// v7 bisect aid: SENJA_NOFAST=1 disables the chunk fast slot path.
static int nofast = 0;

// (JParser is defined after the Slice/Arena types, just before j_hex4,
// because it references both.)



// ---- carried from v5: types/arena/buf (v5 lines 26-122) ----

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

// ------------------------------------------------------------------ slices

typedef struct { const u8 *p; size_t n; } Slice;
static inline Slice sl(const u8 *p, size_t n) { Slice s; s.p = p; s.n = n; return s; }
static inline Slice slc(const char *s) { return sl((const u8 *)s, strlen(s)); }
static inline int sl_eq(Slice a, Slice b) { return a.n == b.n && (a.n == 0 || memcmp(a.p, b.p, a.n) == 0); }
static inline int sl_eq_c(Slice a, const char *b) { size_t n = strlen(b); return a.n == n && (n == 0 || memcmp(a.p, b, n) == 0); }
static inline int sl_ends(Slice a, const char *suf) {
  size_t n = strlen(suf);
  return a.n >= n && (n == 0 || memcmp(a.p + a.n - n, suf, n) == 0);
}
static inline int sl_starts(Slice a, const char *pre) {
  size_t n = strlen(pre);
  return a.n >= n && (n == 0 || memcmp(a.p, pre, n) == 0);
}

// ------------------------------------------------------------------ arena

typedef struct AChunk { struct AChunk *next; size_t used, cap; u8 data[]; } AChunk;
typedef struct { AChunk *head, *cur; } Arena;
#define ARENA_CHUNK (1u << 22) // 4 MiB chunks

static void arena_init(Arena *a) { a->head = a->cur = NULL; }
static void *arena_alloc(Arena *a, size_t n) {
  n = (n + 7) & ~(size_t)7;
  if (n == 0) n = 8;
  AChunk *c = a->cur;
  if (!c || c->used + n > c->cap) {
    size_t cap = ARENA_CHUNK;
    if (n > cap) cap = n;
    c = (AChunk *)malloc(sizeof(AChunk) + cap);
    if (!c) abort();
    c->next = NULL; c->used = 0; c->cap = cap;
    if (a->cur) a->cur->next = c; else a->head = c;
    a->cur = c;
  }
  void *p = c->data + c->used;
  c->used += n;
  return p;
}
static void arena_reset(Arena *a) {
  for (AChunk *c = a->head; c; c = c->next) c->used = 0;
  a->cur = a->head;
}
static void arena_free(Arena *a) {
  AChunk *c = a->head;
  while (c) { AChunk *nx = c->next; free(c); c = nx; }
  a->head = a->cur = NULL;
}
static Slice arena_slice(Arena *a, const u8 *p, size_t n) {
  u8 *q = (u8 *)arena_alloc(a, n ? n : 1);
  if (n) memcpy(q, p, n);
  return sl(q, n);
}
static Slice arena_cat2(Arena *a, Slice x, Slice y) {
  u8 *q = (u8 *)arena_alloc(a, x.n + y.n ? x.n + y.n : 1);
  if (x.n) memcpy(q, x.p, x.n);
  if (y.n) memcpy(q + x.n, y.p, y.n);
  return sl(q, x.n + y.n);
}
static Slice arena_cat3(Arena *a, Slice x, Slice y, Slice z) {
  u8 *q = (u8 *)arena_alloc(a, x.n + y.n + z.n ? x.n + y.n + z.n : 1);
  size_t o = 0;
  if (x.n) { memcpy(q + o, x.p, x.n); o += x.n; }
  if (y.n) { memcpy(q + o, y.p, y.n); o += y.n; }
  if (z.n) { memcpy(q + o, z.p, z.n); o += z.n; }
  return sl(q, x.n + y.n + z.n);
}

// ------------------------------------------------------------------ growable byte buffer (malloc'd)

typedef struct { u8 *p; size_t len, cap; } Buf;
// Small-span copy without a memcpy call: 8-byte word copies + byte tail.
// (memcpy's call + size-dispatch costs ~15ns, more than the copy itself
// for the short spans that dominate md_inline / esc / json_write.)
static inline u8 *copy_span(u8 *dst, const u8 *src, size_t n) {
  while (n >= 8) { __builtin_memcpy(dst, src, 8); dst += 8; src += 8; n -= 8; }
  while (n--) *dst++ = *src++;
  return dst;
}
static inline void buf_reserve(Buf *b, size_t extra) {
  if (b->len + extra > b->cap) {
    size_t nc = b->cap ? b->cap : 256;
    while (nc < b->len + extra) nc *= 2;
    b->p = (u8 *)realloc(b->p, nc);
    if (!b->p) abort();
    b->cap = nc;
  }
}
static inline void buf_put(Buf *b, const void *s, size_t n) {
  if (n == 0) return;
  if (b->len + n > b->cap) buf_reserve(b, n);
  if (n <= 32) b->len = (size_t)(copy_span(b->p + b->len, (const u8 *)s, n) - b->p);
  else { memcpy(b->p + b->len, s, n); b->len += n; }
}
static inline void buf_sl(Buf *b, Slice s) { buf_put(b, s.p, s.n); }
static inline void buf_c(Buf *b, u8 c) { buf_reserve(b, 1); b->p[b->len++] = c; }
static inline void buf_cs(Buf *b, const char *s) { buf_put(b, s, strlen(s)); }
static inline void buf_u64(Buf *b, u64 v) { // decimal, no allocation
  char tmp[24]; int i = 23;
  if (v == 0) { buf_c(b, '0'); return; }
  while (v > 0) { tmp[--i] = (char)('0' + (v % 10)); v /= 10; }
  buf_put(b, tmp + i, 24 - (size_t)i - 1);
}



// ---- carried from v5: utf8/u16/cmp/trim/find_byte (v5 lines 128-247) ----

static inline u32 utf8_decode(const u8 *b, size_t len, size_t i, size_t *used) {
  u8 b0 = b[i];
  if (b0 < 0x80) { *used = 1; return b0; }
  if (b0 >= 0xC2 && b0 < 0xE0 && i + 1 < len && (b[i + 1] & 0xC0) == 0x80) {
    *used = 2; return ((u32)(b0 & 0x1F) << 6) | (u32)(b[i + 1] & 0x3F);
  }
  if (b0 >= 0xE0 && b0 < 0xF0 && i + 2 < len && (b[i + 1] & 0xC0) == 0x80 && (b[i + 2] & 0xC0) == 0x80) {
    u32 cp = ((u32)(b0 & 0x0F) << 12) | ((u32)(b[i + 1] & 0x3F) << 6) | (u32)(b[i + 2] & 0x3F);
    if (cp >= 0x800) { *used = 3; return cp; }
    *used = 1; return 0xFFFD;
  }
  if (b0 >= 0xF0 && b0 < 0xF5 && i + 3 < len
      && (b[i + 1] & 0xC0) == 0x80 && (b[i + 2] & 0xC0) == 0x80 && (b[i + 3] & 0xC0) == 0x80) {
    u32 cp = ((u32)(b0 & 0x07) << 18) | ((u32)(b[i + 1] & 0x3F) << 12)
           | ((u32)(b[i + 2] & 0x3F) << 6) | (u32)(b[i + 3] & 0x3F);
    if (cp >= 0x10000 && cp <= 0x10FFFF) { *used = 4; return cp; }
    *used = 1; return 0xFFFD;
  }
  *used = 1; return 0xFFFD;
}

static inline void utf8_encode(u32 cp, Buf *out) {
  if (cp < 0x80) { buf_c(out, (u8)cp); }
  else if (cp < 0x800) {
    buf_c(out, (u8)(0xC0 | (cp >> 6))); buf_c(out, (u8)(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    buf_c(out, (u8)(0xE0 | (cp >> 12))); buf_c(out, (u8)(0x80 | ((cp >> 6) & 0x3F))); buf_c(out, (u8)(0x80 | (cp & 0x3F)));
  } else {
    buf_c(out, (u8)(0xF0 | (cp >> 18))); buf_c(out, (u8)(0x80 | ((cp >> 12) & 0x3F)));
    buf_c(out, (u8)(0x80 | ((cp >> 6) & 0x3F))); buf_c(out, (u8)(0x80 | (cp & 0x3F)));
  }
}

// UTF-16 code units of a WTF-8 string, one at a time.
typedef struct { const u8 *b; size_t len, i; int has_pending; u32 pending; } U16Iter;
static inline void u16_init(U16Iter *it, Slice s) { it->b = s.p; it->len = s.n; it->i = 0; it->has_pending = 0; it->pending = 0; }
// Returns 1 and sets *u, or 0 at the end.
static int u16_next(U16Iter *it, u32 *u) {
  if (it->has_pending) { it->has_pending = 0; *u = it->pending; return 1; }
  if (it->i >= it->len) return 0;
  size_t used; u32 cp = utf8_decode(it->b, it->len, it->i, &used);
  it->i += used;
  if (cp >= 0x10000) {
    u32 v = cp - 0x10000;
    it->pending = 0xDC00 + (v & 0x3FF); it->has_pending = 1;
    *u = 0xD800 + (v >> 10);
  } else *u = cp;
  return 1;
}

// Compare two WTF-8 strings in UTF-16 code-unit order (JS `<`).
static int cmp_u16(Slice a, Slice b) {
  U16Iter ia, ib; u16_init(&ia, a); u16_init(&ib, b);
  for (;;) {
    u32 x, y; int ha = u16_next(&ia, &x), hb = u16_next(&ib, &y);
    if (!ha && !hb) return 0;
    if (!ha) return -1;
    if (!hb) return 1;
    if (x != y) return x < y ? -1 : 1;
  }
}
static size_t u16_len(Slice s) {
  U16Iter it; u16_init(&it, s); u32 u; size_t n = 0;
  while (u16_next(&it, &u)) n++;
  return n;
}

// JavaScript \s: White_Space plus line terminators.
static inline int is_js_space(u32 cp) {
  return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0xA0 || cp == 0x1680
      || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029
      || cp == 0x202F || cp == 0x205F || cp == 0x3000 || cp == 0xFEFF;
}

// JavaScript String.prototype.trim.
static Slice js_trim(Slice b) {
  size_t a = 0, z = b.n;
  while (a < z) {
    size_t used; u32 cp = utf8_decode(b.p, b.n, a, &used);
    if (!is_js_space(cp)) break;
    a += used;
  }
  while (z > a) {
    size_t s = z - 1;
    while (s > a && (b.p[s] & 0xC0) == 0x80) s--;
    size_t used; u32 cp = utf8_decode(b.p, b.n, s, &used);
    if (!is_js_space(cp) || s + used != z) break;
    z = s;
  }
  return sl(b.p + a, z - a);
}

static inline int is_blank_sp_tab(Slice line) {
  for (size_t i = 0; i < line.n; i++) if (line.p[i] != ' ' && line.p[i] != '\t') return 0;
  return 1;
}

// ------------------------------------------------------------------ SSE2 scans

// First index >= from holding byte b, or n.
__attribute__((target("avx2")))
static size_t find_byte_avx2(const u8 *s, size_t n, size_t from, u8 b) {
  size_t i = from;
  __m256i needle = _mm256_set1_epi8((char)b);
  while (i + 32 <= n) {
    __m256i chunk = _mm256_loadu_si256((const __m256i *)(s + i));
    unsigned mask = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, needle));
    if (mask) return i + (size_t)__builtin_ctz(mask);
    i += 32;
  }
  if (i < n && n >= 32) {
    __m256i chunk = _mm256_loadu_si256((const __m256i *)(s + n - 32));
    unsigned mask = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, needle));
    size_t skip = i - (n - 32);
    mask &= ~((1u << skip) - 1u);
    if (mask) return (n - 32) + (size_t)__builtin_ctz(mask);
    return n;
  }
  while (i < n) { if (s[i] == b) return i; i++; }
  return n;
}

static size_t find_byte(const u8 *s, size_t n, size_t from, u8 b) {
#ifdef SENJA_COUNTERS
  if (have_avx2) cnt_avx2_fb++; else cnt_sse2_fb++;
#endif
  if (have_avx2) return find_byte_avx2(s, n, from, b);
  size_t i = from;
  __m128i needle = _mm_set1_epi8((char)b);
  while (i + 16 <= n) {
    __m128i chunk = _mm_loadu_si128((const __m128i *)(s + i));
    unsigned mask = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(chunk, needle));
    if (mask) return i + (size_t)__builtin_ctz(mask);
    i += 16;
  }
  if (i < n && n >= 16) {
    __m128i chunk = _mm_loadu_si128((const __m128i *)(s + n - 16));
    unsigned mask = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(chunk, needle));
    size_t skip = i - (n - 16);
    mask &= ~((1u << skip) - 1u);
    if (mask) return (n - 16) + (size_t)__builtin_ctz(mask);
    return n;
  }
  while (i < n) { if (s[i] == b) return i; i++; }
  return n;
}



// ---- carried from v5: next_json_esc (v5 lines 252-285) ----

static size_t next_json_esc_avx2(const u8 *s, size_t n, size_t from);
static size_t next_json_esc(const u8 *s, size_t n, size_t from) {
#ifdef SENJA_COUNTERS
  if (have_avx2) cnt_avx2_nje++; else cnt_sse2_nje++;
#endif
  if (have_avx2) return next_json_esc_avx2(s, n, from);
  size_t i = from;
  __m128i quote = _mm_set1_epi8('"');
  __m128i backslash = _mm_set1_epi8('\\');
  __m128i lead_ed = _mm_set1_epi8((char)0xED);
  __m128i lo31 = _mm_set1_epi8(0x1F);
  __m128i zero = _mm_setzero_si128();
  while (i + 16 <= n) {
    __m128i chunk = _mm_loadu_si128((const __m128i *)(s + i));
    __m128i m = _mm_or_si128(_mm_or_si128(_mm_cmpeq_epi8(chunk, quote), _mm_cmpeq_epi8(chunk, backslash)),
                              _mm_or_si128(_mm_cmpeq_epi8(chunk, lead_ed),
                                           _mm_cmpeq_epi8(_mm_subs_epu8(chunk, lo31), zero)));
    unsigned mask = (unsigned)_mm_movemask_epi8(m);
    if (mask) return i + (size_t)__builtin_ctz(mask);
    i += 16;
  }
  if (i < n && n >= 16) {
    __m128i chunk = _mm_loadu_si128((const __m128i *)(s + n - 16));
    __m128i m = _mm_or_si128(_mm_or_si128(_mm_cmpeq_epi8(chunk, quote), _mm_cmpeq_epi8(chunk, backslash)),
                              _mm_or_si128(_mm_cmpeq_epi8(chunk, lead_ed),
                                           _mm_cmpeq_epi8(_mm_subs_epu8(chunk, lo31), zero)));
    unsigned mask = (unsigned)_mm_movemask_epi8(m);
    size_t skip = i - (n - 16);
    mask &= ~((1u << skip) - 1u);
    if (mask) return (n - 16) + (size_t)__builtin_ctz(mask);
    return n;
  }
  while (i < n) {
    u8 c = s[i];
    if (c == '"' || c == '\\' || c < 0x20 || c == 0xED) return i;
    i++;
  }
  return n;
}



// ---- carried from v5: fxhash/strmap (v5 lines 289-350) ----

static inline u64 fx_mix(u64 h, u64 v) { return ((((h << 5) | (h >> 59)) ^ v) * 0x517cc1b727220a95ULL); }
static u64 fx_hash_bytes(const u8 *p, size_t n) {
  u64 h = 0;
  size_t i = 0;
  // 8 bytes per mix (the hash is internal only — bucket order is never
  // observable: the one place that reads map keys sorts them after).
  while (i + 8 <= n) {
    u64 v;
    memcpy(&v, p + i, 8);
    h = fx_mix(h, v);
    i += 8;
  }
  while (i < n) h = fx_mix(h, p[i++]);
  h = fx_mix(h, (u64)n); // Rust Hash for [u8] writes the length last
  return h;
}

// Open-addressing map from byte-string keys to void* values. Keys are
// copied into the arena on insert, so callers may pass scratch slices.
typedef struct { u64 hash; const u8 *key; size_t klen; void *val; } StrSlot;
typedef struct { StrSlot *slots; size_t cap, len; Arena *arena; } StrMap;

static void strmap_init(StrMap *m, Arena *a) { m->slots = NULL; m->cap = 0; m->len = 0; m->arena = a; }
static void strmap_grow(StrMap *m) {
  size_t nc = m->cap ? m->cap * 2 : 32;
  StrSlot *ns = (StrSlot *)arena_alloc(m->arena, nc * sizeof(StrSlot));
  memset(ns, 0, nc * sizeof(StrSlot));
  for (size_t i = 0; i < m->cap; i++) {
    if (m->slots[i].key) {
      size_t j = m->slots[i].hash & (nc - 1);
      while (ns[j].key) j = (j + 1) & (nc - 1);
      ns[j] = m->slots[i];
    }
  }
  m->slots = ns; m->cap = nc;
}
static void *strmap_get(StrMap *m, Slice key) {
  if (!m->cap) return NULL;
  u64 h = fx_hash_bytes(key.p, key.n);
  size_t j = h & (m->cap - 1);
  while (m->slots[j].key) {
    if (m->slots[j].hash == h && m->slots[j].klen == key.n && memcmp(m->slots[j].key, key.p, key.n) == 0)
      return m->slots[j].val;
    j = (j + 1) & (m->cap - 1);
  }
  return NULL;
}
static int strmap_has(StrMap *m, Slice key) { return strmap_get(m, key) != NULL; }
static void strmap_put(StrMap *m, Slice key, void *val) {
  if ((m->len + 1) * 10 > m->cap * 7) strmap_grow(m);
  u64 h = fx_hash_bytes(key.p, key.n);
  size_t j = h & (m->cap - 1);
  while (m->slots[j].key) {
    if (m->slots[j].hash == h && m->slots[j].klen == key.n && memcmp(m->slots[j].key, key.p, key.n) == 0) {
      m->slots[j].val = val; return;
    }
    j = (j + 1) & (m->cap - 1);
  }
  Slice owned = arena_slice(m->arena, key.p, key.n);
  m->slots[j].hash = h; m->slots[j].key = owned.p; m->slots[j].klen = owned.n; m->slots[j].val = val;
  m->len++;
}



// ---- carried from v5: j_hex4 (v5 lines 376-389) ----

// Flat JSON parser cursor (same layout as v5's JParser; the extracted
// j_hex4 uses it, and the v6 flat parser is built on it).
typedef struct { const u8 *b; size_t len, pos; Arena *arena; } JParser;

static int j_hex4(JParser *p, u32 *out) {
  if (p->pos + 4 > p->len) return 0;
  u32 v = 0;
  for (int i = 0; i < 4; i++) {
    u8 c = p->b[p->pos];
    u32 d;
    if (c >= '0' && c <= '9') d = (u32)(c - '0');
    else if (c >= 'a' && c <= 'f') d = (u32)(c - 'a' + 10);
    else if (c >= 'A' && c <= 'F') d = (u32)(c - 'A' + 10);
    else return 0;
    v = (v << 4) | d; p->pos++;
  }
  *out = v; return 1;
}



// ---- carried from v5: json_write/js_num (v5 lines 601-740) ----

// JSON string writer, v7: AVX2 32-byte scan (SSE2 16-byte fallback)
// with mask-draining, and escapes written inline through a raw output
// pointer instead of one buf_put call per escape (the v5/v6 cost: the
// practice answers carry ~159k escapes, ~2.25% of string bytes, and
// each escape cost a buf_put + an emit_json_escape call). The output
// buffer is reserved for the common case (n + 64) up front; the rare
// escape-heavy string falls back to buf_reserve mid-string.
static const char JHEX[] = "0123456789abcdef";

#define JW_ROOM(dst, end, need) \
  do { if ((dst) + (need) > (end)) { \
    out->len = (size_t)((dst) - out->p); buf_reserve(out, (need) + 64); \
    (dst) = out->p + out->len; (end) = out->p + out->cap; } } while (0)

// Plain-run copy with room check: escape expansions can eat the
// upfront reserve, so a long plain run after many escapes must grow.
#define JW_PLAIN(dst, end, p, plain, j) \
  do { size_t _m = (size_t)((j) - (plain)); \
    if (_m) { JW_ROOM(dst, end, _m); (dst) = copy_span((dst), (p) + (plain), _m); } } while (0)

#define JW_ESCAPE(dst, end, p, n, j, plain) \
  do { \
    u8 _c = (p)[(j)]; \
    JW_ROOM(dst, end, 6); \
    if (_c == '"') { *(dst)++ = '\\'; *(dst)++ = '"'; (plain) = (j) + 1; } \
    else if (_c == '\\') { *(dst)++ = '\\'; *(dst)++ = '\\'; (plain) = (j) + 1; } \
    else if (_c == 0x08) { *(dst)++ = '\\'; *(dst)++ = 'b'; (plain) = (j) + 1; } \
    else if (_c == 0x09) { *(dst)++ = '\\'; *(dst)++ = 't'; (plain) = (j) + 1; } \
    else if (_c == 0x0A) { *(dst)++ = '\\'; *(dst)++ = 'n'; (plain) = (j) + 1; } \
    else if (_c == 0x0C) { *(dst)++ = '\\'; *(dst)++ = 'f'; (plain) = (j) + 1; } \
    else if (_c == 0x0D) { *(dst)++ = '\\'; *(dst)++ = 'r'; (plain) = (j) + 1; } \
    else if (_c < 0x20) { \
      *(dst)++ = '\\'; *(dst)++ = 'u'; *(dst)++ = '0'; *(dst)++ = '0'; \
      *(dst)++ = (u8)JHEX[(_c >> 4) & 0xF]; *(dst)++ = (u8)JHEX[_c & 0xF]; \
      (plain) = (j) + 1; } \
    else if (_c == 0xED && (j) + 2 < (n) && (p)[(j) + 1] >= 0xA0 && (p)[(j) + 1] <= 0xBF && ((p)[(j) + 2] & 0xC0) == 0x80) { \
      u32 _cp = 0xD800 + (((u32)((p)[(j) + 1] & 0x3F)) << 6) + (u32)((p)[(j) + 2] & 0x3F); \
      *(dst)++ = '\\'; *(dst)++ = 'u'; \
      *(dst)++ = (u8)JHEX[(_cp >> 12) & 0xF]; *(dst)++ = (u8)JHEX[(_cp >> 8) & 0xF]; \
      *(dst)++ = (u8)JHEX[(_cp >> 4) & 0xF]; *(dst)++ = (u8)JHEX[_cp & 0xF]; \
      (plain) = (j) + 3; } \
    else { *(dst)++ = _c; (plain) = (j) + 1; } \
  } while (0)

// v8: shuffle tables for the quote/backslash fast path in
// json_write_str_avx2. For group mask gm (bit k = input byte k
// is a '"' or '\\' escape): SHUF_IDX permutes the 8 input bytes
// into output order (0x80 = gap where '\\' goes), INS_SLASH
// supplies the '\\' bytes, OUTLEN is the output length.
static const u8 JW_SHUF_IDX[256][16] = {
  { 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x03,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x03,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x03,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x03,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x02,0x80,0x03,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x80,0x03,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x80,0x03,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x80,0x03,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x80,0x03,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x80,0x03,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x02,0x03,0x80,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x03,0x80,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x03,0x80,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x03,0x80,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x03,0x80,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x03,0x80,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x03,0x80,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x03,0x80,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x02,0x80,0x03,0x80,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x80,0x03,0x80,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x80,0x03,0x80,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x80,0x03,0x80,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x05,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x05,0x06,0x07,0x80,0x80,0x80 },
  { 0x00,0x01,0x02,0x03,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x03,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x03,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x03,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x03,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x03,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x03,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x03,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x02,0x80,0x03,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x80,0x03,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x80,0x03,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x80,0x03,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x80,0x03,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x80,0x03,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80 },
  { 0x00,0x01,0x02,0x03,0x80,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x03,0x80,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x03,0x80,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x03,0x80,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x03,0x80,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x03,0x80,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x03,0x80,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x03,0x80,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80 },
  { 0x00,0x01,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x06,0x07,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x06,0x07,0x80,0x80 },
  { 0x00,0x01,0x02,0x03,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x03,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x03,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x03,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x03,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x03,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x03,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x03,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x02,0x80,0x03,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x80,0x03,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x80,0x03,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x80,0x03,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x80,0x03,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x80,0x03,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80 },
  { 0x00,0x01,0x02,0x03,0x80,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x03,0x80,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x03,0x80,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x03,0x80,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x03,0x80,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x03,0x80,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x03,0x80,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x03,0x80,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80 },
  { 0x00,0x01,0x02,0x80,0x03,0x80,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x80,0x03,0x80,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x80,0x03,0x80,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x80,0x03,0x80,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x05,0x80,0x06,0x07,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x05,0x80,0x06,0x07,0x80,0x80 },
  { 0x00,0x01,0x02,0x03,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x03,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x03,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x03,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x03,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x03,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x03,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x03,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80 },
  { 0x00,0x01,0x02,0x80,0x03,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x80,0x03,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x80,0x03,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x80,0x03,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x80,0x03,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x80,0x03,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80 },
  { 0x00,0x01,0x02,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80 },
  { 0x00,0x01,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x07,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x07,0x80 },
  { 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x03,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x03,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x03,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x03,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x03,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x03,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x02,0x80,0x03,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x80,0x03,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x80,0x03,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x80,0x03,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x80,0x03,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x80,0x03,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x00,0x01,0x02,0x03,0x80,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x03,0x80,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x03,0x80,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x03,0x80,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x03,0x80,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x03,0x80,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x03,0x80,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x03,0x80,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x00,0x01,0x02,0x80,0x03,0x80,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x80,0x03,0x80,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x80,0x03,0x80,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x80,0x03,0x80,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x05,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x05,0x06,0x80,0x07,0x80,0x80 },
  { 0x00,0x01,0x02,0x03,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x03,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x03,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x03,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x03,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x03,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x03,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x03,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x00,0x01,0x02,0x80,0x03,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x80,0x03,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x80,0x03,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x80,0x03,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x80,0x03,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x80,0x03,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80 },
  { 0x00,0x01,0x02,0x03,0x80,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x03,0x80,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x03,0x80,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x03,0x80,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x03,0x80,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x03,0x80,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x03,0x80,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x03,0x80,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80 },
  { 0x00,0x01,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x06,0x80,0x07,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x06,0x80,0x07,0x80 },
  { 0x00,0x01,0x02,0x03,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x03,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x03,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x03,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x03,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x03,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x03,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x03,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x00,0x01,0x02,0x80,0x03,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x80,0x03,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x80,0x03,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x80,0x03,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x80,0x03,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x80,0x03,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80 },
  { 0x00,0x01,0x02,0x03,0x80,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x03,0x80,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x03,0x80,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x03,0x80,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x03,0x80,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x03,0x80,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x03,0x80,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x03,0x80,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80 },
  { 0x00,0x01,0x02,0x80,0x03,0x80,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x80,0x03,0x80,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x80,0x03,0x80,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x80,0x03,0x80,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x05,0x80,0x06,0x80,0x07,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x05,0x80,0x06,0x80,0x07,0x80 },
  { 0x00,0x01,0x02,0x03,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x03,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x03,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x03,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x03,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x03,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x03,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x03,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80 },
  { 0x00,0x01,0x02,0x80,0x03,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x80,0x03,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x80,0x03,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x80,0x03,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x80,0x03,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x80,0x03,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80 },
  { 0x00,0x01,0x02,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80 },
  { 0x00,0x01,0x80,0x02,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80 },
  { 0x00,0x01,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80,0x80 },
  { 0x80,0x00,0x01,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80 },
  { 0x00,0x80,0x01,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80 },
  { 0x80,0x00,0x80,0x01,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80 },
  { 0x00,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80,0x80 },
  { 0x80,0x00,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80 },
  { 0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07,0x80 },
  { 0x80,0x00,0x80,0x01,0x80,0x02,0x80,0x03,0x80,0x04,0x80,0x05,0x80,0x06,0x80,0x07 },
};
static const u8 JW_INS_SLASH[256][16] = {
  { 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00 },
  { 0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00 },
  { 0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00,0x00 },
  { 0x5C,0x00,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00 },
  { 0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x00 },
  { 0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00,0x5C,0x00 },
};
static const u8 JW_OUTLEN[256] = {
  8,9,9,10,9,10,10,11,9,10,10,11,10,11,11,12,9,10,10,11,10,11,11,12,10,11,11,12,11,12,12,13,9,10,10,11,10,11,11,12,10,11,11,12,11,12,12,13,10,11,11,12,11,12,12,13,11,12,12,13,12,13,13,14,9,10,10,11,10,11,11,12,10,11,11,12,11,12,12,13,10,11,11,12,11,12,12,13,11,12,12,13,12,13,13,14,10,11,11,12,11,12,12,13,11,12,12,13,12,13,13,14,11,12,12,13,12,13,13,14,12,13,13,14,13,14,14,15,9,10,10,11,10,11,11,12,10,11,11,12,11,12,12,13,10,11,11,12,11,12,12,13,11,12,12,13,12,13,13,14,10,11,11,12,11,12,12,13,11,12,12,13,12,13,13,14,11,12,12,13,12,13,13,14,12,13,13,14,13,14,14,15,10,11,11,12,11,12,12,13,11,12,12,13,12,13,13,14,11,12,12,13,12,13,13,14,12,13,13,14,13,14,14,15,11,12,12,13,12,13,13,14,12,13,13,14,13,14,14,15,12,13,13,14,13,14,14,15,13,14,14,15,14,15,15,16
};

// v9: fill table for \n-only groups in json_write_str_avx2. For \n-mask nm:
// gap positions get 0x5C ('\'), the \n byte positions get 0x64 (0x0A|0x64='n').
static const u8 JW_N_FILL[256][16] = {
  { 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00 },
  { 0x00,0x00,0x00,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00 },
  { 0x00,0x00,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00 },
  { 0x00,0x00,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00 },
  { 0x00,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00 },
  { 0x00,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00 },
  { 0x00,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00 },
  { 0x00,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00,0x00 },
  { 0x5C,0x64,0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00 },
  { 0x00,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00 },
  { 0x00,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00,0x00 },
  { 0x5C,0x64,0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00 },
  { 0x00,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x00 },
  { 0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64,0x5C,0x64 },
};

__attribute__((target("avx2")))
static void json_write_str_avx2(Buf *out, Slice s, int quotes) {
  // v8: shuffle fast path. The old version drained one escape at a time
  // (ctz + JW_PLAIN copy_span + JW_ESCAPE branch chain per escape,
  // ~40 ns each, ~300k escapes/run). Now each 32-byte chunk is split
  // into 4 groups of 8 bytes: a group whose only escapes are '"' and
  // '\' is emitted with ONE shuffle + one OR + one 16-byte store
  // (tables JW_SHUF_IDX / JW_INS_SLASH / JW_OUTLEN); clean groups are
  // a single memcpy; v9: '\n' groups take the shuffle too (JW_N_FILL for
  // \n-only groups, per-bit fixup for mixed groups); only groups with a
  // control byte other than '\n' or a 0xED lead byte take the old scalar
  // path (JW_ESCAPE, byte-exact).
  // The 16-byte store writes up to 7 bytes past the group's logical
  // end; those bytes are inside the pre-sized buffer and are
  // overwritten by the next group (or lie past out->len at the end).
  const u8 *p = s.p; size_t n = s.n;
  buf_reserve(out, n + 64);
  u8 *dst = out->p + out->len;
  u8 *end = out->p + out->cap;
  if (quotes) *dst++ = '"';
  __m256i quote = _mm256_set1_epi8('"');
  __m256i backslash = _mm256_set1_epi8('\\');
  __m256i lead_ed = _mm256_set1_epi8((char)0xED);
  __m256i lo31 = _mm256_set1_epi8(0x1F);
  __m256i newline = _mm256_set1_epi8('\n');
  __m256i zero = _mm256_setzero_si256();
  size_t i = 0;
  while (i + 32 <= n) {
    // Worst case for a chunk: every byte a 6-byte escape (192 bytes).
    if (dst + 200 > end) { out->len = (size_t)(dst - out->p); buf_reserve(out, 256); dst = out->p + out->len; end = out->p + out->cap; }
    __m256i chunk = _mm256_loadu_si256((const __m256i *)(p + i));
    unsigned qmask = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, quote));
    unsigned bmask = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, backslash));
    unsigned cmask = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_subs_epu8(chunk, lo31), zero));
    unsigned edmask = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, lead_ed));
    unsigned nmask = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, newline));
    unsigned hard = (cmask & ~nmask) | edmask;  // v9: \n is shuffle-handled, not hard
    unsigned shuf = qmask | bmask;   // bytes the shuffle handles
    if (!hard && !shuf && !nmask) { memcpy(dst, p + i, 32); dst += 32; i += 32; continue; }
    for (int g = 0; g < 4; g++) {
      unsigned hg = (hard >> (g * 8)) & 0xFFu;
      unsigned sg = (shuf >> (g * 8)) & 0xFFu;
      unsigned ng = (nmask >> (g * 8)) & 0xFFu;
      const u8 *gp = p + i + (size_t)g * 8;
      if (!hg && !sg && !ng) { memcpy(dst, gp, 8); dst += 8; continue; }
      if (!hg) {
        if (!ng) {
          __m128i v = _mm_loadl_epi64((const __m128i *)gp);
          __m128i sh = _mm_loadu_si128((const __m128i *)JW_SHUF_IDX[sg]);
          __m128i sl = _mm_loadu_si128((const __m128i *)JW_INS_SLASH[sg]);
          v = _mm_or_si128(_mm_shuffle_epi8(v, sh), sl);
          _mm_storeu_si128((__m128i *)dst, v);
          dst += JW_OUTLEN[sg];
          continue;
        }
        if (!sg) {
          // v9: \n-only group: same shuffle, JW_N_FILL turns 0x0A into 'n'.
          __m128i v = _mm_loadl_epi64((const __m128i *)gp);
          __m128i sh = _mm_loadu_si128((const __m128i *)JW_SHUF_IDX[ng]);
          __m128i fl = _mm_loadu_si128((const __m128i *)JW_N_FILL[ng]);
          v = _mm_or_si128(_mm_shuffle_epi8(v, sh), fl);
          _mm_storeu_si128((__m128i *)dst, v);
          dst += JW_OUTLEN[ng];
          continue;
        }
        // Mixed " / \ / \n group: standard shuffle, then fix up \n bytes.
        // The shuffle puts each escape BYTE after its own gap (output
        // position b + popcount(escapes before b) + 1); the gap holds
        // '\'. The fixup must overwrite the 0x0A BYTE at that +1
        // position with 'n' — writing at the gap position instead
        // (missing +1) overwrote '\' with 'n' and left a raw 0x0A in
        // the JSON output (found by practice 3/3 failing, 2026-11-24).
        unsigned em = sg | ng;
        __m128i v = _mm_loadl_epi64((const __m128i *)gp);
        __m128i sh = _mm_loadu_si128((const __m128i *)JW_SHUF_IDX[em]);
        __m128i sl = _mm_loadu_si128((const __m128i *)JW_INS_SLASH[em]);
        v = _mm_or_si128(_mm_shuffle_epi8(v, sh), sl);
        _mm_storeu_si128((__m128i *)dst, v);
        unsigned nm2 = ng;
        while (nm2) {
          unsigned b = __builtin_ctz(nm2);
          size_t op = (size_t)b + (size_t)__builtin_popcount(em & ((1u << b) - 1u)) + 1;
          dst[op] = (u8)'n';
          nm2 &= nm2 - 1;
        }
        dst += JW_OUTLEN[em];
        continue;
      }
      // Scalar group: control byte / '\n' / 0xED in here.
      size_t gp0 = i + (size_t)g * 8, plain = gp0;
      for (size_t k = gp0; k < gp0 + 8; k++) {
        u8 c = p[k];
        if (c == '"' || c == '\\' || c < 0x20 || c == 0xED) {
          JW_PLAIN(dst, end, p, plain, k);
          JW_ESCAPE(dst, end, p, n, k, plain);
        }
      }
      JW_PLAIN(dst, end, p, plain, gp0 + 8);
    }
    i += 32;
  }
  // Tail (<= 31 bytes): scalar, as before.
  size_t plain = i;
  while (i < n) {
    u8 c = p[i];
    if (c == '"' || c == '\\' || c < 0x20 || c == 0xED) {
      JW_PLAIN(dst, end, p, plain, i);
      JW_ESCAPE(dst, end, p, n, i, plain);
      i = plain;
    } else i++;
  }
  JW_PLAIN(dst, end, p, plain, n);
  JW_ROOM(dst, end, 1);
  if (quotes) *dst++ = '"';
  out->len = (size_t)(dst - out->p);
}

static void json_write_str_sse2(Buf *out, Slice s, int quotes) {
  const u8 *p = s.p; size_t n = s.n;
  buf_reserve(out, n + 64);
  u8 *dst = out->p + out->len;
  u8 *end = out->p + out->cap;
  if (quotes) *dst++ = '"';
  __m128i quote = _mm_set1_epi8('"');
  __m128i backslash = _mm_set1_epi8('\\');
  __m128i lead_ed = _mm_set1_epi8((char)0xED);
  __m128i lo31 = _mm_set1_epi8(0x1F);
  __m128i zero = _mm_setzero_si128();
  size_t i = 0, plain = 0;
  while (i + 16 <= n) {
    __m128i chunk = _mm_loadu_si128((const __m128i *)(p + i));
    __m128i m = _mm_or_si128(_mm_or_si128(_mm_cmpeq_epi8(chunk, quote), _mm_cmpeq_epi8(chunk, backslash)),
                              _mm_or_si128(_mm_cmpeq_epi8(chunk, lead_ed),
                                           _mm_cmpeq_epi8(_mm_subs_epu8(chunk, lo31), zero)));
    unsigned mask = (unsigned)_mm_movemask_epi8(m);
    if (!mask) { i += 16; continue; }
    size_t base = i;
    while (mask) {
      unsigned b = __builtin_ctz(mask);
      size_t j = base + b;
      JW_PLAIN(dst, end, p, plain, j);
      JW_ESCAPE(dst, end, p, n, j, plain);
      mask &= mask - 1;
    }
    i = plain;
  }
  if (i < n && n >= 16) {
    __m128i chunk = _mm_loadu_si128((const __m128i *)(p + n - 16));
    __m128i m = _mm_or_si128(_mm_or_si128(_mm_cmpeq_epi8(chunk, quote), _mm_cmpeq_epi8(chunk, backslash)),
                              _mm_or_si128(_mm_cmpeq_epi8(chunk, lead_ed),
                                           _mm_cmpeq_epi8(_mm_subs_epu8(chunk, lo31), zero)));
    unsigned mask = (unsigned)_mm_movemask_epi8(m);
    size_t skip = i - (n - 16);
    mask &= ~((1u << skip) - 1u);
    size_t base = n - 16;
    while (mask) {
      unsigned b = __builtin_ctz(mask);
      size_t j = base + b;
      JW_PLAIN(dst, end, p, plain, j);
      JW_ESCAPE(dst, end, p, n, j, plain);
      mask &= mask - 1;
    }
    i = plain;
  }
  while (i < n) {
    u8 c = p[i];
    if (c == '"' || c == '\\' || c < 0x20 || c == 0xED) {
      JW_PLAIN(dst, end, p, plain, i);
      JW_ESCAPE(dst, end, p, n, i, plain);
      i = plain;
    } else i++;
  }
  JW_PLAIN(dst, end, p, plain, n);
  JW_ROOM(dst, end, 1);
  if (quotes) *dst++ = '"';
  out->len = (size_t)(dst - out->p);
}

// JSON string writer: AVX2 when available, SSE2 otherwise.
static void json_write_str(Buf *out, Slice s) {
#ifdef SENJA_PROFILE
  PROF_T0; prof_njw++;
#endif
#ifdef SENJA_COUNTERS
  if (have_avx2) cnt_avx2_jw++; else cnt_sse2_jw++;
#endif
  if (have_avx2) json_write_str_avx2(out, s, 1);
  else json_write_str_sse2(out, s, 1);
#ifdef SENJA_PROFILE
  PROF_ADD(prof_jw);
#endif
}

// v11: JSON-escape a string WITHOUT the surrounding quotes — for the
// fused renderer, which writes page HTML straight into the final
// answers.json buffer (the opening quote is already written by the
// emitter, the closing quote follows the render).
static void json_write_raw(Buf *out, Slice s) {
#ifdef SENJA_COUNTERS
  if (have_avx2) cnt_avx2_jwr++; else cnt_sse2_jwr++;
#endif
  if (have_avx2) json_write_str_avx2(out, s, 0);
  else json_write_str_sse2(out, s, 0);
}

// JavaScript Number.prototype.toString for a double (ECMAScript
// Number::toString): shortest digits that round-trip, then the spec's
// plain/scientific layout. Integers with |n| < 1e21 use plain digits.
// Hand-rolled exponent parse for "%e" output ("[+-]digits"); avoids atoi,
// which GCC rewrites to strtol -> __isoc23_strtol@GLIBC_2.38 (the .node
// would then fail to load on any glibc < 2.38, e.g. the grader's).
static int parse_exp(const char *e) {
  int neg = 0, v = 0;
  if (*e == '-') { neg = 1; e++; }
  else if (*e == '+') e++;
  while (*e >= '0' && *e <= '9') v = v * 10 + (*e++ - '0');
  return neg ? -v : v;
}

static void js_num(Buf *out, double n) {
  if (isnan(n)) { buf_cs(out, "NaN"); return; }
  if (isinf(n)) { buf_cs(out, n < 0 ? "-Infinity" : "Infinity"); return; }
  if (n == 0.0) { buf_c(out, '0'); return; } // -0 prints "0" in JS too
  if (n < 0) { buf_c(out, '-'); js_num(out, -n); return; }
  if (n == trunc(n) && n < 1e21) {
    // Exact integer digits via %.0f (glibc converts exactly).
    char tmp[32];
    int m = snprintf(tmp, sizeof(tmp), "%.0f", n);
    buf_put(out, tmp, (size_t)m);
    return;
  }
  // Shortest round-trip decimal digits: try 15, 16, 17 significant digits.
  char digs[24]; int ndig = 0, exp10 = 0;
  for (int prec = 15; prec <= 17; prec++) {
    char tmp[32];
    snprintf(tmp, sizeof(tmp), "%.*e", prec - 1, n);
    // tmp like "1.2345e+20" or "1e-07".
    char *e = strchr(tmp, 'e');
    exp10 = parse_exp(e + 1);
    int di = 0;
    for (char *p = tmp; p < e; p++) {
      if (*p >= '0' && *p <= '9') digs[di++] = *p;
    }
    while (di > 1 && digs[di - 1] == '0') di--; // strip trailing zeros
    digs[di] = 0;
    // Round-trip check.
    char back[40];
    snprintf(back, sizeof(back), "%se%d", digs, exp10 - (di - 1));
    if (strtod(back, NULL) == n) { ndig = di; break; }
  }
  if (ndig == 0) { // fall back to 17 digits
    char tmp[32];
    snprintf(tmp, sizeof(tmp), "%.16e", n);
    char *e = strchr(tmp, 'e');
    exp10 = parse_exp(e + 1);
    int di = 0;
    for (char *p = tmp; p < e; p++) if (*p >= '0' && *p <= '9') digs[di++] = *p;
    while (di > 1 && digs[di - 1] == '0') di--;
    digs[di] = 0; ndig = di;
  }
  int k = ndig, nn = exp10 + 1; // n = 0.digs * 10^nn
  if (k <= nn && nn <= 21) {
    buf_put(out, digs, (size_t)k);
    for (int i = 0; i < nn - k; i++) buf_c(out, '0');
  } else if (0 < nn && nn <= 21) {
    buf_put(out, digs, (size_t)nn);
    buf_c(out, '.');
    buf_put(out, digs + nn, (size_t)(k - nn));
  } else if (-6 < nn && nn <= 0) {
    buf_cs(out, "0.");
    for (int i = 0; i < -nn; i++) buf_c(out, '0');
    buf_put(out, digs, (size_t)k);
  } else {
    buf_c(out, (u8)digs[0]);
    if (k > 1) { buf_c(out, '.'); buf_put(out, digs + 1, (size_t)(k - 1)); }
    buf_c(out, 'e');
    buf_c(out, (u8)(nn - 1 < 0 ? '-' : '+'));
    int ex = nn - 1; if (ex < 0) ex = -ex;
    char et[8]; int m = snprintf(et, sizeof(et), "%d", ex);
    buf_put(out, et, (size_t)m);
  }
}



// ---- carried from v5: esc/slugify (v5 lines 749-921) ----

static const u8 ESC_AMP_B[16] = { '&','&','&','&','&','&','&','&','&','&','&','&','&','&','&','&' };
static const u8 ESC_LT_B[16] = { '<','<','<','<','<','<','<','<','<','<','<','<','<','<','<','<' };
static const u8 ESC_GT_B[16] = { '>','>','>','>','>','>','>','>','>','>','>','>','>','>','>','>' };
static const u8 ESC_QUOT_B[16] = { '"','"','"','"','"','"','"','"','"','"','"','"','"','"','"','"' };
static const u8 ESC_APOS_B[16] = { '\'','\'','\'','\'','\'','\'','\'','\'','\'','\'','\'','\'','\'','\'','\'','\'' };

static inline void esc_one(Buf *out, u8 c) {
  switch (c) {
    case '&': buf_cs(out, "&amp;"); break;
    case '<': buf_cs(out, "&lt;"); break;
    case '>': buf_cs(out, "&gt;"); break;
    case '"': buf_cs(out, "&quot;"); break;
    default: buf_cs(out, "&#39;"); break;
  }
}


// Shufti tables for esc's 5 specials: & < > " '
// LO[lo] = bitmask of hi nibbles h with ((h<<4)|lo) a special;
// BIT[hi] = 1<<hi.
static const u8 ESC_LO_TBL[16] = { 0x00,0x00,0x04,0x00,0x00,0x00,0x04,0x04,0x00,0x00,0x00,0x00,0x08,0x00,0x08,0x00 };
static const u8 ESC_BIT_TBL[16] = { 0x00,0x00,0x04,0x08,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 };

static void esc(Buf *out, Slice s) {
#ifdef SENJA_PROFILE
  PROF_T0; prof_nesc++;
#endif
  const u8 *p = s.p; size_t n = s.n;
  // v9: raw-pointer output (was buf_put/buf_cs per escape, each with a
  // buf_reserve call) + shufti scan (2 shuffles + AND replace 5 cmpeq).
  // Specials: & < > " ' (lo nibbles 6,C,E,2,7; hi nibbles 2,3).
  __m128i esc_lo = _mm_loadu_si128((const __m128i *)ESC_LO_TBL);
  __m128i esc_bit = _mm_loadu_si128((const __m128i *)ESC_BIT_TBL);
  __m128i lowmask = _mm_set1_epi8(0x0F);
  __m128i zero = _mm_setzero_si128();
  buf_reserve(out, n + n / 4 + 128);
  u8 *wp = out->p + out->len;
  u8 *wend = out->p + out->cap;
#define ESC_ROOM(need) do { if (wp + (size_t)(need) > wend) { \
    out->len = (size_t)(wp - out->p); buf_reserve(out, (size_t)(need) + 256); \
    wp = out->p + out->len; wend = out->p + out->cap; } } while (0)
#define ESC_LIT(str) do { ESC_ROOM(sizeof(str) - 1); wp = copy_span(wp, (const u8 *)(str), sizeof(str) - 1); } while (0)
  size_t plain = 0; // start of the not-yet-copied plain run
  size_t i = 0;
  while (i < n) {
    size_t j;
    if (i + 16 <= n) {
      __m128i chunk = _mm_loadu_si128((const __m128i *)(p + i));
      __m128i lo = _mm_and_si128(chunk, lowmask);
      __m128i hi = _mm_and_si128(_mm_srli_epi16(chunk, 4), lowmask);
      __m128i cand = _mm_and_si128(_mm_shuffle_epi8(esc_lo, lo), _mm_shuffle_epi8(esc_bit, hi));
      unsigned mask = (unsigned)(~_mm_movemask_epi8(_mm_cmpeq_epi8(cand, zero))) & 0xFFFFu;
      if (!mask) { i += 16; continue; }
      j = i + (size_t)__builtin_ctz(mask);
    } else {
      size_t k = i;
      while (k < n) {
        u8 c = p[k];
        if (c == '&' || c == '<' || c == '>' || c == '"' || c == '\'') break;
        k++;
      }
      if (k >= n) break;
      j = k;
    }
    if (j > plain) { ESC_ROOM(j - plain); wp = copy_span(wp, p + plain, j - plain); }
    switch (p[j]) {
      case '&': ESC_LIT("&amp;"); break;
      case '<': ESC_LIT("&lt;"); break;
      case '>': ESC_LIT("&gt;"); break;
      case '"': ESC_LIT("&quot;"); break;
      default: ESC_LIT("&#39;"); break;
    }
    plain = j + 1;
    i = plain;
  }
  if (plain < n) { ESC_ROOM(n - plain); wp = copy_span(wp, p + plain, n - plain); }
  out->len = (size_t)(wp - out->p);
#undef ESC_ROOM
#undef ESC_LIT
#ifdef SENJA_PROFILE
  PROF_ADD(prof_esc);
#endif
}



// ---- v11: esc_j — fused HTML + JSON escape for render slots.
// render_chunks now writes DIRECTLY into the final answers.json buffer
// (no page_buf + json_write_str second pass). A non-raw slot value must
// therefore be HTML-escaped AND JSON-escaped in ONE pass. Composition,
// byte-exact vs esc() followed by json_write_str():
//   & < > " ' -> HTML entities (no JSON specials left in them)
//   \ -> \\ ; 0x08/09/0x0A/0x0C/0x0D -> \b \t \n \f \r ; other <0x20 -> \u00XX
//   0xED + surrogate pattern -> \uDXXX (3 bytes consumed, as JW_ESCAPE);
//   a lone 0xED is emitted literally (also as JW_ESCAPE).
// Tables extend esc's shufti tables with \ (lo C hi 5) and controls
// (hi 0/1, all lo nibbles); 0xED is flagged by a dedicated cmpeq mask
// (like json_write_str_avx2's edmask) so surrogate sequences are seen.
static const u8 ESCJ_LO_TBL[16] = { 0x03,0x03,0x07,0x03,0x03,0x03,0x07,0x07,0x03,0x03,0x03,0x03,0x2B,0x03,0x0B,0x03 };
static const u8 ESCJ_BIT_TBL[16] = { 0x01,0x02,0x04,0x08,0x10,0x20,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 };

static void esc_j(Buf *out, Slice s) {
  const u8 *p = s.p; size_t n = s.n;
  __m128i esc_lo = _mm_loadu_si128((const __m128i *)ESCJ_LO_TBL);
  __m128i esc_bit = _mm_loadu_si128((const __m128i *)ESCJ_BIT_TBL);
  __m128i lowmask = _mm_set1_epi8(0x0F);
  __m128i zero = _mm_setzero_si128();
  __m128i edv = _mm_set1_epi8((char)0xED);
  buf_reserve(out, n + n / 4 + 128);
  u8 *wp = out->p + out->len;
  u8 *wend = out->p + out->cap;
#define ESCJ_ROOM(need) do { if (wp + (size_t)(need) > wend) { \
    out->len = (size_t)(wp - out->p); buf_reserve(out, (size_t)(need) + 256); \
    wp = out->p + out->len; wend = out->p + out->cap; } } while (0)
#define ESCJ_LIT(str) do { ESCJ_ROOM(sizeof(str) - 1); wp = copy_span(wp, (const u8 *)(str), sizeof(str) - 1); } while (0)
  size_t plain = 0, i = 0;
  while (i < n) {
    size_t j;
    if (i + 16 <= n) {
      __m128i chunk = _mm_loadu_si128((const __m128i *)(p + i));
      __m128i lo = _mm_and_si128(chunk, lowmask);
      __m128i hi = _mm_and_si128(_mm_srli_epi16(chunk, 4), lowmask);
      __m128i cand = _mm_and_si128(_mm_shuffle_epi8(esc_lo, lo), _mm_shuffle_epi8(esc_bit, hi));
      unsigned mask = (unsigned)(~_mm_movemask_epi8(_mm_cmpeq_epi8(cand, zero))) & 0xFFFFu;
      mask |= (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(chunk, edv));
      if (!mask) { i += 16; continue; }
      j = i + (size_t)__builtin_ctz(mask);
    } else {
      size_t k = i;
      while (k < n) {
        u8 c = p[k];
        if (c == '&' || c == '<' || c == '>' || c == '"' || c == '\'' || c == '\\' || c < 0x20 || c == 0xED) break;
        k++;
      }
      if (k >= n) break;
      j = k;
    }
    if (j > plain) { ESCJ_ROOM(j - plain); wp = copy_span(wp, p + plain, j - plain); }
    u8 c = p[j];
    if (c == '&') ESCJ_LIT("&amp;");
    else if (c == '<') ESCJ_LIT("&lt;");
    else if (c == '>') ESCJ_LIT("&gt;");
    else if (c == '"') ESCJ_LIT("&quot;");
    else if (c == '\'') ESCJ_LIT("&#39;");
    else if (c == '\\') ESCJ_LIT("\\\\");
    else if (c == 0x08) ESCJ_LIT("\\b");
    else if (c == 0x09) ESCJ_LIT("\\t");
    else if (c == 0x0A) ESCJ_LIT("\\n");
    else if (c == 0x0C) ESCJ_LIT("\\f");
    else if (c == 0x0D) ESCJ_LIT("\\r");
    else if (c < 0x20) {
      ESCJ_ROOM(6);
      *wp++ = '\\'; *wp++ = 'u'; *wp++ = '0'; *wp++ = '0';
      *wp++ = (u8)JHEX[(c >> 4) & 0xF]; *wp++ = (u8)JHEX[c & 0xF];
      plain = j + 1; i = plain; continue;
    } else if (c == 0xED && j + 2 < n && p[j + 1] >= 0xA0 && p[j + 1] <= 0xBF && (p[j + 2] & 0xC0) == 0x80) {
      u32 cp = 0xD800 + (((u32)(p[j + 1] & 0x3F)) << 6) + (u32)(p[j + 2] & 0x3F);
      ESCJ_ROOM(6);
      *wp++ = '\\'; *wp++ = 'u';
      *wp++ = (u8)JHEX[(cp >> 12) & 0xF]; *wp++ = (u8)JHEX[(cp >> 8) & 0xF];
      *wp++ = (u8)JHEX[(cp >> 4) & 0xF]; *wp++ = (u8)JHEX[cp & 0xF];
      plain = j + 3; i = plain; continue;
    } else { ESCJ_ROOM(1); *wp++ = c; }
    plain = j + 1;
    i = plain;
  }
  if (plain < n) { ESCJ_ROOM(n - plain); wp = copy_span(wp, p + plain, n - plain); }
  out->len = (size_t)(wp - out->p);
#undef ESCJ_ROOM
#undef ESCJ_LIT
}

static int is_mark(u32 cp) {
  if (cp < 0x300) return 0; // combining marks start at U+0300
  size_t lo = 0, hi = MARK_N;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    u32 a = MARK_RANGES[mid][0], b = MARK_RANGES[mid][1];
    if (cp < a) hi = mid;
    else if (cp > b) lo = mid + 1;
    else return 1;
  }
  return 0;
}

// Fully-expanded NFKD sequence for cp, or NULL. Returns count via *n.
static const u32 *decomp_of(u32 cp, size_t *n) {
  size_t lo = 0, hi = DECOMP_N;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    u32 k = DECOMP_KEYS[mid];
    if (cp < k) hi = mid;
    else if (cp > k) lo = mid + 1;
    else { *n = (size_t)(DECOMP_OFF[mid + 1] - DECOMP_OFF[mid]); return DECOMP_VALS + DECOMP_OFF[mid]; }
  }
  return NULL;
}

static inline void step_slug(u32 c, Buf *out, int *gap) {
  if (is_mark(c)) return;
  int keep = -1;
  if (c >= 'A' && c <= 'Z') keep = (int)(c + 32);
  else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) keep = (int)c;
  if (keep < 0) { *gap = out->len > 0; }
  else {
    if (*gap) buf_c(out, '-');
    *gap = 0;
    buf_c(out, (u8)keep);
  }
}

// The slug of `text`: NFKD, marks dropped, A-Z lowered, runs of other
// chars one '-', none at either end, cut to 64 with a trailing '-' at the
// cut dropped, 'untitled' when nothing is left.
static Slice slugify(Arena *arena, Slice text) {
#ifdef SENJA_PROFILE
  PROF_T0;
#endif
  Buf out = {0};
  int gap = 0;
  // ASCII fast path: no ASCII code point has an NFKD decomposition and
  // no ASCII char is a mark, so bytes map straight through.
  int ascii = 1;
  for (size_t k = 0; k + 16 <= text.n; k += 16) {
    if (_mm_movemask_epi8(_mm_loadu_si128((const __m128i *)(text.p + k)))) { ascii = 0; break; }
  }
  if (ascii) for (size_t k = text.n & ~(size_t)15; k < text.n; k++) if (text.p[k] >= 0x80) { ascii = 0; break; }
  if (ascii) {
    for (size_t i = 0; i < text.n; i++) {
      u8 c = text.p[i];
      int keep = -1;
      if (c >= 'A' && c <= 'Z') keep = (int)(c + 32);
      else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) keep = (int)c;
      if (keep < 0) { gap = out.len > 0; }
      else {
        if (gap) buf_c(&out, '-');
        gap = 0;
        buf_c(&out, (u8)keep);
      }
    }
  } else {
    size_t i = 0;
    while (i < text.n) {
      size_t used; u32 cp = utf8_decode(text.p, text.n, i, &used);
      i += used;
      size_t n; const u32 *seq = cp < 0x80 ? NULL : decomp_of(cp, &n);
      if (seq) { for (size_t k = 0; k < n; k++) step_slug(seq[k], &out, &gap); }
      else step_slug(cp, &out, &gap);
    }
  }
  if (out.len > 64) out.len = 64;
  while (out.len > 0 && out.p[out.len - 1] == '-') out.len--;
  Slice r;
  if (out.len == 0) r = slc("untitled");
  else r = arena_slice(arena, out.p, out.len);
  free(out.p);
#ifdef SENJA_PROFILE
  PROF_ADD(prof_slugw);
#endif
  return r;
}

// `slug` made unique against `taken` (and added to it).
static Slice unique_slug(Arena *arena, StrMap *taken, Slice slug) {
  if (!strmap_has(taken, slug)) { strmap_put(taken, slug, (void *)1); return slug; }
  u32 n = 2;
  for (;;) {
    char num[16]; int m = snprintf(num, sizeof(num), "%u", n);
    Buf tmp = {0};
    buf_sl(&tmp, slug); buf_c(&tmp, '-'); buf_put(&tmp, num, (size_t)m);
    Slice cand = arena_slice(arena, tmp.p, tmp.len);
    free(tmp.p);
    if (!strmap_has(taken, cand)) { strmap_put(taken, cand, (void *)1); return cand; }
    n++;
  }
}



// ---- carried from v5: long_date/words_of (v5 lines 924-1005) ----

static const char *MONTHS[12] = {
  "January", "February", "March", "April", "May", "June", "July",
  "August", "September", "October", "November", "December",
};

// A YYYY-MM-DD date in words, from the string alone.
static Slice long_date(Arena *arena, Slice date) {
  u32 day = (u32)(date.p[8] - '0') * 10 + (u32)(date.p[9] - '0');
  size_t mon = (size_t)(((u32)(date.p[5] - '0') * 10 + (u32)(date.p[6] - '0')));
  char num[8]; int m = snprintf(num, sizeof(num), "%u", day);
  Buf out = {0};
  buf_put(&out, num, (size_t)m); buf_c(&out, ' ');
  buf_cs(&out, MONTHS[mon - 1]); buf_c(&out, ' ');
  buf_put(&out, date.p, 4);
  Slice r = arena_slice(arena, out.p, out.len);
  free(out.p);
  return r;
}

// The words of a body: runs of characters other than whitespace
// (the reference's /\S+/g over the JS \s set). Pure-ASCII 16-byte
// chunks are counted with an SSSE3 whitespace classifier (a word starts
// where a non-ws byte follows a ws byte); any chunk with a byte >= 0x80
// falls back to the scalar UTF-8 path for that chunk, so the JS \s set
// for non-ASCII is exactly preserved.
static size_t words_of(Slice text) {
#ifdef SENJA_PROFILE
  PROF_T0;
#endif
  static const u8 WS_LO[16] = { 0x04,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x01,0x01,0x01,0x01,0x00,0x00 };
  static const u8 WS_BIT[16] = { 0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 };
  __m128i ws_lo = _mm_loadu_si128((const __m128i *)WS_LO);
  __m128i ws_bit = _mm_loadu_si128((const __m128i *)WS_BIT);
  __m128i lowmask = _mm_set1_epi8(0x0F);
  __m128i zero = _mm_setzero_si128();
  size_t n = 0, i = 0;
  int prev_ws = 1; // as if preceded by whitespace (first word counts)
  while (i < text.n) {
    if (i + 16 <= text.n) {
      __m128i chunk = _mm_loadu_si128((const __m128i *)(text.p + i));
      if (_mm_movemask_epi8(chunk)) {
        // Non-ASCII in this chunk: scalar, character by character.
        // (A character may straddle the chunk end; i then lands on the
        // next character boundary and chunking resumes from there.)
        int inw = !prev_ws;
        size_t end = i + 16;
        while (i < end) {
          u8 c = text.p[i];
          if (c < 0x80) {
            int ws = c == ' ' || (c >= 0x09 && c <= 0x0D);
            if (ws) inw = 0; else if (!inw) { inw = 1; n++; }
            i++;
          } else {
            size_t used; u32 cp = utf8_decode(text.p, text.n, i, &used);
            i += used;
            if (is_js_space(cp)) inw = 0; else if (!inw) { inw = 1; n++; }
          }
        }
        prev_ws = !inw;
        continue;
      }
      __m128i lo = _mm_and_si128(chunk, lowmask);
      __m128i hi = _mm_and_si128(_mm_srli_epi16(chunk, 4), lowmask);
      __m128i cand = _mm_and_si128(_mm_shuffle_epi8(ws_lo, lo), _mm_shuffle_epi8(ws_bit, hi));
      unsigned ws = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(cand, zero));
      ws = ~ws & 0xFFFFu; // bit = 1 iff the byte is whitespace
      unsigned starts = ~ws & ((ws << 1) | (prev_ws ? 1u : 0u)) & 0xFFFFu;
      n += (size_t)__builtin_popcount(starts);
      prev_ws = (int)((ws >> 15) & 1u);
      i += 16;
    } else {
      u8 c = text.p[i];
      if (c < 0x80) {
        int ws = c == ' ' || (c >= 0x09 && c <= 0x0D);
        if (ws) prev_ws = 1; else if (prev_ws) { prev_ws = 0; n++; }
        i++;
      } else {
        size_t used; u32 cp = utf8_decode(text.p, text.n, i, &used);
        i += used;
        if (is_js_space(cp)) prev_ws = 1; else if (prev_ws) { prev_ws = 0; n++; }
      }
    }
  }
  size_t _r = n;
#ifdef SENJA_PROFILE
  PROF_ADD(prof_slugw);
#endif
  return _r;
}



// ---- carried from v5: front_matter (v5 lines 1009-1250) ----

typedef struct {
  int has_title, has_date, has_tags, has_draft, has_layout, has_slug, has_summary;
  Slice title, date, layout, slug, summary;
  Slice *tags; size_t ntags; // arena array
  int draft;
} FmData;

typedef struct { FmData data; Slice body; } Fm;

static int days_in(u32 y, u32 m) {
  if (m == 2) return ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0) ? 29 : 28;
  if (m == 4 || m == 6 || m == 9 || m == 11) return 30;
  return 31;
}

static int is_digit(u8 c) { return c >= '0' && c <= '9'; }

// A string value: "..." with \" and \\ its only escapes, or bare.
// On error returns 0 and sets *err to the reason (static text).
static int fm_string_of(Slice raw, Arena *arena, Slice *out, Slice *err) {
  if (raw.n > 0 && raw.p[0] == '"') {
    if (raw.n < 2 || raw.p[raw.n - 1] != '"') { *err = slc("a quoted string is not closed"); return 0; }
    Buf b = {0};
    size_t i = 1;
    while (i < raw.n - 1) {
      u8 c = raw.p[i];
      if (c == '\\') {
        u8 nx = raw.p[i + 1];
        if ((nx != '"' && nx != '\\') || i + 1 == raw.n - 1) {
          free(b.p); *err = slc("a quoted string holds an escape other than \\\" or \\\\"); return 0;
        }
        buf_c(&b, nx); i += 2;
      } else if (c == '"') {
        free(b.p); *err = slc("a quoted string holds a stray quote"); return 0;
      } else { buf_c(&b, c); i++; }
    }
    *out = arena_slice(arena, b.p ? b.p : (const u8 *)"", b.len);
    free(b.p);
    return 1;
  }
  if (raw.n > 0 && raw.p[0] == '[') { *err = slc("a bare string opens on ["); return 0; }
  *out = raw;
  return 1;
}

// Error line: key "K": reason
static Slice fm_err_key(Arena *arena, Slice key, Slice reason) {
  Buf b = {0};
  buf_cs(&b, "key \""); buf_sl(&b, key); buf_cs(&b, "\": "); buf_sl(&b, reason);
  Slice r = arena_slice(arena, b.p, b.len);
  free(b.p);
  return r;
}

static int fm_value_of(Slice key, Slice raw, FmData *data, Arena *arena, Slice *err) {
  if (sl_eq_c(key, "title") || sl_eq_c(key, "summary")) {
    Slice s;
    if (!fm_string_of(raw, arena, &s, err)) { *err = fm_err_key(arena, key, *err); return 0; }
    if (sl_eq_c(key, "title")) { data->title = s; data->has_title = 1; }
    else { data->summary = s; data->has_summary = 1; }
    return 1;
  }
  if (sl_eq_c(key, "date")) {
    int ok = raw.n == 10 && is_digit(raw.p[0]) && is_digit(raw.p[1]) && is_digit(raw.p[2]) && is_digit(raw.p[3])
          && raw.p[4] == '-' && is_digit(raw.p[5]) && is_digit(raw.p[6]) && raw.p[7] == '-'
          && is_digit(raw.p[8]) && is_digit(raw.p[9]);
    u32 y = 0, mo = 0, d = 0;
    if (ok) {
      y = (u32)(raw.p[0] - '0') * 1000 + (u32)(raw.p[1] - '0') * 100 + (u32)(raw.p[2] - '0') * 10 + (u32)(raw.p[3] - '0');
      mo = (u32)(raw.p[5] - '0') * 10 + (u32)(raw.p[6] - '0');
      d = (u32)(raw.p[8] - '0') * 10 + (u32)(raw.p[9] - '0');
    }
    if (!ok || y < 1970 || y > 2099 || mo < 1 || mo > 12 || d < 1 || d > (u32)days_in(y, mo)) {
      *err = fm_err_key(arena, key, slc("not a date from 1970-01-01 to 2099-12-31 (YYYY-MM-DD)"));
      return 0;
    }
    data->date = raw; data->has_date = 1;
    return 1;
  }
  if (sl_eq_c(key, "tags")) {
    if (!(raw.n >= 2 && raw.p[0] == '[' && raw.p[raw.n - 1] == ']')) {
      *err = fm_err_key(arena, key, slc("not a list [a, b]")); return 0;
    }
    Slice inner = sl(raw.p + 1, raw.n - 2);
    if (js_trim(inner).n == 0) { data->tags = NULL; data->ntags = 0; data->has_tags = 1; return 1; }
    // Split inner on ','.
    size_t cap = 4, n = 0;
    Slice *tags = (Slice *)arena_alloc(arena, cap * sizeof(Slice));
    size_t start = 0;
    for (size_t i = 0; i <= inner.n; i++) {
      if (i == inner.n || inner.p[i] == ',') {
        Slice piece = js_trim(sl(inner.p + start, i - start));
        if (piece.n == 0) { *err = fm_err_key(arena, key, slc("an empty tag, or one holding \" [ or ]")); return 0; }
        for (size_t k = 0; k < piece.n; k++) {
          u8 c = piece.p[k];
          if (c == '"' || c == '[' || c == ']') { *err = fm_err_key(arena, key, slc("an empty tag, or one holding \" [ or ]")); return 0; }
        }
        if (n == cap) {
          size_t nc = cap * 2;
          Slice *nt = (Slice *)arena_alloc(arena, nc * sizeof(Slice));
          memcpy(nt, tags, n * sizeof(Slice));
          tags = nt; cap = nc;
        }
        tags[n++] = piece;
        start = i + 1;
      }
    }
    data->tags = tags; data->ntags = n; data->has_tags = 1;
    return 1;
  }
  if (sl_eq_c(key, "draft")) {
    if (sl_eq_c(raw, "true")) { data->draft = 1; data->has_draft = 1; return 1; }
    if (sl_eq_c(raw, "false")) { data->draft = 0; data->has_draft = 1; return 1; }
    *err = fm_err_key(arena, key, slc("true or false")); return 0;
  }
  if (sl_eq_c(key, "layout")) {
    int ok = raw.n > 0;
    for (size_t i = 0; ok && i < raw.n; i++) {
      u8 c = raw.p[i];
      if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-')) ok = 0;
    }
    if (!ok) { *err = fm_err_key(arena, key, slc("not a template name")); return 0; }
    data->layout = raw; data->has_layout = 1;
    return 1;
  }
  if (sl_eq_c(key, "slug")) {
    int ok = raw.n > 0 && raw.n <= 64;
    if (ok) {
      int prev_dash = 1;
      for (size_t i = 0; i < raw.n; i++) {
        u8 c = raw.p[i];
        if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) prev_dash = 0;
        else if (c == '-' && !prev_dash) prev_dash = 1;
        else { ok = 0; break; }
      }
      if (prev_dash) ok = 0;
    }
    if (!ok) { *err = fm_err_key(arena, key, slc("not a slug")); return 0; }
    data->slug = raw; data->has_slug = 1;
    return 1;
  }
  // Unreachable (KEYS checked by the caller), kept for symmetry.
  Buf b = {0};
  buf_cs(&b, "unknown key \""); buf_sl(&b, key); buf_c(&b, '"');
  *err = arena_slice(arena, b.p, b.len);
  free(b.p);
  return 0;
}

static int fm_has_key(FmData *data, Slice key) {
  if (sl_eq_c(key, "title")) return data->has_title;
  if (sl_eq_c(key, "date")) return data->has_date;
  if (sl_eq_c(key, "tags")) return data->has_tags;
  if (sl_eq_c(key, "draft")) return data->has_draft;
  if (sl_eq_c(key, "layout")) return data->has_layout;
  if (sl_eq_c(key, "slug")) return data->has_slug;
  if (sl_eq_c(key, "summary")) return data->has_summary;
  return 0;
}

static const char *FM_KEYS[7] = { "title", "date", "tags", "draft", "layout", "slug", "summary" };

// Parse front matter: { data, body } or the refusal reason in *err.
// `text` has "\n" line ends. Returns 1 on success.
static int parse_front_matter(Slice text, Arena *arena, Fm *out, Slice *err) {
  if (!(sl_eq_c(text, "---") || sl_starts(text, "---\n"))) { *err = slc("no front matter"); return 0; }
  // Walk the front-matter lines only (one SSE find_byte per line). The
  // body is never scanned: the old version counted/split every line of
  // the whole source and re-scanned for the body start. Collect the FM
  // content lines (between the opening and closing "---") as slices;
  // pass 2 below is unchanged, over fm_lines[i] as line number i + 2.
  size_t cap = 8, nfm = 0;
  Slice *fm_lines = (Slice *)arena_alloc(arena, cap * sizeof(Slice));
  size_t pos = 0, body_start = text.n;
  long close = -1, idx = 0;
  for (;;) {
    size_t nl = find_byte(text.p, text.n, pos, '\n');
    size_t lend = nl < text.n ? nl : text.n;
    if (idx > 0 && lend - pos == 3 && memcmp(text.p + pos, "---", 3) == 0) {
      close = idx;
      body_start = nl < text.n ? nl + 1 : text.n;
      break;
    }
    if (idx > 0) {
      if (nfm == cap) {
        size_t nc = cap * 2;
        Slice *nl2 = (Slice *)arena_alloc(arena, nc * sizeof(Slice));
        memcpy(nl2, fm_lines, nfm * sizeof(Slice));
        fm_lines = nl2; cap = nc;
      }
      fm_lines[nfm++] = sl(text.p + pos, lend - pos);
    }
    if (nl >= text.n) break;
    pos = nl + 1; idx++;
  }
  if (close < 0) { *err = slc("the front matter is not closed"); return 0; }
  // Pass 2: the lines between.
  FmData data; memset(&data, 0, sizeof(data));
  for (size_t li2 = 0; li2 < nfm; li2++) {
    Slice line = fm_lines[li2];
    long line_no = (long)li2 + 2;
    if (is_blank_sp_tab(line) || (line.n > 0 && line.p[0] == '#')) continue;
    size_t ki = 0;
    while (ki < line.n && ((line.p[ki] >= 'a' && line.p[ki] <= 'z') || line.p[ki] == '_')) ki++;
    if (ki == 0 || ki >= line.n || line.p[ki] != ':') {
      char num[64]; int m = snprintf(num, sizeof(num), "line %ld is not \"key: value\"", line_no);
      Buf b = {0}; buf_put(&b, num, (size_t)m);
      *err = arena_slice(arena, b.p, b.len); free(b.p);
      return 0;
    }
    Slice key = sl(line.p, ki);
    Slice raw_full = slc("");
    if (ki + 1 < line.n && line.p[ki + 1] == ' ') raw_full = sl(line.p + ki + 2, line.n - (ki + 2));
    Slice raw = raw_full;
    while (raw.n > 0 && (raw.p[raw.n - 1] == ' ' || raw.p[raw.n - 1] == '\t')) raw.n--;
    int known = 0;
    for (int k = 0; k < 7; k++) if (sl_eq_c(key, FM_KEYS[k])) known = 1;
    if (!known) {
      Buf b = {0}; buf_cs(&b, "unknown key \""); buf_sl(&b, key); buf_c(&b, '"');
      *err = arena_slice(arena, b.p, b.len); free(b.p);
      return 0;
    }
    if (fm_has_key(&data, key)) {
      // Note: no colon here (the reference builds this message directly,
      // unlike value-parser errors which use `key "K": reason`).
      Buf b = {0}; buf_cs(&b, "key \""); buf_sl(&b, key); buf_cs(&b, "\" is given twice");
      *err = arena_slice(arena, b.p, b.len); free(b.p);
      return 0;
    }
    if (raw.n == 0) {
      Buf b = {0}; buf_cs(&b, "key \""); buf_sl(&b, key); buf_cs(&b, "\" has no value");
      *err = arena_slice(arena, b.p, b.len); free(b.p);
      return 0;
    }
    if (!fm_value_of(key, raw, &data, arena, err)) return 0;
  }
  if (!data.has_title) { *err = slc("no title"); return 0; }
  // Body: everything after the closing line's "\n" (already found above).
  out->data = data;
  out->body = sl(text.p + body_start, text.n - body_start);
  return 1;
}



// ---- carried from v5: markdown (v5 lines 1254-1755) ----

typedef struct { u32 level; Slice id, html; } TocEntry;
typedef struct { Slice html; TocEntry *toc; size_t ntoc, captoc; Slice excerpt; size_t words; } MdOut;

// Link resolution context (per source).
typedef struct {
  Slice dir, src_path, base;
  // by_source: sources sorted by path; binary search.
  struct Source *sources; size_t nsources;
  StrMap *static_paths;
  StrMap *broken; // set of report lines
  Arena *arena;
} LinkCtx;

// v12: SIMD newline scan for md_split_lines (was two scalar passes over
// the whole body). AVX2 helpers carry target("avx2") because the shipped
// .node builds at -march=x86-64-v2 (same pattern as json_write_str_avx2).
__attribute__((target("avx2")))
static size_t count_nl_avx2(const u8 *p, size_t n) {
  __m256i nl = _mm256_set1_epi8('\n');
  size_t cnt = 0, i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i c = _mm256_loadu_si256((const __m256i *)(p + i));
    cnt += (size_t)__builtin_popcount((unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(c, nl)));
  }
  for (; i < n; i++) if (p[i] == '\n') cnt++;
  return cnt;
}

__attribute__((target("avx2")))
static void fill_lines_avx2(const u8 *p, size_t n, Slice *lines, size_t *li_io, size_t *start_io) {
  __m256i nl = _mm256_set1_epi8('\n');
  size_t li = *li_io, start = *start_io;
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i c = _mm256_loadu_si256((const __m256i *)(p + i));
    unsigned m = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(c, nl));
    while (m) {
      unsigned b = (unsigned)__builtin_ctz(m);
      size_t pos = i + b;
      lines[li++] = sl(p + start, pos - start);
      start = pos + 1;
      m &= m - 1;
    }
  }
  for (; i < n; i++) if (p[i] == '\n') { lines[li++] = sl(p + start, i - start); start = i + 1; }
  *li_io = li; *start_io = start;
}

static size_t count_nl_sse2(const u8 *p, size_t n) {
  __m128i nl = _mm_set1_epi8('\n');
  size_t cnt = 0, i = 0;
  for (; i + 16 <= n; i += 16) {
    __m128i c = _mm_loadu_si128((const __m128i *)(p + i));
    cnt += (size_t)__builtin_popcount((unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(c, nl)));
  }
  for (; i < n; i++) if (p[i] == '\n') cnt++;
  return cnt;
}

static void fill_lines_sse2(const u8 *p, size_t n, Slice *lines, size_t *li_io, size_t *start_io) {
  __m128i nl = _mm_set1_epi8('\n');
  size_t li = *li_io, start = *start_io;
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m128i c = _mm_loadu_si128((const __m128i *)(p + i));
    unsigned m = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(c, nl));
    while (m) {
      unsigned b = (unsigned)__builtin_ctz(m);
      size_t pos = i + b;
      lines[li++] = sl(p + start, pos - start);
      start = pos + 1;
      m &= m - 1;
    }
  }
  for (; i < n; i++) if (p[i] == '\n') { lines[li++] = sl(p + start, i - start); start = i + 1; }
  *li_io = li; *start_io = start;
}


static const u8 WS2_LO[16] = { 0x04,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x01,0x01,0x01,0x01,0x00,0x00 };
static const u8 WS2_BIT[16] = { 0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 };

// Fused \n-count + word-count: the word algorithm is words_of verbatim
// (same tables, same scalar fallback, same prev_ws carry); only the loop
// skeleton differs (16B word steps inside the 32B \n chunks). A UTF-8 char
// straddling a chunk end is consumed by the scalar path; its continuation
// bytes are >=0x80 (never '\n'), so the \n scan cannot miss one.
__attribute__((target("avx2")))
static size_t count_nl_words_avx2(const u8 *p, size_t n, size_t *out_words) {
  __m256i nl = _mm256_set1_epi8('\n');
  __m128i ws_lo = _mm_loadu_si128((const __m128i *)WS2_LO);
  __m128i ws_bit = _mm_loadu_si128((const __m128i *)WS2_BIT);
  __m128i lowmask = _mm_set1_epi8(0x0F);
  __m128i zero = _mm_setzero_si128();
  size_t cnt = 0, i = 0;
  size_t nw = 0; int prev_ws = 1; size_t wi = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i c = _mm256_loadu_si256((const __m256i *)(p + i));
    cnt += (size_t)__builtin_popcount((unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(c, nl)));
    size_t wend = i + 32;
    while (wi < wend) {
      if (wi + 16 <= n) {
        __m128i chunk = _mm_loadu_si128((const __m128i *)(p + wi));
        if (_mm_movemask_epi8(chunk)) {
          int inw = !prev_ws;
          size_t end = wi + 16;
          while (wi < end) {
            u8 ch = p[wi];
            if (ch < 0x80) {
              int ws = ch == ' ' || (ch >= 0x09 && ch <= 0x0D);
              if (ws) inw = 0; else if (!inw) { inw = 1; nw++; }
              wi++;
            } else {
              size_t used; u32 cp = utf8_decode(p, n, wi, &used);
              wi += used;
              if (is_js_space(cp)) inw = 0; else if (!inw) { inw = 1; nw++; }
            }
          }
          prev_ws = !inw;
          continue;
        }
        __m128i lo = _mm_and_si128(chunk, lowmask);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(chunk, 4), lowmask);
        __m128i cand = _mm_and_si128(_mm_shuffle_epi8(ws_lo, lo), _mm_shuffle_epi8(ws_bit, hi));
        unsigned ws = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(cand, zero));
        ws = ~ws & 0xFFFFu;
        unsigned starts = ~ws & ((ws << 1) | (prev_ws ? 1u : 0u)) & 0xFFFFu;
        nw += (size_t)__builtin_popcount(starts);
        prev_ws = (int)((ws >> 15) & 1u);
        wi += 16;
      } else {
        u8 ch = p[wi];
        if (ch < 0x80) {
          int ws = ch == ' ' || (ch >= 0x09 && ch <= 0x0D);
          if (ws) prev_ws = 1; else if (prev_ws) { prev_ws = 0; nw++; }
          wi++;
        } else {
          size_t used; u32 cp = utf8_decode(p, n, wi, &used);
          wi += used;
          if (is_js_space(cp)) prev_ws = 1; else if (prev_ws) { prev_ws = 0; nw++; }
        }
      }
    }
  }
  for (; i < n; i++) if (p[i] == '\n') cnt++;
  while (wi < n) {
    u8 ch = p[wi];
    if (ch < 0x80) {
      int ws = ch == ' ' || (ch >= 0x09 && ch <= 0x0D);
      if (ws) prev_ws = 1; else if (prev_ws) { prev_ws = 0; nw++; }
      wi++;
    } else {
      size_t used; u32 cp = utf8_decode(p, n, wi, &used);
      wi += used;
      if (is_js_space(cp)) prev_ws = 1; else if (prev_ws) { prev_ws = 0; nw++; }
    }
  }
  *out_words = nw;
  return cnt;
}

static size_t count_nl_words_sse2(const u8 *p, size_t n, size_t *out_words) {
  __m128i nl = _mm_set1_epi8('\n');
  __m128i ws_lo = _mm_loadu_si128((const __m128i *)WS2_LO);
  __m128i ws_bit = _mm_loadu_si128((const __m128i *)WS2_BIT);
  __m128i lowmask = _mm_set1_epi8(0x0F);
  __m128i zero = _mm_setzero_si128();
  size_t cnt = 0, i = 0;
  size_t nw = 0; int prev_ws = 1; size_t wi = 0;
  for (; i + 16 <= n; i += 16) {
    __m128i c = _mm_loadu_si128((const __m128i *)(p + i));
    cnt += (size_t)__builtin_popcount((unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(c, nl)));
    size_t wend = i + 16;
    while (wi < wend) {
      if (wi + 16 <= n) {
        __m128i chunk = _mm_loadu_si128((const __m128i *)(p + wi));
        if (_mm_movemask_epi8(chunk)) {
          int inw = !prev_ws;
          size_t end = wi + 16;
          while (wi < end) {
            u8 ch = p[wi];
            if (ch < 0x80) {
              int ws = ch == ' ' || (ch >= 0x09 && ch <= 0x0D);
              if (ws) inw = 0; else if (!inw) { inw = 1; nw++; }
              wi++;
            } else {
              size_t used; u32 cp = utf8_decode(p, n, wi, &used);
              wi += used;
              if (is_js_space(cp)) inw = 0; else if (!inw) { inw = 1; nw++; }
            }
          }
          prev_ws = !inw;
          continue;
        }
        __m128i lo = _mm_and_si128(chunk, lowmask);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(chunk, 4), lowmask);
        __m128i cand = _mm_and_si128(_mm_shuffle_epi8(ws_lo, lo), _mm_shuffle_epi8(ws_bit, hi));
        unsigned ws = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(cand, zero));
        ws = ~ws & 0xFFFFu;
        unsigned starts = ~ws & ((ws << 1) | (prev_ws ? 1u : 0u)) & 0xFFFFu;
        nw += (size_t)__builtin_popcount(starts);
        prev_ws = (int)((ws >> 15) & 1u);
        wi += 16;
      } else {
        u8 ch = p[wi];
        if (ch < 0x80) {
          int ws = ch == ' ' || (ch >= 0x09 && ch <= 0x0D);
          if (ws) prev_ws = 1; else if (prev_ws) { prev_ws = 0; nw++; }
          wi++;
        } else {
          size_t used; u32 cp = utf8_decode(p, n, wi, &used);
          wi += used;
          if (is_js_space(cp)) prev_ws = 1; else if (prev_ws) { prev_ws = 0; nw++; }
        }
      }
    }
  }
  for (; i < n; i++) if (p[i] == '\n') cnt++;
  while (wi < n) {
    u8 ch = p[wi];
    if (ch < 0x80) {
      int ws = ch == ' ' || (ch >= 0x09 && ch <= 0x0D);
      if (ws) prev_ws = 1; else if (prev_ws) { prev_ws = 0; nw++; }
      wi++;
    } else {
      size_t used; u32 cp = utf8_decode(p, n, wi, &used);
      wi += used;
      if (is_js_space(cp)) prev_ws = 1; else if (prev_ws) { prev_ws = 0; nw++; }
    }
  }
  *out_words = nw;
  return cnt;
}

static Slice *md_split_lines(Arena *arena, Slice src, size_t *out_n, size_t *out_words) {
  // Byte-identical to the old two-scalar-pass version: same slices, same order.
  const u8 *p = src.p; size_t n = src.n;
  size_t words = 0;
  size_t cnt = have_avx2 ? count_nl_words_avx2(p, n, &words) : count_nl_words_sse2(p, n, &words);
#ifdef SENJA_COUNTERS
  if (have_avx2) cnt_avx2_nl++; else cnt_sse2_nl++;
#endif
  if (out_words) *out_words = words;
  Slice *lines = (Slice *)arena_alloc(arena, (cnt + 1) * sizeof(Slice));
  size_t li = 0, start = 0;
  if (have_avx2) fill_lines_avx2(p, n, lines, &li, &start);
  else fill_lines_sse2(p, n, lines, &li, &start);
  lines[li++] = sl(p + start, n - start);
  *out_n = li;
  return lines;
}

static inline int md_is_blank(Slice line) { return is_blank_sp_tab(line); }

static Slice md_trim_line(Slice line) {
  size_t a = 0, b = line.n;
  while (a < b && (line.p[a] == ' ' || line.p[a] == '\t')) a++;
  while (b > a && (line.p[b - 1] == ' ' || line.p[b - 1] == '\t')) b--;
  return sl(line.p + a, b - a);
}

static inline int md_is_lang_char(u8 c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '+' || c == '-';
}

// Fence-open language, or (int)-1 sentinel via *is_fence + lang out.
static int md_fence_lang(Slice line, Slice *lang) {
  if (line.n < 3 || line.p[0] != '`' || line.p[1] != '`' || line.p[2] != '`') return 0;
  size_t i = 3;
  while (i < line.n && md_is_lang_char(line.p[i])) i++;
  *lang = sl(line.p + 3, i - 3);
  while (i < line.n) {
    if (line.p[i] != ' ' && line.p[i] != '\t') return 0;
    i++;
  }
  return 1;
}

static int md_is_fence_close(Slice line) {
  if (line.n < 3 || line.p[0] != '`' || line.p[1] != '`' || line.p[2] != '`') return 0;
  for (size_t i = 3; i < line.n; i++) if (line.p[i] != ' ' && line.p[i] != '\t') return 0;
  return 1;
}

// (depth, text) or 0.
static int md_heading_of(Slice line, u32 *depth, Slice *text) {
  size_t n = 0;
  while (n < 6 && n < line.n && line.p[n] == '#') n++;
  if (n == 0 || n >= line.n || (line.p[n] != ' ' && line.p[n] != '\t')) return 0;
  size_t a = n;
  while (a < line.n && (line.p[a] == ' ' || line.p[a] == '\t')) a++;
  size_t b = line.n;
  while (b > a && (line.p[b - 1] == ' ' || line.p[b - 1] == '\t')) b--;
  if (b <= a) return 0;
  *depth = (u32)n; *text = sl(line.p + a, b - a);
  return 1;
}

static int md_is_hr(Slice line) {
  if (!(sl_starts(line, "---") || sl_starts(line, "***"))) return 0;
  for (size_t i = 3; i < line.n; i++) if (line.p[i] != ' ' && line.p[i] != '\t') return 0;
  return 1;
}

// (defined below md_inline's helpers; forward-declared for the classifier)
static int md_ordered_of(Slice line, u32 *num, Slice *rest);
static int md_is_continued(Slice line);

// v12: line classes. One classification pass memoises the block dispatch
// (was: blank/fence/heading/hr re-checked per line in md_ends_paragraph
// AND in the main loop, ordered re-checked per list line) and records
// trim-free flags so blocks whose lines need no trimming can be inlined
// from the raw source subslice instead of a re-joined copy.
enum { LC_BLANK=0, LC_FENCE=1, LC_FENCE_OPEN=2, LC_HEADING=3, LC_HR=4,
       LC_QUOTE=5, LC_BULLET=6, LC_ORDERED=7, LC_CONT=8, LC_TEXT=9 };
#define LC_TF_LINE 0x80 // whole line needs no trim (edges are not sp/tab)
#define LC_TF_REST 0x40 // bullet/ordered rest needs no trim
#define LC_KIND(c) ((c) & 0x0F)

static u8 *md_classify_lines(Arena *arena, Slice *lines, size_t nlines) {
  u8 *cls = (u8 *)arena_alloc(arena, nlines ? nlines : 1);
  for (size_t i = 0; i < nlines; i++) {
    Slice l = lines[i];
    u8 k; Slice rr = slc("");
    if (md_is_blank(l)) k = LC_BLANK;
    else {
      Slice lang;
      if (md_fence_lang(l, &lang)) k = md_is_fence_close(l) ? LC_FENCE : LC_FENCE_OPEN;
      else {
        u32 d; Slice ht; u32 nn;
        if (md_heading_of(l, &d, &ht)) k = LC_HEADING;
        else if (md_is_hr(l)) k = LC_HR;
        else if (l.n > 0 && l.p[0] == '>') k = LC_QUOTE;
        else if (l.n > 1 && (l.p[0] == '-' || l.p[0] == '*') && l.p[1] == ' ') k = LC_BULLET;
        else if (md_ordered_of(l, &nn, &rr)) k = LC_ORDERED;
        else if (md_is_continued(l)) k = LC_CONT;
        else k = LC_TEXT;
      }
    }
    if (l.n == 0 || (l.p[0] != ' ' && l.p[0] != '\t' && l.p[l.n - 1] != ' ' && l.p[l.n - 1] != '\t'))
      k |= LC_TF_LINE;
    if (LC_KIND(k) == LC_BULLET) {
      Slice r = sl(l.p + 2, l.n - 2);
      if (r.n == 0 || (r.p[0] != ' ' && r.p[0] != '\t' && r.p[r.n - 1] != ' ' && r.p[r.n - 1] != '\t'))
        k |= LC_TF_REST;
    } else if (LC_KIND(k) == LC_ORDERED) {
      if (rr.n == 0 || (rr.p[0] != ' ' && rr.p[0] != '\t' && rr.p[rr.n - 1] != ' ' && rr.p[rr.n - 1] != '\t'))
        k |= LC_TF_REST;
    }
    cls[i] = k;
  }
  return cls;
}

// (number, rest) for an ordered-list line, or 0.
static int md_ordered_of(Slice line, u32 *num, Slice *rest) {
  size_t i = 0;
  while (i < 9 && i < line.n && line.p[i] >= '0' && line.p[i] <= '9') i++;
  if (i == 0 || i + 1 >= line.n || line.p[i] != '.' || line.p[i + 1] != ' ') return 0;
  u32 v = 0;
  for (size_t k = 0; k < i; k++) v = v * 10 + (u32)(line.p[k] - '0');
  *num = v; *rest = sl(line.p + i + 2, line.n - (i + 2));
  return 1;
}

// CONTINUED = /^ {2,}\S/: two or more spaces, then a non-space.
static int md_is_continued(Slice line) {
  size_t i = 0;
  while (i < line.n && line.p[i] == ' ') i++;
  if (i < 2 || i >= line.n) return 0;
  size_t used; u32 cp = utf8_decode(line.p, line.n, i, &used);
  (void)used;
  return !is_js_space(cp);
}

static const u8 MD_ESCAPABLE[] = "\\`*_[]()#+-.!>";
// v8: byte table for the same set (replaces a memchr call per backslash).
static const u8 MD_ESC_TBL[256] = {
  ['\\'] = 1, ['`'] = 1, ['*'] = 1, ['_'] = 1, ['['] = 1, [']'] = 1,
  ['('] = 1, [')'] = 1, ['#'] = 1, ['+'] = 1, ['-'] = 1, ['.'] = 1,
  ['!'] = 1, ['>'] = 1,
};

// Whether position j is escaped: the run of '\' immediately before it
// has odd length.
static int md_is_escaped(const u8 *s, size_t from, size_t j) {
  size_t b = j, n = 0;
  while (b > from && s[b - 1] == '\\') { n++; b--; }
  return (n % 2) == 1;
}

static int md_find_unescaped(const u8 *s, size_t n, const u8 *what, size_t wl, size_t from, size_t *out) {
  if (wl == 0 || n < wl) return 0;
  size_t j = find_byte(s, n, from, what[0]);
  if (j >= n) return 0;
  for (;;) {
    if (j + wl <= n && memcmp(s + j, what, wl) == 0 && !md_is_escaped(s, from, j)) { *out = j; return 1; }
    size_t k = find_byte(s, n, j + 1, what[0]);
    if (k >= n) return 0;
    j = k;
  }
}

static int md_find_single_star(const u8 *s, size_t n, size_t from, size_t *out) {
  size_t j = find_byte(s, n, from, '*');
  if (j >= n) return 0;
  for (;;) {
    if (!md_is_escaped(s, from, j)) {
      if (j + 1 < n && s[j + 1] == '*') {
        size_t k = find_byte(s, n, j + 2, '*');
        if (k >= n) return 0;
        j = k; continue;
      }
      *out = j; return 1;
    }
    size_t k = find_byte(s, n, j + 1, '*');
    if (k >= n) return 0;
    j = k;
  }
}

static int md_has_ws(Slice s) {
  size_t i = 0;
  while (i < s.n) {
    size_t used; u32 cp = utf8_decode(s.p, s.n, i, &used);
    if (is_js_space(cp)) return 1;
    i += used;
  }
  return 0;
}

// Forward: link resolution is supplied by the build (LinkCtx).
typedef Slice (*LinkFn)(void *ctx, Slice target);

static void md_inline_old(Slice s, LinkFn link, void *lctx, Buf *out);

static void md_inline_rec(Slice s, LinkFn link, void *lctx, Buf *out) { md_inline_old(s, link, lctx, out); }

// SSSE3 "shufti" tables for md_inline's 9 specials: \ ` * [ & < > " '
// MD_LO_TBL[lo] = bitmask of hi nibbles h with ((h<<4)|lo) a special;
// MD_BIT_TBL[hi] = 1<<hi. A byte is special iff
// (MD_LO_TBL[lo] & MD_BIT_TBL[hi]) != 0. Two shuffles + one AND replace
// nine cmpeq + nine movemask per 16 bytes.
static const u8 MD_LO_TBL[16] = { 0x40,0x00,0x04,0x00,0x00,0x00,0x04,0x04,0x00,0x00,0x04,0x20,0x28,0x00,0x08,0x00 };
static const u8 MD_BIT_TBL[16] = { 0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00 };

static void md_inline_old(Slice s, LinkFn link, void *lctx, Buf *out) {
#ifdef SENJA_PROFILE
  PROF_T0; prof_nmdin++;
#endif
  __m128i lo_tbl = _mm_loadu_si128((const __m128i *)MD_LO_TBL);
  __m128i bit_tbl = _mm_loadu_si128((const __m128i *)MD_BIT_TBL);
  __m128i lowmask = _mm_set1_epi8(0x0F);
  __m128i zero = _mm_setzero_si128();
  size_t n = s.n, i = 0;
  size_t plain = 0; // start of the not-yet-copied plain run
  unsigned mask = 0; size_t base = 0; // pending chunk mask + its base
  // v8: raw-pointer output. The Buf cursor is hoisted into (wp, wend);
  // plain runs and tags are written directly (no buf_put/buf_c/buf_cs
  // calls, each of which used to cost a buf_reserve call + a memcpy).
  // Before any call that can grow the buffer (recursion, esc), the
  // length is synced back and the cursor reloaded afterwards.
  buf_reserve(out, s.n + s.n / 4 + 128);
  u8 *wp = out->p + out->len;
  u8 *wend = out->p + out->cap;
#define MD_ROOM(need) do { if (wp + (size_t)(need) > wend) { \
    out->len = (size_t)(wp - out->p); buf_reserve(out, (size_t)(need) + 256); \
    wp = out->p + out->len; wend = out->p + out->cap; } } while (0)
#define MD_LIT(str) do { MD_ROOM(sizeof(str) - 1); wp = copy_span(wp, (const u8 *)(str), sizeof(str) - 1); } while (0)
#define MD_SYNC() do { out->len = (size_t)(wp - out->p); } while (0)
#define MD_RELOAD() do { wp = out->p + out->len; wend = out->p + out->cap; } while (0)
  while (i < n) {
    size_t j;
    if (mask) {
      j = base + (size_t)__builtin_ctz(mask);
      mask &= mask - 1;
    } else if (i + 16 <= n) {
      __m128i chunk = _mm_loadu_si128((const __m128i *)(s.p + i));
      __m128i lo = _mm_and_si128(chunk, lowmask);
      __m128i hi = _mm_and_si128(_mm_srli_epi16(chunk, 4), lowmask);
      __m128i cand = _mm_and_si128(_mm_shuffle_epi8(lo_tbl, lo), _mm_shuffle_epi8(bit_tbl, hi));
      mask = (unsigned)(~_mm_movemask_epi8(_mm_cmpeq_epi8(cand, zero))) & 0xFFFFu;
      base = i;
      if (!mask) { i += 16; continue; }
      j = base + (size_t)__builtin_ctz(mask);
      mask &= mask - 1;
    } else {
      // Scalar tail: find the next special byte-wise.
      size_t k = i;
      while (k < n) {
        u8 c = s.p[k];
        if (c == '\\' || c == '`' || c == '*' || c == '[' || c == '&' || c == '<' || c == '>' || c == '"' || c == '\'') break;
        k++;
      }
      if (k >= n) break;
      j = k;
    }
    if (j > plain) { MD_ROOM(j - plain); wp = copy_span(wp, s.p + plain, j - plain); }
    i = j; plain = j;
    u8 c = s.p[i];
    switch (c) {
      case '\\': {
        if (i + 1 < n && MD_ESC_TBL[s.p[i + 1]]) {
          u8 ch = s.p[i + 1];
          if (ch == '>') MD_LIT("&gt;");
          else { MD_ROOM(1); *wp++ = ch; }
          i += 2;
        } else { MD_ROOM(1); *wp++ = '\\'; i += 1; }
        break;
      }
      case '`': {
        size_t k = find_byte(s.p, n, i + 1, '`');
        if (k < n && k > i + 1) {
          MD_LIT("<code>");
          MD_SYNC(); esc(out, sl(s.p + i + 1, k - (i + 1))); MD_RELOAD();
          MD_LIT("</code>");
          i = k + 1;
        } else { MD_ROOM(1); *wp++ = '`'; i += 1; }
        break;
      }
      case '*': {
        if (i + 1 < n && s.p[i + 1] == '*') {
          size_t k;
          if (md_find_unescaped(s.p, n, (const u8 *)"**", 2, i + 2, &k) && k > i + 2) {
            MD_LIT("<strong>");
            MD_SYNC(); md_inline_rec(sl(s.p + i + 2, k - (i + 2)), link, lctx, out); MD_RELOAD();
            MD_LIT("</strong>");
            i = k + 2;
          } else { MD_LIT("**"); i += 2; }
        } else {
          size_t k;
          if (md_find_single_star(s.p, n, i + 1, &k) && k > i + 1) {
            MD_LIT("<em>");
            MD_SYNC(); md_inline_rec(sl(s.p + i + 1, k - (i + 1)), link, lctx, out); MD_RELOAD();
            MD_LIT("</em>");
            i = k + 1;
          } else { MD_ROOM(1); *wp++ = '*'; i += 1; }
        }
        break;
      }
      case '[': {
        size_t k;
        int has_k = md_find_unescaped(s.p, n, (const u8 *)"]", 1, i + 1, &k);
        size_t m = 0; int has_q = 0;
        if (has_k && k + 1 < n && s.p[k + 1] == '(') {
          size_t q = find_byte(s.p, n, k + 2, ')');
          if (q < n) { m = q; has_q = 1; }
        }
        Slice target = slc("");
        if (has_q) target = sl(s.p + k + 2, m - (k + 2));
        if (has_q && target.n > 0 && !md_has_ws(target)) {
          MD_LIT("<a href=\"");
          Slice href = link(lctx, target);
          MD_SYNC(); esc(out, href); MD_RELOAD();
          MD_LIT("\">");
          MD_SYNC(); md_inline_rec(sl(s.p + i + 1, k - (i + 1)), link, lctx, out); MD_RELOAD();
          MD_LIT("</a>");
          i = m + 1;
        } else { MD_ROOM(1); *wp++ = '['; i += 1; }
        break;
      }
      case '&': MD_LIT("&amp;"); i++; break;
      case '<': MD_LIT("&lt;"); i++; break;
      case '>': MD_LIT("&gt;"); i++; break;
      case '"': MD_LIT("&quot;"); i++; break;
      case '\'': MD_LIT("&#39;"); i++; break;
      default: MD_ROOM(1); *wp++ = c; i++; break;
    }
    // Keep the pending chunk mask valid: drop bits at or before the new
    // position (the input bytes never change, so later bits stay valid);
    // if the handler consumed past the chunk, discard the mask.
    if (i >= base + 16) mask = 0;
    else if (i > base) mask &= ~((1u << (i - base)) - 1u);
    plain = i;
  }
  if (plain < n) { MD_ROOM(n - plain); wp = copy_span(wp, s.p + plain, n - plain); }
  MD_SYNC();
#undef MD_ROOM
#undef MD_LIT
#undef MD_SYNC
#undef MD_RELOAD
#ifdef SENJA_PROFILE
  PROF_ADD(prof_mdin);
#endif
}

// ---- v12: iterative single-pass md_inline (stack machine) ----
// Replaces the recursive md_inline_old (kept above for differential
// testing). One left-to-right scan; explicit span stack; no forward
// closer search, no recursive re-scan of interiors. Semantics match
// md_inline_old:
//  * strong/em close the NEAREST matching opener (the recursive handler
//    claims the first closer ahead); an empty interior flushes the opener
//    as a literal and re-opens.
//  * links pair OLDEST-first (the recursive [ handler claims the first ]);
//    a failed [ is flushed as a literal and the next [ is tried.
//  * inside a code span, \ and [ are plain, and ** / * / ] only close an
//    OUTER span (the recursive closer searches are raw byte searches, so
//    a ** or ] inside backticks still closes an outer span); a flushed
//    code span rewinds the input so its interior is re-scanned as
//    markdown (exactly what the recursive version does when the outer
//    span's interior is re-parsed).
//  * a code span's interior is a raw input slice, esc()'d at close.
// A CODE span is always on top of the stack when open (nothing is ever
// pushed above one), so "in code" == (top is CODE).
enum { SP_STRONG = 0, SP_EM = 1, SP_CODE = 2, SP_LINK = 3 };
typedef struct { u8 kind; size_t ipos; size_t opos; } Span;

static const u8 MD2_LO_TBL[16] = { 0x40,0x00,0x04,0x00,0x00,0x00,0x04,0x04,0x00,0x00,0x04,0x20,0x28,0x20,0x08,0x00 };
// (MD_LO_TBL + ']' as a 10th special, so ] can close links.)

// A link at stack index li (] at i, ) at m) is shadowed if an enclosing
// strong/em (below li) closes at or before m: the recursive version lets
// the enclosing span claim the text first, so the link's ) lies outside
// its interior and the link never forms. The closer must be past i (any
// closer at or before i would already have closed the span while scanning).
static int mi_link_shadowed(Slice s, size_t n, Span *st, int li, size_t i, size_t m) {
  int has_strong = 0, has_em = 0;
  for (int t = 0; t < li; t++) {
    if (st[t].kind == SP_STRONG) has_strong = 1;
    else if (st[t].kind == SP_EM) has_em = 1;
  }
  if (has_strong) {
    size_t k;
    if (md_find_unescaped(s.p, n, (const u8 *)"**", 2, i + 1, &k) && k <= m) return 1;
  }
  if (has_em) {
    size_t k;
    if (md_find_single_star(s.p, n, i + 1, &k) && k <= m) return 1;
  }
  return 0;
}

static void md_inline(Slice s, LinkFn link, void *lctx, Buf *out) {
#ifdef SENJA_PROFILE
  PROF_T0; prof_nmdin++;
#endif
  __m128i lo_tbl = _mm_loadu_si128((const __m128i *)MD2_LO_TBL);
  __m128i bit_tbl = _mm_loadu_si128((const __m128i *)MD_BIT_TBL);
  __m128i lowmask = _mm_set1_epi8(0x0F);
  __m128i zero = _mm_setzero_si128();
  size_t n = s.n, i = 0;
  size_t plain = 0;
  unsigned mask = 0; size_t base = 0;
  buf_reserve(out, s.n + s.n / 4 + 128);
  u8 *wp = out->p + out->len;
  u8 *wend = out->p + out->cap;
  Span st[64]; int sp = 0;
  Buf pfx = {0}; // scratch for link prefixes
#define MI_ROOM(need) do { if (wp + (size_t)(need) > wend) { out->len = (size_t)(wp - out->p); buf_reserve(out, (size_t)(need) + 256); wp = out->p + out->len; wend = out->p + out->cap; } } while (0)
#define MI_LIT(str) do { MI_ROOM(sizeof(str) - 1); wp = copy_span(wp, (const u8 *)(str), sizeof(str) - 1); } while (0)
#define MI_SYNC() do { out->len = (size_t)(wp - out->p); } while (0)
#define MI_RELOAD() do { wp = out->p + out->len; wend = out->p + out->cap; } while (0)
#define MI_INSERT(pos, bytes, nbytes) do { MI_SYNC(); buf_reserve(out, (size_t)(nbytes) + 64); memmove(out->p + (pos) + (nbytes), out->p + (pos), out->len - (pos)); memcpy(out->p + (pos), (bytes), (nbytes)); out->len += (nbytes); MI_RELOAD(); } while (0)
#define MI_OUTLEN() ((size_t)(wp - out->p))
  // After inserting len bytes at P (flushing a failed [), spans ABOVE the
  // flushed one shift right by len; spans below keep their position (the
  // flushed [ is part of their interior, so their tag goes before it).
#define MI_SHIFT_ABOVE(t, P, L) do { for (int _u = (t) + 1; _u < sp; _u++) st[_u].opos += (L); (void)(P); } while (0)
  while (i < n || sp > 0) {
    if (i >= n) {
      // Copy any un-copied plain tail first (normal mode only; in code
      // mode the tail is the code interior, buffered as input positions).
      if (!(sp > 0 && st[sp-1].kind == SP_CODE) && plain < n) { MI_ROOM(n - plain); wp = copy_span(wp, s.p + plain, n - plain); plain = n; }
      // End of input: flush the top span as its literal opener.
      Span t = st[--sp];
      if (t.kind == SP_CODE) {
        MI_ROOM(1); *wp++ = '`';
        i = t.ipos; plain = i; mask = 0;
        continue;
      }
      if (t.kind == SP_STRONG) MI_INSERT(t.opos, (const u8 *)"**", 2);
      else if (t.kind == SP_EM) MI_INSERT(t.opos, (const u8 *)"*", 1);
      else MI_INSERT(t.opos, (const u8 *)"[", 1);
      continue;
    }
    size_t j;
    if (mask) {
      j = base + (size_t)__builtin_ctz(mask);
      mask &= mask - 1;
    } else if (i + 16 <= n) {
      __m128i chunk = _mm_loadu_si128((const __m128i *)(s.p + i));
      __m128i lo = _mm_and_si128(chunk, lowmask);
      __m128i hi = _mm_and_si128(_mm_srli_epi16(chunk, 4), lowmask);
      __m128i cand = _mm_and_si128(_mm_shuffle_epi8(lo_tbl, lo), _mm_shuffle_epi8(bit_tbl, hi));
      mask = (unsigned)(~_mm_movemask_epi8(_mm_cmpeq_epi8(cand, zero))) & 0xFFFFu;
      base = i;
      if (!mask) { i += 16; continue; }
      j = base + (size_t)__builtin_ctz(mask);
      mask &= mask - 1;
    } else {
      size_t k = i;
      while (k < n) {
        u8 c = s.p[k];
        if (c == '\\' || c == '`' || c == '*' || c == '[' || c == ']' || c == '&' || c == '<' || c == '>' || c == '"' || c == '\'') break;
        k++;
      }
      if (k >= n) {
        if (!(sp > 0 && st[sp-1].kind == SP_CODE) && plain < n) { MI_ROOM(n - plain); wp = copy_span(wp, s.p + plain, n - plain); }
        i = n; plain = n;
        continue;
      }
      j = k;
    }
    int in_code = (sp > 0 && st[sp-1].kind == SP_CODE);
    if (!in_code && j > plain) { MI_ROOM(j - plain); wp = copy_span(wp, s.p + plain, j - plain); }
    i = j; plain = j;
    u8 c = s.p[i];
    if (in_code) {
      // Inside a code span: only `, **, *, ] are meaningful.
      if (c == '`') {
        Span *cd = &st[sp-1];
        if (i > cd->ipos) {
          size_t a = cd->ipos, b = i;
          sp--;
          MI_LIT("<code>");
          MI_SYNC(); esc(out, sl(s.p + a, b - a)); MI_RELOAD();
          MI_LIT("</code>");
          i++; plain = i;
        } else {
          // Empty interior: flush the opener as a literal, re-open.
          sp--;
          MI_ROOM(1); *wp++ = '`';
          st[sp].kind = SP_CODE; st[sp].ipos = i + 1; st[sp].opos = 0; sp++;
          i++; plain = i;
        }
      } else if (c == '*') {
        // An escaped * (odd run of \ before it) is plain interior: the
        // recursive closer searches (md_find_unescaped /
        // md_find_single_star) skip escaped occurrences.
        if (md_is_escaped(s.p, 0, i)) { i++; plain = i; }
        else {
        int dbl = (i + 1 < n && s.p[i+1] == '*');
        u8 want = dbl ? SP_STRONG : SP_EM;
        int found = -1;
        for (int t = sp - 2; t >= 0; t--) if (st[t].kind == want) { found = t; break; }
        if (found >= 0) {
          Span cd = st[sp-1]; sp--;
          MI_ROOM(1); *wp++ = '`';
          i = cd.ipos; plain = i; mask = 0;
        } else {
          i += dbl ? 2 : 1; plain = i;
        }
        }
      } else if (c == ']') {
        // An escaped ] is plain interior (md_find_unescaped skips it);
        // testing it as a closer would wrongly flush the LINK below.
        if (md_is_escaped(s.p, 0, i)) { i++; plain = i; }
        else {
        // Try the oldest LINK below the code; loop while they fail.
        for (;;) {
          int li = -1;
          for (int t = 0; t < sp - 1; t++) if (st[t].kind == SP_LINK) { li = t; break; }
          if (li < 0) { i++; plain = i; break; }
          if (i + 1 >= n || s.p[i+1] != '(') {
            size_t P = st[li].opos;
            MI_INSERT(P, (const u8 *)"[", 1);
            MI_SHIFT_ABOVE(li, P, 1);
            memmove(&st[li], &st[li+1], (size_t)(sp - 1 - li) * sizeof(Span));
            sp--;
            continue;
          }
          size_t m = find_byte(s.p, n, i + 2, ')');
          if (m >= n) {
            size_t P = st[li].opos;
            MI_INSERT(P, (const u8 *)"[", 1);
            MI_SHIFT_ABOVE(li, P, 1);
            memmove(&st[li], &st[li+1], (size_t)(sp - 1 - li) * sizeof(Span));
            sp--;
            continue;
          }
          Slice target = sl(s.p + i + 2, m - (i + 2));
          if (target.n == 0 || md_has_ws(target)) {
            size_t P = st[li].opos;
            MI_INSERT(P, (const u8 *)"[", 1);
            MI_SHIFT_ABOVE(li, P, 1);
            memmove(&st[li], &st[li+1], (size_t)(sp - 1 - li) * sizeof(Span));
            sp--;
            continue;
          }
          if (mi_link_shadowed(s, n, st, li, i, m)) {
            size_t P = st[li].opos;
            MI_INSERT(P, (const u8 *)"[", 1);
            MI_SHIFT_ABOVE(li, P, 1);
            memmove(&st[li], &st[li+1], (size_t)(sp - 1 - li) * sizeof(Span));
            sp--;
            continue;
          }
          // Valid link below the code: flush the code, re-encounter ].
          Span cd = st[sp-1]; sp--;
          MI_ROOM(1); *wp++ = '`';
          i = cd.ipos; plain = i; mask = 0;
          break;
        }
        }
      } else {
        // \ [ & < > " ' inside code: plain interior bytes.
        i++; plain = i;
      }
      if (i >= base + 16) mask = 0;
      else if (i > base) mask &= ~((1u << (i - base)) - 1u);
      plain = i;
      continue;
    }
    // ---- normal mode dispatch ----
    switch (c) {
      case '\\': {
        if (i + 1 < n && MD_ESC_TBL[s.p[i + 1]]) {
          u8 ch = s.p[i + 1];
          if (ch == '>') MI_LIT("&gt;");
          else { MI_ROOM(1); *wp++ = ch; }
          i += 2;
        } else { MI_ROOM(1); *wp++ = '\\'; i += 1; }
        break;
      }
      case '`': {
        if (sp >= 64) { MI_ROOM(1); *wp++ = '`'; i++; break; }
        st[sp].kind = SP_CODE; st[sp].ipos = i + 1; st[sp].opos = 0; sp++;
        i++; break;
      }
      case '*': {
        int dbl = (i + 1 < n && s.p[i+1] == '*');
        u8 want = dbl ? SP_STRONG : SP_EM;
        int found = -1;
        for (int t = sp - 1; t >= 0; t--) if (st[t].kind == want) { found = t; break; }
        if (found >= 0 && i > st[found].ipos) {
          // Close: flush openers above as literals, insert open tag.
          for (int t = sp - 1; t > found; t--) {
            if (st[t].kind == SP_STRONG) MI_INSERT(st[t].opos, (const u8 *)"**", 2);
            else if (st[t].kind == SP_EM) MI_INSERT(st[t].opos, (const u8 *)"*", 1);
            else MI_INSERT(st[t].opos, (const u8 *)"[", 1);
          }
          size_t op = st[found].opos;
          sp = found;
          if (dbl) { MI_INSERT(op, (const u8 *)"<strong>", 8); MI_LIT("</strong>"); }
          else { MI_INSERT(op, (const u8 *)"<em>", 4); MI_LIT("</em>"); }
          i += dbl ? 2 : 1;
        } else if (found >= 0) {
          // Empty interior (found is top): flush as literal, re-open.
          size_t op = st[found].opos;
          if (dbl) MI_INSERT(op, (const u8 *)"**", 2);
          else MI_INSERT(op, (const u8 *)"*", 1);
          sp = found;
          st[sp].kind = want; st[sp].ipos = i + (dbl ? 2 : 1); st[sp].opos = MI_OUTLEN(); sp++;
          i += dbl ? 2 : 1;
        } else {
          if (sp >= 64) { if (dbl) { MI_LIT("**"); i += 2; } else { MI_ROOM(1); *wp++ = '*'; i += 1; } break; }
          st[sp].kind = want; st[sp].ipos = i + (dbl ? 2 : 1); st[sp].opos = MI_OUTLEN(); sp++;
          i += dbl ? 2 : 1;
        }
        break;
      }
      case '[': {
        if (sp >= 64) { MI_ROOM(1); *wp++ = '['; i++; break; }
        st[sp].kind = SP_LINK; st[sp].ipos = i + 1; st[sp].opos = MI_OUTLEN(); sp++;
        i++; break;
      }
      case ']': {
        // Oldest-first link close.
        int closed = 0;
        for (int t = 0; t < sp; t++) {
          if (st[t].kind != SP_LINK) continue;
          int bad = 0;
          size_t m = n;
          Slice target = slc("");
          if (i + 1 >= n || s.p[i+1] != '(') bad = 1;
          else {
            m = find_byte(s.p, n, i + 2, ')');
            if (m >= n) bad = 1;
            else {
              target = sl(s.p + i + 2, m - (i + 2));
              if (target.n == 0 || md_has_ws(target)) bad = 1;
            }
          }
          if (!bad && mi_link_shadowed(s, n, st, t, i, m)) bad = 1;
          if (bad) {
            size_t P = st[t].opos;
            MI_INSERT(P, (const u8 *)"[", 1);
            MI_SHIFT_ABOVE(t, P, 1);
            memmove(&st[t], &st[t+1], (size_t)(sp - 1 - t) * sizeof(Span));
            sp--; t--;
            continue;
          }
          // Valid: flush openers above as literals, then close.
          for (int u = sp - 1; u > t; u--) {
            if (st[u].kind == SP_STRONG) MI_INSERT(st[u].opos, (const u8 *)"**", 2);
            else if (st[u].kind == SP_EM) MI_INSERT(st[u].opos, (const u8 *)"*", 1);
            else MI_INSERT(st[u].opos, (const u8 *)"[", 1);
          }
          size_t op = st[t].opos;
          sp = t;
          Slice href = link(lctx, target);
          pfx.len = 0;
          buf_cs(&pfx, "<a href=\"");
          esc(&pfx, href);
          buf_cs(&pfx, "\">");
          MI_INSERT(op, pfx.p, pfx.len);
          MI_LIT("</a>");
          i = m + 1;
          closed = 1;
          break;
        }
        if (!closed) { MI_ROOM(1); *wp++ = ']'; i += 1; }
        break;
      }
      case '&': MI_LIT("&amp;"); i++; break;
      case '<': MI_LIT("&lt;"); i++; break;
      case '>': MI_LIT("&gt;"); i++; break;
      case '"': MI_LIT("&quot;"); i++; break;
      case '\'': MI_LIT("&#39;"); i++; break;
      default: MI_ROOM(1); *wp++ = c; i++; break;
    }
    if (i >= base + 16) mask = 0;
    else if (i > base) mask &= ~((1u << (i - base)) - 1u);
    plain = i;
  }
  // Tail: the loop can exit via a no-special 16B chunk jump with the plain
  // run un-copied and no span open.
  if (plain < n) { MI_ROOM(n - plain); wp = copy_span(wp, s.p + plain, n - plain); plain = n; }
  MI_SYNC();
  free(pfx.p);
#undef MI_ROOM
#undef MI_LIT
#undef MI_SYNC
#undef MI_RELOAD
#undef MI_INSERT
#undef MI_OUTLEN
#undef MI_SHIFT_ABOVE
#ifdef SENJA_PROFILE
  PROF_ADD(prof_mdin);
#endif
}


// { html, toc, excerpt } for a Markdown body. slug_cached calls the
// build's slug cache. `arena` owns the toc/id storage.
typedef Slice (*SlugFn)(void *ctx, Slice text);
static MdOut md_render(Slice src, LinkFn link, void *lctx, SlugFn slugfn, void *sctx, Arena *arena, StrMap *ids) {
  size_t nlines;
  size_t md_words = 0;
  Slice *lines = md_split_lines(arena, src, &nlines, &md_words);
  u8 *cls = md_classify_lines(arena, lines, nlines);
  Buf html = {0};
  buf_reserve(&html, src.n + src.n / 2 + 64);
  Buf joined = {0}; // scratch: line-join buffer, reused per block (no malloc per block)
  buf_reserve(&joined, 512);
  MdOut mo; memset(&mo, 0, sizeof(mo));
  int have_excerpt = 0;
  size_t i = 0;
  while (i < nlines) {
    u8 c = cls[i];
    u8 kind = LC_KIND(c);
    Slice line = lines[i];
    if (kind == LC_BLANK) { i++; continue; }
    if (kind == LC_FENCE || kind == LC_FENCE_OPEN) {
      Slice lang; md_fence_lang(line, &lang);
      // v12: fence body is contiguous in src and never trimmed — escape
      // the raw subslice directly, no joined copy.
      i++;
      size_t bstart = i;
      while (i < nlines && LC_KIND(cls[i]) != LC_FENCE) i++;
      size_t bend = i; // first LC_FENCE line (close) or nlines
      i++; // past the close (or past the end)
      buf_cs(&html, "<pre><code");
      if (lang.n > 0) {
        buf_cs(&html, " class=\"language-");
        esc(&html, lang);
        buf_c(&html, '"');
      }
      buf_c(&html, '>');
      if (bend > bstart) {
        Slice body = sl(lines[bstart].p, lines[bend - 1].p + lines[bend - 1].n - lines[bstart].p);
        esc(&html, body);
        buf_c(&html, '\n');
      }
      buf_cs(&html, "</code></pre>\n");
      continue;
    }
    u32 depth; Slice htext;
    if (kind == LC_HEADING && md_heading_of(line, &depth, &htext)) {
      Slice base_slug = slugfn(sctx, htext);
      Slice id = unique_slug(arena, ids, base_slug);
      char hb[16]; int hm = snprintf(hb, sizeof(hb), "<h%u id=\"", depth);
      buf_put(&html, hb, (size_t)hm);
      buf_sl(&html, id);
      buf_cs(&html, "\">");
      size_t h_start = html.len;
      md_inline(htext, link, lctx, &html);
      size_t h_end = html.len;
      hm = snprintf(hb, sizeof(hb), "</h%u>", depth);
      buf_put(&html, hb, (size_t)hm);
      buf_c(&html, '\n');
      if (depth == 2 || depth == 3) {
        if (mo.ntoc == mo.captoc) {
          size_t nc = mo.captoc ? mo.captoc * 2 : 8;
          TocEntry *nt = (TocEntry *)arena_alloc(arena, nc * sizeof(TocEntry));
          if (mo.toc) memcpy(nt, mo.toc, mo.ntoc * sizeof(TocEntry));
          mo.toc = nt; mo.captoc = nc;
        }
        mo.toc[mo.ntoc].level = depth;
        mo.toc[mo.ntoc].id = id;
        mo.toc[mo.ntoc].html = arena_slice(arena, html.p + h_start, h_end - h_start);
        mo.ntoc++;
      }
      i++;
      continue;
    }
    if (kind == LC_HR) { buf_cs(&html, "<hr>\n"); i++; continue; }
    if (kind == LC_QUOTE) {
      size_t cap = 8, nb = 0;
      Slice *body = (Slice *)arena_alloc(arena, cap * sizeof(Slice));
      while (i < nlines && LC_KIND(cls[i]) == LC_QUOTE) {
        Slice l = lines[i]; i++;
        Slice stripped = (l.n >= 2 && l.p[1] == ' ') ? sl(l.p + 2, l.n - 2) : sl(l.p + 1, l.n - 1);
        if (nb == cap) {
          size_t nc = cap * 2;
          Slice *nb2 = (Slice *)arena_alloc(arena, nc * sizeof(Slice));
          memcpy(nb2, body, nb * sizeof(Slice));
          body = nb2; cap = nc;
        }
        body[nb++] = stripped;
      }
      // Trim each, drop empties.
      size_t ni = 0;
      for (size_t k = 0; k < nb; k++) {
        Slice t = md_trim_line(body[k]);
        if (t.n > 0) body[ni++] = t;
      }
      if (ni == 0) { buf_cs(&html, "<blockquote>\n</blockquote>\n"); }
      else {
        joined.len = 0;
        for (size_t k = 0; k < ni; k++) {
          if (k > 0) buf_c(&joined, '\n');
          buf_sl(&joined, body[k]);
        }
        buf_cs(&html, "<blockquote>\n<p>");
        md_inline(sl(joined.p, joined.len), link, lctx, &html);
        buf_cs(&html, "</p>\n</blockquote>\n");
      }
      continue;
    }
    if (kind == LC_BULLET || kind == LC_ORDERED) {
      int is_ordered = (kind == LC_ORDERED);
      u32 start = 1;
      if (is_ordered) { u32 nn; Slice rr; md_ordered_of(line, &nn, &rr); start = nn; }
      // Items: each a list of trimmed line slices.
      typedef struct { Slice *parts; size_t n, cap; int tf; } Item;
      size_t capi = 8, nitems = 0;
      Item *items = (Item *)arena_alloc(arena, capi * sizeof(Item));
      while (i < nlines) {
        Slice l = lines[i];
        u8 cc = cls[i]; u8 kk = LC_KIND(cc);
        Slice rest; int is_item = 0;
        if (is_ordered) {
          if (kk == LC_ORDERED) { u32 nn; Slice rr; md_ordered_of(l, &nn, &rr); rest = rr; is_item = 1; }
        } else if (kk == LC_BULLET) {
          rest = sl(l.p + 2, l.n - 2); is_item = 1;
        }
        if (is_item) {
          if (nitems == capi) {
            size_t nc = capi * 2;
            Item *ni2 = (Item *)arena_alloc(arena, nc * sizeof(Item));
            memcpy(ni2, items, nitems * sizeof(Item));
            items = ni2; capi = nc;
          }
          Item *it = &items[nitems++];
          it->n = 0; it->cap = 4;
          it->tf = (cc & LC_TF_REST) ? 1 : 0;
          it->parts = (Slice *)arena_alloc(arena, it->cap * sizeof(Slice));
          it->parts[it->n++] = md_trim_line(rest);
          i++;
        } else if (kk == LC_CONT) {
          Item *it = &items[nitems - 1];
          if (it->n == it->cap) {
            size_t nc = it->cap * 2;
            Slice *np = (Slice *)arena_alloc(arena, nc * sizeof(Slice));
            memcpy(np, it->parts, it->n * sizeof(Slice));
            it->parts = np; it->cap = nc;
          }
          it->tf &= (cc & LC_TF_LINE) ? 1 : 0;
          it->parts[it->n++] = md_trim_line(l);
          i++;
        } else break;
      }
      const char *tag = is_ordered ? "ol" : "ul";
      if (is_ordered && start != 1) {
        char ob[32]; int m = snprintf(ob, sizeof(ob), "<ol start=\"%u\">", start);
        buf_put(&html, ob, (size_t)m);
      } else {
        buf_c(&html, '<'); buf_cs(&html, tag); buf_c(&html, '>');
      }
      buf_c(&html, '\n');
      for (size_t k = 0; k < nitems; k++) {
        buf_cs(&html, "<li>");
        if (items[k].tf && items[k].n > 0) {
          // v12: all parts trim-free — the item is the raw subslice from
          // the first part's start to the last part's end.
          Slice j = sl(items[k].parts[0].p, items[k].parts[items[k].n - 1].p + items[k].parts[items[k].n - 1].n - items[k].parts[0].p);
          md_inline(j, link, lctx, &html);
        } else {
          joined.len = 0;
          for (size_t q = 0; q < items[k].n; q++) {
            if (q > 0) buf_c(&joined, '\n');
            buf_sl(&joined, items[k].parts[q]);
          }
          md_inline(sl(joined.p, joined.len), link, lctx, &html);
        }
        buf_cs(&html, "</li>\n");
      }
      buf_cs(&html, "</");
      buf_cs(&html, tag);
      buf_cs(&html, ">\n");
      continue;
    }
    // Paragraph.
    size_t cap = 4, nb = 0;
    Slice *body = (Slice *)arena_alloc(arena, cap * sizeof(Slice));
    body[nb++] = md_trim_line(line);
    int tf = (c & LC_TF_LINE) ? 1 : 0;
    i++;
    while (i < nlines) {
      u8 kk = LC_KIND(cls[i]);
      if (kk == LC_BLANK || kk == LC_FENCE || kk == LC_FENCE_OPEN || kk == LC_HEADING || kk == LC_HR) break;
      if (nb == cap) {
        size_t nc = cap * 2;
        Slice *nb2 = (Slice *)arena_alloc(arena, nc * sizeof(Slice));
        memcpy(nb2, body, nb * sizeof(Slice));
        body = nb2; cap = nc;
      }
      body[nb++] = md_trim_line(lines[i]);
      tf &= (cls[i] & LC_TF_LINE) ? 1 : 0;
      i++;
    }
    buf_cs(&html, "<p>");
    size_t p_start = html.len;
    if (tf) {
      // v12: no line needed trimming — the paragraph is the raw subslice.
      Slice j = sl(body[0].p, body[nb - 1].p + body[nb - 1].n - body[0].p);
      md_inline(j, link, lctx, &html);
    } else {
      joined.len = 0;
      for (size_t k = 0; k < nb; k++) {
        if (k > 0) buf_c(&joined, '\n');
        buf_sl(&joined, body[k]);
      }
      md_inline(sl(joined.p, joined.len), link, lctx, &html);
    }
    size_t p_end = html.len;
    if (!have_excerpt) {
      mo.excerpt = arena_slice(arena, html.p + p_start, p_end - p_start);
      have_excerpt = 1;
    }
    buf_cs(&html, "</p>\n");
  }
  free(joined.p);
  mo.words = md_words;
  mo.html = sl(html.p, html.len); // malloc'd; kept alive for the case
  return mo;
}



// ---- carried from v5: dir/resolve_path/slice_bounds/sort (v5 lines 2503-2634) ----

static Slice dir_of(Slice path) {
  size_t i = path.n;
  while (i > 0) {
    i--;
    if (path.p[i] == '/') return sl(path.p, i);
    if (i == 0) break;
  }
  return slc("");
}

static inline int is_ascii_alpha(u8 c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
static inline int is_ascii_alnum(u8 c) { return is_ascii_alpha(c) || (c >= '0' && c <= '9'); }

static int is_scheme(Slice t) {
  if (t.n == 0 || !is_ascii_alpha(t.p[0])) return 0;
  for (size_t i = 1; i < t.n; i++) {
    u8 c = t.p[i];
    if (c == ':') return 1;
    if (!(is_ascii_alnum(c) || c == '+' || c == '.' || c == '-')) return 0;
  }
  return 0;
}

// Resolve `rel` against `dir` (both '/'-separated). Returns a slice with
// p==NULL when ".." would escape the root.
static Slice resolve_path(Arena *a, Slice dir, Slice rel) {
  size_t cap = 8, n = 0;
  Slice *parts = (Slice *)arena_alloc(a, cap * sizeof(Slice));
  if (dir.n > 0) {
    size_t start = 0;
    for (size_t i = 0; i <= dir.n; i++) {
      if (i == dir.n || dir.p[i] == '/') {
        if (n == cap) {
          size_t nc = cap * 2;
          Slice *np = (Slice *)arena_alloc(a, nc * sizeof(Slice));
          memcpy(np, parts, n * sizeof(Slice));
          parts = np; cap = nc;
        }
        parts[n++] = sl(dir.p + start, i - start);
        start = i + 1;
      }
    }
  }
  size_t start = 0;
  for (size_t i = 0; i <= rel.n; i++) {
    if (i == rel.n || rel.p[i] == '/') {
      Slice seg = sl(rel.p + start, i - start);
      start = i + 1;
      if (seg.n == 0 || sl_eq_c(seg, ".")) continue;
      if (sl_eq_c(seg, "..")) {
        if (n == 0) return sl(NULL, 0);
        n--;
      } else {
        if (n == cap) {
          size_t nc = cap * 2;
          Slice *np = (Slice *)arena_alloc(a, nc * sizeof(Slice));
          memcpy(np, parts, n * sizeof(Slice));
          parts = np; cap = nc;
        }
        parts[n++] = seg;
      }
    }
  }
  size_t tl = 0;
  for (size_t i = 0; i < n; i++) tl += parts[i].n + (i > 0 ? 1 : 0);
  u8 *rp = (u8 *)arena_alloc(a, tl ? tl : 1);
  size_t w = 0;
  for (size_t i = 0; i < n; i++) {
    if (i > 0) rp[w++] = '/';
    memcpy(rp + w, parts[i].p, parts[i].n); w += parts[i].n;
  }
  return sl(rp, tl);
}

// JS Array.prototype.slice bounds for one end index pair.
static void js_slice_bounds(size_t len, double start, double end, size_t *a, size_t *b) {
  double lf = (double)len;
  double na, nb;
  double t = trunc(start);
  na = t < 0.0 ? fmax(0.0, lf + t) : fmin(t, lf);
  t = trunc(end);
  nb = t < 0.0 ? fmax(0.0, lf + t) : fmin(t, lf);
  if (na > nb) { *a = (size_t)na; *b = (size_t)na; }
  else { *a = (size_t)na; *b = (size_t)nb; }
}

// Stable merge sort of an index array by a comparator over its values.
// (The old insertion sort was O(n^2): ~1.9M cmp_u16 calls per 3-case run
// for 200-600 files. Merge sort is O(n log n) and stable, so the order
// is identical.) Scratch is malloc'd once per call.
typedef int (*IdxCmp)(void *ctx, size_t a, size_t b);
static void sort_indices_rec(size_t *idx, size_t *tmp, size_t n, void *ctx, IdxCmp cmp) {
  if (n < 2) return;
  size_t m = n / 2;
  sort_indices_rec(idx, tmp, m, ctx, cmp);
  sort_indices_rec(idx + m, tmp + m, n - m, ctx, cmp);
  size_t i = 0, j = m, k = 0;
  while (i < m && j < n) {
    if (cmp(ctx, idx[i], idx[j]) <= 0) tmp[k++] = idx[i++];
    else tmp[k++] = idx[j++];
  }
  while (i < m) tmp[k++] = idx[i++];
  while (j < n) tmp[k++] = idx[j++];
  memcpy(idx, tmp, n * sizeof(size_t));
}
static void sort_indices(size_t *idx, size_t n, void *ctx, IdxCmp cmp) {
  if (n < 2) return;
  size_t *tmp = (size_t *)malloc(n * sizeof(size_t));
  sort_indices_rec(idx, tmp, n, ctx, cmp);
  free(tmp);
}

// Stable merge sort of slices by cmp_u16.
static void sort_slices_rec(Slice *a, Slice *tmp, size_t n) {
  if (n < 2) return;
  size_t m = n / 2;
  sort_slices_rec(a, tmp, m);
  sort_slices_rec(a + m, tmp + m, n - m);
  size_t i = 0, j = m, k = 0;
  while (i < m && j < n) {
    if (cmp_u16(a[i], a[j]) <= 0) tmp[k++] = a[i++];
    else tmp[k++] = a[j++];
  }
  while (i < m) tmp[k++] = a[i++];
  while (j < n) tmp[k++] = a[j++];
  memcpy(a, tmp, n * sizeof(Slice));
}
static void sort_slices(Slice *a, size_t n) {
  if (n < 2) return;
  Slice *tmp = (Slice *)malloc(n * sizeof(Slice));
  sort_slices_rec(a, tmp, n);
  free(tmp);
}



// ---- carried from v5: index_url/crlf_to_lf (v5 lines 2891-2920) ----

static Slice index_url(Arena *a, Slice base, u64 k) {
  Buf b = {0};
  buf_sl(&b, base);
  if (k == 1) buf_c(&b, '/');
  else {
    buf_cs(&b, "/page/");
    char kb[24];
    int m = snprintf(kb, sizeof(kb), "%llu", (unsigned long long)k);
    buf_put(&b, kb, (size_t)m);
    buf_c(&b, '/');
  }
  Slice r = arena_slice(a, b.p, b.len);
  free(b.p);
  return r;
}

// \r\n and lone \r become \n. Files without any \r keep their bytes.
static Slice crlf_to_lf(Arena *a, Slice content) {
  if (find_byte(content.p, content.n, 0, '\r') >= content.n) return content;
  u8 *out = (u8 *)arena_alloc(a, content.n ? content.n : 1);
  size_t o = 0, i = 0;
  while (i < content.n) {
    if (content.p[i] == '\r') {
      out[o++] = '\n';
      if (i + 1 < content.n && content.p[i + 1] == '\n') i += 2;
      else i += 1;
    } else out[o++] = content.p[i++];
  }
  return sl(out, o);
}


// ---- carried from v5: tpl tokenizer/compiler (v5 lines 1835-2172) ----

typedef enum { OP_TEXT, OP_VAR, OP_RAW, OP_JIF, OP_JMP, OP_FOR, OP_NEXT, OP_INC, OP_END } OpCode;
// ---------------------------------------------------------------- template parsing

#define INCLUDE_DEPTH 8

typedef struct {
  OpCode op;
  Slice text;              // OP_TEXT
  Slice *segs; size_t nsegs; // OP_VAR/OP_RAW/OP_JIF/OP_FOR paths
  int not;                 // OP_JIF
  int target;              // OP_JIF (else), OP_JMP (end), OP_FOR (after loop)
  Slice name;              // OP_FOR loop variable
  int body;                // OP_NEXT: first body instruction
  int tpl;                 // OP_INC: template index (-1 = missing)
  int cacheable;           // OP_INC
  int shared_only;         // OP_INC: fragment reads shared roots only
  // ---- v6: constant-folded resolution (filled by resolve_slots) ----
  u8 rkind;                // RK_* for OP_VAR/OP_RAW/OP_JIF/OP_FOR
  i16 rdepth;              // RK_LOOPVAR: compile-time loop depth
  u8 chain[6];             // resolved field codes for segs[1..]
  u8 nchain;               // number of chain entries (0 = the root itself)
} Instr;

typedef struct Chunk Chunk;
typedef struct {
  Slice name;
  Instr *code; size_t n, cap;
  Chunk *chunks; size_t nchunks; // v7: chunked renderer (built from code)
  // Cacheability analysis of this template as an include fragment:
  int has_inc;
  int loop_outside; // reads `loop` with no fragment-local for
  Slice *free_names; size_t nfree, capfree;
} Tpl;

static void tpl_emit(Tpl *t, Arena *a, Instr in) {
  if (t->n == t->cap) {
    size_t nc = t->cap ? t->cap * 2 : 32;
    Instr *ni = (Instr *)arena_alloc(a, nc * sizeof(Instr));
    if (t->code) memcpy(ni, t->code, t->n * sizeof(Instr));
    t->code = ni; t->cap = nc;
  }
  t->code[t->n++] = in;
}

static Slice tmpl_err(Arena *a, Slice name, Slice msg) {
  Buf b = {0};
  buf_cs(&b, "template \""); buf_sl(&b, name); buf_cs(&b, "\": "); buf_sl(&b, msg);
  Slice r = arena_slice(a, b.p, b.len);
  free(b.p);
  return r;
}

static inline int is_path_start(u8 c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
static inline int is_path_char(u8 c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'; }

// PATH = /^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$/
// On success returns 1 and fills *segs/*nsegs (arena). On failure returns
// 0 and sets *err to `"<json>" is not a path` (with the template prefix).
static int path_of(Slice name, Slice text, Arena *a, Slice **segs, size_t *nsegs, Slice *err) {
  int ok = text.n > 0;
  size_t i = 0;
  while (ok && i < text.n) {
    if (!is_path_start(text.p[i])) { ok = 0; break; }
    i++;
    while (i < text.n && is_path_char(text.p[i])) i++;
    if (i < text.n) {
      if (text.p[i] != '.') { ok = 0; break; }
      i++;
      if (i >= text.n) { ok = 0; break; }
    }
  }
  if (!ok) {
    Buf m = {0};
    json_write_str(&m, text);
    buf_cs(&m, " is not a path");
    *err = tmpl_err(a, name, sl(m.p, m.len));
    free(m.p);
    return 0;
  }
  size_t cap = 4, n = 0;
  Slice *ss = (Slice *)arena_alloc(a, cap * sizeof(Slice));
  size_t start = 0;
  for (size_t k = 0; k <= text.n; k++) {
    if (k == text.n || text.p[k] == '.') {
      if (n == cap) {
        size_t nc = cap * 2;
        Slice *ns = (Slice *)arena_alloc(a, nc * sizeof(Slice));
        memcpy(ns, ss, n * sizeof(Slice));
        ss = ns; cap = nc;
      }
      ss[n++] = sl(text.p + start, k - start);
      start = k + 1;
    }
  }
  *segs = ss; *nsegs = n;
  return 1;
}

static inline int only_spaces(Slice s, size_t a, size_t b) {
  for (size_t i = a; i < b; i++) if (s.p[i] != ' ' && s.p[i] != '\t') return 0;
  return 1;
}

static size_t find_subslice(const u8 *hay, size_t n, const u8 *needle, size_t nl, size_t from) {
  if (nl == 0 || n < nl) return n;
  for (size_t i = from; i + nl <= n; i++) if (memcmp(hay + i, needle, nl) == 0) return i;
  return n;
}

typedef enum { TK_TEXT, TK_VAR, TK_RAW, TK_TAG } TokKind;
typedef struct {
  TokKind kind;
  Slice text;              // TK_TEXT
  Slice *segs; size_t nsegs; // TK_VAR/TK_RAW
  Slice tag;               // TK_TAG
} Tok;

typedef struct { Tok *toks; size_t n, cap; } TokVec;
static void tok_push(TokVec *v, Arena *a, Tok t) {
  if (v->n == v->cap) {
    size_t nc = v->cap ? v->cap * 2 : 32;
    Tok *nt = (Tok *)arena_alloc(a, nc * sizeof(Tok));
    if (v->toks) memcpy(nt, v->toks, v->n * sizeof(Tok));
    v->toks = nt; v->cap = nc;
  }
  v->toks[v->n++] = t;
}

static int tokenize(Slice name, Slice src, Arena *a, TokVec *out, Slice *err) {
  Buf text = {0};
  size_t i = 0, n = src.n;
  while (i < n) {
    size_t open = find_byte(src.p, n, i, '{');
    if (open >= n || open == n - 1) {
      buf_put(&text, src.p + i, n - i);
      break;
    }
    u8 nx = src.p[open + 1];
    if (nx != '{' && nx != '%') {
      buf_put(&text, src.p + i, open + 1 - i);
      i = open + 1;
      continue;
    }
    buf_put(&text, src.p + i, open - i);
    if (nx == '{') {
      int raw = (open + 2 < n && src.p[open + 2] == '{');
      const u8 *close_pat = raw ? (const u8 *)"}}}" : (const u8 *)"}}";
      size_t cpl = raw ? 3 : 2;
      size_t start = open + cpl;
      size_t close = find_subslice(src.p, n, close_pat, cpl, start);
      if (close >= n) {
        *err = tmpl_err(a, name, slc(raw ? "a {{{ is not closed" : "a {{ is not closed"));
        free(text.p);
        return 0;
      }
      if (text.len > 0) {
        Tok t; memset(&t, 0, sizeof(t)); t.kind = TK_TEXT;
        t.text = arena_slice(a, text.p, text.len);
        tok_push(out, a, t);
        text.len = 0;
      }
      Slice inner = js_trim(sl(src.p + start, close - start));
      Slice *segs; size_t nsegs;
      if (!path_of(name, inner, a, &segs, &nsegs, err)) { free(text.p); return 0; }
      Tok t; memset(&t, 0, sizeof(t)); t.kind = raw ? TK_RAW : TK_VAR; t.segs = segs; t.nsegs = nsegs;
      tok_push(out, a, t);
      i = close + cpl;
      continue;
    }
    // {% ... %}
    size_t close = find_subslice(src.p, n, (const u8 *)"%}", 2, open + 2);
    if (close >= n) {
      *err = tmpl_err(a, name, slc("a {% is not closed"));
      free(text.p);
      return 0;
    }
    size_t end = close + 2;
    size_t line_start = 0;
    if (open > 0) {
      // rposition of '\n' in src[..open]
      size_t k = open;
      while (k > 0) { k--; if (src.p[k] == '\n') { line_start = k + 1; break; } if (k == 0) break; }
      // Note: the loop above stops at k==0 without checking src[0] if open>0
      // and no '\n' was found before k==0; line_start stays 0 either way.
    }
    size_t nl = find_byte(src.p, n, end, '\n');
    size_t line_end = nl < n ? nl : n;
    int standalone = only_spaces(src, line_start, open) && only_spaces(src, end, line_end);
    if (standalone) {
      size_t cut = open - line_start;
      text.len -= cut;
    }
    if (text.len > 0) {
      Tok t; memset(&t, 0, sizeof(t)); t.kind = TK_TEXT;
      t.text = arena_slice(a, text.p, text.len);
      tok_push(out, a, t);
      text.len = 0;
    }
    Tok t; memset(&t, 0, sizeof(t)); t.kind = TK_TAG;
    t.tag = js_trim(sl(src.p + open + 2, close - (open + 2)));
    tok_push(out, a, t);
    i = standalone ? (nl < n ? nl + 1 : n) : end;
  }
  if (text.len > 0) {
    Tok t; memset(&t, 0, sizeof(t)); t.kind = TK_TEXT;
    t.text = arena_slice(a, text.p, text.len);
    tok_push(out, a, t);
  }
  free(text.p);
  return 1;
}

static int has_js_ws(Slice s) {
  size_t i = 0;
  while (i < s.n) {
    size_t used; u32 cp = utf8_decode(s.p, s.n, i, &used);
    if (is_js_space(cp)) return 1;
    i += used;
  }
  return 0;
}

// /^for ([A-Za-z_][A-Za-z0-9_]*) in (\S+)$/
static int parse_for(Slice tag, Slice *name, Slice *path) {
  if (!sl_starts(tag, "for ")) return 0;
  Slice rest = sl(tag.p + 4, tag.n - 4);
  size_t i = 0;
  if (i >= rest.n || !is_path_start(rest.p[i])) return 0;
  i++;
  while (i < rest.n && is_path_char(rest.p[i])) i++;
  *name = sl(rest.p, i);
  if (i + 4 <= rest.n && memcmp(rest.p + i, " in ", 4) == 0) {
    Slice p = sl(rest.p + i + 4, rest.n - (i + 4));
    if (p.n > 0 && !has_js_ws(p)) { *path = p; return 1; }
  }
  return 0;
}

// /^if (not )?(\S+)$/
static int parse_if(Slice tag, int *not, Slice *path) {
  if (!sl_starts(tag, "if ")) return 0;
  Slice rest = sl(tag.p + 3, tag.n - 3);
  int nt = 0;
  if (sl_starts(rest, "not ")) { nt = 1; rest = sl(rest.p + 4, rest.n - 4); }
  if (rest.n > 0 && !has_js_ws(rest)) { *not = nt; *path = rest; return 1; }
  return 0;
}

// /^include "([a-z0-9-]+)"$/
static int parse_include(Slice tag, Slice *name) {
  if (!sl_starts(tag, "include \"") || tag.n < 11 || tag.p[tag.n - 1] != '"') return 0;
  Slice inner = sl(tag.p + 9, tag.n - 10);
  if (inner.n == 0) return 0;
  for (size_t i = 0; i < inner.n; i++) {
    u8 c = inner.p[i];
    if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-')) return 0;
  }
  *name = inner;
  return 1;
}

// Compile tokens to flat bytecode. Returns 1 on success, 0 with *err.
static int compile_template(Slice name, TokVec *toks, Arena *a, Tpl *tpl, Slice *err) {
  memset(tpl, 0, sizeof(*tpl));
  tpl->name = name;
  typedef struct { int kind; int idx; int jmp_idx; int in_else; } CFrame; // kind 0=for 1=if
  size_t capf = 8, nf = 0;
  CFrame *stack = (CFrame *)arena_alloc(a, capf * sizeof(CFrame));
  Instr in;
  for (size_t ti = 0; ti < toks->n; ti++) {
    Tok *tok = &toks->toks[ti];
    memset(&in, 0, sizeof(in));
    switch (tok->kind) {
      case TK_TEXT: in.op = OP_TEXT; in.text = tok->text; tpl_emit(tpl, a, in); break;
      case TK_VAR: in.op = OP_VAR; in.segs = tok->segs; in.nsegs = tok->nsegs; tpl_emit(tpl, a, in); break;
      case TK_RAW: in.op = OP_RAW; in.segs = tok->segs; in.nsegs = tok->nsegs; tpl_emit(tpl, a, in); break;
      case TK_TAG: {
        Slice tag = tok->tag;
        Slice fname, fpath; int not; Slice ipath, iname;
        if (parse_for(tag, &fname, &fpath)) {
          Slice *segs; size_t nsegs;
          if (!path_of(name, fpath, a, &segs, &nsegs, err)) return 0;
          memset(&in, 0, sizeof(in)); in.op = OP_FOR; in.name = fname; in.segs = segs; in.nsegs = nsegs; in.target = -1;
          tpl_emit(tpl, a, in);
          if (nf == capf) { size_t nc = capf * 2; CFrame *ns = (CFrame *)arena_alloc(a, nc * sizeof(CFrame)); memcpy(ns, stack, nf * sizeof(CFrame)); stack = ns; capf = nc; }
          stack[nf].kind = 0; stack[nf].idx = (int)tpl->n - 1; stack[nf].jmp_idx = -1; stack[nf].in_else = 0; nf++;
        } else if (parse_if(tag, &not, &ipath)) {
          Slice *segs; size_t nsegs;
          if (!path_of(name, ipath, a, &segs, &nsegs, err)) return 0;
          memset(&in, 0, sizeof(in)); in.op = OP_JIF; in.not = not; in.segs = segs; in.nsegs = nsegs; in.target = -1;
          tpl_emit(tpl, a, in);
          if (nf == capf) { size_t nc = capf * 2; CFrame *ns = (CFrame *)arena_alloc(a, nc * sizeof(CFrame)); memcpy(ns, stack, nf * sizeof(CFrame)); stack = ns; capf = nc; }
          stack[nf].kind = 1; stack[nf].idx = (int)tpl->n - 1; stack[nf].jmp_idx = -1; stack[nf].in_else = 0; nf++;
        } else if (sl_eq_c(tag, "else")) {
          if (nf == 0 || stack[nf - 1].kind != 1 || stack[nf - 1].in_else) {
            *err = tmpl_err(a, name, slc("an else outside an if")); return 0;
          }
          // JIF jumps to the first instruction of the else branch, which
          // is the instruction AFTER the JMP emitted here.
          memset(&in, 0, sizeof(in)); in.op = OP_JMP; in.target = -1;
          tpl_emit(tpl, a, in);
          tpl->code[stack[nf - 1].idx].target = (int)tpl->n;
          stack[nf - 1].jmp_idx = (int)tpl->n - 1;
          stack[nf - 1].in_else = 1;
        } else if (sl_eq_c(tag, "endfor") || sl_eq_c(tag, "endif")) {
          int want = sl_eq_c(tag, "endfor") ? 0 : 1;
          if (nf == 0 || stack[nf - 1].kind != want) {
            Buf m = {0};
            buf_cs(&m, "an "); buf_sl(&m, tag); buf_cs(&m, " with no open ");
            buf_put(&m, tag.p + 3, tag.n - 3);
            *err = tmpl_err(a, name, sl(m.p, m.len)); free(m.p);
            return 0;
          }
          CFrame fr = stack[--nf];
          if (want == 0) {
            memset(&in, 0, sizeof(in)); in.op = OP_NEXT; in.body = fr.idx + 1;
            tpl_emit(tpl, a, in);
            tpl->code[fr.idx].target = (int)tpl->n;
          } else {
            if (fr.in_else) tpl->code[fr.jmp_idx].target = (int)tpl->n;
            else tpl->code[fr.idx].target = (int)tpl->n;
          }
        } else if (parse_include(tag, &iname)) {
          memset(&in, 0, sizeof(in)); in.op = OP_INC; in.tpl = -2; in.cacheable = 0;
          // Stash the include name in `text` until resolution.
          in.text = iname;
          tpl_emit(tpl, a, in);
        } else {
          Buf m = {0};
          buf_cs(&m, "{% "); buf_sl(&m, tag); buf_cs(&m, " %} is not a tag");
          *err = tmpl_err(a, name, sl(m.p, m.len)); free(m.p);
          return 0;
        }
        break;
      }
    }
  }
  if (nf > 0) {
    Buf m = {0};
    buf_cs(&m, "a ");
    buf_cs(&m, stack[nf - 1].kind == 0 ? "for" : "if");
    buf_cs(&m, " is not closed");
    *err = tmpl_err(a, name, sl(m.p, m.len)); free(m.p);
    return 0;
  }
  memset(&in, 0, sizeof(in)); in.op = OP_END;
  tpl_emit(tpl, a, in);
  return 1;
}


// ---- carried from v5: include resolve/cache analysis (v5 lines 2358-2481) ----
static void resolve_includes(Tpl *tpls, size_t ntpls) {
  for (size_t i = 0; i < ntpls; i++) {
    Tpl *t = &tpls[i];
    for (size_t j = 0; j < t->n; j++) {
      Instr *in = &t->code[j];
      if (in->op == OP_INC) {
        int found = -1;
        for (size_t k = 0; k < ntpls; k++) if (sl_eq(tpls[k].name, in->text)) { found = (int)k; break; }
        in->tpl = found;
      }
    }
  }
}

// Analysis pass A: for template t, compute has_inc / loop_outside / free names.
static void analyze_fragment(Tpl *t, Arena *a) {
  // Loop-name stack (arena array of Slices).
  size_t cap = 8, n = 0;
  Slice *stack = (Slice *)arena_alloc(a, cap * sizeof(Slice));
  t->free_names = NULL; t->nfree = 0; t->capfree = 0;
  for (size_t i = 0; i < t->n; i++) {
    Instr *in = &t->code[i];
    switch (in->op) {
      case OP_VAR: case OP_RAW: case OP_JIF: {
        if (in->nsegs == 0) break;
        Slice f = in->segs[0];
        if (sl_eq_c(f, "loop")) {
          if (n == 0) t->loop_outside = 1;
        } else {
          int bound = 0;
          for (size_t k = 0; k < n; k++) if (sl_eq(stack[k], f)) { bound = 1; break; }
          if (!bound) {
            int seen = 0;
            for (size_t k = 0; k < t->nfree; k++) if (sl_eq(t->free_names[k], f)) { seen = 1; break; }
            if (!seen) {
              if (t->nfree == t->capfree) {
                size_t nc = t->capfree ? t->capfree * 2 : 8;
                Slice *ns = (Slice *)arena_alloc(a, nc * sizeof(Slice));
                if (t->free_names) memcpy(ns, t->free_names, t->nfree * sizeof(Slice));
                t->free_names = ns; t->capfree = nc;
              }
              t->free_names[t->nfree++] = f;
            }
          }
        }
        break;
      }
      case OP_FOR: {
        if (in->nsegs > 0) {
          Slice f = in->segs[0];
          if (sl_eq_c(f, "loop")) {
            if (n == 0) t->loop_outside = 1;
          } else {
            int bound = 0;
            for (size_t k = 0; k < n; k++) if (sl_eq(stack[k], f)) { bound = 1; break; }
            if (!bound) {
              int seen = 0;
              for (size_t k = 0; k < t->nfree; k++) if (sl_eq(t->free_names[k], f)) { seen = 1; break; }
              if (!seen) {
                if (t->nfree == t->capfree) {
                  size_t nc = t->capfree ? t->capfree * 2 : 8;
                  Slice *ns = (Slice *)arena_alloc(a, nc * sizeof(Slice));
                  if (t->free_names) memcpy(ns, t->free_names, t->nfree * sizeof(Slice));
                  t->free_names = ns; t->capfree = nc;
                }
                t->free_names[t->nfree++] = f;
              }
            }
          }
        }
        if (n == cap) {
          size_t nc = cap * 2;
          Slice *ns = (Slice *)arena_alloc(a, nc * sizeof(Slice));
          memcpy(ns, stack, n * sizeof(Slice));
          stack = ns; cap = nc;
        }
        stack[n++] = in->name;
        break;
      }
      case OP_NEXT: if (n > 0) n--; break;
      case OP_INC: t->has_inc = 1; break;
      default: break;
    }
  }
}

// Analysis pass B: for each template, walk its code with a loop stack and
// mark each OP_INC cacheable when the fragment's free names are all shared
// (site/tags/recent, unshadowed) or the innermost loop variable.
static void analyze_cacheable(Tpl *tpls, size_t ntpls, Arena *a) {
  for (size_t ti = 0; ti < ntpls; ti++) {
    Tpl *t = &tpls[ti];
    size_t cap = 8, n = 0;
    Slice *stack = (Slice *)arena_alloc(a, cap * sizeof(Slice));
    for (size_t i = 0; i < t->n; i++) {
      Instr *in = &t->code[i];
      if (in->op == OP_FOR) {
        if (n == cap) {
          size_t nc = cap * 2;
          Slice *ns = (Slice *)arena_alloc(a, nc * sizeof(Slice));
          memcpy(ns, stack, n * sizeof(Slice));
          stack = ns; cap = nc;
        }
        stack[n++] = in->name;
      } else if (in->op == OP_NEXT) {
        if (n > 0) n--;
      } else if (in->op == OP_INC) {
        if (in->tpl < 0) continue;
        Tpl *frag = &tpls[in->tpl];
        if (frag->has_inc || frag->loop_outside) continue;
        int ok = 1;
        for (size_t fi = 0; fi < frag->nfree && ok; fi++) {
          Slice f = frag->free_names[fi];
          int in_loop_stack = 0;
          for (size_t k = 0; k < n; k++) if (sl_eq(stack[k], f)) { in_loop_stack = 1; break; }
          int shared = (sl_eq_c(f, "site") || sl_eq_c(f, "tags") || sl_eq_c(f, "recent")) && !in_loop_stack;
          int innermost = n > 0 && sl_eq(stack[n - 1], f);
          if (!shared && !innermost) ok = 0;
        }
        in->cacheable = ok;
        // shared_only: the fragment's output cannot depend on any loop
        // item at this include site (all its free names are roots, and
        // none of them is shadowed by an enclosing loop variable).
        // Such fragments share ONE card-cache entry per template.
        int so = !frag->loop_outside;
        for (size_t fi = 0; fi < frag->nfree && so; fi++) {
          Slice f = frag->free_names[fi];
          if (!(sl_eq_c(f, "site") || sl_eq_c(f, "tags") || sl_eq_c(f, "recent")
                || sl_eq_c(f, "page") || sl_eq_c(f, "posts") || sl_eq_c(f, "pager")
                || sl_eq_c(f, "tag")))
            so = 0;
          for (size_t k = 0; k < n && so; k++) if (sl_eq(stack[k], f)) so = 0;
        }
        in->shared_only = so;
      }
    }
  }
}

// ---- v6: AVX2 32-byte twin of next_json_esc (same classes, same tail) ----
// AVX2 twin of next_json_esc: 32-byte lanes, same return contract.
// Compiled with target("avx2") — the translation unit targets
// x86-64-v2, so the attribute is what makes the AVX2 intrinsics legal
// here; the caller only reaches this function when have_avx2 is set.
__attribute__((target("avx2")))
static size_t next_json_esc_avx2(const u8 *s, size_t n, size_t from) {
  size_t i = from;
  __m256i quote = _mm256_set1_epi8('"');
  __m256i backslash = _mm256_set1_epi8('\\');
  __m256i lead_ed = _mm256_set1_epi8((char)0xED);
  __m256i lo31 = _mm256_set1_epi8(0x1F);
  __m256i zero = _mm256_setzero_si256();
  while (i + 32 <= n) {
    __m256i chunk = _mm256_loadu_si256((const __m256i *)(s + i));
    __m256i m = _mm256_or_si256(_mm256_or_si256(_mm256_cmpeq_epi8(chunk, quote), _mm256_cmpeq_epi8(chunk, backslash)),
                              _mm256_or_si256(_mm256_cmpeq_epi8(chunk, lead_ed),
                                           _mm256_cmpeq_epi8(_mm256_subs_epu8(chunk, lo31), zero)));
    unsigned mask = (unsigned)_mm256_movemask_epi8(m);
    if (mask) return i + (size_t)__builtin_ctz(mask);
    i += 32;
  }
  if (i < n && n >= 32) {
    __m256i chunk = _mm256_loadu_si256((const __m256i *)(s + n - 32));
    __m256i m = _mm256_or_si256(_mm256_or_si256(_mm256_cmpeq_epi8(chunk, quote), _mm256_cmpeq_epi8(chunk, backslash)),
                              _mm256_or_si256(_mm256_cmpeq_epi8(chunk, lead_ed),
                                           _mm256_cmpeq_epi8(_mm256_subs_epu8(chunk, lo31), zero)));
    unsigned mask = (unsigned)_mm256_movemask_epi8(m);
    size_t skip = i - (n - 32);
    mask &= ~((1u << skip) - 1u);
    if (mask) return (n - 32) + (size_t)__builtin_ctz(mask);
    return n;
  }
  while (i < n) {
    u8 c = s[i];
    if (c == '"' || c == '\\' || c < 0x20 || c == 0xED) return i;
    i++;
  }
  return n;
}

// ------------------------------------------------------------------ v6 flat JSON parser
// Parses cases.json straight into flat structs — no Json tree, no
// per-value allocation. Strings without escapes are direct slices of
// the input buffer (which outlives the whole buildCases call); decoded
// strings go to the parse arena. The grammar scan mirrors v5's
// j_number/j_string exactly (same acceptance, same doubles via strtod).
typedef struct { Slice name, src; } TplSrc;
typedef struct { Slice path, content; } InFile;
typedef struct {
  Slice title, base_url;
  double per_page, recent;
  int has_title, has_base_url, has_per_page, has_recent;
} SiteIn;
typedef struct {
  SiteIn site;
  TplSrc *tpls; size_t ntpls;
  InFile *files; size_t nfiles;
} CaseIn;
typedef struct { CaseIn *items; Slice *ids; size_t n; } CaseVec;

static inline void fp_ws(JParser *p) {
  while (p->pos < p->len) {
    u8 c = p->b[p->pos];
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') p->pos++;
    else break;
  }
}

static int fp_lit(JParser *p, const char *lit) {
  size_t n = strlen(lit);
  if (p->pos + n <= p->len && memcmp(p->b + p->pos, lit, n) == 0) { p->pos += n; return 1; }
  return 0;
}

// Parse a JSON string (pos at the opening '"'). Same zero-copy fast
// path as v5's j_string.
static int fp_string(JParser *p, Slice *out) {
  size_t start = p->pos + 1;
  size_t j = next_json_esc(p->b, p->len, start);
  if (j < p->len && p->b[j] == '"') {
    *out = sl(p->b + start, j - start);
    p->pos = j + 1;
    return 1;
  }
  p->pos++;
  Buf tmp = {0};
  for (;;) {
    size_t k = next_json_esc(p->b, p->len, p->pos);
    if (k > p->pos) { buf_put(&tmp, p->b + p->pos, k - p->pos); p->pos = k; }
    if (p->pos >= p->len) { free(tmp.p); return 0; }
    u8 c = p->b[p->pos];
    if (c == '"') { p->pos++; break; }
    if (c == '\\') {
      p->pos++;
      if (p->pos >= p->len) { free(tmp.p); return 0; }
      u8 e = p->b[p->pos]; p->pos++;
      switch (e) {
        case '"': buf_c(&tmp, '"'); break;
        case '\\': buf_c(&tmp, '\\'); break;
        case '/': buf_c(&tmp, '/'); break;
        case 'b': buf_c(&tmp, 0x08); break;
        case 'f': buf_c(&tmp, 0x0C); break;
        case 'n': buf_c(&tmp, '\n'); break;
        case 'r': buf_c(&tmp, '\r'); break;
        case 't': buf_c(&tmp, '\t'); break;
        case 'u': {
          u32 cp;
          if (!j_hex4(p, &cp)) { free(tmp.p); return 0; }
          if (cp >= 0xD800 && cp < 0xDC00) {
            if (p->pos + 1 < p->len && p->b[p->pos] == '\\' && p->b[p->pos + 1] == 'u') {
              size_t save = p->pos;
              p->pos += 2;
              u32 lo;
              if (j_hex4(p, &lo) && lo >= 0xDC00 && lo < 0xE000) {
                utf8_encode(0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00), &tmp);
              } else { p->pos = save; utf8_encode(cp, &tmp); }
            } else utf8_encode(cp, &tmp);
          } else utf8_encode(cp, &tmp);
          break;
        }
        default: free(tmp.p); return 0;
      }
    } else if (c < 0x20) { free(tmp.p); return 0; }
    else { buf_c(&tmp, c); p->pos++; }
  }
  *out = arena_slice(p->arena, tmp.p ? tmp.p : (const u8 *)"", tmp.len);
  free(tmp.p);
  return 1;
}

static int fp_number(JParser *p, double *out) {
  size_t start = p->pos;
  if (p->pos < p->len && p->b[p->pos] == '-') p->pos++;
  if (p->pos >= p->len) return 0;
  if (p->b[p->pos] == '0') p->pos++;
  else if (p->b[p->pos] >= '1' && p->b[p->pos] <= '9') {
    while (p->pos < p->len && p->b[p->pos] >= '0' && p->b[p->pos] <= '9') p->pos++;
  } else return 0;
  if (p->pos < p->len && p->b[p->pos] == '.') {
    p->pos++;
    if (p->pos >= p->len || p->b[p->pos] < '0' || p->b[p->pos] > '9') return 0;
    while (p->pos < p->len && p->b[p->pos] >= '0' && p->b[p->pos] <= '9') p->pos++;
  }
  if (p->pos < p->len && (p->b[p->pos] == 'e' || p->b[p->pos] == 'E')) {
    p->pos++;
    if (p->pos < p->len && (p->b[p->pos] == '+' || p->b[p->pos] == '-')) p->pos++;
    if (p->pos >= p->len || p->b[p->pos] < '0' || p->b[p->pos] > '9') return 0;
    while (p->pos < p->len && p->b[p->pos] >= '0' && p->b[p->pos] <= '9') p->pos++;
  }
  char numbuf[64];
  size_t n = p->pos - start;
  if (n >= sizeof(numbuf)) return 0;
  memcpy(numbuf, p->b + start, n); numbuf[n] = 0;
  *out = strtod(numbuf, NULL);
  return 1;
}

static int fp_skip_value(JParser *p, int depth) {
  if (depth > 400 || p->pos >= p->len) return 0;
  u8 c = p->b[p->pos];
  switch (c) {
    case '{': {
      p->pos++;
      fp_ws(p);
      if (p->pos < p->len && p->b[p->pos] == '}') { p->pos++; return 1; }
      for (;;) {
        fp_ws(p);
        if (p->pos >= p->len || p->b[p->pos] != '"') return 0;
        Slice k;
        if (!fp_string(p, &k)) return 0;
        fp_ws(p);
        if (p->pos >= p->len || p->b[p->pos] != ':') return 0;
        p->pos++;
        fp_ws(p);
        if (!fp_skip_value(p, depth + 1)) return 0;
        fp_ws(p);
        if (p->pos >= p->len) return 0;
        if (p->b[p->pos] == ',') { p->pos++; continue; }
        if (p->b[p->pos] == '}') { p->pos++; return 1; }
        return 0;
      }
    }
    case '[': {
      p->pos++;
      fp_ws(p);
      if (p->pos < p->len && p->b[p->pos] == ']') { p->pos++; return 1; }
      for (;;) {
        fp_ws(p);
        if (!fp_skip_value(p, depth + 1)) return 0;
        fp_ws(p);
        if (p->pos >= p->len) return 0;
        if (p->b[p->pos] == ',') { p->pos++; continue; }
        if (p->b[p->pos] == ']') { p->pos++; return 1; }
        return 0;
      }
    }
    case '"': { Slice s; return fp_string(p, &s); }
    case 't': return fp_lit(p, "true");
    case 'f': return fp_lit(p, "false");
    case 'n': return fp_lit(p, "null");
    default: {
      double d;
      if (c == '-' || (c >= '0' && c <= '9')) return fp_number(p, &d);
      return 0;
    }
  }
}

static int fp_parse_site(JParser *p, SiteIn *s) {
  memset(s, 0, sizeof(*s));
  if (p->pos >= p->len || p->b[p->pos] != '{') return 0;
  p->pos++;
  fp_ws(p);
  if (p->pos < p->len && p->b[p->pos] == '}') { p->pos++; return 1; }
  for (;;) {
    fp_ws(p);
    if (p->pos >= p->len || p->b[p->pos] != '"') return 0;
    Slice k;
    if (!fp_string(p, &k)) return 0;
    fp_ws(p);
    if (p->pos >= p->len || p->b[p->pos] != ':') return 0;
    p->pos++;
    fp_ws(p);
    if (sl_eq_c(k, "title")) {
      if (!fp_string(p, &s->title)) return 0;
      s->has_title = 1;
    } else if (sl_eq_c(k, "base_url")) {
      if (!fp_string(p, &s->base_url)) return 0;
      s->has_base_url = 1;
    } else if (sl_eq_c(k, "per_page")) {
      if (!fp_number(p, &s->per_page)) return 0;
      s->has_per_page = 1;
    } else if (sl_eq_c(k, "recent")) {
      if (!fp_number(p, &s->recent)) return 0;
      s->has_recent = 1;
    } else {
      if (!fp_skip_value(p, 0)) return 0;
    }
    fp_ws(p);
    if (p->pos >= p->len) return 0;
    if (p->b[p->pos] == ',') { p->pos++; continue; }
    if (p->b[p->pos] == '}') { p->pos++; return 1; }
    return 0;
  }
}

static int fp_parse_templates(JParser *p, TplSrc **out, size_t *nout) {
  if (p->pos >= p->len || p->b[p->pos] != '{') return 0;
  p->pos++;
  size_t cap = 16, n = 0;
  TplSrc *arr = (TplSrc *)arena_alloc(p->arena, cap * sizeof(TplSrc));
  fp_ws(p);
  if (p->pos < p->len && p->b[p->pos] == '}') { p->pos++; *out = arr; *nout = 0; return 1; }
  for (;;) {
    fp_ws(p);
    if (p->pos >= p->len || p->b[p->pos] != '"') return 0;
    Slice k;
    if (!fp_string(p, &k)) return 0;
    fp_ws(p);
    if (p->pos >= p->len || p->b[p->pos] != ':') return 0;
    p->pos++;
    fp_ws(p);
    Slice v;
    if (p->pos >= p->len || p->b[p->pos] != '"') return 0;
    if (!fp_string(p, &v)) return 0;
    if (n == cap) {
      size_t nc = cap * 2;
      TplSrc *na = (TplSrc *)arena_alloc(p->arena, nc * sizeof(TplSrc));
      memcpy(na, arr, n * sizeof(TplSrc));
      arr = na; cap = nc;
    }
    arr[n].name = k; arr[n].src = v; n++;
    fp_ws(p);
    if (p->pos >= p->len) return 0;
    if (p->b[p->pos] == ',') { p->pos++; continue; }
    if (p->b[p->pos] == '}') { p->pos++; *out = arr; *nout = n; return 1; }
    return 0;
  }
}

static int fp_parse_files(JParser *p, InFile **out, size_t *nout) {
  if (p->pos >= p->len || p->b[p->pos] != '[') return 0;
  p->pos++;
  size_t cap = 64, n = 0;
  InFile *arr = (InFile *)arena_alloc(p->arena, cap * sizeof(InFile));
  fp_ws(p);
  if (p->pos < p->len && p->b[p->pos] == ']') { p->pos++; *out = arr; *nout = 0; return 1; }
  for (;;) {
    fp_ws(p);
    if (p->pos >= p->len || p->b[p->pos] != '{') return 0;
    p->pos++;
    Slice path = slc(""), content = slc("");
    int has_path = 0, has_content = 0;
    fp_ws(p);
    if (p->pos < p->len && p->b[p->pos] == '}') { p->pos++; return 0; }
    for (;;) {
      fp_ws(p);
      if (p->pos >= p->len || p->b[p->pos] != '"') return 0;
      Slice k;
      if (!fp_string(p, &k)) return 0;
      fp_ws(p);
      if (p->pos >= p->len || p->b[p->pos] != ':') return 0;
      p->pos++;
      fp_ws(p);
      if (sl_eq_c(k, "path")) {
        if (p->pos >= p->len || p->b[p->pos] != '"') return 0;
        if (!fp_string(p, &path)) return 0;
        has_path = 1;
      } else if (sl_eq_c(k, "content")) {
        if (p->pos >= p->len || p->b[p->pos] != '"') return 0;
        if (!fp_string(p, &content)) return 0;
        has_content = 1;
      } else {
        if (!fp_skip_value(p, 0)) return 0;
      }
      fp_ws(p);
      if (p->pos >= p->len) return 0;
      if (p->b[p->pos] == ',') { p->pos++; continue; }
      if (p->b[p->pos] == '}') { p->pos++; break; }
      return 0;
    }
    if (!has_path || !has_content) return 0;
    if (n == cap) {
      size_t nc = cap * 2;
      InFile *na = (InFile *)arena_alloc(p->arena, nc * sizeof(InFile));
      memcpy(na, arr, n * sizeof(InFile));
      arr = na; cap = nc;
    }
    arr[n].path = path; arr[n].content = content; n++;
    fp_ws(p);
    if (p->pos >= p->len) return 0;
    if (p->b[p->pos] == ',') { p->pos++; continue; }
    if (p->b[p->pos] == ']') { p->pos++; *out = arr; *nout = n; return 1; }
    return 0;
  }
}

static int fp_parse_case_input(JParser *p, CaseIn *c) {
  // Consumes the input object's own opening '{' (the caller only
  // consumed the enclosing case object's '{', if any).
  memset(c, 0, sizeof(*c));
  if (p->pos >= p->len || p->b[p->pos] != '{') return 0;
  p->pos++;
  fp_ws(p);
  if (p->pos < p->len && p->b[p->pos] == '}') { p->pos++; return 1; }
  for (;;) {
    fp_ws(p);
    if (p->pos >= p->len || p->b[p->pos] != '"') return 0;
    Slice k;
    if (!fp_string(p, &k)) return 0;
    fp_ws(p);
    if (p->pos >= p->len || p->b[p->pos] != ':') return 0;
    p->pos++;
    fp_ws(p);
    if (sl_eq_c(k, "site")) {
      if (!fp_parse_site(p, &c->site)) return 0;
    } else if (sl_eq_c(k, "templates")) {
      if (!fp_parse_templates(p, &c->tpls, &c->ntpls)) return 0;
    } else if (sl_eq_c(k, "files")) {
      if (!fp_parse_files(p, &c->files, &c->nfiles)) return 0;
    } else {
      if (!fp_skip_value(p, 0)) return 0;
    }
    fp_ws(p);
    if (p->pos >= p->len) return 0;
    if (p->b[p->pos] == ',') { p->pos++; continue; }
    if (p->b[p->pos] == '}') { p->pos++; return 1; }
    return 0;
  }
}

static int fp_parse_cases(const char *json, size_t jlen, Arena *arena, CaseVec *out) {
  JParser pv = { (const u8 *)json, jlen, 0, arena };
  JParser *p = &pv;
  if (p->pos >= p->len || p->b[p->pos] != '[') return 0;
  p->pos++;
  size_t cap = 4, n = 0;
  CaseIn *arr = (CaseIn *)arena_alloc(arena, cap * sizeof(CaseIn));
  Slice *ids = (Slice *)arena_alloc(arena, cap * sizeof(Slice));
  fp_ws(p);
  if (p->pos < p->len && p->b[p->pos] == ']') { p->pos++; out->items = arr; out->ids = ids; out->n = 0; return 1; }
  for (;;) {
    fp_ws(p);
    if (p->pos >= p->len || p->b[p->pos] != '{') return 0;
    p->pos++;
    CaseIn ci;
    memset(&ci, 0, sizeof(ci));
    Slice id = slc("");
    int has_input = 0, has_id = 0;
    fp_ws(p);
    if (p->pos < p->len && p->b[p->pos] == '}') { p->pos++; return 0; }
    for (;;) {
      fp_ws(p);
      if (p->pos >= p->len || p->b[p->pos] != '"') return 0;
      Slice k;
      if (!fp_string(p, &k)) return 0;
      fp_ws(p);
      if (p->pos >= p->len || p->b[p->pos] != ':') return 0;
      p->pos++;
      fp_ws(p);
      if (sl_eq_c(k, "input")) {
        if (!fp_parse_case_input(p, &ci)) return 0;
        has_input = 1;
      } else if (sl_eq_c(k, "id")) {
        if (!fp_string(p, &id)) return 0;
        has_id = 1;
      } else {
        if (!fp_skip_value(p, 0)) return 0;
      }
      fp_ws(p);
      if (p->pos >= p->len) return 0;
      if (p->b[p->pos] == ',') { p->pos++; continue; }
      if (p->b[p->pos] == '}') { p->pos++; break; }
      return 0;
    }
    if (!has_input || !has_id) return 0;
    if (n == cap) {
      size_t nc = cap * 2;
      CaseIn *na = (CaseIn *)arena_alloc(arena, nc * sizeof(CaseIn));
      memcpy(na, arr, n * sizeof(CaseIn));
      arr = na;
      Slice *ni = (Slice *)arena_alloc(arena, nc * sizeof(Slice));
      memcpy(ni, ids, n * sizeof(Slice));
      ids = ni; cap = nc;
    }
    arr[n] = ci;
    ids[n] = id;
    n++;
    fp_ws(p);
    if (p->pos >= p->len) return 0;
    if (p->b[p->pos] == ',') { p->pos++; continue; }
    if (p->b[p->pos] == ']') { p->pos++; out->items = arr; out->ids = ids; out->n = n; return 1; }
    return 0;
  }
}

// ------------------------------------------------------------------ v6 flat values
// No Val tree: template values are small by-value registers (RVal)
// pointing at precomputed flat records. A "card" is just the source
// record itself; a "tag" is the Tag struct; the page object is the
// record. Nothing is allocated per card/page render.

// Field codes (resolved at template compile time).
enum {
  F_TITLE = 1, F_URL, F_DATE, F_DATE_LONG, F_SUMMARY, F_EXCERPT,
  F_READING_MINUTES, F_TAGS, F_SLUG, F_CONTENT, F_TOC, F_NEWER, F_OLDER,
  F_IS_POST, F_NAME, F_COUNT, F_POSTS, F_LEVEL, F_ID, F_HTML,
  F_NUMBER, F_TOTAL, F_NEWER_URL, F_OLDER_URL, F_LENGTH,
  F_INDEX, F_FIRST, F_LAST, F_BASE_URL
};
// Compile-time value types.
enum { T_UNK, T_STR, T_NUM, T_BOOL, T_ARR_REC, T_ARR_TAG, T_ARR_TAGREF, T_ARR_TOC, T_REC, T_TAG, T_TAGREF, T_TOC, T_PAGER, T_SITE, T_LOOP };
// Resolved root kinds (Instr.rkind).
enum { RK_NOTFOUND = 0, RK_DYN, RK_SITE, RK_TAGS, RK_RECENT, RK_PAGE, RK_POSTS, RK_PAGER, RK_TAG, RK_LOOP, RK_LOOPVAR };
// Runtime value kinds (RVal.kind).
enum { RV_UNDEF = 0, RV_NULL, RV_BOOL, RV_NUM, RV_STR, RV_ARR, RV_REC, RV_TAG, RV_TAGREF, RV_TOC, RV_PAGER, RV_SITE, RV_LOOP };
// Array element kinds (RVal.elem for RV_ARR).
enum { AE_REC_IDX = 1, AE_TAG, AE_TAG_PTR, AE_TAGREF, AE_TOC };

typedef struct {
  u8 kind;
  Slice s;        // RV_STR
  double num;     // RV_NUM
  int boolean;    // RV_BOOL
  const void *ptr; // RV_REC/TAG/TAGREF/TOC/PAGER/SITE/LOOP, or RV_ARR items
  size_t len;     // RV_ARR length
  u8 elem;        // RV_ARR element kind
} RVal;

typedef struct { Slice name, slug, url; } TagRef;

typedef struct {
  Slice path;
  FmData data;
  Slice body;      // after front-matter strip
  int post;
  Slice layout;
  Slice slug, out_dir, url;
  Slice html;      // markdown html (malloc'd buffer; leaked per case)
  TocEntry *toc; size_t ntoc;
  Slice excerpt;
  Slice date_long; // precomputed ("" when no date)
  double reading_minutes; // precomputed
  int newer, older; // source indices, -1 = none
  TagRef *tag_refs; size_t ntag_refs, captag_refs;
} CRec;

typedef struct {
  Slice name, slug, url;
  size_t *posts; size_t nposts, capposts; // source indices
} Tag;

typedef struct { double number, total; Slice url, newer_url, older_url; } Pager;

// (DynBind moved up: BuildCtx owns the renderer's loop/frame stacks.)
typedef struct { Slice name; RVal val; } DynBind;

typedef struct LoopState LoopState;
struct LoopState {
  RVal arr;          // the array being iterated (kind RV_ARR)
  size_t idx;        // current index
  size_t len;        // cached = arr.len
  Slice var_name;
  u32 saved_shadow;  // shadow mask before this loop
};

// v9 C3: DYN-slot resolution memo. A dynamic slot (RK_DYN — a free name
// in an include fragment, e.g. `p.title` where `p` is the caller's loop
// variable) is re-resolved by string comparison on EVERY evaluation
// (22,222 evals/run). The resolution is a pure function of (segs,
// frame-stack shape), and the shape is stable across items in a loop,
// so memoize it: on a hit the value is fetched directly (fast Slice
// read for single-step string fields, else walk_chain with the cached
// field codes) with no string work at all. The memo lives in BuildCtx
// (fresh per case); entries store the full shape for verification, so
// a hash collision can only cost a re-resolve, never a wrong value.
#define DYN_MEMO_N 64
#define DYN_MEMO_MAXFRAMES 8
typedef struct {
  int valid;
  const Slice *segs; size_t nsegs;
  u8 nframes; u8 nloops;
  const u8 *fname_p[DYN_MEMO_MAXFRAMES]; size_t fname_n[DYN_MEMO_MAXFRAMES];
  u8 ftype[DYN_MEMO_MAXFRAMES];
  u8 rkind; i16 rdepth; // absolute frame index for RK_LOOPVAR, -1 otherwise
  u8 chain[6]; u8 nchain;
  u8 fast; u8 base; u16 off;
  u8 notfound;
} DynMemo;

typedef struct {
  Arena *arena;
  Slice *tpl_names; Tpl *tpls; int *tpl_ok; size_t ntpls;
  StrMap *slug_cache, *date_cache, *card_cache;
  CRec *sources; size_t nsources, capsources;
  struct { Slice path, content; } *statics; size_t nstatics, capstatics;
  StrMap *by_source, *static_paths, *broken;
  Slice base;        // base_url with trailing '/' trimmed
  Buf *out;
  int first;         // emit comma tracking
  Slice *report; size_t nreport, capreport;
  Buf page_buf;      // reusable per-page render buffer
  LoopState *loops; size_t loops_cap;   // renderer loop stack, reused across pages (v8)
  DynBind *frames; size_t frames_cap;   // renderer frame stack, reused across pages (v8)
  Buf scratch;                          // renderer scratch buffer, reused across pages (v8)
  SiteIn *site;
  RVal tags_rv, recent_rv; // per-case globals (arrays)
  DynMemo dyn_memo[DYN_MEMO_N]; // v9 C3: DYN-slot resolution memo
} BuildCtx;

static int field_code_for(Slice seg) {
  // Linear over the field names (only used for DYN paths and at
  // compile time — never in the hot render loop).
  static const struct { const char *n; int f; } tab[] = {
    {"title", F_TITLE}, {"url", F_URL}, {"date", F_DATE}, {"date_long", F_DATE_LONG},
    {"summary", F_SUMMARY}, {"excerpt", F_EXCERPT}, {"reading_minutes", F_READING_MINUTES},
    {"tags", F_TAGS}, {"slug", F_SLUG}, {"content", F_CONTENT}, {"toc", F_TOC},
    {"newer", F_NEWER}, {"older", F_OLDER}, {"is_post", F_IS_POST}, {"name", F_NAME},
    {"count", F_COUNT}, {"posts", F_POSTS}, {"level", F_LEVEL}, {"id", F_ID},
    {"html", F_HTML}, {"number", F_NUMBER}, {"total", F_TOTAL},
    {"newer_url", F_NEWER_URL}, {"older_url", F_OLDER_URL}, {"length", F_LENGTH},
    {"index", F_INDEX}, {"first", F_FIRST}, {"last", F_LAST}, {"base_url", F_BASE_URL},
  };
  for (size_t i = 0; i < sizeof(tab) / sizeof(tab[0]); i++)
    if (sl_eq_c(seg, tab[i].n)) return tab[i].f;
  return 0;
}

// One field step on a runtime value. Returns 0 when the field does
// not exist on that value (v5's obj_get returning 0).
static int rval_step(BuildCtx *ctx, RVal v, int f, RVal *out) {
  memset(out, 0, sizeof(*out));
  switch (v.kind) {
    case RV_REC: {
      CRec *s = (CRec *)v.ptr;
      switch (f) {
        case F_TITLE: out->kind = RV_STR; out->s = s->data.title; return 1;
        case F_URL: out->kind = RV_STR; out->s = s->url; return 1;
        case F_DATE: out->kind = RV_STR; out->s = s->data.has_date ? s->data.date : slc(""); return 1;
        case F_DATE_LONG: out->kind = RV_STR; out->s = s->date_long; return 1;
        case F_SUMMARY: out->kind = RV_STR; out->s = s->data.has_summary ? s->data.summary : slc(""); return 1;
        case F_EXCERPT: out->kind = RV_STR; out->s = s->excerpt; return 1;
        case F_READING_MINUTES: out->kind = RV_NUM; out->num = s->reading_minutes; return 1;
        case F_TAGS: out->kind = RV_ARR; out->ptr = s->tag_refs; out->len = s->ntag_refs; out->elem = AE_TAGREF; return 1;
        case F_SLUG: out->kind = RV_STR; out->s = s->slug; return 1;
        case F_CONTENT: out->kind = RV_STR; out->s = s->html; return 1;
        case F_TOC: out->kind = RV_ARR; out->ptr = s->toc; out->len = s->ntoc; out->elem = AE_TOC; return 1;
        case F_NEWER:
          if (s->newer < 0) { out->kind = RV_NULL; return 1; }
          out->kind = RV_REC; out->ptr = &ctx->sources[s->newer]; return 1;
        case F_OLDER:
          if (s->older < 0) { out->kind = RV_NULL; return 1; }
          out->kind = RV_REC; out->ptr = &ctx->sources[s->older]; return 1;
        case F_IS_POST: out->kind = RV_BOOL; out->boolean = s->post; return 1;
        default: return 0;
      }
    }
    case RV_TAG: {
      Tag *t = (Tag *)v.ptr;
      switch (f) {
        case F_NAME: out->kind = RV_STR; out->s = t->name; return 1;
        case F_SLUG: out->kind = RV_STR; out->s = t->slug; return 1;
        case F_URL: out->kind = RV_STR; out->s = t->url; return 1;
        case F_COUNT: out->kind = RV_NUM; out->num = (double)t->nposts; return 1;
        case F_POSTS: out->kind = RV_ARR; out->ptr = t->posts; out->len = t->nposts; out->elem = AE_REC_IDX; return 1;
        default: return 0;
      }
    }
    case RV_TAGREF: {
      TagRef *t = (TagRef *)v.ptr;
      switch (f) {
        case F_NAME: out->kind = RV_STR; out->s = t->name; return 1;
        case F_SLUG: out->kind = RV_STR; out->s = t->slug; return 1;
        case F_URL: out->kind = RV_STR; out->s = t->url; return 1;
        default: return 0;
      }
    }
    case RV_TOC: {
      TocEntry *t = (TocEntry *)v.ptr;
      switch (f) {
        case F_LEVEL: out->kind = RV_NUM; out->num = (double)t->level; return 1;
        case F_ID: out->kind = RV_STR; out->s = t->id; return 1;
        case F_HTML: out->kind = RV_STR; out->s = t->html; return 1;
        default: return 0;
      }
    }
    case RV_PAGER: {
      Pager *pg = (Pager *)v.ptr;
      switch (f) {
        case F_NUMBER: out->kind = RV_NUM; out->num = pg->number; return 1;
        case F_TOTAL: out->kind = RV_NUM; out->num = pg->total; return 1;
        case F_URL: out->kind = RV_STR; out->s = pg->url; return 1;
        case F_NEWER_URL: out->kind = RV_STR; out->s = pg->newer_url; return 1;
        case F_OLDER_URL: out->kind = RV_STR; out->s = pg->older_url; return 1;
        default: return 0;
      }
    }
    case RV_SITE: {
      BuildCtx *c = (BuildCtx *)v.ptr;
      switch (f) {
        case F_TITLE: out->kind = RV_STR; out->s = c->site->title; return 1;
        case F_BASE_URL: out->kind = RV_STR; out->s = c->base; return 1;
        default: return 0;
      }
    }
    case RV_ARR:
      if (f == F_LENGTH) { out->kind = RV_NUM; out->num = (double)v.len; return 1; }
      return 0;
    case RV_LOOP: {
      LoopState *ls = (LoopState *)v.ptr;
      switch (f) {
        case F_INDEX: out->kind = RV_NUM; out->num = (double)ls->idx + 1.0; return 1;
        case F_FIRST: out->kind = RV_BOOL; out->boolean = (ls->idx == 0); return 1;
        case F_LAST: out->kind = RV_BOOL; out->boolean = (ls->idx + 1 >= ls->len); return 1;
        case F_LENGTH: out->kind = RV_NUM; out->num = (double)ls->len; return 1;
        default: return 0;
      }
    }
    default: return 0;
  }
}

// The item of an array RVal at index idx.
static int arr_item(BuildCtx *ctx, RVal *arr, size_t idx, RVal *out) {
  if (arr->kind != RV_ARR || idx >= arr->len) return 0;
  memset(out, 0, sizeof(*out));
  switch (arr->elem) {
    case AE_REC_IDX: {
      size_t si = ((const size_t *)arr->ptr)[idx];
      out->kind = RV_REC; out->ptr = &ctx->sources[si];
      return 1;
    }
    case AE_TAG:
      out->kind = RV_TAG; out->ptr = &((const Tag *)arr->ptr)[idx];
      return 1;
    case AE_TAG_PTR:
      out->kind = RV_TAG; out->ptr = ((Tag *const *)arr->ptr)[idx];
      return 1;
    case AE_TAGREF:
      out->kind = RV_TAGREF; out->ptr = &((const TagRef *)arr->ptr)[idx];
      return 1;
    case AE_TOC:
      out->kind = RV_TOC; out->ptr = &((const TocEntry *)arr->ptr)[idx];
      return 1;
    default: return 0;
  }
}

// ------------------------------------------------------------------ v6 slot compiler
// Constant folding: every template path is resolved at compile time to
// a root kind + a chain of field codes, so the hot render loop does no
// string comparison and no frame search for the common paths. Anything
// that cannot be resolved statically stays RK_DYN and is looked up at
// runtime (free names of include fragments, loop variables of unknown
// item type, over-long paths).

typedef struct { Slice name; int item_type; } CLoopVar;

// Map one field step from a current compile-time type. Returns 0 when
// the field does not exist on that type (the whole path is then
// NOTFOUND, exactly like v5's lookup returning 0).
static int field_step_type(int type, Slice seg, int *field_out, int *next_out) {
  int f = field_code_for(seg);
  if (!f) return 0;
  switch (type) {
    case T_REC:
      switch (f) {
        case F_TITLE: case F_URL: case F_DATE: case F_DATE_LONG:
        case F_SUMMARY: case F_EXCERPT: case F_SLUG: case F_CONTENT:
          *field_out = f; *next_out = T_STR; return 1;
        case F_READING_MINUTES: *field_out = f; *next_out = T_NUM; return 1;
        case F_TAGS: *field_out = f; *next_out = T_ARR_TAGREF; return 1;
        case F_TOC: *field_out = f; *next_out = T_ARR_TOC; return 1;
        case F_NEWER: case F_OLDER: *field_out = f; *next_out = T_REC; return 1;
        case F_IS_POST: *field_out = f; *next_out = T_BOOL; return 1;
        default: return 0;
      }
    case T_TAG:
      switch (f) {
        case F_NAME: case F_SLUG: case F_URL: *field_out = f; *next_out = T_STR; return 1;
        case F_COUNT: *field_out = f; *next_out = T_NUM; return 1;
        case F_POSTS: *field_out = f; *next_out = T_ARR_REC; return 1;
        default: return 0;
      }
    case T_TAGREF:
      switch (f) {
        case F_NAME: case F_SLUG: case F_URL: *field_out = f; *next_out = T_STR; return 1;
        default: return 0;
      }
    case T_TOC:
      switch (f) {
        case F_LEVEL: *field_out = f; *next_out = T_NUM; return 1;
        case F_ID: case F_HTML: *field_out = f; *next_out = T_STR; return 1;
        default: return 0;
      }
    case T_PAGER:
      switch (f) {
        case F_NUMBER: case F_TOTAL: *field_out = f; *next_out = T_NUM; return 1;
        case F_URL: case F_NEWER_URL: case F_OLDER_URL: *field_out = f; *next_out = T_STR; return 1;
        default: return 0;
      }
    case T_SITE:
      switch (f) {
        case F_TITLE: case F_BASE_URL: *field_out = f; *next_out = T_STR; return 1;
        default: return 0;
      }
    case T_LOOP:
      switch (f) {
        case F_INDEX: case F_LENGTH: *field_out = f; *next_out = T_NUM; return 1;
        case F_FIRST: case F_LAST: *field_out = f; *next_out = T_BOOL; return 1;
        default: return 0;
      }
    case T_ARR_REC: case T_ARR_TAG: case T_ARR_TAGREF: case T_ARR_TOC:
      if (f == F_LENGTH) { *field_out = f; *next_out = T_NUM; return 1; }
      return 0;
    default: return 0;
  }
}

// Resolve a path's root (first segment) against the compile-time loop
// stack and the fixed roots. Returns the root's value type.
static int resolve_root(Slice s0, CLoopVar *lv, size_t nlv, u8 *rk, i16 *rdepth) {
  if (sl_eq_c(s0, "loop")) { *rk = RK_LOOP; *rdepth = 0; return T_LOOP; }
  for (size_t i = nlv; i-- > 0;) {
    if (sl_eq(lv[i].name, s0)) { *rk = RK_LOOPVAR; *rdepth = (i16)i; return lv[i].item_type; }
  }
  if (sl_eq_c(s0, "site")) { *rk = RK_SITE; return T_SITE; }
  if (sl_eq_c(s0, "tags")) { *rk = RK_TAGS; return T_ARR_TAG; }
  if (sl_eq_c(s0, "recent")) { *rk = RK_RECENT; return T_ARR_REC; }
  if (sl_eq_c(s0, "page")) { *rk = RK_PAGE; return T_REC; }
  if (sl_eq_c(s0, "posts")) { *rk = RK_POSTS; return T_ARR_REC; }
  if (sl_eq_c(s0, "pager")) { *rk = RK_PAGER; return T_PAGER; }
  if (sl_eq_c(s0, "tag")) { *rk = RK_TAG; return T_TAG; }
  *rk = RK_DYN; return T_UNK;
}

// Resolve one path into an instruction's slot fields. Returns the value
// type of the resolved path (T_UNK for DYN/NOTFOUND).
static int resolve_path_slots(Slice *segs, size_t nsegs, CLoopVar *lv, size_t nlv, Instr *in) {
  in->rkind = RK_NOTFOUND; in->rdepth = 0; in->nchain = 0;
  if (nsegs == 0) return T_UNK;
  u8 rk; i16 rd = 0;
  int type = resolve_root(segs[0], lv, nlv, &rk, &rd);
  if (rk == RK_DYN) { in->rkind = RK_DYN; return T_UNK; }
  in->rkind = rk; in->rdepth = rd;
  for (size_t i = 1; i < nsegs; i++) {
    int f, nx;
    if (type == T_UNK || !field_step_type(type, segs[i], &f, &nx)) {
      // Unresolvable tail: a loop variable of unknown item type stays
      // dynamic; anything else is a path that does not exist.
      if (rk == RK_LOOPVAR && type == T_UNK) { in->rkind = RK_DYN; in->nchain = 0; return T_UNK; }
      in->rkind = RK_NOTFOUND; in->nchain = 0; return T_UNK;
    }
    if (in->nchain >= 6) { in->rkind = RK_DYN; in->nchain = 0; return T_UNK; }
    in->chain[in->nchain++] = (u8)f;
    type = nx;
  }
  return type;
}

// Constant-fold every path in a compiled template. Walks the code with
// a compile-time loop-variable stack; OP_FOR pushes its variable (typed
// by its source array), OP_NEXT pops it.
static void resolve_slots(Tpl *t) {
  CLoopVar stack[32];
  size_t depth = 0, nlv = 0;
  for (size_t i = 0; i < t->n; i++) {
    Instr *in = &t->code[i];
    switch (in->op) {
      case OP_VAR: case OP_RAW: case OP_JIF:
        // With a loop stack deeper than the table, no loop variable is
        // statically known — resolve without it (roots/DYN still work).
        resolve_path_slots(in->segs, in->nsegs, stack, depth > 32 ? 0 : nlv, in);
        break;
      case OP_FOR: {
        int type = resolve_path_slots(in->segs, in->nsegs, stack, depth > 32 ? 0 : nlv, in);
        int item = T_UNK;
        if (type == T_ARR_REC) item = T_REC;
        else if (type == T_ARR_TAG) item = T_TAG;
        else if (type == T_ARR_TAGREF) item = T_TAGREF;
        else if (type == T_ARR_TOC) item = T_TOC;
        if (depth < 32) { stack[depth].name = in->name; stack[depth].item_type = item; nlv = depth + 1; }
        depth++;
        break;
      }
      case OP_NEXT:
        if (depth > 0) depth--;
        if (depth < 32) nlv = depth;
        break;
      default: break;
    }
  }
}

// ------------------------------------------------------------------ v6 template runtime

typedef struct { const u8 *ptr; size_t len; } CacheEnt;

typedef struct {
  Tpl *tpls; size_t ntpls;
  StrMap *card_cache;
  Buf *out;           // per-page buffer (reset per render_page call)
  Arena *arena;
  BuildCtx *bctx;
  // Roots bound for this render call (site/tags/recent always bound).
  RVal site, tags, recent, page, posts, pager, tag;
  u32 root_mask;      // bit per RK_* root kind bound
  u32 shadow_mask;    // root names shadowed by active loop vars
  LoopState *loops; size_t nloops, caploops;
  DynBind *frames; size_t nframes, capframes;
  int depth;
  Buf scratch;
} Renderer;

static u32 root_bit_for_name(Slice name) {
  if (sl_eq_c(name, "site")) return 1u << RK_SITE;
  if (sl_eq_c(name, "tags")) return 1u << RK_TAGS;
  if (sl_eq_c(name, "recent")) return 1u << RK_RECENT;
  if (sl_eq_c(name, "page")) return 1u << RK_PAGE;
  if (sl_eq_c(name, "posts")) return 1u << RK_POSTS;
  if (sl_eq_c(name, "pager")) return 1u << RK_PAGER;
  if (sl_eq_c(name, "tag")) return 1u << RK_TAG;
  return 0;
}

static int root_rval(Renderer *r, int rk, RVal *out) {
  if (!(r->root_mask & (1u << rk))) return 0;
  switch (rk) {
    case RK_SITE: *out = r->site; return 1;
    case RK_TAGS: *out = r->tags; return 1;
    case RK_RECENT: *out = r->recent; return 1;
    case RK_PAGE: *out = r->page; return 1;
    case RK_POSTS: *out = r->posts; return 1;
    case RK_PAGER: *out = r->pager; return 1;
    case RK_TAG: *out = r->tag; return 1;
    default: return 0;
  }
}

static int walk_chain(Renderer *r, RVal v, const u8 *chain, size_t nchain, RVal *out) {
  for (size_t i = 0; i < nchain; i++) {
    RVal nx;
    if (!rval_step(r->bctx, v, chain[i], &nx)) return 0;
    v = nx;
  }
  *out = v;
  return 1;
}

// Dynamic path evaluation (RK_DYN, and shadowed roots): full frame
// search like v5's lookup, then field steps by name.
static int dyn_eval_memo(Renderer *r, Slice *segs, size_t nsegs, RVal *out); // v9 C3, defined after slot_fast_map
static int dyn_eval(Renderer *r, Slice *segs, size_t nsegs, RVal *out) {
  if (nsegs == 0) return 0;
  RVal v;
  if (sl_eq_c(segs[0], "loop")) {
    if (r->nloops == 0) return 0;
    memset(&v, 0, sizeof(v));
    v.kind = RV_LOOP; v.ptr = &r->loops[r->nloops - 1];
  } else {
    int found = 0;
    for (size_t i = r->nframes; i-- > 0;) {
      if (sl_eq(r->frames[i].name, segs[0])) { v = r->frames[i].val; found = 1; break; }
    }
    if (!found) {
      u8 rk = 0;
      if (sl_eq_c(segs[0], "site")) rk = RK_SITE;
      else if (sl_eq_c(segs[0], "tags")) rk = RK_TAGS;
      else if (sl_eq_c(segs[0], "recent")) rk = RK_RECENT;
      else if (sl_eq_c(segs[0], "page")) rk = RK_PAGE;
      else if (sl_eq_c(segs[0], "posts")) rk = RK_POSTS;
      else if (sl_eq_c(segs[0], "pager")) rk = RK_PAGER;
      else if (sl_eq_c(segs[0], "tag")) rk = RK_TAG;
      else return 0;
      if (!root_rval(r, rk, &v)) return 0;
    }
  }
  for (size_t i = 1; i < nsegs; i++) {
    int f = field_code_for(segs[i]);
    if (!f) return 0;
    RVal nx;
    if (!rval_step(r->bctx, v, f, &nx)) return 0;
    v = nx;
  }
  *out = v;
  return 1;
}

// Evaluate a resolved slot to an RVal. Returns 0 when the path does
// not exist (renders as nothing / falsy, like v5's lookup returning 0).
// The slot fields are passed explicitly so the chunked renderer (v7)
// can share this path without an Instr.
static int eval_slot_f(Renderer *r, Slice *segs, size_t nsegs, u8 rkind, i16 rdepth,
                       const u8 *chain, size_t nchain, size_t loop_base, RVal *out) {
#ifdef SENJA_PROFILE
  prof_neval++;
#endif
  switch (rkind) {
    case RK_NOTFOUND: return 0;
    case RK_DYN: return dyn_eval_memo(r, segs, nsegs, out); // v9 C3: memoized resolution
    case RK_LOOP: {
      if (r->nloops == 0) return 0;
      RVal v; memset(&v, 0, sizeof(v));
      v.kind = RV_LOOP; v.ptr = &r->loops[r->nloops - 1];
      return walk_chain(r, v, chain, nchain, out);
    }
    case RK_LOOPVAR: {
      size_t idx = loop_base + (size_t)rdepth;
      if (idx >= r->nloops) return 0;
      // The dyn frame at the same index holds the current item.
      RVal v = r->frames[idx].val;
      return walk_chain(r, v, chain, nchain, out);
    }
    default: {
      // A statically resolved root: if a same-named loop variable
      // shadows it (only possible via a caller's loop in a fragment),
      // fall back to the dynamic path.
      if (r->shadow_mask & (1u << rkind)) return dyn_eval(r, segs, nsegs, out);
      RVal v;
      if (!root_rval(r, rkind, &v)) return 0;
      return walk_chain(r, v, chain, nchain, out);
    }
  }
}

// JavaScript's String(v) for template values, written into `out`.
static void write_rval(Buf *out, BuildCtx *ctx, RVal *v) {
  switch (v->kind) {
    case RV_STR: buf_sl(out, v->s); break;
    case RV_NUM: js_num(out, v->num); break;
    case RV_BOOL: buf_cs(out, v->boolean ? "true" : "false"); break;
    case RV_ARR: {
      for (size_t i = 0; i < v->len; i++) {
        if (i > 0) buf_cs(out, ", ");
        RVal item;
        if (arr_item(ctx, v, i, &item)) write_rval(out, ctx, &item);
      }
      break;
    }
    default: break; // objects, null, undef -> ""
  }
}


// v11: JSON-mode write_rval — same as write_rval, but string items are
// JSON-escaped inline (no quotes) instead of copied raw. Used by the
// fused renderer for raw slots whose value is not a plain string.
static void write_rval_j(Buf *out, BuildCtx *ctx, RVal *v) {
  switch (v->kind) {
    case RV_STR: json_write_raw(out, v->s); break;
    case RV_NUM: js_num(out, v->num); break;
    case RV_BOOL: buf_cs(out, v->boolean ? "true" : "false"); break;
    case RV_ARR: {
      for (size_t i = 0; i < v->len; i++) {
        if (i > 0) buf_cs(out, ", ");
        RVal item;
        if (arr_item(ctx, v, i, &item)) write_rval_j(out, ctx, &item);
      }
      break;
    }
    default: break; // objects, null, undef -> ""
  }
}

static int rval_truthy(RVal *v) {
  switch (v->kind) {
    case RV_NUM: return v->num != 0.0;
    case RV_ARR: return v->len > 0;
    case RV_STR: return v->s.n > 0;
    case RV_BOOL: return v->boolean;
    case RV_NULL: case RV_UNDEF: return 0;
    default: return 1; // objects are truthy
  }
}

// ------------------------------------------------------------------ v7 chunked renderer
// A template compiles to a flat Chunk array (built once per case from
// the Instr array). Rendering walks chunks with computed-goto dispatch
// (one indirect jump per chunk, no switch range check, no loop
// back-edge) instead of the v5/v6 switch VM. Slots that resolve to a
// single plain Slice field of a known record carry a precomputed byte
// offset: the hot path is one masked check + one load + esc/memcpy,
// with no eval_slot / walk_chain / rval_step calls. Include fragments
// that only read shared roots are inlined into the caller's chunk
// stream at build time (no call, no card-cache lookup per render).

typedef enum { CK_TEXT, CK_SLOT, CK_IF, CK_JMP, CK_FOR, CK_NEXT, CK_INC, CK_END } ChunkKind;

// Fast-path base records (offsets are precomputed at build time).
enum { CB_NONE = 0, CB_PAGE, CB_LOOPVAR, CB_TAG, CB_PAGER, CB_SITE_TITLE, CB_SITE_BASE };

struct Chunk {
  u8 kind; u8 raw; u8 rkind; u8 fast; u8 base; u8 nchain; u8 chain[6];
  i16 rdepth; u16 off;
  int target; int tpl; int cacheable; int shared_only; int not; int body;
  Slice text; Slice jtext; Slice name; Slice *segs; size_t nsegs;
};

// Map a single-step slice field to (base, offset). Returns 1 when the
// (root kind, item type, field) triple is a plain Slice field of a
// fixed-layout record; 0 otherwise (generic path).
static int slot_fast_map(u8 rkind, int item_type, u8 field, u8 *base_out, u16 *off_out) {
  // CRec fields.
  if ((rkind == RK_PAGE) || (rkind == RK_LOOPVAR && item_type == T_REC)) {
    u8 base = (rkind == RK_PAGE) ? CB_PAGE : CB_LOOPVAR;
    size_t off = 0;
    switch (field) {
      case F_TITLE: off = offsetof(CRec, data) + offsetof(FmData, title); break;
      case F_URL: off = offsetof(CRec, url); break;
      case F_DATE: off = offsetof(CRec, data) + offsetof(FmData, date); break;
      case F_DATE_LONG: off = offsetof(CRec, date_long); break;
      case F_SUMMARY: off = offsetof(CRec, data) + offsetof(FmData, summary); break;
      case F_EXCERPT: off = offsetof(CRec, excerpt); break;
      case F_SLUG: off = offsetof(CRec, slug); break;
      case F_CONTENT: off = offsetof(CRec, html); break;
      default: return 0;
    }
    *base_out = base; *off_out = (u16)off; return 1;
  }
  // Tag / TagRef fields (identical layout: name, slug, url).
  if ((rkind == RK_TAG) || (rkind == RK_LOOPVAR && item_type == T_TAG)
      || (rkind == RK_LOOPVAR && item_type == T_TAGREF)) {
    u8 base = (rkind == RK_TAG) ? CB_TAG : CB_LOOPVAR;
    size_t off = 0;
    switch (field) {
      case F_NAME: off = offsetof(Tag, name); break;
      case F_SLUG: off = offsetof(Tag, slug); break;
      case F_URL: off = offsetof(Tag, url); break;
      default: return 0;
    }
    *base_out = base; *off_out = (u16)off; return 1;
  }
  // TocEntry fields.
  if (rkind == RK_LOOPVAR && item_type == T_TOC) {
    size_t off = 0;
    switch (field) {
      case F_ID: off = offsetof(TocEntry, id); break;
      case F_HTML: off = offsetof(TocEntry, html); break;
      default: return 0;
    }
    *base_out = CB_LOOPVAR; *off_out = (u16)off; return 1;
  }
  // Pager fields.
  if (rkind == RK_PAGER) {
    size_t off = 0;
    switch (field) {
      case F_URL: off = offsetof(Pager, url); break;
      case F_NEWER_URL: off = offsetof(Pager, newer_url); break;
      case F_OLDER_URL: off = offsetof(Pager, older_url); break;
      default: return 0;
    }
    *base_out = CB_PAGER; *off_out = (u16)off; return 1;
  }
  // Site fields: title lives in SiteIn, base_url in BuildCtx.base.
  if (rkind == RK_SITE) {
    if (field == F_TITLE) { *base_out = CB_SITE_TITLE; *off_out = (u16)offsetof(SiteIn, title); return 1; }
    if (field == F_BASE_URL) { *base_out = CB_SITE_BASE; *off_out = 0; return 1; }
    return 0;
  }
  return 0;
}

// Try to give a slot chunk its fast offset path. Only single-step
// string slots qualify; everything else keeps the generic eval path.
static void chunk_try_fast(Chunk *c, Instr *in, CLoopVar *lv, size_t nlv) {
  c->fast = 0; c->base = CB_NONE; c->off = 0;
  if (in->nchain != 1) return;
  int item_type = T_UNK;
  if (in->rkind == RK_LOOPVAR) {
    if ((size_t)in->rdepth >= nlv) return;
    item_type = lv[in->rdepth].item_type;
  }
  u8 base; u16 off;
  if (slot_fast_map(in->rkind, item_type, in->chain[0], &base, &off)) {
    c->fast = 1; c->base = base; c->off = off;
  }
}

// 1:1 conversion of a compiled template's Instr array to chunks.
static void build_chunks(Tpl *t, Arena *a) {
  Chunk *ch = (Chunk *)arena_alloc(a, (t->n ? t->n : 1) * sizeof(Chunk));
  CLoopVar stack[32];
  size_t depth = 0, nlv = 0;
  for (size_t i = 0; i < t->n; i++) {
    Instr *in = &t->code[i];
    Chunk *c = &ch[i];
    memset(c, 0, sizeof(*c));
    c->segs = in->segs; c->nsegs = in->nsegs;
    c->rkind = in->rkind; c->rdepth = in->rdepth;
    c->nchain = in->nchain;
    memcpy(c->chain, in->chain, sizeof(c->chain));
    switch (in->op) {
      case OP_TEXT: {
        c->kind = CK_TEXT; c->text = in->text;
        // v11: JSON-escape the static text ONCE at compile time. The
        // fused renderer emits pages straight into answers.json, so
        // static text is emitted as a plain memcpy of its escaped form
        // instead of being re-scanned by json_write_str on every page.
        Buf jb = {0};
        json_write_raw(&jb, in->text);
        c->jtext = arena_slice(a, jb.p, jb.len);
        free(jb.p);
        break;
      }
      case OP_VAR:
        c->kind = CK_SLOT; c->raw = 0;
        chunk_try_fast(c, in, stack, depth > 32 ? 0 : nlv);
        break;
      case OP_RAW:
        c->kind = CK_SLOT; c->raw = 1;
        chunk_try_fast(c, in, stack, depth > 32 ? 0 : nlv);
        break;
      case OP_JIF: c->kind = CK_IF; c->not = in->not; c->target = in->target; break;
      case OP_JMP: c->kind = CK_JMP; c->target = in->target; break;
      case OP_FOR: {
        c->kind = CK_FOR; c->name = in->name; c->target = in->target;
        Instr dummy; memset(&dummy, 0, sizeof(dummy));
        int type = resolve_path_slots(in->segs, in->nsegs, stack, depth > 32 ? 0 : nlv, &dummy);
        int item = T_UNK;
        if (type == T_ARR_REC) item = T_REC;
        else if (type == T_ARR_TAG) item = T_TAG;
        else if (type == T_ARR_TAGREF) item = T_TAGREF;
        else if (type == T_ARR_TOC) item = T_TOC;
        if (depth < 32) { stack[depth].name = in->name; stack[depth].item_type = item; nlv = depth + 1; }
        depth++;
        break;
      }
      case OP_NEXT:
        c->kind = CK_NEXT; c->body = in->body;
        if (depth > 0) depth--;
        if (depth < 32) nlv = depth;
        break;
      case OP_INC:
        c->kind = CK_INC; c->tpl = in->tpl; c->cacheable = in->cacheable;
        c->shared_only = in->shared_only;
        break;
      case OP_END: c->kind = CK_END; break;
    }
  }
  t->chunks = ch; t->nchunks = t->n;
}

// ------------------------------------------------------------------ v9 C3: DYN-slot resolution memo
//
// A dynamic slot's resolution is a pure function of (segs, frame-stack
// shape). Resolve it once per shape (mirroring dyn_eval + the
// compile-time field_step_type rules), memoize it, and fetch the value
// from the cached resolution afterwards. The value fetch always uses
// the CURRENT frames/roots, so a memo hit can never return a stale
// item — only the resolution (which frame, which fields) is cached.
//
// Resolve result: 1 = resolved (entry filled, fetch from it),
// 0 = not found (entry filled with notfound), -1 = unresolvable
// (do not memoize; the caller falls back to dyn_eval).
static int dyn_rval_type(int kind) {
  switch (kind) {
    case RV_REC: return T_REC;
    case RV_TAG: return T_TAG;
    case RV_TAGREF: return T_TAGREF;
    case RV_TOC: return T_TOC;
    case RV_PAGER: return T_PAGER;
    case RV_LOOP: return T_LOOP;
    case RV_SITE: return T_SITE;
    default: return T_UNK;
  }
}

static int dyn_memo_resolve(Renderer *r, Slice *segs, size_t nsegs, DynMemo *m) {
  u8 rk; int type; RVal root; int abs_idx = -1;
  if (sl_eq_c(segs[0], "loop")) {
    if (r->nloops == 0) return 0;
    rk = RK_LOOP; type = T_LOOP;
    memset(&root, 0, sizeof(root)); root.kind = RV_LOOP; root.ptr = &r->loops[r->nloops - 1];
  } else {
    int found = -1;
    for (size_t i = r->nframes; i-- > 0;) if (sl_eq(r->frames[i].name, segs[0])) { found = (int)i; break; }
    if (found >= 0) {
      rk = RK_LOOPVAR; abs_idx = found; root = r->frames[found].val;
      type = dyn_rval_type(root.kind);
      if (type == T_UNK) return -1;
    } else {
      if (sl_eq_c(segs[0], "site")) rk = RK_SITE;
      else if (sl_eq_c(segs[0], "tags")) rk = RK_TAGS;
      else if (sl_eq_c(segs[0], "recent")) rk = RK_RECENT;
      else if (sl_eq_c(segs[0], "page")) rk = RK_PAGE;
      else if (sl_eq_c(segs[0], "posts")) rk = RK_POSTS;
      else if (sl_eq_c(segs[0], "pager")) rk = RK_PAGER;
      else if (sl_eq_c(segs[0], "tag")) rk = RK_TAG;
      else return 0;
      if (!root_rval(r, rk, &root)) return 0;
      switch (rk) {
        case RK_SITE: type = T_SITE; break;
        case RK_TAGS: type = T_ARR_TAG; break;
        case RK_RECENT: case RK_POSTS: type = T_ARR_REC; break;
        case RK_PAGE: type = T_REC; break;
        case RK_PAGER: type = T_PAGER; break;
        case RK_TAG: type = T_TAG; break;
        default: type = T_UNK; break;
      }
    }
  }
  int root_type = type;
  m->rkind = rk; m->rdepth = (i16)abs_idx; m->nchain = 0; m->notfound = 0;
  for (size_t i = 1; i < nsegs; i++) {
    int f, nx;
    if (type == T_UNK || !field_step_type(type, segs[i], &f, &nx)) {
      if (rk == RK_LOOPVAR && type == T_UNK) return -1;
      return 0;
    }
    if (m->nchain >= 6) return -1;
    m->chain[m->nchain++] = (u8)f;
    type = nx;
  }
  m->fast = 0; m->base = CB_NONE; m->off = 0;
  if (m->nchain == 1) {
    u8 base; u16 off;
    if (slot_fast_map(rk, root_type, m->chain[0], &base, &off)) {
      m->fast = 1; m->base = base; m->off = off;
    }
  }
  return 1;
}

static const void *dyn_memo_base(Renderer *r, DynMemo *m) {
  switch (m->base) {
    case CB_LOOPVAR: {
      size_t idx = (size_t)m->rdepth;
      if (idx >= r->nframes) return NULL;
      return r->frames[idx].val.ptr;
    }
    case CB_PAGE: return r->page.ptr;
    case CB_TAG: return r->tag.ptr;
    case CB_PAGER: return r->pager.ptr;
    case CB_SITE_TITLE: return (const void *)r->bctx->site;
    case CB_SITE_BASE: return (const void *)&r->bctx->base;
    default: return NULL;
  }
}

static int dyn_memo_fetch(Renderer *r, DynMemo *m, RVal *out) {
  if (m->notfound) return 0;
  if (m->fast) {
    const void *bp = dyn_memo_base(r, m);
    if (!bp) return 0;
    memset(out, 0, sizeof(*out));
    out->kind = RV_STR;
    out->s = *(const Slice *)((const u8 *)bp + m->off);
    return 1;
  }
  RVal v;
  if (m->rkind == RK_LOOPVAR) {
    size_t idx = (size_t)m->rdepth;
    if (idx >= r->nframes) return 0;
    v = r->frames[idx].val;
  } else if (m->rkind == RK_LOOP) {
    if (r->nloops == 0) return 0;
    memset(&v, 0, sizeof(v)); v.kind = RV_LOOP; v.ptr = &r->loops[r->nloops - 1];
  } else {
    if (!root_rval(r, m->rkind, &v)) return 0;
  }
  return walk_chain(r, v, m->chain, m->nchain, out);
}

static int dyn_eval_memo(Renderer *r, Slice *segs, size_t nsegs, RVal *out) {
  if (nsegs == 0) return 0;
  BuildCtx *ctx = r->bctx;
  if (r->nframes > DYN_MEMO_MAXFRAMES) return dyn_eval(r, segs, nsegs, out);
  // Hash: segs identity + frame-stack shape (name pointers + item types).
  uintptr_t h = (uintptr_t)segs;
  h ^= (uintptr_t)nsegs * 0x9e3779b97f4a7c15ULL;
  h ^= (uintptr_t)r->nframes * 0xbf58476d1ce4e5b9ULL;
  h ^= (uintptr_t)r->nloops * 0x94d049bb133111ebULL;
  for (size_t i = 0; i < r->nframes; i++) {
    h ^= (uintptr_t)r->frames[i].name.p + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= (uintptr_t)(size_t)dyn_rval_type(r->frames[i].val.kind) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  }
  DynMemo *m = &ctx->dyn_memo[(size_t)(h & (DYN_MEMO_N - 1))];
  if (m->valid && m->segs == segs && m->nsegs == nsegs
      && m->nframes == (u8)r->nframes && m->nloops == (u8)r->nloops) {
    int ok = 1;
    for (size_t i = 0; i < r->nframes; i++) {
      if (m->fname_p[i] != r->frames[i].name.p || m->fname_n[i] != r->frames[i].name.n
          || m->ftype[i] != (u8)dyn_rval_type(r->frames[i].val.kind)) { ok = 0; break; }
    }
    if (ok) return dyn_memo_fetch(r, m, out);
  }
  // Miss: resolve, record the shape, store (unless unresolvable).
  DynMemo nm;
  memset(&nm, 0, sizeof(nm));
  int res = dyn_memo_resolve(r, segs, nsegs, &nm);
  if (res < 0) return dyn_eval(r, segs, nsegs, out);
  nm.segs = segs; nm.nsegs = nsegs;
  nm.nframes = (u8)r->nframes; nm.nloops = (u8)r->nloops;
  for (size_t i = 0; i < r->nframes; i++) {
    nm.fname_p[i] = r->frames[i].name.p; nm.fname_n[i] = r->frames[i].name.n;
    nm.ftype[i] = (u8)dyn_rval_type(r->frames[i].val.kind);
  }
  nm.notfound = res == 0 ? 1 : 0;
  nm.valid = 1;
  *m = nm;
  return dyn_memo_fetch(r, &nm, out);
}

// A fragment is inlineable when it is straight-line (no for, no
// include), never reads `loop`, and every free name is a root — so its
// chunks mean the same thing spliced into any caller at the call site
// (the dynamic environment is identical: same frames, same roots, and
// the shadow check still runs at render time).
static int frag_inlineable(Tpl *frag) {
  if (frag->has_inc || frag->loop_outside) return 0;
  for (size_t i = 0; i < frag->n; i++) if (frag->code[i].op == OP_FOR) return 0;
  for (size_t f = 0; f < frag->nfree; f++) {
    Slice s = frag->free_names[f];
    if (!(sl_eq_c(s, "site") || sl_eq_c(s, "tags") || sl_eq_c(s, "recent")
          || sl_eq_c(s, "page") || sl_eq_c(s, "posts") || sl_eq_c(s, "pager")
          || sl_eq_c(s, "tag")))
      return 0;
  }
  return 1;
}

// Splice inlineable include fragments into a template's chunk array,
// rebasing jump targets (prefix-sum remap: a target at old index T
// moves to newidx[T]; a fragment chunk at frag index j spliced at old
// position i moves to newidx[i] + j).
static void inline_chunks(Tpl *t, Tpl *tpls, size_t ntpls, Arena *a) {
  (void)ntpls;
  size_t n = t->nchunks;
  if (n == 0) return;
  size_t *splice = (size_t *)arena_alloc(a, n * sizeof(size_t));
  size_t total = n;
  for (size_t i = 0; i < n; i++) {
    splice[i] = 0;
    Chunk *c = &t->chunks[i];
    if (c->kind == CK_INC && c->tpl >= 0 && (size_t)c->tpl < ntpls) {
      Tpl *frag = &tpls[c->tpl];
      if (frag != t && frag->chunks && frag_inlineable(frag) && frag->nchunks >= 1) {
        splice[i] = frag->nchunks - 1; // fragment chunks copied (drop CK_END)
        // Net change: +splice[i] copied chunks, -1 for the INC chunk removed.
        // (Off-by-one here used to leave an unwritten HOLE after each splice:
        //  the hole read as a harmless empty text on fresh mmap pages, but
        //  as garbage text on reused malloc pages -> crash on rep 3.)
        if (splice[i]) total += splice[i] - 1;
      }
    }
  }
  if (total == n) return;
  size_t *newidx = (size_t *)arena_alloc(a, (n + 1) * sizeof(size_t));
  size_t run = 0;
  for (size_t i = 0; i <= n; i++) { newidx[i] = i + run; if (i < n && splice[i]) run += splice[i] - 1; }
  Chunk *nw = (Chunk *)arena_alloc(a, total * sizeof(Chunk));
  for (size_t i = 0; i < n; i++) {
    if (splice[i]) {
      Tpl *frag = &tpls[t->chunks[i].tpl];
      for (size_t j = 0; j < splice[i]; j++) {
        Chunk fc = frag->chunks[j];
        if (fc.kind == CK_IF || fc.kind == CK_JMP) fc.target = (int)(newidx[i] + (size_t)fc.target);
        nw[newidx[i] + j] = fc;
      }
    } else {
      Chunk c = t->chunks[i];
      if (c.kind == CK_IF || c.kind == CK_JMP || c.kind == CK_FOR)
        c.target = (int)newidx[c.target];
      else if (c.kind == CK_NEXT)
        c.body = (int)newidx[c.body];
      nw[newidx[i]] = c;
    }
  }
  t->chunks = nw; t->nchunks = total;
}

// Structural validation of chunk arrays (cheap, once per case):
// catches a bad splice/remap before it can loop or crash the render.
// Debug dump of template "head" chunk array (SENJA_DUMPCHUNK=1).
static void dump_head_chunks(Tpl *tpls, size_t ntpls, const char *tag) {
  for (size_t ti = 0; ti < ntpls; ti++) {
    Tpl *t = &tpls[ti];
    if (!sl_eq_c(t->name, "head") || !t->chunks) continue;
    fprintf(stderr, "DUMP-%s head nchunks=%zu chunks=%p\n", tag, t->nchunks, (void *)t->chunks);
    for (size_t i = 0; i < t->nchunks; i++) {
      Chunk *c = &t->chunks[i];
      fprintf(stderr, "  [%zu] kind=%d raw=%d rkind=%d fast=%d base=%d nchain=%d target=%d tpl=%d text=(%p,%zu)\n",
              i, (int)c->kind, (int)c->raw, (int)c->rkind, (int)c->fast, (int)c->base, (int)c->nchain,
              c->target, c->tpl, (const void *)c->text.p, c->text.n);
    }
  }
}

static void validate_chunks(Tpl *tpls, size_t ntpls) {
  for (size_t ti = 0; ti < ntpls; ti++) {
    Tpl *t = &tpls[ti];
    if (!t->chunks) continue;
    if (t->nchunks < 1 || t->chunks[t->nchunks - 1].kind != CK_END) {
      fprintf(stderr, "CHUNK-BAD %.*s: nchunks=%zu last-kind=%d\n",
              (int)t->name.n, t->name.p, t->nchunks,
              t->nchunks ? (int)t->chunks[t->nchunks - 1].kind : -1);
      abort();
    }
    for (size_t i = 0; i < t->nchunks; i++) {
      Chunk *c = &t->chunks[i];
      if (c->kind > CK_END || c->nchain > 6 || c->rkind > RK_LOOPVAR) {
        fprintf(stderr, "CHUNK-BAD %.*s[%zu]: kind=%d nchain=%d rkind=%d\n",
                (int)t->name.n, t->name.p, i, (int)c->kind, (int)c->nchain, (int)c->rkind);
        abort();
      }
      if ((c->kind == CK_IF || c->kind == CK_JMP || c->kind == CK_FOR)
          && ((size_t)c->target >= t->nchunks)) {
        fprintf(stderr, "CHUNK-BAD %.*s[%zu]: kind=%d target=%d nchunks=%zu\n",
                (int)t->name.n, t->name.p, i, (int)c->kind, c->target, t->nchunks);
        abort();
      }
      if (c->kind == CK_NEXT && (size_t)c->body >= t->nchunks) {
        fprintf(stderr, "CHUNK-BAD %.*s[%zu]: NEXT body=%d nchunks=%zu\n",
                (int)t->name.n, t->name.p, i, c->body, t->nchunks);
        abort();
      }
      if (c->kind == CK_INC && (c->tpl < -1 || c->tpl >= (int)ntpls)) {
        // tpl == -1 is legitimate: include target not found (renders nothing).
        fprintf(stderr, "CHUNK-BAD %.*s[%zu]: INC tpl=%d ntpls=%zu\n",
                (int)t->name.n, t->name.p, i, c->tpl, ntpls);
        abort();
      }
      if (c->kind == CK_FOR && (size_t)c->target <= i) {
        fprintf(stderr, "CHUNK-BAD %.*s[%zu]: FOR target=%d not forward\n",
                (int)t->name.n, t->name.p, i, c->target);
        abort();
      }
      if (c->kind == CK_NEXT && (size_t)c->body >= i) {
        fprintf(stderr, "CHUNK-BAD %.*s[%zu]: NEXT body=%d not backward\n",
                (int)t->name.n, t->name.p, i, c->body);
        abort();
      }
    }
  }
}

static void render_chunks(Renderer *r, Tpl *tpl) {
#ifdef SENJA_PROFILE
  prof_nrender++;
#endif
  Buf *out = r->out;
  size_t base = r->nloops; // loop-stack depth at this template's entry
  Chunk *ch = tpl->chunks;
  size_t pc = 0;
  size_t steps = 0;
  static void *dispatch[] = { &&L_TEXT, &&L_SLOT, &&L_IF, &&L_JMP, &&L_FOR, &&L_NEXT, &&L_INC, &&L_END };
#define NEXT_CHUNK() do { \
    if (++steps > 20000000u) return; \
    c = &ch[++pc]; \
    goto *dispatch[c->kind]; } while (0)
  Chunk *c = &ch[0];
  goto *dispatch[c->kind];

L_TEXT:
  buf_sl(out, c->jtext);
  c = &ch[++pc];
  if (++steps > 20000000u) return;
  goto *dispatch[c->kind];

L_SLOT: {
  int done = 0;
#ifdef SENJA_PROFILE
  prof_nslot++;
#endif
  if (!nofast && c->fast && !(r->shadow_mask & (1u << c->rkind))) {
    const void *bp = NULL;
    if (c->base == CB_LOOPVAR) {
      size_t idx = base + (size_t)c->rdepth;
      if (idx < r->nframes) bp = r->frames[idx].val.ptr;
    } else if (r->root_mask & (1u << c->rkind)) {
      switch (c->base) {
        case CB_PAGE: bp = r->page.ptr; break;
        case CB_TAG: bp = r->tag.ptr; break;
        case CB_PAGER: bp = r->pager.ptr; break;
        case CB_SITE_TITLE: bp = (const void *)r->bctx->site; break;
        case CB_SITE_BASE: bp = (const void *)&r->bctx->base; break;
        default: break;
      }
    }
    if (bp) {
      Slice s = *(const Slice *)((const u8 *)bp + c->off);
      if (c->raw) json_write_raw(out, s);
      else esc_j(out, s);
      done = 1;
#ifdef SENJA_PROFILE
      prof_nfast++;
#endif
    }
  }
  if (!done) {
#ifdef SENJA_PROFILE
    prof_ngen++;
#endif
    RVal ref;
    if (eval_slot_f(r, c->segs, c->nsegs, c->rkind, c->rdepth, c->chain, c->nchain, base, &ref)) {
      if (c->raw) write_rval_j(out, r->bctx, &ref);
      else if (ref.kind == RV_STR) esc_j(out, ref.s);
      else {
        r->scratch.len = 0;
        write_rval(&r->scratch, r->bctx, &ref);
        esc_j(out, sl(r->scratch.p, r->scratch.len));
      }
    }
  }
  c = &ch[++pc];
  if (++steps > 20000000u) return;
  goto *dispatch[c->kind];
}

L_IF: {
  RVal ref;
  int t = eval_slot_f(r, c->segs, c->nsegs, c->rkind, c->rdepth, c->chain, c->nchain, base, &ref) ? rval_truthy(&ref) : 0;
  if (t != c->not) { c = &ch[++pc]; }
  else { pc = (size_t)c->target; c = &ch[pc]; }
  if (++steps > 20000000u) return;
  goto *dispatch[c->kind];
}

L_JMP:
  pc = (size_t)c->target;
  c = &ch[pc];
  if (++steps > 20000000u) return;
  goto *dispatch[c->kind];

L_FOR: {
  RVal ref;
  if (!eval_slot_f(r, c->segs, c->nsegs, c->rkind, c->rdepth, c->chain, c->nchain, base, &ref)
      || ref.kind != RV_ARR || ref.len == 0) {
    pc = (size_t)c->target;
    c = &ch[pc];
    if (++steps > 20000000u) return;
    goto *dispatch[c->kind];
  }
  if (r->nloops == r->caploops) {
    size_t nc = r->caploops ? r->caploops * 2 : 8;
    r->loops = (LoopState *)realloc(r->loops, nc * sizeof(LoopState));
    r->caploops = nc;
  }
  if (r->nframes == r->capframes) {
    size_t nc = r->capframes ? r->capframes * 2 : 8;
    r->frames = (DynBind *)realloc(r->frames, nc * sizeof(DynBind));
    r->capframes = nc;
  }
  LoopState *ls = &r->loops[r->nloops++];
  memset(ls, 0, sizeof(*ls));
  ls->arr = ref; ls->idx = 0; ls->len = ref.len; ls->var_name = c->name;
  ls->saved_shadow = r->shadow_mask;
  u32 bit = root_bit_for_name(c->name);
  if (bit) r->shadow_mask |= bit;
  RVal item;
  arr_item(r->bctx, &ref, 0, &item);
  r->frames[r->nframes].name = c->name;
  r->frames[r->nframes].val = item;
  r->nframes++;
  c = &ch[++pc];
  if (++steps > 20000000u) return;
  goto *dispatch[c->kind];
}

L_NEXT: {
  LoopState *ls = &r->loops[r->nloops - 1];
  ls->idx++;
  if (ls->idx >= ls->len) {
    r->shadow_mask = ls->saved_shadow;
    r->nloops--;
    r->nframes--;
    c = &ch[++pc];
  } else {
    RVal item;
    arr_item(r->bctx, &ls->arr, ls->idx, &item);
    r->frames[r->nframes - 1].val = item;
    pc = (size_t)c->body;
    c = &ch[pc];
  }
  if (++steps > 20000000u) return;
  goto *dispatch[c->kind];
}

L_INC: {
  if (r->depth >= INCLUDE_DEPTH || c->tpl < 0) { NEXT_CHUNK(); }
  Tpl *frag = &r->tpls[c->tpl];
  if (c->cacheable) {
#ifdef SENJA_PROFILE
    c_inc_c++;
#endif
    const void *itemptr = NULL;
    if (!c->shared_only && r->nloops) {
      RVal it = r->frames[r->nframes - 1].val;
      itemptr = it.ptr;
    }
    u8 key[12];
    memcpy(key, &c->tpl, 4);
    uintptr_t up = (uintptr_t)itemptr;
    memcpy(key + 4, &up, 8);
    Slice k = sl(key, 12);
    CacheEnt *hit = (CacheEnt *)strmap_get(r->card_cache, k);
    if (hit) {
#ifdef SENJA_PROFILE
      c_hit++; c_hit_bytes += (long)hit->len; if (c->shared_only) c_hit_sh++; else c_hit_it++;
#endif
      buf_put(out, hit->ptr, hit->len); NEXT_CHUNK(); }
    size_t start = out->len;
    r->depth++;
    render_chunks(r, frag);
    r->depth--;
    CacheEnt *ne = (CacheEnt *)arena_alloc(r->arena, sizeof(CacheEnt));
    ne->len = out->len - start;
#ifdef SENJA_PROFILE
    c_miss++; c_miss_bytes += (long)ne->len;
#endif
    ne->ptr = arena_slice(r->arena, out->p + start, ne->len).p;
    strmap_put(r->card_cache, k, ne);
    NEXT_CHUNK();
  }
#ifdef SENJA_PROFILE
  c_inc++;
#endif
  r->depth++;
  render_chunks(r, frag);
  r->depth--;
  NEXT_CHUNK();
}

L_END:
  return;
#undef NEXT_CHUNK
}


// ------------------------------------------------------------------ v6 build pipeline

static StrMap *strmap_new(Arena *a) {
  StrMap *m = (StrMap *)arena_alloc(a, sizeof(StrMap));
  strmap_init(m, a);
  return m;
}

static size_t strmap_keys(StrMap *m, Slice *out, size_t cap) {
  size_t n = 0;
  for (size_t i = 0; i < m->cap; i++) {
    if (m->slots[i].key) {
      if (n < cap) out[n] = sl(m->slots[i].key, m->slots[i].klen);
      n++;
    }
  }
  return n;
}

static Slice slug_cached(BuildCtx *ctx, Slice text) {
  void *hit = strmap_get(ctx->slug_cache, text);
  if (hit) return *(Slice *)hit;
  Slice s = slugify(ctx->arena, text);
  Slice *sp = (Slice *)arena_alloc(ctx->arena, sizeof(Slice));
  *sp = s;
  strmap_put(ctx->slug_cache, text, sp);
  return s;
}

static Slice long_date_cached(BuildCtx *ctx, Slice date) {
  void *hit = strmap_get(ctx->date_cache, date);
  if (hit) return *(Slice *)hit;
  Slice s = long_date(ctx->arena, date);
  Slice *sp = (Slice *)arena_alloc(ctx->arena, sizeof(Slice));
  *sp = s;
  strmap_put(ctx->date_cache, date, sp);
  return s;
}

static Slice slug_cached_fn(void *vctx, Slice text) {
  return slug_cached((BuildCtx *)vctx, text);
}

typedef struct {
  Slice dir, src_path, base;
  CRec *sources; size_t nsources;
  StrMap *by_source, *static_paths, *broken;
  Arena *arena;
} LinkCtx2;

static Slice build_link(void *vctx, Slice target) {
  LinkCtx2 *ctx = (LinkCtx2 *)vctx;
  if (is_scheme(target) || sl_starts(target, "/") || sl_starts(target, "#")) return target;
  size_t hash = find_byte(target.p, target.n, 0, '#');
  Slice rel, frag;
  if (hash < target.n) { rel = sl(target.p, hash); frag = sl(target.p + hash, target.n - hash); }
  else { rel = target; frag = slc(""); }
  int have_path = 0;
  Slice path = sl(NULL, 0);
  if (rel.n > 0) {
    Slice rp = resolve_path(ctx->arena, ctx->dir, rel);
    if (rp.p) { path = rp; have_path = 1; }
  }
  if (have_path) {
    void *hit = strmap_get(ctx->by_source, path);
    if (hit) {
      size_t si = (size_t)((uintptr_t)hit - 1);
      Buf b = {0};
      buf_sl(&b, ctx->sources[si].url);
      buf_sl(&b, frag);
      Slice r = arena_slice(ctx->arena, b.p, b.len);
      free(b.p);
      return r;
    }
    if (strmap_get(ctx->static_paths, path)) {
      Buf b = {0};
      buf_sl(&b, ctx->base);
      buf_c(&b, '/');
      buf_sl(&b, path);
      buf_sl(&b, frag);
      Slice r = arena_slice(ctx->arena, b.p, b.len);
      free(b.p);
      return r;
    }
  }
  Buf line = {0};
  buf_cs(&line, "broken-link ");
  buf_sl(&line, ctx->src_path);
  buf_cs(&line, " -> ");
  buf_sl(&line, target);
  Slice ls = arena_slice(ctx->arena, line.p, line.len);
  free(line.p);
  strmap_put(ctx->broken, ls, (void *)1);
  return target;
}

// Each tag name once per slug, in its own order (dedup by slug).
static size_t own_tags(BuildCtx *ctx, size_t si, Slice **out_names) {
  CRec *s = &ctx->sources[si];
  StrMap *seen = strmap_new(ctx->arena);
  size_t cap = 4, n = 0;
  Slice *names = (Slice *)arena_alloc(ctx->arena, cap * sizeof(Slice));
  if (s->data.has_tags) {
    for (size_t i = 0; i < s->data.ntags; i++) {
      Slice name = s->data.tags[i];
      Slice slug = slug_cached(ctx, name);
      if (strmap_get(seen, slug)) continue;
      strmap_put(seen, slug, (void *)1);
      if (n == cap) {
        size_t nc = cap * 2;
        Slice *nn = (Slice *)arena_alloc(ctx->arena, nc * sizeof(Slice));
        memcpy(nn, names, n * sizeof(Slice));
        names = nn; cap = nc;
      }
      names[n++] = name;
    }
  }
  *out_names = names;
  return n;
}

static void tagref_push(CRec *s, Arena *a, Slice name, Slice slug, Slice url) {
  if (s->ntag_refs == s->captag_refs) {
    size_t nc = s->captag_refs ? s->captag_refs * 2 : 4;
    TagRef *nr = (TagRef *)arena_alloc(a, nc * sizeof(TagRef));
    if (s->tag_refs) memcpy(nr, s->tag_refs, s->ntag_refs * sizeof(TagRef));
    s->tag_refs = nr; s->captag_refs = nc;
  }
  s->tag_refs[s->ntag_refs].name = name;
  s->tag_refs[s->ntag_refs].slug = slug;
  s->tag_refs[s->ntag_refs].url = url;
  s->ntag_refs++;
}

typedef struct {
  RVal page, posts, pager, tag;
  u32 mask; // ROOT_BIT per bound root (page/posts/pager/tag)
} Roots;

// Render the template at index ti with the given roots (+ site/tags/recent)
// v11: render a page STRAIGHT into the final answers.json buffer.
// render_chunks writes JSON-escaped output (static text pre-escaped at
// compile time, slots via json_write_raw / esc_j, cached fragments
// stored JSON-ready), so there is no page_buf and no json_write_str
// second pass over the page HTML. The caller (emit_page_c) has already
// written `"key":"` and closes the string after this returns.
static int render_page_c(BuildCtx *ctx, int ti, Roots *roots) {
#ifdef SENJA_PROFILE
  c_pages++;
#endif
  if (ti < 0 || !ctx->tpl_ok[ti]) return 0;
  Renderer r;
  memset(&r, 0, sizeof(r));
  r.tpls = ctx->tpls; r.ntpls = ctx->ntpls;
  r.card_cache = ctx->card_cache; r.out = ctx->out; r.arena = ctx->arena;
  r.bctx = ctx; r.depth = 0;
  memset(&r.site, 0, sizeof(r.site));
  r.site.kind = RV_SITE; r.site.ptr = ctx;
  r.tags = ctx->tags_rv;
  r.recent = ctx->recent_rv;
  r.page = roots->page; r.posts = roots->posts; r.pager = roots->pager; r.tag = roots->tag;
  r.root_mask = (1u << RK_SITE) | (1u << RK_TAGS) | (1u << RK_RECENT) | roots->mask;
  // v8: reuse the case's loop/frame/scratch buffers instead of
  // malloc/freeing them per page (8k renders per run).
  r.loops = ctx->loops; r.caploops = ctx->loops_cap;
  r.frames = ctx->frames; r.capframes = ctx->frames_cap;
  r.scratch = ctx->scratch;
  render_chunks(&r, &ctx->tpls[ti]);
  ctx->loops = r.loops; ctx->loops_cap = r.caploops;
  ctx->frames = r.frames; ctx->frames_cap = r.capframes;
  ctx->scratch = r.scratch;
  return 1;
}

static void emit_c(BuildCtx *ctx, Slice k, Slice v) {
  if (!ctx->first) buf_c(ctx->out, ',');
  ctx->first = 0;
  json_write_str(ctx->out, k);
  buf_c(ctx->out, ':');
  json_write_str(ctx->out, v);
}

static int template_index(BuildCtx *ctx, Slice layout) {
  for (size_t i = 0; i < ctx->ntpls; i++) {
    if (sl_eq(ctx->tpl_names[i], layout)) return (int)i;
  }
  return -1;
}

// v11: emit one rendered page as a JSON member: `"key":"<html>"`.
// The page HTML is rendered straight into ctx->out by render_page_c
// (JSON-escaped inline); this only writes the key, the opening quote
// and the closing quote. Returns 0 when the template is missing or
// failed (nothing written, ctx->first untouched — same contract as the
// old render-then-emit_c sequence).
static int emit_page_c(BuildCtx *ctx, Slice p, Slice layout, Roots *roots) {
  int ti = template_index(ctx, layout);
  if (ti < 0 || !ctx->tpl_ok[ti]) return 0;
  if (!ctx->first) buf_c(ctx->out, ',');
  ctx->first = 0;
  json_write_str(ctx->out, p);
  buf_c(ctx->out, ':');
  buf_c(ctx->out, '"');
  if (!render_page_c(ctx, ti, roots)) return 0;
  buf_c(ctx->out, '"');
  return 1;
}

static void report_push(BuildCtx *ctx, Slice line) {
  if (ctx->nreport == ctx->capreport) {
    size_t nc = ctx->capreport ? ctx->capreport * 2 : 16;
    Slice *nr = (Slice *)arena_alloc(ctx->arena, nc * sizeof(Slice));
    if (ctx->report) memcpy(nr, ctx->report, ctx->nreport * sizeof(Slice));
    ctx->report = nr; ctx->capreport = nc;
  }
  ctx->report[ctx->nreport++] = line;
}

static int template_exists(BuildCtx *ctx, Slice layout) {
  for (size_t i = 0; i < ctx->ntpls; i++) {
    if (sl_eq(ctx->tpl_names[i], layout)) return ctx->tpl_ok[i];
  }
  return 0;
}

static int cmp_tpl_names(void *vctx, size_t a, size_t b) {
  TplSrc *t = (TplSrc *)vctx;
  return cmp_u16(t[a].name, t[b].name);
}

static int cmp_files(void *vctx, size_t a, size_t b) {
  InFile *f = (InFile *)vctx;
  return cmp_u16(f[a].path, f[b].path);
}

static int cmp_posts(void *vctx, size_t a, size_t b) {
  BuildCtx *ctx = (BuildCtx *)vctx;
  CRec *sa = &ctx->sources[a], *sb = &ctx->sources[b];
  int c = cmp_u16(sb->data.date, sa->data.date);
  if (c) return c;
  return cmp_u16(sa->slug, sb->slug);
}

static int cmp_tags(void *vctx, size_t a, size_t b) {
  Tag *tags = (Tag *)vctx;
  Tag *ta = &tags[a], *tb = &tags[b];
  if (ta->nposts != tb->nposts) return ta->nposts > tb->nposts ? -1 : 1;
  return cmp_u16(ta->slug, tb->slug);
}

static void source_push(BuildCtx *ctx, CRec s) {
  if (ctx->nsources == ctx->capsources) {
    size_t nc = ctx->capsources ? ctx->capsources * 2 : 64;
    ctx->sources = (CRec *)realloc(ctx->sources, nc * sizeof(CRec));
    ctx->capsources = nc;
  }
  ctx->sources[ctx->nsources++] = s;
}

static void static_push(BuildCtx *ctx, Slice path, Slice content) {
  if (ctx->nstatics == ctx->capstatics) {
    size_t nc = ctx->capstatics ? ctx->capstatics * 2 : 16;
    ctx->statics = realloc(ctx->statics, nc * sizeof(*ctx->statics));
    ctx->capstatics = nc;
  }
  ctx->statics[ctx->nstatics].path = path;
  ctx->statics[ctx->nstatics].content = content;
  ctx->nstatics++;
}

// ------------------------------------------------------------------ build_site
static int build_site(CaseIn *input, Buf *out, Arena *arena) {
#ifdef SENJA_PROFILE
  double _t_bs = prof_now(), _t_s;
  double _c0 = prof_compile, _f0 = prof_fm, _m0 = prof_md, _r0 = prof_render;
  double _c_bs = senja_cpu_now(), _c_prev = senja_cpu_now();
  double _cc0 = c_compile, _cf0 = c_fm, _cm0 = c_md, _cr0 = c_render;
  c_cases++;
#endif
  BuildCtx ctx;
  memset(&ctx, 0, sizeof(ctx));
  ctx.arena = arena; ctx.out = out; ctx.first = 1;
  ctx.slug_cache = strmap_new(arena);
  ctx.date_cache = strmap_new(arena);
  ctx.card_cache = strmap_new(arena);
  ctx.by_source = strmap_new(arena);
  ctx.static_paths = strmap_new(arena);
  ctx.broken = strmap_new(arena);

  SiteIn *site = &input->site;
  if (!site->has_title || !site->has_base_url || !site->has_per_page || !site->has_recent) return 0;
  Slice site_title = site->title;
  Slice base_url = site->base_url;
  double per_page = site->per_page;
  double recent_n = site->recent;
  ctx.site = site;

  Slice base = base_url;
  while (base.n > 0 && base.p[base.n - 1] == '/') base.n--;
  ctx.base = base;

#ifdef SENJA_PROFILE
  _t_s = prof_now();
#endif
  // Templates, in code-unit order of name; one that does not parse is
  // reported and left out.
  size_t nt = input->ntpls;
#ifdef SENJA_PROFILE
  c_tpls += (long)nt;
#endif
  size_t *torder = (size_t *)malloc((nt ? nt : 1) * sizeof(size_t));
  for (size_t i = 0; i < nt; i++) torder[i] = i;
  sort_indices(torder, nt, input->tpls, cmp_tpl_names);
  ctx.ntpls = nt;
  ctx.tpl_names = (Slice *)arena_alloc(arena, (nt ? nt : 1) * sizeof(Slice));
  ctx.tpls = (Tpl *)arena_alloc(arena, (nt ? nt : 1) * sizeof(Tpl));
  ctx.tpl_ok = (int *)arena_alloc(arena, (nt ? nt : 1) * sizeof(int));
  // Zero the Tpl array: a template that fails to tokenize/compile leaves
  // its slot untouched (continue below), and arena memory is NOT zeroed —
  // without this, a failed template's slot holds stale bytes from a
  // previous call's freed arena block, and validate_chunks (or any reader)
  // would see a garbage chunks pointer / nchunks. (Exposed by the
  // tpl-unclosed-var mutation: fresh mmap reads as zeros and hides it;
  // reused malloc pages expose it as CHUNK-BAD / a crash.)
  memset(ctx.tpls, 0, (nt ? nt : 1) * sizeof(Tpl));
  for (size_t i = 0; i < nt; i++) {
    size_t mi = torder[i];
    Slice name = input->tpls[mi].name;
    ctx.tpl_names[i] = name;
    ctx.tpls[i].name = name;
    Slice src = input->tpls[mi].src;
    TokVec toks; memset(&toks, 0, sizeof(toks));
    Slice err;
    if (!tokenize(name, src, arena, &toks, &err)) {
      report_push(&ctx, err);
      ctx.tpl_ok[i] = 0;
      continue;
    }
    if (!compile_template(name, &toks, arena, &ctx.tpls[i], &err)) {
      report_push(&ctx, err);
      ctx.tpl_ok[i] = 0;
      continue;
    }
    resolve_slots(&ctx.tpls[i]);
    ctx.tpl_ok[i] = 1;
  }
  free(torder);
  resolve_includes(ctx.tpls, nt);
  for (size_t i = 0; i < nt; i++) if (ctx.tpl_ok[i]) analyze_fragment(&ctx.tpls[i], arena);
  analyze_cacheable(ctx.tpls, nt, arena);
  for (size_t i = 0; i < nt; i++) if (ctx.tpl_ok[i]) build_chunks(&ctx.tpls[i], arena);
  if (getenv("SENJA_DUMPCHUNK")) dump_head_chunks(ctx.tpls, nt, "BEFORE");
  if (getenv("SENJA_INLINE"))
    for (size_t i = 0; i < nt; i++) if (ctx.tpl_ok[i]) inline_chunks(&ctx.tpls[i], ctx.tpls, nt, arena);
  if (getenv("SENJA_DUMPCHUNK")) dump_head_chunks(ctx.tpls, nt, "AFTER");
  validate_chunks(ctx.tpls, nt);
#ifdef SENJA_PROFILE
  prof_compile += prof_now() - _t_s;
  { double _cn = senja_cpu_now(); c_compile += _cn - _c_prev; _c_prev = _cn; }
#endif

#ifdef SENJA_PROFILE
  _t_s = prof_now();
#endif
  // Sources, in code-unit order of path.
  size_t nf = input->nfiles;
  size_t *forder = (size_t *)malloc((nf ? nf : 1) * sizeof(size_t));
  for (size_t i = 0; i < nf; i++) forder[i] = i;
  sort_indices(forder, nf, input->files, cmp_files);
  for (size_t i = 0; i < nf; i++) {
    InFile *f = &input->files[forder[i]];
    Slice path = f->path, content = f->content;
    if (sl_ends(path, ".md")) {
      CRec s;
      memset(&s, 0, sizeof(s));
      s.path = path;
      s.body = crlf_to_lf(arena, content);
      s.newer = -1; s.older = -1;
      source_push(&ctx, s);
    } else {
      static_push(&ctx, path, content);
    }
  }
  free(forder);

  // Front matter + refusals, in path order.
  size_t w = 0;
  for (size_t r = 0; r < ctx.nsources; r++) {
    CRec *s = &ctx.sources[r];
    Fm fm;
    Slice err;
    if (!parse_front_matter(s->body, arena, &fm, &err)) {
      Buf line = {0};
      buf_cs(&line, "refused ");
      buf_sl(&line, s->path);
      buf_cs(&line, ": ");
      buf_sl(&line, err);
      report_push(&ctx, arena_slice(arena, line.p, line.len));
      free(line.p);
      continue;
    }
    s->data = fm.data;
    s->body = fm.body;
    s->post = sl_starts(s->path, "posts/") ? 1 : 0;
    const char *refused = NULL;
    if (sl_starts(s->path, "tags/")) refused = "tags/ is the generator's own";
    else if (sl_starts(s->path, "page/")) refused = "page/ is the generator's own";
    if (refused) {
      Buf line = {0};
      buf_cs(&line, "refused ");
      buf_sl(&line, s->path);
      buf_cs(&line, ": ");
      buf_cs(&line, refused);
      report_push(&ctx, arena_slice(arena, line.p, line.len));
      free(line.p);
      continue;
    }
    if (s->post && !s->data.has_date) {
      Buf line = {0};
      buf_cs(&line, "refused ");
      buf_sl(&line, s->path);
      buf_cs(&line, ": a post needs a date");
      report_push(&ctx, arena_slice(arena, line.p, line.len));
      free(line.p);
      continue;
    }
    Slice layout;
    if (s->data.has_layout) layout = s->data.layout;
    else layout = s->post ? slc("post") : slc("page");
    if (!template_exists(&ctx, layout)) {
      Buf line = {0};
      buf_cs(&line, "refused ");
      buf_sl(&line, s->path);
      buf_cs(&line, ": layout \"");
      buf_sl(&line, layout);
      buf_cs(&line, "\" is no template that parses");
      report_push(&ctx, arena_slice(arena, line.p, line.len));
      free(line.p);
      continue;
    }
    s->layout = layout;
    if (s->data.draft) {
      Buf line = {0};
      buf_cs(&line, "draft ");
      buf_sl(&line, s->path);
      report_push(&ctx, arena_slice(arena, line.p, line.len));
      free(line.p);
      continue;
    }
    ctx.sources[w++] = *s;
  }
  ctx.nsources = w;

#ifdef SENJA_PROFILE
  prof_fm += prof_now() - _t_s;
  { double _cn = senja_cpu_now(); c_fm += _cn - _c_prev; _c_prev = _cn; }
#endif
  // Permalinks (only the slug part is deduplicated).
  StrMap *taken = strmap_new(arena);
  for (size_t idx = 0; idx < ctx.nsources; idx++) {
    CRec *s = &ctx.sources[idx];
    Slice dir = s->post ? slc("posts") : dir_of(s->path);
    StrMap *tt = (StrMap *)strmap_get(taken, dir);
    if (!tt) {
      tt = strmap_new(arena);
      if (dir.n == 0) {
        strmap_put(tt, slc("posts"), (void *)1);
        strmap_put(tt, slc("tags"), (void *)1);
        strmap_put(tt, slc("page"), (void *)1);
      }
      strmap_put(taken, dir, tt);
    }
    Slice base_slug = s->data.has_slug ? s->data.slug : slug_cached(&ctx, s->data.title);
    Slice sg = unique_slug(arena, tt, base_slug);
    s->slug = sg;
    Slice out_dir;
    if (dir.n == 0) out_dir = sg;
    else {
      Buf b = {0};
      buf_sl(&b, dir); buf_c(&b, '/'); buf_sl(&b, sg);
      out_dir = arena_slice(arena, b.p, b.len);
      free(b.p);
    }
    s->out_dir = out_dir;
    Buf u = {0};
    buf_sl(&u, base); buf_c(&u, '/'); buf_sl(&u, out_dir); buf_c(&u, '/');
    s->url = arena_slice(arena, u.p, u.len);
    free(u.p);
    strmap_put(ctx.by_source, s->path, (void *)(uintptr_t)(idx + 1));
  }

#ifdef SENJA_PROFILE
  _t_s = prof_now();
#endif
  // Markdown for every source; then precompute the flat card fields
  // (reading time, long date) once, here, instead of per render.
  for (size_t i = 0; i < ctx.nstatics; i++)
    strmap_put(ctx.static_paths, ctx.statics[i].path, (void *)1);
  for (size_t idx = 0; idx < ctx.nsources; idx++) {
    CRec *s = &ctx.sources[idx];
    Slice dir = dir_of(s->path);
    LinkCtx2 lctx;
    memset(&lctx, 0, sizeof(lctx));
    lctx.dir = dir; lctx.src_path = s->path; lctx.base = base;
    lctx.sources = ctx.sources; lctx.nsources = ctx.nsources;
    lctx.by_source = ctx.by_source; lctx.static_paths = ctx.static_paths;
    lctx.broken = ctx.broken; lctx.arena = arena;
    StrMap *ids = strmap_new(arena);
    MdOut md = md_render(s->body, build_link, &lctx, slug_cached_fn, &ctx, arena, ids);
    s->html = md.html;
    s->toc = md.toc;
    s->ntoc = md.ntoc;
    s->excerpt = md.excerpt;
    size_t nw = md.words; // fused into md_split_lines (p_words prototype)
    u32 reading = (u32)((nw + 199) / 200);
    if (reading < 1) reading = 1;
    s->reading_minutes = (double)reading;
    s->date_long = s->data.has_date ? long_date_cached(&ctx, s->data.date) : slc("");
  }

#ifdef SENJA_PROFILE
  prof_md += prof_now() - _t_s;
  { double _cn = senja_cpu_now(); c_md += _cn - _c_prev; _c_prev = _cn; }
#endif
  // Collections.
  size_t *posts = (size_t *)malloc((ctx.nsources ? ctx.nsources : 1) * sizeof(size_t));
  size_t nposts = 0;
  for (size_t i = 0; i < ctx.nsources; i++) if (ctx.sources[i].post) posts[nposts++] = i;
  sort_indices(posts, nposts, &ctx, cmp_posts);
  for (size_t k = 0; k < nposts; k++) {
    size_t pi = posts[k];
    ctx.sources[pi].newer = k == 0 ? -1 : (int)posts[k - 1];
    ctx.sources[pi].older = k + 1 == nposts ? -1 : (int)posts[k + 1];
  }

  // Tags. Each post's tags (deduped by slug, in the post's own order)
  // join the tag's post list; tags are then sorted by post count, then
  // slug. Non-post sources get tag references without a post list.
  StrMap *by_slug = strmap_new(arena);
  size_t captags = 16, ntags = 0;
  Tag *tags_arr = (Tag *)malloc(captags * sizeof(Tag));
  size_t *tag_order = (size_t *)malloc((ctx.nsources ? ctx.nsources : 1) * sizeof(size_t));
  size_t ntag_order = 0;
  for (size_t k = 0; k < nposts; k++) {
    size_t pi = posts[k];
    Slice *names;
    size_t nnames = own_tags(&ctx, pi, &names);
    for (size_t q = 0; q < nnames; q++) {
      Slice name = names[q];
      Slice slug = slug_cached(&ctx, name);
      void *hit = strmap_get(by_slug, slug);
      size_t ti;
      if (!hit) {
        if (ntags == captags) { captags *= 2; tags_arr = (Tag *)realloc(tags_arr, captags * sizeof(Tag)); }
        ti = ntags++;
        tags_arr[ti].name = name;
        tags_arr[ti].slug = slug;
        Buf u = {0};
        buf_sl(&u, base); buf_cs(&u, "/tags/"); buf_sl(&u, slug); buf_c(&u, '/');
        tags_arr[ti].url = arena_slice(arena, u.p, u.len);
        free(u.p);
        tags_arr[ti].posts = NULL; tags_arr[ti].nposts = 0; tags_arr[ti].capposts = 0;
        strmap_put(by_slug, slug, (void *)(uintptr_t)(ti + 1));
        tag_order[ntag_order++] = ti;
      } else ti = (size_t)((uintptr_t)hit - 1);
      Tag *t = &tags_arr[ti];
      if (t->nposts == t->capposts) {
        size_t nc = t->capposts ? t->capposts * 2 : 4;
        t->posts = (size_t *)realloc(t->posts, nc * sizeof(size_t));
        t->capposts = nc;
      }
      t->posts[t->nposts++] = pi;
      tagref_push(&ctx.sources[pi], arena, t->name, t->slug, t->url);
    }
  }
  for (size_t i = 0; i < ctx.nsources; i++) {
    if (ctx.sources[i].post) continue;
    Slice *names;
    size_t nnames = own_tags(&ctx, i, &names);
    for (size_t q = 0; q < nnames; q++) {
      Slice name = names[q];
      Slice slug = slug_cached(&ctx, name);
      void *hit = strmap_get(by_slug, slug);
      if (hit) {
        Tag *t = &tags_arr[(size_t)((uintptr_t)hit - 1)];
        tagref_push(&ctx.sources[i], arena, t->name, t->slug, t->url);
      } else {
        tagref_push(&ctx.sources[i], arena, name, slug, slc(""));
      }
    }
  }
  sort_indices(tag_order, ntag_order, tags_arr, cmp_tags);

  // Per-case globals (flat arrays, no Val construction).
  {
    Tag **sorted = (Tag **)arena_alloc(arena, (ntag_order ? ntag_order : 1) * sizeof(Tag *));
    for (size_t k = 0; k < ntag_order; k++) sorted[k] = &tags_arr[tag_order[k]];
    memset(&ctx.tags_rv, 0, sizeof(ctx.tags_rv));
    ctx.tags_rv.kind = RV_ARR; ctx.tags_rv.ptr = sorted; ctx.tags_rv.len = ntag_order; ctx.tags_rv.elem = AE_TAG_PTR;
  }
  {
    size_t r0, r1;
    js_slice_bounds(nposts, 0.0, recent_n, &r0, &r1);
    memset(&ctx.recent_rv, 0, sizeof(ctx.recent_rv));
    ctx.recent_rv.kind = RV_ARR; ctx.recent_rv.ptr = posts + r0; ctx.recent_rv.len = r1 - r0; ctx.recent_rv.elem = AE_REC_IDX;
  }

#ifdef SENJA_PROFILE
  _t_s = prof_now();
#endif
  StrMap *out_paths = strmap_new(arena);
  // Layouts.
  for (size_t idx = 0; idx < ctx.nsources; idx++) {
    CRec *s = &ctx.sources[idx];
    Roots roots;
    memset(&roots, 0, sizeof(roots));
    roots.page.kind = RV_REC; roots.page.ptr = s;
    roots.mask = (1u << RK_PAGE);
    Buf pb = {0};
    buf_sl(&pb, s->out_dir);
    buf_cs(&pb, "/index.html");
    Slice p = arena_slice(arena, pb.p, pb.len);
    free(pb.p);
    if (!emit_page_c(&ctx, p, s->layout, &roots)) return 0;
    strmap_put(out_paths, p, (void *)1);
  }
  // Index pages.
  double pages_n = per_page == 0.0 ? INFINITY : fmax(1.0, ceil((double)nposts / per_page));
  {
    u64 k = 1;
    while ((double)k <= pages_n) {
      size_t a, b;
      js_slice_bounds(nposts, (double)(k - 1) * per_page, (double)k * per_page, &a, &b);
      Pager pager;
      pager.number = (double)k;
      pager.total = pages_n;
      pager.url = index_url(arena, base, k);
      pager.newer_url = k == 1 ? slc("") : index_url(arena, base, k - 1);
      pager.older_url = (double)k == pages_n ? slc("") : index_url(arena, base, k + 1);
      Roots roots;
      memset(&roots, 0, sizeof(roots));
      roots.posts.kind = RV_ARR; roots.posts.ptr = posts + a; roots.posts.len = b - a; roots.posts.elem = AE_REC_IDX;
      roots.pager.kind = RV_PAGER; roots.pager.ptr = &pager;
      roots.mask = (1u << RK_POSTS) | (1u << RK_PAGER);
      Slice p;
      if (k == 1) p = slc("index.html");
      else {
        Buf pb = {0};
        buf_cs(&pb, "page/");
        char kb[24];
        int m = snprintf(kb, sizeof(kb), "%llu", (unsigned long long)k);
        buf_put(&pb, kb, (size_t)m);
        buf_cs(&pb, "/index.html");
        p = arena_slice(arena, pb.p, pb.len);
        free(pb.p);
      }
      if (emit_page_c(&ctx, p, slc("index"), &roots)) strmap_put(out_paths, p, (void *)1);
      k++;
      if (!isfinite(pages_n)) break;
    }
  }
  // Tag pages.
  for (size_t ti = 0; ti < ntag_order; ti++) {
    Tag *t = &tags_arr[tag_order[ti]];
    Roots roots;
    memset(&roots, 0, sizeof(roots));
    roots.tag.kind = RV_TAG; roots.tag.ptr = t;
    roots.mask = (1u << RK_TAG);
    Buf pb = {0};
    buf_cs(&pb, "tags/");
    buf_sl(&pb, t->slug);
    buf_cs(&pb, "/index.html");
    Slice p = arena_slice(arena, pb.p, pb.len);
    free(pb.p);
    if (emit_page_c(&ctx, p, slc("tag"), &roots)) strmap_put(out_paths, p, (void *)1);
  }
  // Tags index.
  {
    Roots roots;
    memset(&roots, 0, sizeof(roots));
    if (emit_page_c(&ctx, slc("tags/index.html"), slc("tags"), &roots))
      strmap_put(out_paths, slc("tags/index.html"), (void *)1);
  }
  // Statics (shadow check): a static whose path a built page already
  // has is reported, not emitted.
  for (size_t i = 0; i < ctx.nstatics; i++) {
    Slice p = ctx.statics[i].path;
    if (strmap_get(out_paths, p)) {
      Buf line = {0};
      buf_cs(&line, "shadowed ");
      buf_sl(&line, p);
      buf_cs(&line, ": a built page has its path");
      report_push(&ctx, arena_slice(arena, line.p, line.len));
      free(line.p);
    } else {
      strmap_put(out_paths, p, (void *)1);
      emit_c(&ctx, p, ctx.statics[i].content);
    }
  }
  // Report: all report lines plus all broken-link lines, sorted
  // together, then the "built N pages..." summary line.
  {
    size_t nbroken = ctx.broken->len;
    Slice *lines = (Slice *)arena_alloc(arena, (ctx.nreport + nbroken + 1) * sizeof(Slice));
    size_t nlines = 0;
    for (size_t i = 0; i < ctx.nreport; i++) lines[nlines++] = ctx.report[i];
    nlines += strmap_keys(ctx.broken, lines + nlines, nbroken);
    sort_slices(lines, nlines);
    Buf rt = {0};
    for (size_t i = 0; i < nlines; i++) {
      buf_sl(&rt, lines[i]);
      buf_c(&rt, '\n');
    }
    size_t page_count = 0;
    for (size_t i = 0; i < ctx.nsources; i++) if (!ctx.sources[i].post) page_count++;
    char last[256];
    int m = snprintf(last, sizeof(last),
                     "built %zu pages, %zu posts, %zu tags and %zu static files",
                     page_count, nposts, ntag_order, ctx.nstatics);
    buf_put(&rt, last, (size_t)m);
    buf_c(&rt, '\n');
    emit_c(&ctx, slc("build-report.txt"), sl(rt.p, rt.len));
    free(rt.p);
  }
  (void)site_title;
#ifdef SENJA_PROFILE
  prof_render += prof_now() - _t_s;
  prof_other += (prof_now() - _t_bs) - ((prof_compile - _c0) + (prof_fm - _f0) + (prof_md - _m0) + (prof_render - _r0));
  { double _cn = senja_cpu_now(); c_render += _cn - _c_prev; _c_prev = _cn; c_other += (_cn - _c_bs) - ((c_compile - _cc0) + (c_fm - _cf0) + (c_md - _cm0) + (c_render - _cr0)); }
#endif
  free(ctx.sources); free(ctx.statics); free(posts); free(tags_arr); free(tag_order);
  free(ctx.page_buf.p);
  free(ctx.loops); free(ctx.frames); free(ctx.scratch.p); // v8 ctx-owned renderer buffers
  return 1;
}

// ------------------------------------------------------------------ run_cases / run_one
//
// answers.json is a JSON ARRAY of answer objects, one per case, in the
// cases' order (the baseline driver: answers.push(run(c.input))).
// run_one answers ONE case input object ({site, templates, files}) with
// one answer object — the solve.js run() contract.

// The whole cases.json array at once -> answers.json array bytes.
static int run_cases(const char *json, size_t jlen, Buf *out) {
  // Runtime dispatch for the standalone harness (the N-API path sets
  // this in senja_init; harmless to re-set here — same value, once).
  { static int _cpu_init = 0; if (!_cpu_init) { have_avx2 = senja_have_avx2(); _cpu_init = 1; nofast = getenv("SENJA_NOFAST") ? 1 : 0; } }
#ifdef SENJA_PROFILE
  double _t_all = prof_now();
  double _c_all = senja_cpu_now();
#endif
  Arena arena_storage;
  Arena *arena = &arena_storage;
  arena_init(arena);
  buf_reserve(out, jlen * 3 + 1024); // answers ~2.7x input: avoid doubling copies
  CaseVec cases; memset(&cases, 0, sizeof(cases));
#ifdef SENJA_PROFILE
  double _t_p = prof_now();
  double _c_p = senja_cpu_now();
#endif
  if (!fp_parse_cases(json, jlen, arena, &cases)) { arena_free(arena); return 0; }
#ifdef SENJA_PROFILE
  prof_parse += prof_now() - _t_p;
  c_parse += senja_cpu_now() - _c_p;
#endif
  // v8: pre-size the output buffer. Answers run ~2.5x the input bytes;
  // growing the buffer by doubling copied ~8 MB per runAll and paid a
  // buf_reserve call per JSON string. Growth still works if the ratio
  // is higher on an unseen set (correctness unaffected).
  buf_reserve(out, jlen * 3 + (1 << 20));
  buf_c(out, '[');
  for (size_t ci = 0; ci < cases.n; ci++) {
    if (ci > 0) buf_c(out, ',');
    buf_c(out, '{');
    if (!build_site(&cases.items[ci], out, arena)) { arena_free(arena); return 0; }
    buf_c(out, '}');
  }
  buf_c(out, ']');
  arena_free(arena);
#ifdef SENJA_PROFILE
  fprintf(stderr, "PROFILE total=%.1fms parse=%.1fms compile=%.1fms fm=%.1fms md=%.1fms md_inline(incl)=%.1fms render=%.1fms json_write=%.1fms esc=%.1fms slugw=%.1fms other=%.1fms | instr=%ld eval=%ld render_calls=%ld esc_calls=%ld jw_calls=%ld mdin_calls=%ld\n",
    (prof_now() - _t_all) * 1000.0, prof_parse * 1000.0, prof_compile * 1000.0, prof_fm * 1000.0, prof_md * 1000.0, prof_mdin * 1000.0, prof_render * 1000.0, prof_jw * 1000.0, prof_esc * 1000.0, prof_slugw * 1000.0, prof_other * 1000.0,
    prof_ninstr, prof_neval, prof_nrender, prof_nesc, prof_njw, prof_nmdin);
  fprintf(stderr, "PROFILE slots=%ld fast=%ld generic=%ld\n", prof_nslot, prof_nfast, prof_ngen);
  fprintf(stderr, "CPUPROF parse=%.1fms compile=%.1fms fm=%.1fms md=%.1fms render=%.1fms other=%.1fms\n",
    c_parse * 1000.0, c_compile * 1000.0, c_fm * 1000.0, c_md * 1000.0, c_render * 1000.0, c_other * 1000.0);
  fprintf(stderr, "CPUWORDS words_of=%.1fms\n", c_words * 1000.0);
  fprintf(stderr, "CPUCOUNT cases=%ld tpls=%ld pages=%ld inc=%ld inc_cacheable=%ld card_hit=%ld (shared=%ld item=%ld) card_miss=%ld hit_bytes=%ld miss_bytes=%ld render_calls=%ld\n",
    c_cases, c_tpls, c_pages, c_inc, c_inc_c, c_hit, c_hit_sh, c_hit_it, c_miss, c_hit_bytes, c_miss_bytes, prof_nrender);
#endif
  return 1;
}

// One case input object at once -> one answer object.
static int run_one(const char *json, size_t jlen, Buf *out) {
  Arena arena_storage;
  Arena *arena = &arena_storage;
  arena_init(arena);
  buf_reserve(out, jlen * 3 + 1024); // answers ~2.7x input: avoid doubling copies
  JParser p = { (const u8 *)json, jlen, 0, arena };
  fp_ws(&p);
  if (p.pos >= p.len || p.b[p.pos] != '{') { arena_free(arena); return 0; }
  CaseIn ci;
  memset(&ci, 0, sizeof(ci));
  int ok = fp_parse_case_input(&p, &ci);
  if (!ok) { arena_free(arena); return 0; }
  // v8: pre-size (same reasoning as run_cases).
  buf_reserve(out, jlen * 3 + (1 << 20));
  buf_c(out, '{');
  ok = build_site(&ci, out, arena);
  buf_c(out, '}');
  arena_free(arena);
  return ok;
}

#ifndef SENJA_STANDALONE
// ------------------------------------------------------------------ N-API glue

// The output buffer is malloc'd by the engine; hand it to JS as an
// external Buffer that frees it on GC instead of copying 7+ MB.
static void free_buf_finalize(napi_env env, void *data, void *hint) {
  (void)env; (void)hint;
  free(data);
}

static napi_value js_build_cases(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value argv[1];
  if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc < 1) {
    napi_throw_error(env, NULL, "buildCases expects one Buffer argument");
    return NULL;
  }
  void *data = NULL;
  size_t len = 0;
  if (napi_get_buffer_info(env, argv[0], &data, &len) != napi_ok) {
    napi_throw_error(env, NULL, "buildCases expects a Buffer");
    return NULL;
  }
  Buf out = {0};
  if (!run_cases((const char *)data, len, &out)) {
    free(out.p);
    abort(); // structural error on a valid case is impossible; match the reference
  }
  napi_value result = NULL;
  if (napi_create_external_buffer(env, out.len, out.p, free_buf_finalize, NULL, &result) != napi_ok) {
    free(out.p);
    napi_throw_error(env, NULL, "cannot create result buffer");
    return NULL;
  }
  return result;
}

static napi_value js_build_one(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value argv[1];
  if (napi_get_cb_info(env, info, &argc, argv, NULL, NULL) != napi_ok || argc < 1) {
    napi_throw_error(env, NULL, "buildOne expects one Buffer argument");
    return NULL;
  }
  void *data = NULL;
  size_t len = 0;
  if (napi_get_buffer_info(env, argv[0], &data, &len) != napi_ok) {
    napi_throw_error(env, NULL, "buildOne expects a Buffer");
    return NULL;
  }
  Buf out = {0};
  if (!run_one((const char *)data, len, &out)) {
    free(out.p);
    abort();
  }
  napi_value result = NULL;
  if (napi_create_external_buffer(env, out.len, out.p, free_buf_finalize, NULL, &result) != napi_ok) {
    free(out.p);
    napi_throw_error(env, NULL, "cannot create result buffer");
    return NULL;
  }
  return result;
}

static napi_value senja_init(napi_env env, napi_value exports) {
  have_avx2 = senja_have_avx2();
  napi_value fn1, fn2, ran;
  napi_create_function(env, "buildCases", NAPI_AUTO_LENGTH, js_build_cases, NULL, &fn1);
  napi_set_named_property(env, exports, "buildCases", fn1);
  napi_create_function(env, "buildOne", NAPI_AUTO_LENGTH, js_build_one, NULL, &fn2);
  napi_set_named_property(env, exports, "buildOne", fn2);
  napi_create_int32(env, 0, &ran);
  napi_set_named_property(env, exports, "cliRan", ran);

  // CLI fast path: when the addon's loader is the entry's main script
  // (node /bot/main.ts), run the whole pipeline here at load time and
  // exit via the exit_group syscall, skipping Node teardown. The write
  // must complete first; on any failure return normally so the JS
  // driver can fall back to the JS/WASM path.
  // Read the process argv from /proc/self/cmdline (Linux).
  {
    int fd = open("/proc/self/cmdline", O_RDONLY);
    if (fd >= 0) {
      char cb[1024];
      ssize_t n = read(fd, cb, sizeof(cb) - 1);
      close(fd);
      if (n > 0) {
        cb[n] = 0;
        // Count NUL-separated fields.
        int count = 0;
        for (ssize_t i = 0; i < n; i++) if (cb[i] == 0) count++;
        if (count >= 2) {
          // argv[1] is the second field.
          const char *p = cb + strlen(cb) + 1;
          size_t len = strlen(p);
          if (len >= 7 && strcmp(p + len - 7, "main.ts") == 0) {
            const char *in_dir = getenv("MAKEFUN_INPUT_DIR");
            const char *out_dir = getenv("MAKEFUN_OUTPUT_DIR");
            if (!in_dir) in_dir = "/input";
            if (!out_dir) out_dir = "/output";
            char in_path[4096], out_path[4096];
            snprintf(in_path, sizeof(in_path), "%s/cases.json", in_dir);
            snprintf(out_path, sizeof(out_path), "%s/answers.json", out_dir);
            int ifd = open(in_path, O_RDONLY);
            if (ifd >= 0) {
              // v10: zero-copy mmap input (bgobray pattern). Size via
              // lseek (fstat needs GLIBC_2.33; lseek/mmap are
              // GLIBC_2.2.5, so the .node stays <= 2.14). The parser is
              // bounds-safe (proven by the exact-length Buffer path),
              // so an exact-size mapping is safe. Any failure falls
              // back to the old grow-until-EOF read loop.
              char *json = NULL;
              size_t got = 0;
              int use_mmap = 0;
              off_t sz = lseek(ifd, 0, SEEK_END);
              if (sz > 0) {
                void *mp = mmap(NULL, (size_t)sz, PROT_READ, MAP_PRIVATE, ifd, 0);
                if (mp != MAP_FAILED) { json = (char *)mp; got = (size_t)sz; use_mmap = 1; }
              }
              if (!use_mmap) {
                (void)lseek(ifd, 0, SEEK_SET);
                size_t cap = 1u << 20;
                got = 0;
                json = (char *)malloc(cap + 1);
                if (json) {
                  for (;;) {
                    if (got == cap) { cap *= 2; char *nj = (char *)realloc(json, cap + 1); if (!nj) break; json = nj; }
                    ssize_t r = read(ifd, json + got, cap - got);
                    if (r <= 0) break;
                    got += (size_t)r;
                  }
                }
              }
              close(ifd);
              if (json && got > 0) {
                Buf out = {0};
                int wok = 0;
                if (run_cases(json, (size_t)got, &out)) {
                  int ofd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
                  if (ofd >= 0) {
                    size_t off = 0;
                    wok = 1;
                    while (off < out.len) {
                      ssize_t w = write(ofd, out.p + off, out.len - off);
                      if (w <= 0) { wok = 0; break; }
                      off += (size_t)w;
                    }
                    close(ofd);
                  }
                }
                free(out.p);
                if (wok) {
                  // Path marker (stderr only; never touches answers.json):
                  // lets a local run confirm the native path actually ran.
                  fprintf(stderr, "[senja] path=native\n");
                  napi_create_int32(env, 1, &ran);
                  napi_set_named_property(env, exports, "cliRan", ran);
                  syscall(SYS_exit_group, 0);
                }
              }
              if (use_mmap) munmap(json, got); else free(json);
            }
          }
        }
      }
    }
  }
  return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, senja_init)
#endif // SENJA_STANDALONE
