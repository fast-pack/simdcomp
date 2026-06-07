/**
 * This code is released under a BSD License.
 *
 * neon128.h -- a small, self-contained ARM NEON implementation of the handful
 * of 128-bit Intel SSE2/SSSE3/SSE4.1 intrinsics that simdcomp actually uses.
 *
 * This is NOT a general-purpose SSE-to-NEON translation layer and it does NOT
 * pull in any third-party project (such as sse2neon). It only defines what the
 * simdcomp sources reference, written directly against <arm_neon.h>. The Intel
 * spellings (__m128i, _mm_*) are kept so that the rest of the library compiles
 * unchanged on AArch64 / ARM NEON.
 *
 * Semantics mirror the corresponding SSE instructions, including:
 *   - 32-bit lane shifts saturate to 0 for shift counts >= 32, and accept a
 *     run-time (non-immediate) count;
 *   - _mm_shuffle_epi8 zeroes any output byte whose control byte has bit 7 set;
 *   - _mm_alignr_epi8 / _mm_slli_si128 / _mm_srli_si128 are whole-register byte
 *     shifts that fill with zero.
 */
#ifndef SIMDCOMP_NEON128_H_
#define SIMDCOMP_NEON128_H_

#include <arm_neon.h>
#include <stdint.h>

/* __inline__ (rather than the bare "inline" keyword) keeps this valid under
 * strict -std=c89, which simdcomp builds with. */
#if defined(__GNUC__) || defined(__clang__)
#define SIMDCOMP_NEON_INLINE __inline__ __attribute__((always_inline))
#else
#define SIMDCOMP_NEON_INLINE inline
#endif

/* The canonical 128-bit integer vector. simdcomp treats this as four 32-bit
 * lanes for arithmetic and as raw 128 bits for the logical/byte operations. */
typedef int32x4_t __m128i;
/* Only used as the bridge for _mm_castsi128_ps -> _mm_movemask_ps. */
typedef float32x4_t __m128;

/* ---- loads and stores (aligned and unaligned are identical on AArch64) ---- */

static SIMDCOMP_NEON_INLINE __m128i _mm_loadu_si128(const __m128i *p) {
  return vld1q_s32((const int32_t *)p);
}
static SIMDCOMP_NEON_INLINE __m128i _mm_load_si128(const __m128i *p) {
  return vld1q_s32((const int32_t *)p);
}
static SIMDCOMP_NEON_INLINE void _mm_storeu_si128(__m128i *p, __m128i a) {
  vst1q_s32((int32_t *)p, a);
}
static SIMDCOMP_NEON_INLINE void _mm_store_si128(__m128i *p, __m128i a) {
  vst1q_s32((int32_t *)p, a);
}

/* ---- constructors ---- */

static SIMDCOMP_NEON_INLINE __m128i _mm_setzero_si128(void) {
  return vdupq_n_s32(0);
}
static SIMDCOMP_NEON_INLINE __m128i _mm_set1_epi32(int v) {
  return vdupq_n_s32(v);
}
/* lane 0 == e0, matching _mm_setr_epi32 (the "reversed" Intel constructor). */
static SIMDCOMP_NEON_INLINE __m128i _mm_setr_epi32(int e0, int e1, int e2,
                                                   int e3) {
  int32_t tmp[4];
  tmp[0] = e0;
  tmp[1] = e1;
  tmp[2] = e2;
  tmp[3] = e3;
  return vld1q_s32(tmp);
}

/* ---- bitwise logic over all 128 bits ---- */

static SIMDCOMP_NEON_INLINE __m128i _mm_and_si128(__m128i a, __m128i b) {
  return vandq_s32(a, b);
}
static SIMDCOMP_NEON_INLINE __m128i _mm_or_si128(__m128i a, __m128i b) {
  return vorrq_s32(a, b);
}

/* ---- 32-bit lane integer arithmetic ---- */

static SIMDCOMP_NEON_INLINE __m128i _mm_add_epi32(__m128i a, __m128i b) {
  return vaddq_s32(a, b);
}
static SIMDCOMP_NEON_INLINE __m128i _mm_sub_epi32(__m128i a, __m128i b) {
  return vsubq_s32(a, b);
}
static SIMDCOMP_NEON_INLINE __m128i _mm_min_epu32(__m128i a, __m128i b) {
  return vreinterpretq_s32_u32(
      vminq_u32(vreinterpretq_u32_s32(a), vreinterpretq_u32_s32(b)));
}
static SIMDCOMP_NEON_INLINE __m128i _mm_max_epu32(__m128i a, __m128i b) {
  return vreinterpretq_s32_u32(
      vmaxq_u32(vreinterpretq_u32_s32(a), vreinterpretq_u32_s32(b)));
}

/* ---- per-lane logical shifts by a (possibly run-time) count ----
 * NEON's vshlq with a negative count performs a right shift; a count whose
 * magnitude is >= 32 yields 0, which matches _mm_slli_epi32/_mm_srli_epi32. */
static SIMDCOMP_NEON_INLINE __m128i _mm_slli_epi32(__m128i a, int count) {
  return vreinterpretq_s32_u32(
      vshlq_u32(vreinterpretq_u32_s32(a), vdupq_n_s32(count)));
}
static SIMDCOMP_NEON_INLINE __m128i _mm_srli_epi32(__m128i a, int count) {
  return vreinterpretq_s32_u32(
      vshlq_u32(vreinterpretq_u32_s32(a), vdupq_n_s32(-count)));
}

/* ---- signed 32-bit lane compare (a < b -> all ones, else zero) ---- */
static SIMDCOMP_NEON_INLINE __m128i _mm_cmplt_epi32(__m128i a, __m128i b) {
  return vreinterpretq_s32_u32(vcltq_s32(a, b));
}

/* ---- lane extraction ---- */
static SIMDCOMP_NEON_INLINE int _mm_cvtsi128_si32(__m128i a) {
  return vgetq_lane_s32(a, 0);
}
/* _mm_extract_epi32 and _mm_shuffle_epi32 need their immediate to fold to a
 * compile-time constant for vgetq_lane / lane indexing, so they are macros. */
#define _mm_extract_epi32(a, imm) vgetq_lane_s32((a), (imm))

/* Reorder 32-bit lanes per the 2-bit fields of the Intel control byte. */
#define _mm_shuffle_epi32(a, imm)                                              \
  vsetq_lane_s32(                                                              \
      vgetq_lane_s32((a), ((imm) >> 6) & 3),                                   \
      vsetq_lane_s32(                                                          \
          vgetq_lane_s32((a), ((imm) >> 4) & 3),                              \
          vsetq_lane_s32(                                                      \
              vgetq_lane_s32((a), ((imm) >> 2) & 3),                          \
              vsetq_lane_s32(vgetq_lane_s32((a), (imm) & 3), vdupq_n_s32(0),  \
                             0),                                               \
              1),                                                             \
          2),                                                                 \
      3)

/* ---- byte (whole-register) shuffles and shifts ----
 * vextq_u8(lo, hi, n) yields the 16 bytes starting at offset n of lo:hi
 * (lo occupying the low bytes), which is exactly Intel's PALIGNR(hi, lo, n). */
#define _mm_alignr_epi8(a, b, imm)                                             \
  vreinterpretq_s32_u8(                                                        \
      vextq_u8(vreinterpretq_u8_s32(b), vreinterpretq_u8_s32(a), (imm)))
/* shift the whole register right by imm bytes, shifting in zeroes */
#define _mm_srli_si128(a, imm)                                                 \
  vreinterpretq_s32_u8(                                                        \
      vextq_u8(vreinterpretq_u8_s32(a), vdupq_n_u8(0), (imm)))
/* shift the whole register left by imm bytes, shifting in zeroes */
#define _mm_slli_si128(a, imm)                                                 \
  vreinterpretq_s32_u8(                                                        \
      vextq_u8(vdupq_n_u8(0), vreinterpretq_u8_s32(a), 16 - (imm)))

/* Per-byte table lookup. Masking the control with 0x8F reproduces PSHUFB
 * exactly: a control byte with bit 7 set (or any value >= 16) selects 0, and
 * otherwise only the low nibble indexes the source. */
static SIMDCOMP_NEON_INLINE __m128i _mm_shuffle_epi8(__m128i a, __m128i mask) {
  uint8x16_t idx = vandq_u8(vreinterpretq_u8_s32(mask), vdupq_n_u8(0x8F));
  return vreinterpretq_s32_u8(vqtbl1q_u8(vreinterpretq_u8_s32(a), idx));
}

/* ---- float reinterpretation + sign-bit gather (for the search kernels) ---- */
#define _mm_castsi128_ps(a) vreinterpretq_f32_s32(a)

static SIMDCOMP_NEON_INLINE int _mm_movemask_ps(__m128 a) {
  /* keep the top bit of each 32-bit lane, then weight lanes 0..3 by 1,2,4,8 */
  const int32x4_t shift = {0, 1, 2, 3};
  uint32x4_t top = vshrq_n_u32(vreinterpretq_u32_f32(a), 31);
  uint32x4_t weighted = vshlq_u32(top, shift);
#if defined(__aarch64__)
  return (int)vaddvq_u32(weighted);
#else
  uint32x2_t s = vadd_u32(vget_low_u32(weighted), vget_high_u32(weighted));
  return (int)vget_lane_u32(vpadd_u32(s, s), 0);
#endif
}

#endif /* SIMDCOMP_NEON128_H_ */
