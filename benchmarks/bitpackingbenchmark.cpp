/**
 * Bit-packing micro-benchmark.
 *
 * Timing is provided by lemire/counters (https://github.com/lemire/counters),
 * which reads hardware performance counters where available (Linux perf, Apple
 * Silicon kperf -- the latter usually needs sudo). When counters are available
 * we report CPU cycles per integer; we always report ns per integer and the
 * throughput in millions of integers per second.
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "counters/bench.h"

/* simdcomp is a C library. */
extern "C" {
#include "simdcomp.h"
}

using counters::bench;
using counters::event_aggregate;

static uint32_t mask_for_bit(uint32_t bit) {
  return (uint32_t)((UINT64_C(1) << bit) - 1);
}

static uint32_t *get_random_array_from_bit_width(uint32_t length, uint32_t bit) {
  uint32_t *answer = (uint32_t *)malloc(sizeof(uint32_t) * length);
  uint32_t mask = mask_for_bit(bit);
  uint32_t i;
  for (i = 0; i < length; ++i) {
    answer[i] = ((uint32_t)rand()) & mask;
  }
  return answer;
}

static uint32_t *get_random_array_from_bit_width_d1(uint32_t length,
                                                    uint32_t bit) {
  uint32_t *answer = (uint32_t *)malloc(sizeof(uint32_t) * length);
  uint32_t mask = mask_for_bit(bit);
  uint32_t i;
  answer[0] = ((uint32_t)rand()) & mask;
  for (i = 1; i < length; ++i) {
    answer[i] = answer[i - 1] + (((uint32_t)rand()) & mask);
  }
  return answer;
}

static void print_header(const char *name, uint32_t length) {
  printf("# --- %s (%u integers per block)\n", name, length);
  printf("# bit  pack(cyc/int)  unpack(cyc/int)  pack(ns/int)  "
         "unpack(ns/int)  pack(Mint/s)  unpack(Mint/s)\n");
}

static void print_row(uint32_t bit, uint32_t length,
                      const event_aggregate &pack,
                      const event_aggregate &unpack, bool have_counters) {
  printf("%3u  ", bit);
  if (have_counters) {
    printf("%12.3f  %14.3f  ", pack.fastest_cycles() / length,
           unpack.fastest_cycles() / length);
  } else {
    printf("%12s  %14s  ", "n/a", "n/a");
  }
  printf("%11.3f  %13.3f  ", pack.fastest_elapsed_ns() / length,
         unpack.fastest_elapsed_ns() / length);
  printf("%11.1f  %12.1f\n",
         length / pack.fastest_elapsed_ns() * 1000.0,
         length / unpack.fastest_elapsed_ns() * 1000.0);
}

static void demo128(bool have_counters) {
  const uint32_t length = 128;
  print_header("demo128 (SSE2 / NEON)", length);
  for (uint32_t bit = 1; bit <= 32; ++bit) {
    uint32_t *data = get_random_array_from_bit_width(length, bit);
    __m128i *buffer = (__m128i *)malloc(length * sizeof(uint32_t));
    uint32_t *backdata = (uint32_t *)malloc(length * sizeof(uint32_t));

    event_aggregate pack =
        bench([&] { simdpackwithoutmask(data, buffer, bit); });
    event_aggregate unpack = bench([&] { simdunpack(buffer, backdata, bit); });

    for (uint32_t z = 0; z < length; ++z) {
      assert(backdata[z] == data[z]);
    }
    print_row(bit, length, pack, unpack, have_counters);
    free(data);
    free(buffer);
    free(backdata);
  }
  printf("\n\n"); /* two blank lines please gnuplot */
}

static void demo128_d1(bool have_counters) {
  const uint32_t length = 128;
  print_header("demo128_d1 (differential, SSE2 / NEON)", length);
  for (uint32_t bit = 1; bit <= 32; ++bit) {
    uint32_t *data = get_random_array_from_bit_width_d1(length, bit);
    __m128i *buffer = (__m128i *)malloc(length * sizeof(uint32_t));
    uint32_t *backdata = (uint32_t *)malloc(length * sizeof(uint32_t));

    event_aggregate pack =
        bench([&] { simdpackwithoutmaskd1(0, data, buffer, bit); });
    event_aggregate unpack =
        bench([&] { simdunpackd1(0, buffer, backdata, bit); });

    for (uint32_t z = 0; z < length; ++z) {
      assert(backdata[z] == data[z]);
    }
    print_row(bit, length, pack, unpack, have_counters);
    free(data);
    free(buffer);
    free(backdata);
  }
  printf("\n\n");
}

#ifdef __AVX2__
static void demo256(bool have_counters) {
  const uint32_t length = 256;
  print_header("demo256 (AVX2)", length);
  for (uint32_t bit = 1; bit <= 32; ++bit) {
    uint32_t *data = get_random_array_from_bit_width(length, bit);
    __m256i *buffer = (__m256i *)malloc(length * sizeof(uint32_t));
    uint32_t *backdata = (uint32_t *)malloc(length * sizeof(uint32_t));

    event_aggregate pack =
        bench([&] { avxpackwithoutmask(data, buffer, bit); });
    event_aggregate unpack = bench([&] { avxunpack(buffer, backdata, bit); });

    for (uint32_t z = 0; z < length; ++z) {
      assert(backdata[z] == data[z]);
    }
    print_row(bit, length, pack, unpack, have_counters);
    free(data);
    free(buffer);
    free(backdata);
  }
  printf("\n\n");
}
#endif /* __AVX2__ */

#ifdef __AVX512F__
static void demo512(bool have_counters) {
  const uint32_t length = 512;
  print_header("demo512 (AVX-512)", length);
  for (uint32_t bit = 1; bit <= 32; ++bit) {
    uint32_t *data = get_random_array_from_bit_width(length, bit);
    __m512i *buffer = (__m512i *)malloc(length * sizeof(uint32_t));
    uint32_t *backdata = (uint32_t *)malloc(length * sizeof(uint32_t));

    event_aggregate pack =
        bench([&] { avx512packwithoutmask(data, buffer, bit); });
    event_aggregate unpack =
        bench([&] { avx512unpack(buffer, backdata, bit); });

    for (uint32_t z = 0; z < length; ++z) {
      assert(backdata[z] == data[z]);
    }
    print_row(bit, length, pack, unpack, have_counters);
    free(data);
    free(buffer);
    free(backdata);
  }
  printf("\n\n");
}
#endif /* __AVX512F__ */

int main() {
  bool have_counters = counters::has_performance_counters();
  if (!have_counters) {
    printf("# Note: hardware performance counters are unavailable, so cycle\n"
           "# counts are shown as n/a. On Apple Silicon / Linux, try sudo.\n");
  }
  demo128(have_counters);
  demo128_d1(have_counters);
#ifdef __AVX2__
  demo256(have_counters);
#endif
#ifdef __AVX512F__
  demo512(have_counters);
#endif
  return EXIT_SUCCESS;
}
