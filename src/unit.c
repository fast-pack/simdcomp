/**
 * This code is released under a BSD License.
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "simdcomp.h"


int test() {
    int N = 5000 * SIMDBlockSize, gap;
    __m128i * buffer = malloc(SIMDBlockSize * sizeof(uint32_t));
    uint32_t * datain = malloc(N * sizeof(uint32_t));
    uint32_t * backbuffer = malloc(SIMDBlockSize * sizeof(uint32_t));
    for (gap = 1; gap <= 387420489; gap *= 3) {
        int k;
        printf(" gap = %u \n", gap);
        for (k = 0; k < N; ++k)
            datain[k] = k * gap;
        for (k = 0; k * SIMDBlockSize < N; ++k) {
            /*
               First part works for general arrays (sorted or unsorted)
            */
            int j;
       	    /* we compute the bit width */
            const uint32_t b = maxbits(datain + k * SIMDBlockSize);
            /* we read 128 integers at "datain + k * SIMDBlockSize" and
               write b 128-bit vectors at "buffer" */
            simdpackwithoutmask(datain + k * SIMDBlockSize, buffer, b);
            /* we read back b1 128-bit vectors at "buffer" and write 128 integers at backbuffer */
            simdunpack(buffer, backbuffer, b);/* uncompressed */
            for (j = 0; j < SIMDBlockSize; ++j) {
                if (backbuffer[j] != datain[k * SIMDBlockSize + j]) {
                    printf("bug in simdpack\n");
                    return -2;
                }
            }

	    {
                /*
                 next part assumes that the data is sorted (uses differential coding)
                */
                uint32_t offset = 0;
                /* we compute the bit width */
                const uint32_t b1 = simdmaxbitsd1(offset,
                    datain + k * SIMDBlockSize);
               /* we read 128 integers at "datain + k * SIMDBlockSize" and
                  write b1 128-bit vectors at "buffer" */
               simdpackwithoutmaskd1(offset, datain + k * SIMDBlockSize, buffer,
                    b1);
               /* we read back b1 128-bit vectors at "buffer" and write 128 integers at backbuffer */
               simdunpackd1(offset, buffer, backbuffer, b1);
               for (j = 0; j < SIMDBlockSize; ++j) {
                   if (backbuffer[j] != datain[k * SIMDBlockSize + j]) {
                       printf("bug in simdpack d1\n");
                       return -3;
                   }
               }
               offset = datain[k * SIMDBlockSize + SIMDBlockSize - 1];
	    }
        }
    }
    free(buffer);
    free(datain);
    free(backbuffer);
    printf("Code looks good.\n");
    return 0;
}

#define MAX 300
int test_simdmaxbitsd1_length() {
    uint32_t result, buffer[MAX + 1];
    int i, j;

    memset(&buffer[0], 0xff, sizeof(buffer));

    /* this test creates buffers of different length; each buffer is
     * initialized to result in the following deltas:
     * length 1: 2
     * length 2: 1 2
     * length 3: 1 1 2
     * length 4: 1 1 1 2
     * length 5: 1 1 1 1 2
     * etc. Each sequence's "maxbits" is 2. */
    for (i = 0; i < MAX; i++) {
      for (j = 0; j < i; j++)
        buffer[j] = j + 1;
      buffer[i] = i + 2;

      result = simdmaxbitsd1_length(0, &buffer[0], i + 1);
      if (result != 2) {
        printf("simdmaxbitsd1_length: unexpected result %u in loop %d\n",
                result, i);
        return -1;
      }
    }
    printf("simdmaxbitsd1_length: ok\n");
    return 0;
}

int test_simdpackedsearch() {
    uint32_t buffer[128];
    uint32_t result, initial = 0;
    int b, i;

    /* initialize the buffer */
    for (i = 0; i < 128; i++)
        buffer[i] = (uint32_t)(i + 1);

    /* this test creates delta encoded buffers with different bits, then
     * performs lower bound searches for each key */
    for (b = 1; b <= 32; b++) {
        uint32_t out[128];
        /* delta-encode to 'i' bits */
        simdpackwithoutmaskd1(initial, buffer, (__m128i *)out, b);

        printf("simdsearchd1: %d bits\n", b);

        /* now perform the searches */
        assert(simdsearchd1(initial, (__m128i *)out, b, 128, 0, &result) == 0);
        assert(result > 0);

        for (i = 1; i <= 128; i++) {
            assert(simdsearchd1(initial, (__m128i *)out, b, 128,
                                    (uint32_t)i, &result) == i - 1);
            assert(result == (unsigned)i);
        }

        assert(simdsearchd1(initial, (__m128i *)out, b, 128, 200, &result)
                        == 128);
        assert(result > 200);
    }
    printf("simdsearchd1: ok\n");
    return 0;
}

int test_simdpackedselect() {
    uint32_t buffer[128];
    uint32_t initial = 33;
    int b, i;

    /* initialize the buffer */
    for (i = 0; i < 128; i++)
        buffer[i] = (uint32_t)(initial + i);

    /* this test creates delta encoded buffers with different bits, then
     * performs lower bound searches for each key */
    for (b = 1; b <= 32; b++) {
        uint32_t out[128];
        /* delta-encode to 'i' bits */
        simdpackwithoutmaskd1(initial, buffer, (__m128i *)out, b);

        printf("simdselectd1: %d bits\n", b);

        /* now perform the searches */
        for (i = 0; i < 128; i++) {
            assert(simdselectd1(initial, (__m128i *)out, b, (uint32_t)i)
                            == initial + i);
        }
    }
    printf("simdselectd1: ok\n");
    return 0;
}

int main() {
    int r;
   
    r = test();
    if (r)
        return r;

    r = test_simdmaxbitsd1_length();
    if (r)
        return r;

    r = test_simdpackedsearch();
    if (r)
        return r;

    r = test_simdpackedselect();
    if (r)
        return r;

    return 0;
}
