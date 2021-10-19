#pragma once
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)

#endif

#define N1 4
#include <x86intrin.h>

typedef size_t labeltype;

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif

 float compare00(const float *a, const float *b, unsigned size){
    float result = 0;
	/*
#ifdef __GNUC__
#ifdef __AVX__
#define AVX_DOT(addr1, addr2, dest, tmp1, tmp2) \
  tmp1 = _mm256_loadu_ps(addr1);                \
  tmp2 = _mm256_loadu_ps(addr2);                \
  tmp1 = _mm256_mul_ps(tmp1, tmp2);             \
  dest = _mm256_add_ps(dest, tmp1);

    __m256 sum;
    __m256 l0, l1;
    __m256 r0, r1;
    unsigned D = (size + 7) & ~7U;
    unsigned DR = D % 16;
    unsigned DD = D - DR;
    const float *l = a;
    const float *r = b;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};

    sum = _mm256_loadu_ps(unpack);
    if (DR) {
      AVX_DOT(e_l, e_r, sum, l0, r0);
    }

    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
      AVX_DOT(l, r, sum, l0, r0);
      AVX_DOT(l + 8, r + 8, sum, l1, r1);
    }
    _mm256_storeu_ps(unpack, sum);
    result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] +
             unpack[5] + unpack[6] + unpack[7];

#else
#ifdef __SSE2__
#define SSE_DOT(addr1, addr2, dest, tmp1, tmp2) \
  tmp1 = _mm_loadu_ps(addr1);                \
  tmp2 = _mm_loadu_ps(addr2);                \
  tmp1 = _mm_mul_ps(tmp1, tmp2);             \
  dest = _mm_add_ps(dest, tmp1);
    __m128 sum;
    __m128 l0, l1, l2, l3;
    __m128 r0, r1, r2, r3;
    unsigned D = (size + 3) & ~3U;
    unsigned DR = D % 16;
    unsigned DD = D - DR;
    const float *l = a;
    const float *r = b;
    const float *e_l = l + DD;
    const float *e_r = r + DD;
    float unpack[4] __attribute__((aligned(16))) = {0, 0, 0, 0};

    sum = _mm_load_ps(unpack);
    switch (DR) {
      case 12:
        SSE_DOT(e_l + 8, e_r + 8, sum, l2, r2);
      case 8:
        SSE_DOT(e_l + 4, e_r + 4, sum, l1, r1);
      case 4:
        SSE_DOT(e_l, e_r, sum, l0, r0);
      default:
        break;
    }
    for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
      SSE_DOT(l, r, sum, l0, r0);
      SSE_DOT(l + 4, r + 4, sum, l1, r1);
      SSE_DOT(l + 8, r + 8, sum, l2, r2);
      SSE_DOT(l + 12, r + 12, sum, l3, r3);
    }
    _mm_storeu_ps(unpack, sum);
    result += unpack[0] + unpack[1] + unpack[2] + unpack[3];
#else
*/
    float dot0, dot1, dot2, dot3;
    const float* last = a + size;
    const float* unroll_group = last - 3;

    // Process 4 items with each loop for efficiency.

    //printf("a[0] = %f; b[0] = %f; a[959] = %f; b[959] = %f\n", a[0], b[0], a[959], b[959]);	
    while (a < unroll_group) {
      dot0 = a[0] * b[0];
      dot1 = a[1] * b[1];
      dot2 = a[2] * b[2];
      dot3 = a[3] * b[3];
      result += dot0 + dot1 + dot2 + dot3;
      a += 4;
      b += 4;
    }
    // Process last 0-3 pixels.  Not needed for standard vector lengths. 
    while (a < last) {
      result += *a++ * *b++;
    }
//#endif
//#endif
//#endif
    return result;
  }


namespace hnswlib {
	//typedef void *labeltype;
	//typedef float(*DISTFUNC) (const void *, const void *, const void *);
	template<typename MTYPE>
	using DISTFUNC = MTYPE(*) (const void *, const void *, const void *);



	template <typename dist_t>  class AlgorithmInterface {
	public:
		//virtual void addPoint(void *, labeltype) = 0;
		virtual std::priority_queue< std::pair< dist_t, labeltype >> searchKnn(void *,int, double*) = 0;
	};
	template<typename MTYPE>
	class SpaceInterface {
	public:
		//virtual void search(void *);
		virtual size_t get_data_size() = 0;
		virtual DISTFUNC<MTYPE> get_dist_func() = 0;
		virtual void *get_dist_func_param() = 0;

	};
}
#include "L2space.h"
#include "hnswalg.h"
