#ifndef FMA_HPP
#define FMA_HPP

#include "avx2.hpp"

#include "immintrin.h"

namespace simd {

// vfmadd
static inline m128 vfmaddss(m128 a, m128 b, m128 c) {
    return _mm_fmadd_ss(a, b, c);
}
static inline m128 vfmaddps(m128 a, m128 b, m128 c) {
    return _mm_fmadd_ps(a, b, c);
}
static inline m256 vfmaddps128(m256 a, m256 b, m256 c) {
    return _mm256_fmadd_ps(a, b, c);
}
static inline m128d vfmaddsd(m128d a, m128d b, m128d c) {
    return _mm_fmadd_sd(a, b, c);
}
static inline m128d vfmaddpd128(m128d a, m128d b, m128d c) {
    return _mm_fmadd_pd(a, b, c);
}
static inline m256d vfmaddpd(m256d a, m256d b, m256d c) {
    return _mm256_fmadd_pd(a, b, c);
}

// vfnmadd
static inline m128 vfnmaddss(m128 a, m128 b, m128 c) {
    return _mm_fnmadd_ss(a, b, c);
}
static inline m128 vfnmaddps(m128 a, m128 b, m128 c) {
    return _mm_fnmadd_ps(a, b, c);
}
static inline m256 vfnmaddps128(m256 a, m256 b, m256 c) {
    return _mm256_fnmadd_ps(a, b, c);
}
static inline m128d vfnmaddsd(m128d a, m128d b, m128d c) {
    return _mm_fnmadd_sd(a, b, c);
}
static inline m128d vfnmaddpd128(m128d a, m128d b, m128d c) {
    return _mm_fnmadd_pd(a, b, c);
}
static inline m256d vfnmaddpd(m256d a, m256d b, m256d c) {
    return _mm256_fnmadd_pd(a, b, c);
}

// vfmsub
static inline m128 vfmsubss(m128 a, m128 b, m128 c) {
    return _mm_fmsub_ss(a, b, c);
}
static inline m128 vfmsubps(m128 a, m128 b, m128 c) {
    return _mm_fmsub_ps(a, b, c);
}
static inline m256 vfmsubps128(m256 a, m256 b, m256 c) {
    return _mm256_fmsub_ps(a, b, c);
}
static inline m128d vfmsubsd(m128d a, m128d b, m128d c) {
    return _mm_fmsub_sd(a, b, c);
}
static inline m128d vfmsubpd128(m128d a, m128d b, m128d c) {
    return _mm_fmsub_pd(a, b, c);
}
static inline m256d vfmsubpd(m256d a, m256d b, m256d c) {
    return _mm256_fmsub_pd(a, b, c);
}

// vfnmsub
static inline m128 vfnmsubss(m128 a, m128 b, m128 c) {
    return _mm_fnmsub_ss(a, b, c);
}
static inline m128 vfnmsubps(m128 a, m128 b, m128 c) {
    return _mm_fnmsub_ps(a, b, c);
}
static inline m256 vfnmsubps128(m256 a, m256 b, m256 c) {
    return _mm256_fnmsub_ps(a, b, c);
}
static inline m128d vfnmsubsd(m128d a, m128d b, m128d c) {
    return _mm_fnmsub_sd(a, b, c);
}
static inline m128d vfnmsubpd128(m128d a, m128d b, m128d c) {
    return _mm_fnmsub_pd(a, b, c);
}
static inline m256d vfnmsubpd(m256d a, m256d b, m256d c) {
    return _mm256_fnmsub_pd(a, b, c);
}

// vfmaddsub
static inline m128 vfmaddsubps128(m128 a, m128 b, m128 c) {
    return _mm_fmaddsub_ps(a, b, c);
}
static inline m256 vfmaddsubps(m256 a, m256 b, m256 c) {
    return _mm256_fmaddsub_ps(a, b, c);
}
static inline m128d vfmaddsubpd128(m128d a, m128d b, m128d c) {
    return _mm_fmaddsub_pd(a, b, c);
}
static inline m256d vfmaddsubpd(m256d a, m256d b, m256d c) {
    return _mm256_fmaddsub_pd(a, b, c);
}

// vfmsubadd
static inline m128 vfmsubaddps128(m128 a, m128 b, m128 c) {
    return _mm_fmsubadd_ps(a, b, c);
}
static inline m256 vfmsubaddps(m256 a, m256 b, m256 c) {
    return _mm256_fmsubadd_ps(a, b, c);
}
static inline m128d vfmsubaddpd128(m128d a, m128d b, m128d c) {
    return _mm_fmsubadd_pd(a, b, c);
}
static inline m256d vfmsubaddpd(m256d a, m256d b, m256d c) {
    return _mm256_fmsubadd_pd(a, b, c);
}

}

#endif
