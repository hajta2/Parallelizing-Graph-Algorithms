#include <VCL2/vectorclass.h>
#include <mkl_spblas.h>
#include <x86intrin.h>

#include <CLI11.hpp>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <ostream>
#include <random>
#include <tuple>
#include <vector>

#include "mmio_cpp.h"
#include "timer.h"

constexpr int VECTOR_SIZE = 16;

inline void csrMKL(const sparse_matrix_t &csrA, const float *v, float *flow) {
  matrix_descr descrA;
  descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
  mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f, csrA, descrA, v, 0.0f,
                  flow);
}

inline void csrMultiRow_avx_ideal(const int NOVertices, const int *csrRowPtr,
                                  const int *csrColInd, const float *m,
                                  const float *v, float *flow) {
#pragma omp parallel for
  for (int i = 0; i < NOVertices / 2; i++) {
    // load 16 consecutive elemnets: 2x8 element--> two rows
    // do add, and reduce the two half
    int start = csrRowPtr[2 * i];
    __m512 res = _mm512_setzero_ps();
    __m512 row = _mm512_loadu_ps(m + start);
    __m512i vidx = _mm512_loadu_epi32(csrColInd + start);
    __m512 values = _mm512_i32gather_ps(vidx, v, sizeof(float));
    res += values * row;
    Vec16f res_vcl(res);
    flow[2 * i] = horizontal_add(res_vcl.get_low());
    flow[2 * i + 1] = horizontal_add(res_vcl.get_high());
  }
}

inline void compute_single_row(int row_idx, int start, int end, const float *m,
                               const int *csrColInd, const float *v,
                               float *flow) {
  int dataSize = end - start;
  if (dataSize != 0) {
    // rounding down to the nearest lower multiple of VECTOR_SIZE
    int regularPart = dataSize & (-VECTOR_SIZE);
    // initalize the vectors and the data
    Vec16f row, weight, multiplication = 0;
    Vec16i index = 0;
    for (int j = 0; j < regularPart; j += VECTOR_SIZE) {
      row.load(m + start + j);
      index.load(csrColInd + start + j);
      weight = lookup<std::numeric_limits<int>::max()>(index, v);
      multiplication += row * weight;
    }
    row.load_partial(dataSize - regularPart, m + start + regularPart);
    index.load_partial(dataSize - regularPart, csrColInd + start + regularPart);
    weight = lookup<std::numeric_limits<int>::max()>(index, v);
    multiplication += weight * row;
    // add the multiplication to res[i]
    flow[row_idx] += horizontal_add(multiplication);
  }
}

void print(const __m512 p) {
  for (int c = 0; c < 16; ++c) {
    printf("%3.2g ", p[c]);
  }
  std::cout << "\n";
}

void print(const __m512i p) {
  int arr[16] = {};
  _mm512_store_epi32(arr, p);
  for (int c = 0; c < 16; ++c) {
    printf("%3d ", arr[c]);
  }
  std::cout << "\n";
}

inline void csrMultiRow_avx_v2(const int NOVertices, const int *csrRowPtr,
                               const int *csrColInd, const float *m,
                               const float *v, float *flow) {
#pragma omp parallel for
  for (int i = 0; i < (NOVertices - 1) / 2 + 1; i++) {
    if (2 * i + 1 == NOVertices) {
      compute_single_row(2 * i, csrRowPtr[2 * i], csrRowPtr[2 * i + 1], m,
                         csrColInd, v, flow);
    } else {
      int start = csrRowPtr[2 * i];
      int start2 = csrRowPtr[2 * i + 1];
      int end = csrRowPtr[2 * i + 2];

      int idx_start[16] = {
          start,      start + 1,  start + 2,  start + 3,
          start + 4,  start + 5,  start + 6,  start + 7,
          start2,     start2 + 1, start2 + 2, start2 + 3,
          start2 + 4, start2 + 5, start2 + 6, start2 + 7,
      };
      int idx_end[16] = {
          start2, start2, start2, start2, start2, start2, start2, start2,
          end,    end,    end,    end,    end,    end,    end,    end,
      };
      __m512i idx = _mm512_loadu_epi32(idx_start);
      __m512i lim = _mm512_loadu_epi32(idx_end);
      __m512i inc = _mm512_set1_epi32(8);
      __m512 zeros = _mm512_setzero_ps();
      __m512i zerosi = _mm512_setzero_epi32();

      __m512 res = _mm512_set1_ps(0.0f);
      int numiter = std::max(start2 - start, end - start2);
      for (int j = 0; j < numiter; j += 8, _mm512_add_epi32(idx, inc)) {
        __mmask16 mask = _mm512_cmplt_epi32_mask(idx, lim);

        __m512 row =
            _mm512_mask_i32gather_ps(zeros, mask, idx, m, sizeof(float));
        __m512i vidx = _mm512_mask_i32gather_epi32(zerosi, mask, idx, csrColInd,
                                                   sizeof(int));
        __m512 values =
            _mm512_mask_i32gather_ps(zeros, mask, vidx, v, sizeof(float));
        res += values * row;
      }
      Vec16f res_vcl(res);
      flow[2 * i] = horizontal_add(res_vcl.get_low());
      flow[2 * i + 1] = horizontal_add(res_vcl.get_high());
    }
  }
}

inline void csrMultiRow_avx_v3(const int NOVertices, const int *csrRowPtr,
                               const int *csrColInd, const float *m,
                               const float *v, float *flow) {
#pragma omp parallel for
  for (int i = 0; i < (NOVertices - 1) / 2 + 1; i++) {
    if (2 * i + 1 == NOVertices) {
      compute_single_row(2 * i, csrRowPtr[2 * i], csrRowPtr[2 * i + 1], m,
                         csrColInd, v, flow);
    } else {
      int start = csrRowPtr[2 * i];
      int start2 = csrRowPtr[2 * i + 1];
      int end = csrRowPtr[2 * i + 2];

      __m512 zeros = _mm512_setzero_ps();

      __m512 res = _mm512_setzero_ps();
      int numiter = std::max(start2 - start, end - start2);
      for (int j = 0; j < numiter; j += 8) {
        int ind1 = start + j;
        int ind2 = start2 + j;
        __mmask8 masklo = __mmask8((1 << (start2 - ind1)) - 1);
        __mmask8 maskhi = __mmask8((1 << (end - ind2)) - 1);
        __mmask16 mask = __mmask16((maskhi << 8) + masklo);

        __m256 rowlo = _mm256_maskz_loadu_ps(masklo, m + ind1);
        __m256 rowhi = _mm256_maskz_loadu_ps(maskhi, m + ind2);
        // __m256 rowlo = _mm256_loadu_ps(m + ind1);
        // __m256 rowhi = _mm256_loadu_ps(m + ind2);
        __m512 row =
            _mm512_insertf32x8(_mm512_castps256_ps512(rowlo), rowhi, 1);

        __m256i vidxlo = _mm256_maskz_loadu_epi32(masklo, csrColInd + ind1);
        __m256i vidxhi = _mm256_maskz_loadu_epi32(maskhi, csrColInd + ind2);
        // __m256i vidxlo = _mm256_loadu_epi32(csrColInd + ind1);
        // __m256i vidxhi = _mm256_loadu_epi32(csrColInd + ind2);
        __m512i vidx =
            _mm512_inserti32x8(_mm512_castsi256_si512(vidxlo), vidxhi, 1);

        __m512 values =
            _mm512_mask_i32gather_ps(zeros, mask, vidx, v, sizeof(float));
        // __m512 values = _mm512_i32gather_ps(vidx, v, sizeof(float));
        res += values * row;
      }
      Vec16f res_vcl(res);
      flow[2 * i] = horizontal_add(res_vcl.get_low());
      flow[2 * i + 1] = horizontal_add(res_vcl.get_high());
    }
  }
}

inline void csrMultiRow_avx_nolast(const int NOVertices, const int *csrRowPtr,
                                   const int *csrColInd, const float *m,
                                   const float *v, float *flow) {
#pragma omp parallel for
  for (int i = 0; i < (NOVertices - 1) / 2 + 1; i++) {
    int start = csrRowPtr[2 * i];
    int start2 = csrRowPtr[2 * i + 1];
    int end = csrRowPtr[2 * i + 2];

    int idx_start[16] = {
        start,      start + 1,  start + 2,  start + 3,  start + 4,  start + 5,
        start + 6,  start + 7,  start2,     start2 + 1, start2 + 2, start2 + 3,
        start2 + 4, start2 + 5, start2 + 6, start2 + 7,
    };
    int idx_end[16] = {
        start2, start2, start2, start2, start2, start2, start2, start2,
        end,    end,    end,    end,    end,    end,    end,    end,
    };
    __m512i idx = _mm512_loadu_epi32(idx_start);
    __m512i lim = _mm512_loadu_epi32(idx_end);
    __m512i inc = _mm512_set1_epi32(8);
    __m512 zeros = _mm512_setzero_ps();
    __m512i zerosi = _mm512_setzero_epi32();

    __m512 res = _mm512_set1_ps(0.0f);
    int numiter = std::max(start2 - start, end - start2);
    for (int j = 0; j < numiter; j += 8, _mm512_add_epi32(idx, inc)) {
      __mmask16 mask = _mm512_cmplt_epi32_mask(idx, lim);

      __m512 row = _mm512_mask_i32gather_ps(zeros, mask, idx, m, sizeof(float));
      __m512i vidx = _mm512_mask_i32gather_epi32(zerosi, mask, idx, csrColInd,
                                                 sizeof(int));
      __m512 values =
          _mm512_mask_i32gather_ps(zeros, mask, vidx, v, sizeof(float));
      res += values * row;
    }
    Vec16f res_vcl(res);
    flow[2 * i] = horizontal_add(res_vcl.get_low());
    flow[2 * i + 1] = horizontal_add(res_vcl.get_high());
  }

  if (NOVertices % 2) {
    compute_single_row(NOVertices - 1, csrRowPtr[NOVertices - 1],
                       csrRowPtr[NOVertices], m, csrColInd, v, flow);
  }
}

inline void csrMultiRow_avx(const int NOVertices, const int *csrRowPtr,
                            const int *csrColInd, const float *m,
                            const float *v, float *flow) {
#pragma omp parallel for
  for (int i = 0; i < NOVertices; i += 2) {
    if (i + 1 == NOVertices) {
      compute_single_row(2 * i, csrRowPtr[2 * i], csrRowPtr[2 * i + 1], m,
                         csrColInd, v, flow);
    } else {
      int start = csrRowPtr[i];
      int start2 = csrRowPtr[i + 1];
      int end = csrRowPtr[i + 2];

      // Vec16f res = 0;
      __m512 res = _mm512_set1_ps(0.0f);
      for (int j = 0; j + start < start2 || j + start2 < end; j += 8) {
        int ind1 = start + j;
        int ind2 = start2 + j;
        // __m512i idx = _mm512_set_epi32(ind1 + 0, ind1 + 1, ind1 + 2, ind1 +
        // 3,
        //                                ind1 + 4, ind1 + 5, ind1 + 6, ind1 +
        //                                7, ind2 + 0, ind2 + 1, ind2 + 2, ind2
        //                                + 3, ind2 + 4, ind2 + 5, ind2 + 6,
        //                                ind2 + 7);

        __m512i idx = _mm512_set_epi32(ind2 + 7, ind2 + 6, ind2 + 5, ind2 + 4,
                                       ind2 + 3, ind2 + 2, ind2 + 1, ind2 + 0,
                                       ind1 + 7, ind1 + 6, ind1 + 5, ind1 + 4,
                                       ind1 + 3, ind1 + 2, ind1 + 1, ind1 + 0);
        __m512i lim = _mm512_set_epi32(end, end, end, end, end, end, end, end,
                                       start2, start2, start2, start2, start2,
                                       start2, start2, start2);
        __mmask16 mask = _mm512_cmplt_epi32_mask(idx, lim);

        __m512 zeros = _mm512_set1_ps(0.0f);
        __m512i zerosi = _mm512_set1_epi32(0);
        __m512 row =
            _mm512_mask_i32gather_ps(zeros, mask, idx, m, sizeof(float));
        __m512i vidx = _mm512_mask_i32gather_epi32(zerosi, mask, idx, csrColInd,
                                                   sizeof(int));
        __m512 values =
            _mm512_mask_i32gather_ps(zeros, mask, vidx, v, sizeof(float));
        res += values * row;

        // Vec16i idx = {ind1 + 0, ind1 + 1, ind1 + 2, ind1 + 3,
        //               ind1 + 4, ind1 + 5, ind1 + 6, ind1 + 7,
        //               ind2 + 0, ind2 + 1, ind2 + 2, ind2 + 3,
        //               ind2 + 4, ind2 + 5, ind2 + 6, ind2 + 7};

        // Vec8f low, high;
        // Vec8i lindex, hindex;
        // if (start2 - ind1 >= 8) {
        //   low.load(m + ind1);
        //   lindex.load(csrColInd + ind1);
        // } else {
        //   low.load_partial(std::max(0, start2 - ind1), m + ind1);
        //   lindex.load_partial(std::max(0, start2 - ind1), csrColInd + ind1);
        // }
        // if (end - ind2 >= 8) {
        //   high.load(m + ind2);
        //   hindex.load(csrColInd + ind2);
        // } else {
        //   high.load_partial(std::max(0, end - ind2), m + ind2);
        //   hindex.load_partial(std::max(0, end - ind2), csrColInd + ind2);
        // }
        //
        // Vec16f row(low, high);
        //
        // Vec16i index_weights(lindex, hindex);
        // Vec16f values =
        //     lookup<std::numeric_limits<int>::max()>(index_weights, v);
        // res += values * row;
      }
      unsigned mlo = 0b0000000011111111;
      __mmask16 mask_lo = _cvtu32_mask16(mlo);
      unsigned mhi = 0b1111111100000000;
      __mmask16 mask_hi = _cvtu32_mask16(mhi);
      flow[i] = _mm512_mask_reduce_add_ps(mask_lo, res);
      flow[i + 1] = _mm512_mask_reduce_add_ps(mask_hi, res);

      // flow[i] = _mm512_castps512_ps256(res);
      // flow[i + 1] = _mm512_extractf32x8(res, 1);
      // flow[i] = horizontal_add(res.get_low());
      // flow[i + 1] = horizontal_add(res.get_high());
    }
  }
}

inline void csrMultiRow(const int NOVertices, const int *csrRowPtr,
                        const int *csrColInd, const float *m, const float *v,
                        float *flow) {
#pragma omp parallel for
  for (int i = 0; i < NOVertices; i += 2) {
    if (i + 1 == NOVertices) {
      int start = csrRowPtr[i];
      int end = csrRowPtr[i + 1];
      // Number of elements in the row
      int dataSize = end - start;
      if (dataSize != 0) {
        // rounding down to the nearest lower multiple of VECTOR_SIZE
        int regularPart = dataSize & (-VECTOR_SIZE);
        // initalize the vectors and the data
        Vec16f row, weight, multiplication = 0;
        Vec16i index = 0;
        for (int j = 0; j < regularPart; j += VECTOR_SIZE) {
          row.load(m + start + j);
          index.load(csrColInd + start + j);
          weight = lookup<std::numeric_limits<int>::max()>(index, v);
          multiplication += row * weight;
        }
        row.load_partial(dataSize - regularPart, m + start + regularPart);
        index.load_partial(dataSize - regularPart,
                           csrColInd + start + regularPart);
        weight = lookup<std::numeric_limits<int>::max()>(index, v);
        multiplication += weight * row;
        // add the multiplication to res[i]
        flow[i] += horizontal_add(multiplication);
      }
    } else {
      int start = csrRowPtr[i];
      int start2 = csrRowPtr[i + 1];
      int end = csrRowPtr[i + 2];

      Vec16f res = 0;
      for (int j = 0; j + start < start2 || j + start2 < end; j += 8) {
        int ind1 = start + j;
        int ind2 = start2 + j;
        Vec8f low, high;
        Vec8i lindex, hindex;
        if (start2 - ind1 >= 8) {
          low.load(m + ind1);
          lindex.load(csrColInd + ind1);
        } else {
          low.load_partial(std::max(0, start2 - ind1), m + ind1);
          lindex.load_partial(std::max(0, start2 - ind1), csrColInd + ind1);
        }
        if (end - ind2 >= 8) {
          high.load(m + ind2);
          hindex.load(csrColInd + ind2);
        } else {
          high.load_partial(std::max(0, end - ind2), m + ind2);
          hindex.load_partial(std::max(0, end - ind2), csrColInd + ind2);
        }

        // float weightList[VECTOR_SIZE] = {};
        // for (int k = 0; k < VECTOR_SIZE / 2; ++k) {
        //   if (start + j + k < start2)
        //     weightList[k] = v[csrColInd[start + j + k]];
        //   if (start2 + j + k < end)
        //     weightList[VECTOR_SIZE / 2 + k] = v[csrColInd[start2 + j + k]];
        // }
        Vec16f row(low, high);

        Vec16i index_weights(lindex, hindex);
        Vec16f values =
            lookup<std::numeric_limits<int>::max()>(index_weights, v);
        // Vec16f values;
        // values.load(weightList);

        res += values * row;
      }
      flow[i] = horizontal_add(res.get_low());
      flow[i + 1] = horizontal_add(res.get_high());
    }
  }
}

inline void csrMultiRow_no_branch(const int NOVertices, const int *csrRowPtr,
                                  const int *csrColInd, const float *m,
                                  const float *v, float *flow) {
#pragma omp parallel for
  for (int i = 0; i < NOVertices - 1; i += 2) {
    int start = csrRowPtr[i];
    int start2 = csrRowPtr[i + 1];
    int end = csrRowPtr[i + 2];

    Vec16f res = 0;
    for (int j = 0; j + start < start2 || j + start2 < end; j += 8) {
      int ind1 = start + j;
      int ind2 = start2 + j;
      Vec8f low, high;
      Vec8i lindex, hindex;
      if (start2 - ind1 >= 8) {
        low.load(m + ind1);
        lindex.load(csrColInd + ind1);
      } else {
        low.load_partial(std::max(0, start2 - ind1), m + ind1);
        lindex.load_partial(std::max(0, start2 - ind1), csrColInd + ind1);
      }
      if (end - ind2 >= 8) {
        high.load(m + ind2);
        hindex.load(csrColInd + ind2);
      } else {
        high.load_partial(std::max(0, end - ind2), m + ind2);
        hindex.load_partial(std::max(0, end - ind2), csrColInd + ind2);
      }

      // float weightList[VECTOR_SIZE] = {};
      // for (int k = 0; k < VECTOR_SIZE / 2; ++k) {
      //   if (start + j + k < start2)
      //     weightList[k] = v[csrColInd[start + j + k]];
      //   if (start2 + j + k < end)
      //     weightList[VECTOR_SIZE / 2 + k] = v[csrColInd[start2 + j + k]];
      // }
      Vec16f row(low, high);

      Vec16i index_weights(lindex, hindex);
      Vec16f values = lookup<std::numeric_limits<int>::max()>(index_weights, v);
      // Vec16f values;
      // values.load(weightList);

      res += values * row;
    }
    flow[i] = horizontal_add(res.get_low());
    flow[i + 1] = horizontal_add(res.get_high());
  }

  if (NOVertices % 2) {
    int start = csrRowPtr[NOVertices - 1];
    int end = csrRowPtr[NOVertices];
    // Number of elements in the row
    int dataSize = end - start;
    if (dataSize != 0) {
      // rounding down to the nearest lower multiple of VECTOR_SIZE
      int regularPart = dataSize & (-VECTOR_SIZE);
      // initalize the vectors and the data
      Vec16f row, weight, multiplication = 0;
      Vec16i index = 0;
      for (int j = 0; j < regularPart; j += VECTOR_SIZE) {
        row.load(m + start + j);
        index.load(csrColInd + start + j);
        weight = lookup<std::numeric_limits<int>::max()>(index, v);
        multiplication += row * weight;
      }
      row.load_partial(dataSize - regularPart, m + start + regularPart);
      index.load_partial(dataSize - regularPart,
                         csrColInd + start + regularPart);
      weight = lookup<std::numeric_limits<int>::max()>(index, v);
      multiplication += weight * row;
      // add the multiplication to res[i]
      flow[NOVertices - 1] += horizontal_add(multiplication);
    }
  }
}

void csr_vcl_16_row(const int NOVertices, const int *csrRowPtr,
                    const int *csrColInd, const float *csrVal,
                    const float *weights, float *flow) {
#pragma omp parallel for
  for (int i = 0; i < NOVertices; ++i) {
    int start = csrRowPtr[i];
    int end = csrRowPtr[i + 1];
    // Number of elements in the row
    int dataSize = end - start;
    if (dataSize != 0) {
      // rounding down to the nearest lower multiple of VECTOR_SIZE
      int regularPart = dataSize & (-VECTOR_SIZE);
      // initalize the vectors and the data
      Vec16f multiplication = 0;
      for (int j = 0; j < regularPart; j += VECTOR_SIZE) {
        Vec16f row, weight;
        float list[VECTOR_SIZE];
        float weightList[VECTOR_SIZE];
        for (int k = 0; k < VECTOR_SIZE; ++k) {
          list[k] = (csrVal[start + j + k]);
          weightList[k] = weights[csrColInd[start + j + k]];
        }
        row.load(list);
        weight.load(weightList);
        multiplication += row * weight;
      }
      // add the multiplication to flow[i]
      flow[i] = horizontal_add(multiplication);
      for (int j = regularPart; j < dataSize; ++j) {
        flow[i] += csrVal[start + j] * weights[csrColInd[start + j]];
      }
    }
  }
}

void csr_vcl_16_row_load(const int NOVertices, const int *csrRowPtr,
                         const int *csrColInd, const float *csrVal,
                         const float *weights, float *flow) {
#pragma omp parallel for
  for (int i = 0; i < NOVertices; ++i) {
    int start = csrRowPtr[i];
    int end = csrRowPtr[i + 1];
    // Number of elements in the row
    int dataSize = end - start;
    if (dataSize != 0) {
      // rounding down to the nearest lower multiple of VECTOR_SIZE
      int regularPart = dataSize & (-VECTOR_SIZE);
      // initalize the vectors and the data
      // Vec16f row, weight, multiplication = 0;
      Vec16f multiplication = 0;
      for (int j = 0; j < regularPart; j += VECTOR_SIZE) {
        Vec16f row, weight;
        float weightList[VECTOR_SIZE];
        for (int k = 0; k < VECTOR_SIZE; ++k) {
          weightList[k] = weights[csrColInd[start + j + k]];
        }
        row.load(csrVal + start + j);
        weight.load(weightList);
        multiplication += row * weight;
      }
      // add the multiplication to flow[i]
      flow[i] = horizontal_add(multiplication);
      for (int j = regularPart; j < dataSize; ++j) {
        flow[i] += csrVal[start + j] * weights[csrColInd[start + j]];
      }
    }
  }
}

void csr_vcl_16_row_lookup(const int NOVertices, const int *csrRowPtr,
                           const int *csrColInd, const float *csrVal,
                           const float *weights, float *flow) {
#pragma omp parallel for
  for (int i = 0; i < NOVertices; ++i) {
    int start = csrRowPtr[i];
    int end = csrRowPtr[i + 1];
    // Number of elements in the row
    int dataSize = end - start;
    if (dataSize != 0) {
      // rounding down to the nearest lower multiple of VECTOR_SIZE
      int regularPart = dataSize & (-VECTOR_SIZE);
      // initalize the vectors and the data
      Vec16f multiplication = 0;
      for (int j = 0; j < regularPart; j += VECTOR_SIZE) {
        Vec16f row, weight;
        float list[VECTOR_SIZE];
        for (int k = 0; k < VECTOR_SIZE; ++k) {
          list[k] = (csrVal[start + j + k]);
        }
        row.load(list);
        Vec16i index;
        index.load(csrColInd + start + j);
        weight = lookup<std::numeric_limits<int>::max()>(index, weights);
        multiplication += row * weight;
      }
      // add the multiplication to flow[i]
      flow[i] = horizontal_add(multiplication);
      for (int j = regularPart; j < dataSize; ++j) {
        flow[i] += csrVal[start + j] * weights[csrColInd[start + j]];
      }
    }
  }
}

void csr_vcl_16_row_load_lookup(const int NOVertices, const int *csrRowPtr,
                                const int *csrColInd, const float *csrVal,
                                const float *weights, float *flow) {
#pragma omp parallel for
  for (int i = 0; i < NOVertices; ++i) {
    int start = csrRowPtr[i];
    int end = csrRowPtr[i + 1];
    // Number of elements in the row
    int dataSize = end - start;
    if (dataSize != 0) {
      // rounding down to the nearest lower multiple of VECTOR_SIZE
      int regularPart = dataSize & (-VECTOR_SIZE);
      // initalize the vectors and the data
      Vec16f multiplication = 0;
      for (int j = 0; j < regularPart; j += VECTOR_SIZE) {
        Vec16f row, weight;
        row.load(csrVal + start + j);
        Vec16i index;
        index.load(csrColInd + start + j);
        weight = lookup<std::numeric_limits<int>::max()>(index, weights);
        multiplication += row * weight;
      }
      // add the multiplication to flow[i]
      flow[i] = horizontal_add(multiplication);
      for (int j = regularPart; j < dataSize; ++j) {
        flow[i] += csrVal[start + j] * weights[csrColInd[start + j]];
      }
    }
  }
}

void csr_vcl_16_row_ll_vec_end(const int NOVertices, const int *csrRowPtr,
                               const int *csrColInd, const float *csrVal,
                               const float *weights, float *flow) {
#pragma omp parallel for
  for (int i = 0; i < NOVertices; ++i) {
    int start = csrRowPtr[i];
    int end = csrRowPtr[i + 1];
    // Number of elements in the row
    int dataSize = end - start;
    if (dataSize != 0) {
      // rounding down to the nearest lower multiple of VECTOR_SIZE
      int regularPart = dataSize & (-VECTOR_SIZE);
      // initalize the vectors and the data
      Vec16f multiplication = 0;
      for (int j = 0; j < regularPart; j += VECTOR_SIZE) {
        Vec16f row, weight;
        row.load(csrVal + start + j);
        Vec16i index;
        index.load(csrColInd + start + j);
        weight = lookup<std::numeric_limits<int>::max()>(index, weights);
        multiplication += row * weight;
      }
      // add the multiplication to flow[i]
      Vec16f row;
      row.load_partial(dataSize - regularPart, csrVal + start + regularPart);
      Vec16i index;
      index.load_partial(dataSize - regularPart,
                         csrColInd + start + +regularPart);

      multiplication +=
          row * lookup<std::numeric_limits<int>::max()>(index, weights);
      flow[i] = horizontal_add(multiplication);
    }
  }
}

template <typename F>
double measure(const F &func) {
  constexpr int amortizationCount = 100;
  constexpr int runCount = 20;
  // constexpr int amortizationCount = 1;
  // constexpr int runCount = 1;

  double seconds = 0.0;

  for (int i = 0; i < runCount; ++i) {
    high_resolution_timer timer;
    for (int j = 0; j < amortizationCount; ++j) {
      func();
    }
    seconds += timer.elapsed();
  }

  return seconds;
}

void coo_to_csr(const std::vector<int> &row, const std::vector<int> &col,
                const std::vector<double> &vals, const int N_x, const int N_y,
                std::vector<int> &csrRowPtr, std::vector<int> &csrColInd,
                std::vector<float> &csrVal) {
  assert(N_y == N_x);
  std::vector<std::tuple<int, int, double>> v;
  v.reserve(row.size());
  for (size_t i = 0; i < row.size(); ++i) {
    v.push_back(std::make_tuple(row[i], col[i], vals[i]));
  }
  std::sort(v.begin(), v.end());

  csrRowPtr.reserve(static_cast<size_t>(N_x + 1));
  csrColInd.reserve(v.size());
  csrVal.reserve(v.size());
  int row_idx = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    const auto &[r, c, val] = v[i];
    while (row_idx != r) {
      row_idx++;
      csrRowPtr.push_back(static_cast<int>(i));
    }
    csrColInd.push_back(c);
    csrVal.push_back(static_cast<float>(val));
  }
}

int input_parse(int argc, const char *argv[], std::vector<int> &csrRowPtr,
                std::vector<int> &csrColInd, std::vector<float> &csrVal,
                std::vector<float> &weights, std::string &output_file) {
  std::string input_file = "";
  size_t N = 8192;
  size_t l = 8;

  CLI::App app;
  app.add_option("-o,--output", output_file, "Output file csv");

  auto opt_if =
      app.add_option("-i,--input", input_file, "Input mtx file to measure");
  auto opt_n =
      app.add_option("-n", N, "Input matrix size for generated matrices");
  auto opt_l = app.add_option("-l", l, "row length");

  opt_if->excludes(opt_n);
  opt_if->excludes(opt_l);
  opt_n->excludes(opt_if);
  opt_l->excludes(opt_if);

  CLI11_PARSE(app, argc, argv);

  std::mt19937_64 gen(42);
  std::uniform_real_distribution<float> dist(0, 1);
  std::uniform_int_distribution<int> dist_ind(0, 10);
  if (opt_if->count()) {
    int N_x = 0, N_y = 0;
    std::vector<int> row;
    std::vector<int> col;
    std::vector<double> vals;
    mm_read_mtx_crd_vec(input_file.c_str(), &N_x, &N_y, row, col, vals);
    coo_to_csr(row, col, vals, N_x, N_y, csrRowPtr, csrColInd, csrVal);
  } else {
    output_file = std::to_string(N) + "x" + std::to_string(l);
    csrVal.resize(l * N, 1.0);
    csrRowPtr.resize(N + 1, 0);
    for (size_t i = 1; i < N + 1; ++i) {
      csrRowPtr[i] = static_cast<int>(l * i);
    }
    csrColInd.resize(l * N, 0);
    for (size_t i = 0; i < N; ++i) {
      int start = std::max(0, static_cast<int>(i) - static_cast<int>(l));
      for (size_t j = 0; j < l; ++j) {
        csrColInd[i * l + j] = start + static_cast<int>(j);
      }
    }
    for (size_t i = 0; i < l * N; ++i) {
      csrVal[i] = dist(gen);
    }
  }
  weights.resize(csrRowPtr.size() - 1);
  for (size_t i = 0; i < N; ++i) {
    weights[i] = dist(gen);
  }

  return 0;
}

void compare_versions(int N, std::vector<int> &csrRowPtr,
                      std::vector<int> &csrColInd, std::vector<float> &csrVal,
                      const std::vector<float> &weights,
                      std::vector<float> &flow, std::string_view output_file,
                      std::ostream &out) {
  out << output_file << "; Avg row_l; "
      << csrRowPtr.back() / static_cast<double>(N) << ";";

  auto rup = [](int num, int num_by = VECTOR_SIZE) {
    return (num - 1) / num_by + 1;
  };

  int num_vectors_used = 0;
  for (int i = 0; i < N; ++i) {
    num_vectors_used += rup(csrRowPtr[i + 1] - csrRowPtr[i]);
  }
  int num_vectors_used_mr2 = 0;
  for (int i = 0; i < N; i += 2) {
    num_vectors_used_mr2 +=
        std::max(rup(csrRowPtr[i + 1] - csrRowPtr[i], VECTOR_SIZE / 2),
                 rup(csrRowPtr[i + 2] - csrRowPtr[i + 1], VECTOR_SIZE / 2));
  }
  std::cout << "efficiency;"
            << csrRowPtr[N] / static_cast<float>(num_vectors_used * VECTOR_SIZE)
            << ";"
            << csrRowPtr[N] /
                   static_cast<float>(num_vectors_used_mr2 * VECTOR_SIZE)
            << "\n";

  sparse_matrix_t csrA;
  mkl_sparse_s_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, N, N, csrRowPtr.data(),
                          csrRowPtr.data() + 1, csrColInd.data(),
                          csrVal.data());
  auto mkl = [&]() { csrMKL(csrA, weights.data(), flow.data()); };

  auto f = [&]() {
    csrMultiRow(N, csrRowPtr.data(), csrColInd.data(), csrVal.data(),
                weights.data(), flow.data());
  };
  auto f1 = [&]() {
    csrMultiRow_no_branch(N, csrRowPtr.data(), csrColInd.data(), csrVal.data(),
                          weights.data(), flow.data());
  };
  auto f2 = [&]() {
    csr_vcl_16_row(N, csrRowPtr.data(), csrColInd.data(), csrVal.data(),
                   weights.data(), flow.data());
  };
  auto f3 = [&]() {
    csr_vcl_16_row_load(N, csrRowPtr.data(), csrColInd.data(), csrVal.data(),
                        weights.data(), flow.data());
  };
  auto f4 = [&]() {
    csr_vcl_16_row_lookup(N, csrRowPtr.data(), csrColInd.data(), csrVal.data(),
                          weights.data(), flow.data());
  };
  auto f51 = [&]() {
    csr_vcl_16_row_ll_vec_end(N, csrRowPtr.data(), csrColInd.data(),
                              csrVal.data(), weights.data(), flow.data());
  };
  auto f6 = [&]() {
    csrMultiRow_avx(N, csrRowPtr.data(), csrColInd.data(), csrVal.data(),
                    weights.data(), flow.data());
  };
  auto f7 = [&]() {
    csrMultiRow_avx_v2(N, csrRowPtr.data(), csrColInd.data(), csrVal.data(),
                       weights.data(), flow.data());
  };

  auto f71 = [&]() {
    csrMultiRow_avx_nolast(N, csrRowPtr.data(), csrColInd.data(), csrVal.data(),
                           weights.data(), flow.data());
  };
  auto f72 = [&]() {
    csrMultiRow_avx_v3(N, csrRowPtr.data(), csrColInd.data(), csrVal.data(),
                       weights.data(), flow.data());
  };
  auto f8 = [&]() {
    csrMultiRow_avx_ideal(N, csrRowPtr.data(), csrColInd.data(), csrVal.data(),
                          weights.data(), flow.data());
  };
  (void)measure(mkl);
  std::vector reference(flow);
  (void)measure(f);
  out << "MKL; " << measure(mkl) << "\n";
  out << "VCL_MULTIROW; " << measure(f) << "\n";
  // out << "VCL_MULTIROW_NO_LAST; " << measure(f1) << "\n";
  // out << "VCL_MULTIROW_AVX; " << measure(f6) << "\n";
  out << "VCL_MULTIROW_AVX2; " << measure(f7) << "\n";
  // out << "VCL_MULTIROW_AVX2_NOLAST; " << measure(f71) << "\n";
  out << "VCL_MULTIROW_AVX3; " << measure(f72) << "\n";
  out << "VCL_MULTIROW_AVX_fix8; " << measure(f8) << "\n";
  // out << "VCL_16_ROW; " << measure(f2) << "\n";
  // out << "VCL_16_ROW_LOAD; " << measure(f3) << "\n";
  // out << "VCL_16_ROW_LOOKUP; " << measure(f4) << "\n";
  out << "VCL_16_ROW_LLV; " << measure(f51) << "\n";

  // float err;
  // for (size_t i = 0; i < flow.size(); ++i) {
  //   err = std::max(std::abs(flow[i] - reference[i]), err);
  // }
  // out << "err: " << err << "\n";
}

int main(int argc, const char *argv[]) {
  std::vector<int> csrRowPtr;
  std::vector<int> csrColInd;
  std::vector<float> csrVal;
  std::vector<float> weights;

  std::string output_file = "";
  int err = input_parse(argc, argv, csrRowPtr, csrColInd, csrVal, weights,
                        output_file);
  if (err) {
    return err;
  }

  std::vector<float> flow(weights.size(), 0);

  if (!output_file.ends_with(".mtx")) {
    compare_versions(static_cast<int>(weights.size()), csrRowPtr, csrColInd,
                     csrVal, weights, flow, output_file, std::cout);
  } else {
    std::ofstream ofile(output_file, std::ios_base::app);
    compare_versions(static_cast<int>(weights.size()), csrRowPtr, csrColInd,
                     csrVal, weights, flow, output_file, ofile);
  }
  return 0;
}
