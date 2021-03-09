#ifndef MMIO_CPP_INCLUDED
#define MMIO_CPP_INCLUDED
#include <cstdio>
#include <vector>

#include "mmio.h"

// Rreads and mtx file and copies the values to a vector. Duplicating the values
// for symmetric matrices.
template <typename Float>
void mm_read_mtx_crd_vec(const char *fname, int *M, int *N,
                         std::vector<int> &row, std::vector<int> &col,
                         std::vector<Float> &val) {
  int *I = nullptr;
  int *J = nullptr;
  MM_typecode matcode;
  double *vals = nullptr;
  int num_nonzeros = 0;
  mm_read_mtx_crd(fname, M, N, &num_nonzeros, &I, &J, &vals, &matcode);
  assert(mm_is_real(matcode));
  assert(mm_is_symmetric(matcode) || mm_is_general(matcode));
  if (mm_is_symmetric(matcode)) {
    row.reserve(num_nonzeros * 2);
    col.reserve(num_nonzeros * 2);
    val.reserve(num_nonzeros * 2);
  } else {
    row.reserve(num_nonzeros);
    col.reserve(num_nonzeros);
    val.reserve(num_nonzeros);
  }
  for (int i = 0; i < num_nonzeros; ++i) {
    row.push_back(I[i]);
    col.push_back(J[i]);
    val.push_back(vals[i]);
    if (mm_is_symmetric(matcode) && I[i] != J[i]) {
      row.push_back(J[i]);
      col.push_back(I[i]);
      val.push_back(vals[i]);
    }
  }
  free(I);
  free(J);
  free(vals);
}

#endif /* ifndef MMIO_CPP_INCLUDED */
