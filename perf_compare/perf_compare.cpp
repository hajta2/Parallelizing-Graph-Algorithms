#include <VCL2/vectorclass.h>
#include <mkl_spblas.h>

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

inline void csrMultiRow(const int NOVertices, const int *csrRowPtr,
                 const int *csrColInd, const float *m,
                 const float *v, float *flow) { 
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

        Vec16f row(low, high);

        Vec16i index_weights(lindex, hindex);
        Vec16f values =
            lookup<std::numeric_limits<int>::max()>(index_weights, v);

        res += values * row;
      }
      flow[i] = horizontal_add(res.get_low());
      flow[i + 1] = horizontal_add(res.get_high());
    }
  }
}

inline void vcl_16_row_load(const int NOVertices, const int *csrRowPtr,
                     const int *csrColInd, const float *csrVal,
                     const float *weights, float *flow){
    #pragma omp parallel for
    for(int i = 0; i < NOVertices; ++i) {
        int start = csrRowPtr[i];
        int end = csrRowPtr[i + 1];
        //Number of elements in the row
        int dataSize = end - start;                                    
        if (dataSize != 0) {
            //rounding down to the nearest lower multiple of VECTOR_SIZE
            int regularPart = dataSize & (-VECTOR_SIZE);
            //initalize the vectors and the data
            // Vec16f row, weight, multiplication = 0;
            Vec16f multiplication = 0;
            for(int j = 0; j < regularPart; j += VECTOR_SIZE) {
                Vec16f row, weight;
                float list[VECTOR_SIZE];
                float weightList[VECTOR_SIZE];
                for(int k = 0; k < VECTOR_SIZE; ++k) {
                    list[k] = (csrVal[start + j + k]);
                    weightList[k] = weights[csrColInd[start + j + k]];
                }
                row.load(list);
                weight.load(weightList);
                multiplication += row * weight;
            }
            //add the multiplication to flow[i]
            flow[i] = horizontal_add(multiplication);
            for(int j = regularPart; j < dataSize; ++j) {
                flow[i] += csrVal[start + j] * weights[csrColInd[start + j]];
            }
        }
    }
}

inline void vcl_16_row_lookup(const int NOVertices, const int *csrRowPtr,
                       const int *csrColInd, const float *csrVal,
                       const float *weights, float *flow){
    #pragma omp parallel for
    for(int i = 0; i < NOVertices; ++i) {
        int start = csrRowPtr[i];
        int end = csrRowPtr[i + 1];
        int dataSize = end - start;                                    
        if (dataSize != 0) {
            int regularPart = dataSize & (-VECTOR_SIZE);
            Vec16f row, weight, multiplication = 0;
            Vec16i index;
            for(int j = 0; j < regularPart; j += VECTOR_SIZE) {
                row.load(&(csrVal[start + j]));
                index.load(&(csrColInd[start + j]));
                weight = lookup<std::numeric_limits<int>::max()>(index, weights);
                multiplication += row * weight;
            }
            flow[i] = horizontal_add(multiplication);
            for(int j = regularPart; j < dataSize; ++j) {
                flow[i] += csrVal[start + j] * weights[csrColInd[start + j]];
            }
        }
    }
}

inline void vcl_16_row_partial_load(const int NOVertices, const int *csrRowPtr,
                             const int *csrColInd, const float *csrVal,
                             const float *weights, float *flow){
    #pragma omp parallel for
    for(int i = 0; i < NOVertices; ++i) {
        int start = csrRowPtr[i];
        int end = csrRowPtr[i + 1];
        int dataSize = end - start;                                    
        int regularPart = dataSize & (-VECTOR_SIZE);
        Vec16f multiplication = 0;
        Vec16f row, weight;
        Vec16i index;
        for(int j = 0; j < regularPart; j += VECTOR_SIZE) {
            row.load(&(csrVal[start + j]));
            index.load(&(csrColInd[start + j]));
            weight = lookup<std::numeric_limits<int>::max()>(index, weights);
            multiplication += row * weight;
        }
        row.load_partial(dataSize - regularPart, &(csrVal[start + regularPart]));
        index.load_partial(dataSize - regularPart, &(csrColInd[start + regularPart]));
        weight = lookup<std::numeric_limits<int>::max()>(index, weights);
        multiplication += row * weight;
        //add the multiplication to flow[i]
        flow[i] = horizontal_add(multiplication);
    }
}

inline void vcl_16_row_cutoff(const int NOVertices, const int *csrRowPtr,
                       const int *csrColInd, const float *csrVal,
                       const float *weights, float *flow){
    #pragma omp parallel for
    for(int i = 0; i < NOVertices; ++i) {
        int start = csrRowPtr[i];
        int end = csrRowPtr[i + 1];
        int dataSize = end - start;                                    
        Vec16f multiplication = 0;
        Vec16f row, weight;
        Vec16i index;
        for(int j = 0; j < dataSize; j += VECTOR_SIZE) {
            row.load(&(csrVal[start + j]));
            index.load(&(csrColInd[start + j]));
            weight = lookup<std::numeric_limits<int>::max()>(index, weights);
            if (dataSize - j < VECTOR_SIZE) {
                for(int k = j; k < dataSize; ++k) {
                    flow[i] += csrVal[start + k] * weights[csrColInd[start + k]];
                }
            }
            multiplication += row * weight;
        }
        flow[i] = horizontal_add(multiplication);
    }
}

inline void vcl_16_row_multiple_load(const int NOVertices, const int *csrRowPtr,
                              const int *csrColInd, const float *csrVal,
                              const float *weights, float *flow){
    #pragma omp parallel for
    for(int i = 0; i < NOVertices; ++i) {
        int start = csrRowPtr[i];
        int end = csrRowPtr[i + 1];
        int dataSize = end - start;                                    
        int regularPart = dataSize & (-VECTOR_SIZE * 4);
        Vec16f multiplication1 = 0;
        Vec16f multiplication2 = 0;
        Vec16f multiplication3 = 0;
        Vec16f multiplication4 = 0;
        Vec16f row1, weight1, row2, weight2, row3, weight3, row4, weight4;
        Vec16i index1, index2, index3, index4;
        int j;
        for(j = 0; j < regularPart; j += VECTOR_SIZE * 4) {
            row1.load(&(csrVal[start + j]));
            index1.load(&(csrColInd[start + j]));
            weight1 = lookup<std::numeric_limits<int>::max()>(index1, weights);
            multiplication1 += row1 * weight1;
            row2.load(&(csrVal[start + j + VECTOR_SIZE]));
            index2.load(&(csrColInd[start + j + VECTOR_SIZE]));
            weight2 = lookup<std::numeric_limits<int>::max()>(index2, weights);
            multiplication2 += row2 * weight2;
            row3.load(&(csrVal[start + j + 2 * VECTOR_SIZE]));
            index3.load(&(csrColInd[start + j + 2 * VECTOR_SIZE]));
            weight3 = lookup<std::numeric_limits<int>::max()>(index3, weights);
            multiplication3 += row3 * weight3;
            row4.load(&(csrVal[start + j + 3 * VECTOR_SIZE]));
            index4.load(&(csrColInd[start + j + 3 * VECTOR_SIZE]));
            weight4 = lookup<std::numeric_limits<int>::max()>(index4, weights);
            multiplication4 += row4 * weight4;
        }
        while (dataSize - j >= VECTOR_SIZE) {
            row1.load(&(csrVal[start + j]));
            index1.load(&(csrColInd[start + j]));
            weight1 = lookup<std::numeric_limits<int>::max()>(index1, weights);
            multiplication1 += row1 * weight1;
            j += VECTOR_SIZE;
        }
        row2.load_partial(dataSize - j, &(csrVal[start + j]));
        index2.load_partial(dataSize - j, &(csrColInd[start + j]));
        weight2 = lookup<std::numeric_limits<int>::max()>(index2, weights);
        multiplication2 += row2 * weight2;
        //add the multiplication to flow[i]
        flow[i] = horizontal_add(multiplication1 + multiplication2 + multiplication3 + multiplication4);
    }
}

template <typename F>
double measure(const F &func) {
    constexpr int amortizationCount = 100;
    constexpr int runCount = 20;

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
        << csrRowPtr.back() / static_cast<double>(N) << "\n";

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
    vcl_16_row_load(N, csrRowPtr.data(), csrColInd.data(), csrVal.data(),
                    weights.data(), flow.data());
    };

    auto f2 = [&]() {
    vcl_16_row_lookup(N, csrRowPtr.data(), csrColInd.data(), csrVal.data(),
                      weights.data(), flow.data());
    };

    auto f3 = [&]() {
    vcl_16_row_partial_load(N, csrRowPtr.data(), csrColInd.data(), csrVal.data(),
                            weights.data(), flow.data());
    };

    auto f4 = [&]() {
    vcl_16_row_cutoff(N, csrRowPtr.data(), csrColInd.data(), csrVal.data(),
                      weights.data(), flow.data());
    };

    auto f5 = [&]() {
    vcl_16_row_multiple_load(N, csrRowPtr.data(), csrColInd.data(), csrVal.data(),
                            weights.data(), flow.data());
    };

    (void)measure(mkl);
    (void)measure(f);
    out << "MKL; " << measure(mkl) << "\n";
    out << "MULTIROW; " << measure(f) << "\n";
    out << "LOAD;" << measure(f1) << "\n";
    out << "LOOKUP;" << measure(f2) << "\n";
    out << "PARTIAL_LOAD;" << measure(f3) << "\n";
    out << "CUTOFF;" << measure(f4) << "\n";
    out << "MULTIPLE_LOAD;" << measure(f5) << "\n";

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