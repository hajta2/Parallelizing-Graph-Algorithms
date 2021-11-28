#include <armpl.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif /* __ARM_FEATURE_SVE */

#include <CLI11.hpp>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <ostream>
#include <random>
#include <tuple>
#include <vector>

#include "timer.h"

inline void armPL(const armpl_spmat_t &csrA, const float *weights, float *flow) {
    armpl_spmv_exec_s(ARMPL_SPARSE_OPERATION_NOTRANS, 1.0f, csrA, weights, 0.0f,
                      flow
        );
}

inline void sve_1(const int NOVertices, const int *csrRowPtr,
           const int *csrColInd, const float *csrVal,
           const float *weights, float *flow) {
    #pragma omp parallel for
    for (int i = 0; i < NOVertices; ++i) {
        uint32_t start = csrRowPtr[i];
        uint32_t end = csrRowPtr[i + 1];

        svbool_t pg;
        svfloat32_t svsum = svdup_f32(0.0);
        for (uint32_t j = start; j < end; j += svcntw()) {
            pg = svwhilelt_b32(j, end);
            svfloat32_t value = svld1_f32(pg, csrVal + j);
            svint32_t column = svld1_s32(pg, csrColInd + j);
            svfloat32_t x_val = svld1_gather_index(pg, weights, column);
            svsum = svmla_m(pg, svsum, value, x_val);
        }
        flow[i] = svaddv_f32(svptrue_b32(), svsum);
    } 
}

inline void sve_2(const int NOVertices, const int *csrRowPtr,
           const int *csrColInd, const float *csrVal,
           const float *weights, float *flow) {
    #pragma omp parallel for
    for (int i = 0; i < NOVertices; ++i) {
        uint32_t start = csrRowPtr[i];
        uint32_t end = csrRowPtr[i + 1];

        svbool_t pg;
        svfloat32_t svsum1 = svdup_f32(0.0);
        svfloat32_t svsum2 = svdup_f32(0.0);
        for (uint32_t j = start; j < end; j += 2 * svcntw()) {
            pg = svwhilelt_b32(j, end);
            svfloat32_t value1 = svld1_f32(pg, csrVal + j);
            svfloat32_t value2 = svld1_f32(pg, csrVal + j + svcntw());
            svint32_t column1 = svld1_s32(pg, csrColInd + j);
            svint32_t column2 = svld1_s32(pg, csrColInd + j + 1 * svcntw());
            svfloat32_t x_val1 = svld1_gather_index(pg, weights, column1);
            svfloat32_t x_val2 = svld1_gather_index(pg, weights, column2);
            svsum1 = svmla_m(pg, svsum1, value1, x_val1);
            svsum2 = svmla_m(pg, svsum2, value2, x_val2);
        }
        flow[i] = svaddv_f32(svptrue_b32(), svsum1);
        flow[i] = svaddv_f32(svptrue_b32(), svsum2);
    }
}

inline void sve_3(const int NOVertices, const int *csrRowPtr,
           const int *csrColInd, const float *csrVal,
           const float *weights, float *flow) {
    #pragma omp parallel for
    for (int i = 0; i < NOVertices; ++i) {
        uint32_t start = csrRowPtr[i];
        uint32_t end = csrRowPtr[i + 1];

        svbool_t pg;
        svfloat32_t svsum1 = svdup_f32(0.0);
        svfloat32_t svsum2 = svdup_f32(0.0);
        svfloat32_t svsum3 = svdup_f32(0.0);
        for (uint32_t j = start; j < end; j += 3 * svcntw()) {
            pg = svwhilelt_b32(j, end);
            svfloat32_t value1 = svld1_f32(pg, csrVal + j);
            svfloat32_t value2 = svld1_f32(pg, csrVal + j + svcntw());
            svfloat32_t value3 = svld1_f32(pg, csrVal + j + 2 * svcntw());
            svint32_t column1 = svld1_s32(pg, csrColInd + j);
            svint32_t column2 = svld1_s32(pg, csrColInd + j + 1 * svcntw());
            svint32_t column3 = svld1_s32(pg, csrColInd + j + 2 * svcntw());
            svfloat32_t x_val1 = svld1_gather_index(pg, weights, column1);
            svfloat32_t x_val2 = svld1_gather_index(pg, weights, column2);
            svfloat32_t x_val3 = svld1_gather_index(pg, weights, column3);
            svsum1 = svmla_m(pg, svsum1, value1, x_val1);
            svsum2 = svmla_m(pg, svsum2, value2, x_val2);
            svsum3 = svmla_m(pg, svsum3, value3, x_val3);
        }
        flow[i] = svaddv_f32(svptrue_b32(), svsum1);
        flow[i] = svaddv_f32(svptrue_b32(), svsum2);
        flow[i] = svaddv_f32(svptrue_b32(), svsum3);
    }
}

inline void sve_4(const int NOVertices, const int *csrRowPtr,
           const int *csrColInd, const float *csrVal,
           const float *weights, float *flow) {
    #pragma omp parallel for
    for (int i = 0; i < NOVertices; ++i) {
        uint32_t start = csrRowPtr[i];
        uint32_t end = csrRowPtr[i + 1];

        svbool_t pg;
        svfloat32_t svsum1 = svdup_f32(0.0);
        svfloat32_t svsum2 = svdup_f32(0.0);
        svfloat32_t svsum3 = svdup_f32(0.0);
        svfloat32_t svsum4 = svdup_f32(0.0);
        for (uint32_t j = start; j < end; j += 4 * svcntw()) {
            pg = svwhilelt_b32(j, end);
            svfloat32_t value1 = svld1_f32(pg, csrVal + j);
            svfloat32_t value2 = svld1_f32(pg, csrVal + j + svcntw());
            svfloat32_t value3 = svld1_f32(pg, csrVal + j + 2 * svcntw());
            svfloat32_t value4 = svld1_f32(pg, csrVal + j + 3 * svcntw());
            svint32_t column1 = svld1_s32(pg, csrColInd + j);
            svint32_t column2 = svld1_s32(pg, csrColInd + j + 1 * svcntw());
            svint32_t column3 = svld1_s32(pg, csrColInd + j + 2 * svcntw());
            svint32_t column4 = svld1_s32(pg, csrColInd + j + 3 * svcntw());
            svfloat32_t x_val1 = svld1_gather_index(pg, weights, column1);
            svfloat32_t x_val2 = svld1_gather_index(pg, weights, column2);
            svfloat32_t x_val3 = svld1_gather_index(pg, weights, column3);
            svfloat32_t x_val4 = svld1_gather_index(pg, weights, column4);
            svsum1 = svmla_m(pg, svsum1, value1, x_val1);
            svsum2 = svmla_m(pg, svsum2, value2, x_val2);
            svsum3 = svmla_m(pg, svsum3, value3, x_val3);
            svsum4 = svmla_m(pg, svsum4, value4, x_val4);
        }
        flow[i] = svaddv_f32(svptrue_b32(), svsum1);
        flow[i] = svaddv_f32(svptrue_b32(), svsum2);
        flow[i] = svaddv_f32(svptrue_b32(), svsum3);
        flow[i] = svaddv_f32(svptrue_b32(), svsum4);
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
        std::cout << "Not implemented\n";
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

    armpl_spmat_t csrA;
    armpl_spmat_create_csr_s(&csrA, N, N, csrRowPtr, csrColInd, csrVal, 0
        );
    auto arm = [&]() { armPL(csrA, weights.data(), flow.data()); };

    auto f1 = [&]() {
    sve_1(N, csrRowPtr.data(), csrColInd.data(), csrVal.data(),
          weights.data(), flow.data());
    };

    auto f2 = [&]() {
    sve_2(N, csrRowPtr.data(), csrColInd.data(), csrVal.data(),
          weights.data(), flow.data());
    };

    auto f3 = [&]() {
    sve_3(N, csrRowPtr.data(), csrColInd.data(), csrVal.data(),
          weights.data(), flow.data());
    };

    auto f4 = [&]() {
    sve_4(N, csrRowPtr.data(), csrColInd.data(), csrVal.data(),
          weights.data(), flow.data());
    };



    (void)measure(arm);
    (void)measure(f1);
    out << "ARM; " << measure(arm) << "\n";
    out << "SVE_1; " << measure(f1) << "\n";
    out << "SVE_2; " << measure(f2) << "\n";
    out << "SVE_3; " << measure(f3) << "\n";
    out << "SVE_4; " << measure(f4) << "\n";

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