#ifndef PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP

#include "graphCOO.hpp"
#include <omp.h>
#include <cassert>
#include <armpl.h>
#include <limits>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif /* __ARM_FEATURE_SVE */ 

void naive(const int NOVertices, const int *csrRowPtr,
           const int *csrColInd, const double *csrVal,
           const double *weights, double *flow) {
    for (int i = 0; i < NOVertices; ++i) {
            int start = csrRowPtr[i];
            int end = csrRowPtr[i + 1];
            for (int j = start; j < end; ++j) {
                flow[i] += csrVal[j] * weights[csrColInd[j]];
            }
    }
}

void openmp(const int NOVertices, const int *csrRowPtr,
            const int *csrColInd, const double *csrVal,
            const double *weights, double *flow) {
    #pragma omp parallel for
    for (int i = 0; i < NOVertices; ++i) {
        int start = csrRowPtr[i];
        int end = csrRowPtr[i + 1];
        for (int j = start; j < end; ++j) {
            flow[i] += csrVal[j] * weights[csrColInd[j]];
        }
    }
}

void sve_1(const int NOVertices, const int *csrRowPtr,
           const int *csrColInd, const double *csrVal,
           const double *weights, double *flow) {
    #pragma omp parallel for
    for (int i = 0; i < NOVertices; ++i) {
        uint64_t idx = 0;
        uint64_t start = csrRowPtr[i];
        uint64_t end = csrRowPtr[i + 1];

        svbool_t pg;
        const double *val = &(csrVal[start]);
        const int *col = &(csrColInd[start]);
        svfloat64_t svsum = svdup_f64(0.0);
        for (uint64_t j = start; j < end; j += svcntd()) {
            pg = svwhilelt_b64(idx, end - start);
            svfloat64_t value = svld1_f64(pg, val + idx);
            svuint64_t column = svld1sw_u64(pg, col + idx);
            svfloat64_t x_val = svld1_gather_index(pg, weights, column);
            svsum = svmla_m(pg, svsum, value, x_val);
            idx += svcntd();
        }
        svst1(svptrue_b64(), &flow[i], svsum);
    } 
}

// void sve_4(const int NOVertices, const int *csrRowPtr,
//            const int *csrColInd, const double *csrVal,
//            const double *weights, double *flow) {
//     #pragma omp parallel for
//     for (int i = 0; i < NOVertices; ++i) {
//         uint64_t idx = 0;
//         uint64_t start = csrRowPtr[i];
//         uint64_t end = csrRowPtr[i + 1];

//         svfloat64_t sum0,sum1,sum2,sum3 = svdup_f64(0.0);
//         svbool_t pg = svwhilelt_b64(idx, end-start);
//         const double *val0 = &(csrVal[start + 0*svcntd()]);
//         const double *val1 = &(csrVal[start + 1*svcntd()]);
//         const double *val2 = &(csrVal[start + 2*svcntd()]);
//         const double *val3 = &(csrVal[start + 3*svcntd()]);
//         const int *col0 = &(csrColInd[start + 0*svcntd()]);
//         const int *col1 = &(csrColInd[start + 1*svcntd()]);
//         const int *col2 = &(csrColInd[start + 2*svcntd()]);
//         const int *col3 = &(csrColInd[start + 3*svcntd()]);

//         do {
//             svfloat64_t value0 = svld1_f64(pg, val0 + idx);
//             svfloat64_t value1 = svld1_f64(pg, val1 + idx);
//             svfloat64_t value2 = svld1_f64(pg, val2 + idx);
//             svfloat64_t value3 = svld1_f64(pg, val3 + idx);
//             svuint64_t column0 = svld1sw_u64(pg, col0 + idx);
//             svuint64_t column1 = svld1sw_u64(pg, col1 + idx);
//             svuint64_t column2 = svld1sw_u64(pg, col2 + idx);
//             svuint64_t column3 = svld1sw_u64(pg, col3 + idx);
//             svfloat64_t x_val0 = svld1_gather_index(pg, weights, column0);
//             svfloat64_t x_val1 = svld1_gather_index(pg, weights, column1);
//             svfloat64_t x_val2 = svld1_gather_index(pg, weights, column2);
//             svfloat64_t x_val3 = svld1_gather_index(pg, weights, column3);
//             sum0 = svmla_m(pg, sum0, value0, x_val0);
//             sum1 = svmla_m(pg, sum1, value1, x_val1);
//             sum2 = svmla_m(pg, sum2, value2, x_val2);
//             sum3 = svmla_m(pg, sum3, value3, x_val3);
//             idx += 4*svcntd();
//             pg = svwhilelt_b64(idx, end-start);
//         } while(svptest_any(svptrue_b64(), pg));
//         svst1(svptrue_b64(), &flow[i + 0*svcntd()], sum0);
//         svst1(svptrue_b64(), &flow[i + 1*svcntd()], sum1);
//         svst1(svptrue_b64(), &flow[i + 2*svcntd()], sum2);
//         svst1(svptrue_b64(), &flow[i + 3*svcntd()], sum3);
//     }
// }

class GraphCSR : public AbstractGraph {
private:
    std::vector<double> csrVal;
    std::vector<int> csrColInd;
    std::vector<int> csrRowPtr;
    std::vector<double> weights;
    std::vector<double> flow;
    // std::vector<double> doubleWeights;
    // std::vector<double> doubleVal;
    std::vector<double> doubleFlow;
    Type type;
    const int NOVertices;
    armpl_spmat_t csrA;

    void getWeightedFlowARM() {
        armpl_status_t result = armpl_spmv_exec_d(ARMPL_SPARSE_OPERATION_NOTRANS,
            1.0f,
            csrA,
            weights.data(),
            0.0f,
            flow.data()
        );
        assert(result == ARMPL_STATUS_SUCCESS);
    }

    void getWeightedFlow() override {
        assert(NOVertices == flow.size());
        if (type == NAIVE) {
            naive(NOVertices, csrRowPtr.data(), csrColInd.data(),
                  csrVal.data(), weights.data(), flow.data());
        } else if (type == OPENMP){
            openmp(NOVertices, csrRowPtr.data(), csrColInd.data(),
                   csrVal.data(), weights.data(), flow.data());;
        } else if (type == SVE) {
            sve_1(NOVertices, csrRowPtr.data(), csrColInd.data(),
                  csrVal.data(), weights.data(), doubleFlow.data());
        }
    }

public:
    explicit GraphCSR(GraphCOO& graph, Type t) : NOVertices(graph.getNOVertices()), type(t) {
        std::vector<value> matrix = graph.getNeighbourMatrix();
        std::vector<float> tmpWeights = graph.getWeights();
        flow.resize(tmpWeights.size());
        weights.resize(tmpWeights.size());
        int actualRow = 0;
        csrRowPtr.push_back(0);
        for(value const &v : matrix){
            while(v.row != actualRow){
                actualRow++;
                csrRowPtr.push_back(static_cast<int>(csrColInd.size()));
            }
            csrColInd.push_back(static_cast<int>(v.col));
            csrVal.push_back(static_cast<double>(v.val));
        }
        int NONonZeros = static_cast<int>(csrColInd.size());
        csrRowPtr.push_back(NONonZeros);
        armpl_status_t armMatrix = armpl_spmat_create_csr_d(
            &csrA,
            NOVertices,
            NOVertices,
            csrRowPtr.data(),
            csrColInd.data(),
            csrVal.data(),
            0
        );
        assert( armMatrix == ARMPL_STATUS_SUCCESS );
        std::copy(tmpWeights.begin(), tmpWeights.end(), weights.begin());
        // std::vector<double> tmpVal(csrVal.size());
        // std::vector<double> tmpWeights(weights.size());
        std::vector<double> tmpFlow(csrVal.size());
        // std::copy(csrVal.begin(),csrVal.end(),tmpVal.begin());
        // std::copy(weights.begin(),weights.end(),tmpWeights.begin());
        // doubleVal = tmpVal;
        // doubleWeights = tmpWeights;
        doubleFlow = tmpFlow;
    }

    ~GraphCSR() { armpl_spmat_destroy(csrA); }

    double bandWidth() {
        double time = this -> measure().first.mean;
        double bytes = 4 * (weights.size() + csrVal.size() + 2 * flow.size());
        //Gigabyte per second
        return (bytes / 1000) / time;
    }

    double getBandWidth(double time_s) override {
        double bytes = 4 * (weights.size() + csrVal.size() + 2 * flow.size());
        //Gigabyte per second
        return bytes * 1e-9 / time_s;

    }

    // float *getResult() override {
    //     // std::vector<float> res(NOVertices);
    //     // std::copy(flow.begin(),flow.end(), res.begin());
    //     // return res.data();
    //     return;
    // }

   std::pair<measurement_result, measurement_result> measureARM_and_bw() {
      constexpr int amortizationCount = 10;
      auto measure = [&]() {
        for (int n = 0; n < amortizationCount; ++n) {
          getWeightedFlowARM();
        }
      };

      auto getbw = [&](double time_s) { return getBandWidth(time_s); };

      return measure_func(measure, getbw, amortizationCount);
    }

};



#endif//PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP
