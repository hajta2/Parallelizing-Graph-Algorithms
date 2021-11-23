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
           const int *csrColInd, const float *csrVal,
           const float *weights, float *flow) {
    for (int i = 0; i < NOVertices; ++i) {
            int start = csrRowPtr[i];
            int end = csrRowPtr[i + 1];
            for (int j = start; j < end; ++j) {
                flow[i] += csrVal[j] * weights[csrColInd[j]];
            }
    }
}

void openmp(const int NOVertices, const int *csrRowPtr,
            const int *csrColInd, const float *csrVal,
            const float *weights, float *flow) {
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
            svfloat32_t value = svld1_f32(pg, &(csrVal[j]));
            svint32_t column = svld1_s32(pg, &(csrColInd[j]));
            svfloat32_t x_val = svld1_gather_index(pg, weights, column);
            svsum = svmla_m(pg, svsum, value, x_val);
        }
        flow[i] = svaddv_f32(svptrue_b32(), svsum);
    } 
}

void sve_4(const int NOVertices, const int *csrRowPtr,
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
            svfloat32_t value1 = svld1_f32(pg, &(csrVal[j]));
            svfloat32_t value2 = svld1_f32(pg, &(csrVal[j + svcntw()]));
            svfloat32_t value3 = svld1_f32(pg, &(csrVal[j + 2 * svcntw()]));
            svfloat32_t value4 = svld1_f32(pg, &(csrVal[j + 3 * svcntw()]));
            svint32_t column1 = svld1_s32(pg, &(csrColInd[j]));
            svint32_t column2 = svld1_s32(pg, &(csrColInd[j + 1 * svcntw()]));
            svint32_t column3 = svld1_s32(pg, &(csrColInd[j + 2 * svcntw()]));
            svint32_t column4 = svld1_s32(pg, &(csrColInd[j + 3 * svcntw()]));
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

void sve_multiRow(const int NOVertices, const int *csrRowPtr,
                  const int *csrColInd, const float *csrVal,
                  const float *weights, float *flow) {
    #pragma omp parallel for
    for (int i = 0; i < NOVertices; i += 2) {
        if (i + 1 == NOVertices) {
            continue;
        } else {
            uint32_t start=csrRowPtr[i];
            uint32_t start1=csrRowPtr[i + 1];
            uint32_t end = csrRowPtr[end];

            svbool_t pg16;
            svbool_t pg32;
            svfloat32_t svsum = svdup_f32(0.0);
            int numiter = std::max(start1 - start, end - start1);
            for (int j = start; j < numiter; j+= svcntw()/2) {
                int indlo = start + j;
                int indhi = start1 + j;
                pg1 = svwhilelt_b32(indlo, start2);
                pg2 = svwhilelt_b32(indhi, end);
                svfloat32_t valuelo = svld1_f32(pg16, csrVal + indlo);
                svfloat32_t valuehi = svld1_f32(pg16, csrVal + indhi);

                svfloat32_t value = svsplice_f32(pg32, valuelo, valuehi);

                svint32_t columnlo = svld1_s32(pg16, csrColInd + indlo);
                svint32_t columnhi = svld1_s32(pg16, csrColInd + indhi);

                svint32_t column = svsplice_s32(pg32, columnlo, columnhi);
                svfloat32_t x_val = svld1_gather_index(pg32, weights, column);

                svsum = svmla_m(pg32, svsum, svsplice_f32(pg32, valuelo, valuehi), x_val);
            }
            flow[2 * i] = svaddv_f32(svdupq_b32(true, true, false, false), svsum);
            flow[2* i + 1] = svaddv_f32(svdupq_b32(false, false, true, true), svsum);
        }
    }
}

class GraphCSR : public AbstractGraph {
private:
    std::vector<float> csrVal;
    std::vector<int> csrColInd;
    std::vector<int> csrRowPtr;
    std::vector<float> weights;
    std::vector<float> flow;    
    Type type;
    const int NOVertices;
    armpl_spmat_t csrA;

    void getWeightedFlowARM() {
        armpl_status_t result = armpl_spmv_exec_s(ARMPL_SPARSE_OPERATION_NOTRANS,
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
                  csrVal.data(), weights.data(), flow.data());
        } else if (type == SVE4) {
            sve_4(NOVertices, csrRowPtr.data(), csrColInd.data(),
                  csrVal.data(), weights.data(), flow.data());
        } else if (type == SVE_MULTIROW) {
            sve_multiRow(NOVertices, csrRowPtr.data(), csrColInd.data(),
                         csrVal.data(), weights.data(), flow.data());
        }
    }

public:
    explicit GraphCSR(GraphCOO& graph, Type t) : NOVertices(graph.getNOVertices()), type(t) {
        std::vector<value> matrix = graph.getNeighbourMatrix();
        weights = graph.getWeights();
        flow.resize(weights.size());
        int actualRow = 0;
        csrRowPtr.push_back(0);
        for(value const &v : matrix){
            while(v.row != actualRow){
                actualRow++;
                csrRowPtr.push_back(static_cast<int>(csrColInd.size()));
            }
            csrColInd.push_back(static_cast<int>(v.col));
            csrVal.push_back(static_cast<float>(v.val));
        }
        int NONonZeros = static_cast<int>(csrColInd.size());
        csrRowPtr.push_back(NONonZeros);
        armpl_status_t armMatrix = armpl_spmat_create_csr_s(
            &csrA,
            NOVertices,
            NOVertices,
            csrRowPtr.data(),
            csrColInd.data(),
            csrVal.data(),
            0
        );
        assert( armMatrix == ARMPL_STATUS_SUCCESS );
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
