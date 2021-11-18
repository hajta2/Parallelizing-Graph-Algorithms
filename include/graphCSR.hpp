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
        uint32_t idx = 0;
        uint32_t start = csrRowPtr[i];
        uint32_t end = csrRowPtr[i + 1];

        svbool_t pg;
        const float *val = &(csrVal[start]);
        const int *col = &(csrColInd[start]);
        svfloat32_t svsum = svdup_f32(0.0);
        for (uint32_t j = start; j < end; j += svcntd()) {
            pg = svwhilelt_b32(idx, end - start);
            svfloat32_t value = svld1_f32(pg, val + idx);
            svint32_t column = svld1_s32(pg, col + idx); //32 bit load
            svfloat32_t x_val = svld1_gather_index(pg, weights, column);
            svsum = svmla_m(pg, svsum, value, x_val); //vegen horizontal add like avx
            idx += svcntd();
        }
        flow[i] = svaddv_f32(svptrue_b32(), svsum);
        //svst1(svptrue_b64(), &flow[i], svsum);
    } 
}

class GraphCSR : public AbstractGraph {
private:
    std::vector<float> csrVal;
    std::vector<int> csrColInd;
    std::vector<int> csrRowPtr;
    std::vector<float> weights;
    std::vector<float> flow;
    
    std::vector<float> sveFlow;
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
                  csrVal.data(), weights.data(), sveFlow.data());
        }
    }

public:
    explicit GraphCSR(GraphCOO& graph, Type t) : NOVertices(graph.getNOVertices()), type(t) {
        std::vector<value> matrix = graph.getNeighbourMatrix();
        weights = graph.getWeights();
        flow.resize(weights.size());
        sveFlow.resize(weights.size());
        //weights.resize(tmpWeights.size());
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
        //std::copy(tmpWeights.begin(), tmpWeights.end(), weights.begin());
        //std::vector<double> tmpFlow(csrVal.size());
        //doubleFlow = tmpFlow;
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
