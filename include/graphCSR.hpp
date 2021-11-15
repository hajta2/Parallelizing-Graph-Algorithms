#ifndef PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP

#include "graphCOO.hpp"
#include <omp.h>
#include <cassert>
#include <armpl.h>
#include <limits>
#include <arm_sve.h> 

class GraphCSR : public AbstractGraph {
private:
    std::vector<float> csrVal;
    std::vector<int> csrColInd;
    std::vector<int> csrRowPtr;
    std::vector<float> weights;
    std::vector<float> flow;
    std::vector<double> doubleWeights;
    std::vector<double> doubleVal;
    std::vector<double> doubleFlow;
    Type type;
    const int NOVertices;
    armpl_spmat_t csrA;

    void getWeightedFlowARM() {
        std::vector<float> res(NOVertices);
        armpl_status_t result = armpl_spmv_exec_s(ARMPL_SPARSE_OPERATION_NOTRANS,
            1.0f,
            csrA,
            weights.data(),
            0.0f,
            res.data()
        );
        assert(result == ARMPL_STATUS_SUCCESS);
    }

    void naiveFlow() {
        std::vector<float> res(NOVertices);
        for (int i = 0; i < NOVertices; ++i) {
            int start = csrRowPtr[i];
            int end = csrRowPtr[i + 1];
            for (int j = start; j < end; ++j) {
                res[i] += csrVal[j] * weights[csrColInd[j]];
            }
        }
    }

    void OPENMPFlow() {
        std::vector<float> res(NOVertices);
        #pragma omp parallel for
        for (int i = 0; i < NOVertices; ++i) {
            int start = csrRowPtr[i];
            int end = csrRowPtr[i + 1];
            for (int j = start; j < end; ++j) {
                res[i] += csrVal[j] * weights[csrColInd[j]];
            }
        }
    }

    void SVEFlow() {
        //std::vector<double> res(NOVertices);
        #pragma omp parallel for
        for (int i = 0; i < NOVertices; ++i) {
            uint64_t idx = 0;
            uint64_t start = csrRowPtr[i];
            uint64_t end = csrRowPtr[i + 1];

            svbool_t pg;
            double *val = &(doubleVal[start]);
            int *col = &(csrColInd[start]);
            svfloat64_t svsum = svdup_f64(0.0);
            for (uint64_t j = start; j < end; j += svcntd()) {
                pg = svwhilelt_b64(idx, end - start);
                svfloat64_t value = svld1_f64(pg, val + idx);
                svuint64_t column = svld1sw_u64(pg, col + idx);
                svfloat64_t x_val = svld1_gather_index(pg, doubleWeights.data(), column);
                svsum = svmla_m(pg, svsum, value, x_val);
                idx += svcntd();
            }
            svst1(svptrue_b64(), &doubleFlow[i], svsum);
        } 
    }

    void getWeightedFlow() override {
        if (type == NAIVE) {
            naiveFlow();
        } else if (type == OPENMP){
            OPENMPFlow();
        } else if (type == SVE) {
            SVEFlow();
        }
    }

public:
    explicit GraphCSR(GraphCOO& graph, Type t) : NOVertices(graph.getNOVertices()), type(t) {
        std::vector<value> matrix = graph.getNeighbourMatrix();
        weights = graph.getWeights();
        int actualRow = 0;
        csrRowPtr.push_back(0);
        for(value const &v : matrix){
            while(v.row != actualRow){
                actualRow++;
                csrRowPtr.push_back(static_cast<int>(csrColInd.size()));
            }
            csrColInd.push_back(v.col);
            csrVal.push_back(v.val);
        }
        //NONonZeroElements
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

        std::vector<double> tmpVal(csrVal.size());
        std::vector<double> tmpWeights(weights.size());
        std::vector<double> tmpFlow(csrVal.size());
        std::copy(csrVal.begin(),csrVal.end(),tmpVal.begin());
        std::copy(weights.begin(),weights.end(),tmpWeights.begin());
        doubleVal = tmpVal;
        doubleWeights = tmpWeights;
        doubleFlow = tmpFlow;
    }

    ~GraphCSR() { armpl_spmat_destroy(csrA); }

    double bandWidth() {
        double time = this -> measure().first.mean;
        double bytes = 4 * (weights.size() + csrVal.size() + 2 * flow.size());
        //Gigabyte per second
        return (bytes / 1000) / time;
    }

    // double bandWidthARM() {
    //     double time = this -> measureARM();
    //     double bytes = 4 * (weights.size() + csrVal.size() + 2 * flow.size());
    //     //Gigabyte per second
    //     return (bytes / 1000) / time;
    // }

    double getBandWidth(double time_s) override {
        double bytes = 4 * (weights.size() + csrVal.size() + 2 * flow.size());
        //Gigabyte per second
        return bytes / 1000 / time_s;
    }

    float *getResult() override {
      return flow.data();
    }

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
