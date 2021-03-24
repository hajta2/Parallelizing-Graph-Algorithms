#ifndef PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP

#include "graphCOO.hpp"
#include <omp.h>
#include "VCL/vectorclass.h"
#include <cassert>

class GraphCSR : public AbstractGraph {
private:
    std::vector<float> csrVal;
    std::vector<int> csrColInd;
    std::vector<int> csrRowPtr;
    std::vector<float> weights;
    const int NOVertices;
    struct matrix_descr descrA;
    sparse_matrix_t csrA;


    void getWeightedFlowMKL(){
        std::vector<float> res(NOVertices);
        descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE,
            1.0f,
            csrA,
            descrA,
            weights.data(),
            0.0f,
            res.data()
        );
    }

    // void getWeightedFlow() override {
    //     std::vector<int> res(NOVertices);
    //     #pragma omp parallel for
    //     for (int i = 0; i < NOVertices-1; ++i) {
    //         int start = csrRowPtr[i];
    //         int end = csrRowPtr[i + 1];
    //         for (int j = start; j < end; ++j) {
    //             res[i] += csrVal[j] * weights[csrColInd[j]];
    //         }
    //     }
    // }

        void getWeightedFlow() override {
            std::vector<float> res(NOVertices);

            for (int i = 0; i < NOVertices - 1; ++i) {
                int start = csrRowPtr[i];
                int end = csrRowPtr[i + 1];
                assert(end - start == 16);
                float list[16];
                int idx = 0;
                for (int j = start; j < end; ++j) {
                    list[idx] = (csrVal[j]);
                    idx++;
                }
                Vec16f row, weight;
                row.load(list);
                weight = weights[i];
                Vec16f multiplication = row * weight;
                res[i] = horizontal_add(multiplication);
  
            }
        }

public:
    explicit GraphCSR(GraphCOO& graph) : NOVertices(graph.getNOVertices()){
       std::vector<value> matrix = graph.getNeighbourMatrix();
       weights = graph.getWeights();
       int actualRow = 0;
       csrRowPtr.push_back(0);
       for(value const &v : matrix){
           while(v.row != actualRow){
               actualRow++;
               csrRowPtr.push_back(csrColInd.size());
           }
           csrColInd.push_back(v.col);
           csrVal.push_back(v.val);
       }
       //NONonZeroElements
       csrRowPtr.push_back(csrColInd.size());
       mkl_sparse_s_create_csr(&csrA, 
           SPARSE_INDEX_BASE_ZERO,
           NOVertices,
           NOVertices,
           csrRowPtr.data(),
           csrRowPtr.data() + 1,
           csrColInd.data(),
           csrVal.data());
    }

    double measureMKL() {
        std::vector<float> res;
        for (int i = 0; i < 10; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            getWeightedFlowMKL();
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            res.push_back((float)duration.count());
        }
        double sum = 0;
        for (float re : res) { sum += re; }
        return sum / res.size();
    }

};



#endif//PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP