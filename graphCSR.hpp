#ifndef PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP

#include "graphCOO.hpp"
#include <omp.h>

class GraphCSR : public AbstractGraph {
private:
    std::vector<float> csrVal;
    std::vector<int> csrColInd;
    std::vector<int> csrRowPtr;
    std::vector<float> weights;
    const int NOVertices;
    struct matrix_descr descrA;
    sparse_matrix_t csrA;


    void getWeightedFlow() override {
        std::vector<float> res(NOVertices);
        descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE,
            1.0f,
            csrA,
            descrA,
            weights.data(),
            0.0f,
            nullptr
        );
    }

    /*void getWeightedFlow() override {
        std::vector<int> res(NOVertices);
         #pragma omp parallel for
         for (int i = 0; i < NOVertices; ++i) {
             int start = csrRowPtr[i];
             int end;
            //check if it is the last index
             end = (i == NOVertices - 1) ? (int)(csrColInd.size() - 1) : (csrRowPtr[i + 1] - csrRowPtr[i]);
             for (int j = start; j < end; ++j) {
                 res[i] += csrVal[start] * weights[csrColInd[start]];
             }
         }
     }*/

public:
    explicit GraphCSR(GraphCOO& graph) : NOVertices(graph.getNOVertices()){
       std::vector<value> matrix = graph.getNeighbourMatrix();
       weights = graph.getWeights();
       int actualRow = 0;
       csrRowPtr.push_back(0);
       for(value const &v : matrix){
           if(v.row != actualRow){
               actualRow++;
               csrRowPtr.push_back(csrColInd.size());
           }
           csrColInd.push_back(v.col);
           csrVal.push_back(v.val);
       }
       int NOZeros = NOVertices * NOVertices - csrColInd.size();
       csrRowPtr.push_back(NOZeros);
       mkl_sparse_s_create_csr(&csrA, 
           SPARSE_INDEX_BASE_ZERO,
           NOVertices,
           NOVertices,
           csrRowPtr.data(),
           csrRowPtr.data() + 1,
           csrColInd.data(),
           csrVal.data());
    }

};



#endif//PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP
