#ifndef PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP

#include "graphCOO.hpp"
#include <omp.h>
#include "VCL2/vectorclass.h"
#include <cassert>


class GraphCSR : public AbstractGraph {
private:
    std::vector<float> csrVal;
    std::vector<int> csrColInd;
    std::vector<int> csrRowPtr;
    std::vector<float> weights;
    std::vector<float> flow;
    const int NOVertices;
    struct matrix_descr descrA;
    sparse_matrix_t csrA;


    void getWeightedFlowMKL(){
        std::vector<float> res(NOVertices);
        descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
        MKL_INT result = mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE,
            1.0f,
            csrA,
            descrA,
            weights.data(),
            0.0f,
            res.data()
        );
        assert(result == SPARSE_STATUS_SUCCESS);
    }
    //Calculating the matrix-vector multiplication w/ OMP 
    /*
    void getWeightedFlow() override {

        std::vector<int> res(NOVertices);
        #pragma omp parallel for
        for (int i = 0; i < NOVertices-1; ++i) {
            int start = csrRowPtr[i];
            int end = csrRowPtr[i + 1];
            for (int j = start; j < end; ++j) {
                res[i] += csrVal[j] * weights[csrColInd[j]];
            }
        }
    }*/
    //Calculating the matrix-vector multiplication w/ 
    //vector class library (the matrix has 16 elements in a row)
    //making vector from the row and the  from the rows
    // void getWeightedFlow() override {

    //     std::vector<float> res(NOVertices);

    //     for (int i = 0; i < NOVertices - 1; ++i) {
    //         int start = csrRowPtr[i];
    //         int end = csrRowPtr[i + 1];
    //         //every row size should be 16
    //         assert(end - start == VECTOR_SIZE);
    //         //init the vectors
    //         Vec16f row, weight, multiplication;
    //         //creating the subarray
    //         float list[VECTOR_SIZE];
    //         float weightList[VECTOR_SIZE];
    //         int idx = 0;
    //         for (int j = start; j < end; ++j) {
    //             list[idx] = (csrVal[j]);
    //             weightList[idx] = weights[csrColInd[j]];
    //             idx++;
    //         }
    //         //loading the subarray into the vector
    //         row.load(list);
    //         weight.load(weightList);

    //         multiplication = row * weight;
    //         //sum of the elements in the multi
    //         res[i] = horizontal_add(multiplication);
    //     }

    //     flow = res;
    // }
    
    void getWeightedFlow() override {
        
        std::vector<float> res(NOVertices);
        for (int i = 0; i < NOVertices; i+=VECTOR_SIZE) {
            Vec16f multiplication = 0;
            for(int j = 0; j < VECTOR_SIZE; ++j) {
            Vec16f col, weight;
            float list[VECTOR_SIZE];
            float weightList[VECTOR_SIZE];
            for (int k = 0; k < VECTOR_SIZE; ++k) {
                list[k] = csrVal[i * VECTOR_SIZE + j + k * VECTOR_SIZE];
                weightList[k] = weights[csrColInd[i * VECTOR_SIZE + j + k * VECTOR_SIZE]];
            }
            col.load(list);
            weight.load(weightList);
            multiplication = col * weight + multiplication;
            }
            multiplication.store(res.data() + i);
        }

        flow = res;
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

    double bandWidth() {

        double time = this -> measure();
        double bytes = 4 * (weights.size() + csrVal.size() + 2 * flow.size());
        //Gigabyte per second
        return (bytes / 1000) / time;
    }

};



#endif//PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP