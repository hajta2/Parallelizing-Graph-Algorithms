#ifndef PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP

#include "graphCOO.hpp"
#include <omp.h>
#include "VCL2/vectorclass.h"
#include <cassert>

enum Type {
    OPENMP,
    CONST_VCL16_ROW,
    CONST_VCL16_TRANSPOSE,
    VCL_16_ROW,
    VCL_16_TRANSPOSE
};

std::string enumString[] = {
    "OPENMP", 
    "CONST_VCL16_ROW", 
    "CONST_VCL16_TRANSPOSE",
    "VCL_16_ROW",
    "VCL_16_TRANSPOSE"
};


class GraphCSR : public AbstractGraph {
private:
    std::vector<float> csrVal;
    std::vector<int> csrColInd;
    std::vector<int> csrRowPtr;
    std::vector<float> weights;
    std::vector<float> flow;
    Type type;
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
    
    void getWeightedFlow() override {
        std::vector<float> res(NOVertices);
        //Calculating the matrix-vector multiplication w/ OMP 
        if (type == OPENMP){
            #pragma omp parallel for
            for (int i = 0; i < NOVertices-1; ++i) {
                int start = csrRowPtr[i];
                int end = csrRowPtr[i + 1];
                for (int j = start; j < end; ++j) {
                    res[i] += csrVal[j] * weights[csrColInd[j]];
                }
            }
        } else if(type == CONST_VCL16_ROW) {
            for (int i = 0; i < NOVertices - 1; ++i) {
                int start = csrRowPtr[i];
                int end = csrRowPtr[i + 1];
                //every row size should be 16
                assert(end - start == VECTOR_SIZE);
                //init the vectors
                Vec16f row, weight, multiplication;
                //creating the subarray
                float list[VECTOR_SIZE];
                float weightList[VECTOR_SIZE];
                int idx = 0;
                for (int j = start; j < end; ++j) {
                    list[idx] = (csrVal[j]);
                    weightList[idx] = weights[csrColInd[j]];
                    idx++;
                }
                //loading the subarray into the vector
                row.load(list);
                weight.load(weightList);
                multiplication = row * weight;
                //sum of the elements in the multiplication
                res[i] = horizontal_add(multiplication);
            }

        } else if (type == CONST_VCL16_TRANSPOSE) { 
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
        } else if (type == VCL_16_ROW) {
            #pragma omp parallel for
            for(int i = 0; i < NOVertices - 1; ++i) {
                int start = csrRowPtr[i];
                int end = csrRowPtr[i + 1];
                //Number of elements in the row
                int dataSize = end - start;                                    
                if (dataSize != 0) {
                    //rounding down to the nearest lower multiple of VECTOR_SIZE
                    int regularPart = dataSize & (-VECTOR_SIZE);
                    //initalize the vectors and the data
                    Vec16f row, weight, multiplication;
                    for(int j = 0; j < regularPart; j += VECTOR_SIZE) {
                        float list[VECTOR_SIZE];
                        float weightList[VECTOR_SIZE];
                        for(int k = 0; k < VECTOR_SIZE; ++k) {
                            list[k] = (csrVal[start + j + k]);
                            weightList[k] = weights[csrColInd[start + j + k]];
                        }
                        row.load(list);
                        weight.load(weightList);
                        multiplication = row * weight;
                    }
                    for(int j = regularPart - 1; j < dataSize; ++j) {
                        res[i] += csrVal[start + j] * weights[csrColInd[start + j]];
                    }
                    //add the multiplication to res[i]
                    res[i] += horizontal_add(multiplication);
                }
            }
        } else if (type == VCL_16_TRANSPOSE) {
            // round up NOVertices to nearest higher multiple of vectorsize
            int rowSize = (NOVertices + VECTOR_SIZE - 1) & (-VECTOR_SIZE);
            #pragma omp parallel for
            for (int i = 0; i < rowSize; i += VECTOR_SIZE) {
                Vec16f multiplication = 0;
                //searching the longest row's element
                int maxElements = 0;
                for (int j = 0; j < VECTOR_SIZE; ++j) {
                    if (i + j < NOVertices) { 
                        int NOElements = csrRowPtr[i + j + 1] - csrRowPtr[i + j];
                        if(NOElements > maxElements) {
                            maxElements = NOElements;
                        }
                    }
                }
                //summing the elements of the row's
                //int regularPart = maxElements & (-VECTOR_SIZE);
                for (int j = 0; j < maxElements; ++j) {
                    Vec16f col, weight;
                    float list[VECTOR_SIZE];
                    float weightList[VECTOR_SIZE];
                    for (int k = 0; k < VECTOR_SIZE; ++k) {
                        if( i + k > NOVertices - 1) {
                            break;
                        } else {
                            if (csrRowPtr[i + k + 1] - csrRowPtr[i + k] < j + 1) {
                                list[k] = 0;
                                weightList[k] = 0;
                            } else {
                                list[k] = csrVal[j + csrRowPtr[i + k]];
                                weightList[k] = weights[csrColInd[j + csrRowPtr[i + k]]];
                            }
                        }
                    }
                    col.load(list);
                    weight.load(weightList);
                    multiplication = col * weight + multiplication;
                }
                if (i + VECTOR_SIZE >= rowSize) {
                    for (int j = 0; j < multiplication.size(); ++j) {
                        if (i + j < NOVertices) {
                            res[i + j] = multiplication[j];
                        } else {
                            break;
                        }
                    }
                } else {
                    multiplication.store(res.data() + i);
                }
            }
        }
        flow = res;
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