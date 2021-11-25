#ifndef PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP

#include "graphCOO.hpp"
#include <omp.h>
#ifdef USE_VCL_LIB
#include <VCL2/vectorclass.h>
#endif
#include <cassert>
#include <limits>
#include "mkl_spblas.h"

void csrMultiRow(const int NOVertices, const int *csrRowPtr,
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
            const float *weights, float *flow){
    #pragma omp parallel for
    for (int i = 0; i < NOVertices; ++i) {
        int start = csrRowPtr[i];
        int end = csrRowPtr[i + 1];
        for (int j = start; j < end; ++j) {
            flow[i] += csrVal[j] * weights[csrColInd[j]];
        }
    }
}

void const_vcl_16_row(const int NOVertices, const int *csrRowPtr,
                      const int *csrColInd, const float *csrVal,
                      const float *weights, float *flow){
    #pragma omp parallel for
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
        flow[i] = horizontal_add(multiplication);
    }
}

void const_vcl_16_transpose(const int NOVertices, const int *csrRowPtr,
                            const int *csrColInd, const float *csrVal,
                            const float *weights, float *flow){
    #pragma omp parallel for
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
        multiplication.store(flow + i);
    }
}

void vcl_16_transpose(const int NOVertices, const int *csrRowPtr,
                      const int *csrColInd, const float *csrVal,
                      const float *weights, float *flow){
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
                    flow[i + j] = multiplication[j];
                } else {
                    break;
                }
            }
        } else {
            multiplication.store(flow + i);
        }
    }
}

void vcl_16_row_load(const int NOVertices, const int *csrRowPtr,
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

void vcl_16_row_lookup(const int NOVertices, const int *csrRowPtr,
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

void vcl_16_row_partial_load(const int NOVertices, const int *csrRowPtr,
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

void vcl_16_row_cutoff(const int NOVertices, const int *csrRowPtr,
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

void vcl_16_row_multiple_load(const int NOVertices, const int *csrRowPtr,
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

class GraphCSR : public AbstractGraph {
private:
    std::vector<float> csrVal;
    std::vector<int> csrColInd;
    std::vector<int> csrRowPtr;
    std::vector<float> weights;
    std::vector<float> flow;
    Type type;
    const int NOVertices;
    int NONonZeros;
    int maxLength = 0;
    int vectorLanes;
    const int vectorRownum = 2; //number of rows that a vector calculate simultaneously
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
        assert(NOVertices == flow.size());
        if (type == NAIVE) {
            naive(NOVertices, csrRowPtr.data(), csrColInd.data(),
                  csrVal.data(), weights.data(), flow.data());
        } else if (type == OPENMP){
            openmp(NOVertices, csrRowPtr.data(), csrColInd.data(),
                   csrVal.data(), weights.data(), flow.data());
        } else if(type == CONST_VCL16_ROW) {
            const_vcl_16_row(NOVertices, csrRowPtr.data(), csrColInd.data(),
                             csrVal.data(), weights.data(), flow.data());
        } else if (type == CONST_VCL16_TRANSPOSE) { 
            const_vcl_16_transpose(NOVertices, csrRowPtr.data(), csrColInd.data(),
                                   csrVal.data(), weights.data(), flow.data());
        } else if (type == VCL_16_ROW) {
            vcl_16_row_load(NOVertices, csrRowPtr.data(), csrColInd.data(),
                            csrVal.data(), weights.data(), flow.data());
        } else if (type == VCL_16_ROW_LOOKUP) {
            vcl_16_row_lookup(NOVertices, csrRowPtr.data(), csrColInd.data(),
                              csrVal.data(), weights.data(), flow.data()); 
        } else if (type == VCL_16_ROW_PARTIAL_LOAD) {
            vcl_16_row_partial_load(NOVertices, csrRowPtr.data(), csrColInd.data(),
                                    csrVal.data(), weights.data(), flow.data()); 
        } else if (type == VCL_16_ROW_CUTOFF) {
            vcl_16_row_cutoff(NOVertices, csrRowPtr.data(), csrColInd.data(),
                              csrVal.data(), weights.data(), flow.data()); 
        } else if (type == VCL_16_ROW_MULTIPLE_LOAD) {
            vcl_16_row_multiple_load(NOVertices, csrRowPtr.data(), csrColInd.data(),
                                     csrVal.data(), weights.data(), flow.data());
        } else if (type == VCL_MULTIROW) {
            csrMultiRow(NOVertices, csrRowPtr.data(), csrColInd.data(),
                        csrVal.data(), weights.data(), flow.data());
        } else if (type == VCL_16_TRANSPOSE) {
            vcl_16_row_load(NOVertices, csrRowPtr.data(), csrColInd.data(),
                            csrVal.data(), weights.data(), flow.data());
        }
    }

public:
    explicit GraphCSR(GraphCOO& graph, Type t) : NOVertices(graph.getNOVertices()), type(t), flow(NOVertices) {
        if (t == VCL_16_TRANSPOSE) graph.transpose();
        std::vector<value> matrix = graph.getNeighbourMatrix();
        weights = graph.getWeights();
        flow.resize(weights.size());
        int actualRow = 0;
        csrRowPtr.push_back(0);
        int counter = 0;
        for(value const &v : matrix){
            while(v.row != actualRow){
                actualRow++;
                counter = 0;
                csrRowPtr.push_back(static_cast<int>(csrColInd.size()));
            }
            counter++;
            if (counter > maxLength) {
                maxLength = counter;
            }
            if (t == VCL_16_TRANSPOSE) {
                csrColInd.push_back(v.row);
            } else {
                csrColInd.push_back(v.col);
            }
            csrVal.push_back(v.val);
        }
        // NONonZeroElements
        csrRowPtr.push_back(static_cast<int>(csrColInd.size()));
        mkl_sparse_s_create_csr(&csrA, 
            SPARSE_INDEX_BASE_ZERO,
            NOVertices,
            NOVertices,
            csrRowPtr.data(),
            csrRowPtr.data() + 1,
            csrColInd.data(),
            csrVal.data());
    }

    ~GraphCSR() { mkl_sparse_destroy(csrA); }

    double measureMKL() {
        std::vector<double> res;
        for (int i = 0; i < 10; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            getWeightedFlowMKL();
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            res.push_back(duration.count());
        }
        double sum = 0;
        for (double re : res) { sum += re; }
        return sum / res.size();
    }

    double bandWidth() {
      double time = this->measure().first.mean;
      double bytes = static_cast<double>(
          sizeof(float) * (weights.size() + csrVal.size() + 2 * flow.size()));
      // Gigabyte per second
      return (bytes / 1000) / time;
    }

    double bandWidthMKL() {
      double time = this->measureMKL();
      double bytes = static_cast<double>(
          sizeof(float) * (weights.size() + csrVal.size() + 2 * flow.size()));
      // Gigabyte per second
      return (bytes / 1000) / time;
    }

    double getBandWidth(double time_s) override {
      double bytes = static_cast<double>(
          sizeof(float) * (weights.size() + csrVal.size() + 2 * flow.size()));
      // Gigabyte per second
      return bytes * 1e-9 / time_s;
    }

    float *getResult() override {
      return flow.data();
    }

    std::pair<measurement_result, measurement_result> measureMKL_and_bw() {
      constexpr int amortizationCount = 10;
      auto measure = [&]() {
        for (int n = 0; n < amortizationCount; ++n) {
          getWeightedFlowMKL();
        }
      };

      auto getbw = [&](double time_s) { return getBandWidth(time_s); };

      return measure_func(measure, getbw, amortizationCount);
    }

};



#endif//PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP
