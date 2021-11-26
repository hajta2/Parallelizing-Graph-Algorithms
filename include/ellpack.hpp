#ifndef PARALLELIZING_GRAPH_ALGORITHMS_ELLPACK_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_ELLPACK_HPP

#include "graphCOO.hpp"
#ifdef USE_VCL_LIB
#include <VCL2/vectorclass.h>
#endif

void naive(const int NOVertices, const int *ellpackColInd,
           const float *ellpackVal,  const float *weights,
           float *flow, int rowLength) {
    for (int i = 0; i < NOVertices; ++i) {
        for (int j = 0; j < rowLength; ++j) {
            flow[i] += ellpackVal[i * rowLength + j] * weights[ellpackColInd[i * rowLength + j]];
        }
    }
}

void openmp(const int NOVertices, const int *ellpackColInd,
            const float *ellpackVal,  const float *weights,
            float *flow, int rowLength){
    #pragma omp parallel for
    for (int i = 0; i < NOVertices; ++i) {
        for (int j = 0; j < rowLength; ++j) {
            flow[i] += ellpackVal[i * rowLength + j] * weights[ellpackColInd[i * rowLength + j]];
        }
    }
}

void ellpack(const int NOVertices, const int *ellpackColInd,
             const float *ellpackVal,  const float *weights,
             float *flow, int rowLength) {
    int regularPart = rowLength & (-VECTOR_SIZE);
    #pragma omp parallel for
    for (int i = 0; i < NOVertices; ++i) {
        Vec16f row, weight, multiplication;
        Vec16i index;
        for (int j = 0; j < regularPart; j += VECTOR_SIZE) {
            row.load(ellpackVal + i * rowLength + j);
            index.load(ellpackColInd + i * rowLength + j);
            weight = lookup<std::numeric_limits<int>::max()>(index, weights);
            multiplication = row * weight;
        }
        flow[i] += horizontal_add(multiplication);
        for (int j = regularPart - 1; j < rowLength; ++j) {
            flow[i] += ellpackVal[i * rowLength + j] * weights[ellpackColInd[i * rowLength +j ]];
        }
    }
}

class Ellpack : public AbstractGraph {
private:
    std::vector<float> weights;
    std::vector<float> ellpackVal;
    std::vector<int> ellpackColInd;
    const int NOVertices;
    int rowLength;
    std::vector<float> flow;
    Type type;
    
public:
    void getWeightedFlow() override {
        if (type == NAIVE) {
            naive(NOVertices, ellpackColInd.data(), ellpackVal.data(),
                  weights.data(), flow.data(), rowLength);
        } else if (type == OPENMP) {
            openmp(NOVertices, ellpackColInd.data(), ellpackVal.data(),
                   weights.data(), flow.data(), rowLength);
        } else if (type == VCL_16_ROW) {
            ellpack(NOVertices, ellpackColInd.data(), ellpackVal.data(),
                    weights.data(), flow.data(), rowLength);
        }
    }

    explicit Ellpack(GraphCOO graph, Type t, bool transposed) : NOVertices(graph.getNOVertices()), type(t) {
        if(transposed) {
            graph.convertToELLPACK();
        } else {
            graph.convertToTransposedELLPACK();
        }
        std::vector<value> matrix = graph.getNeighbourMatrix();
        weights = graph.getWeights();
        rowLength = graph.getEllpackRow();
        flow.resize(weights.size());
        ellpackVal.resize(rowLength * weights.size());
        ellpackColInd.resize(rowLength * weights.size());
        int actualRow = 0;
        int elementsInRow = 0;
        for (value v : matrix){
            if(actualRow != v.row) {
                actualRow++;
                elementsInRow = 0;
            }
            ellpackVal[v.row * rowLength + elementsInRow] = v.val;
            ellpackColInd[v.row * rowLength + elementsInRow] = v.col;
            elementsInRow++;
        }
    }

    double getBandWidth(double time_s) override {
      double bytes = static_cast<double>(
          sizeof(float) * (weights.size() + ellpackVal.size() + 2 * flow.size()));
      return bytes * 1e-9 / time_s;
    }
  
};

#endif//PARALLELIZING_GRAPH_ALGORITHMS_ELLPACK_HPP
