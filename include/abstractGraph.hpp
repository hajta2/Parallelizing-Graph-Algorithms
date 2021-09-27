#ifndef PARALLELIZING_GRAPH_ALGORITHMS_ABSTRACTGRAPH_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_ABSTRACTGRAPH_HPP

#include <chrono>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include "mkl_spblas.h"

enum Type {
    NAIVE,
    OPENMP,
    CONST_VCL16_ROW,
    CONST_VCL16_TRANSPOSE,
    VCL_16_ROW,
    VCL_16_TRANSPOSE
};

std::string enumString[] = {
    "NAIVE",
    "OPENMP", 
    "CONST_VCL16_ROW", 
    "CONST_VCL16_TRANSPOSE",
    "VCL_16_ROW",
    "VCL_16_TRANSPOSE"
};

class AbstractGraph {
public:
    virtual void getWeightedFlow() = 0;
    double measure() {
        std::vector<float> res;
        for (int i = 0; i < 10; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            getWeightedFlow();
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            res.push_back((float)duration.count());
        }
        double sum = 0;
        for (float re : res) { sum += re; }
        return sum / res.size();
    }
};

#endif//PARALLELIZING_GRAPH_ALGORITHMS_ABSTRACTGRAPH_HPP