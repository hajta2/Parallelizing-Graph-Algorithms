#ifndef PARALLELIZING_GRAPH_ALGORITHMS_ABSTRACTGRAPH_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_ABSTRACTGRAPH_HPP

#include <chrono>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

class AbstractGraph {
public:
    virtual void getWeightedFlow() = 0;
    double measure() {
        std::vector<double> res;
        for (int i = 0; i < 10; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            getWeightedFlow();
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = duration_cast<std::chrono::microseconds>(stop - start);
            res.push_back(duration.count());
        }
        double sum = 0;
        for (int re : res) { sum += re; }
        return sum / res.size();
    }
    virtual double getDensity() = 0;
};

#endif//PARALLELIZING_GRAPH_ALGORITHMS_ABSTRACTGRAPH_HPP
