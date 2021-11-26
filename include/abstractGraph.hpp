#ifndef PARALLELIZING_GRAPH_ALGORITHMS_ABSTRACTGRAPH_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_ABSTRACTGRAPH_HPP

#include <utility>

#include "timer.h"
#include "util.h"

enum Type {
    NAIVE,
    OPENMP,
    CONST_VCL16_ROW,
    CONST_VCL16_TRANSPOSE,
    VCL_16_ROW,
    VCL_16_ROW_LOOKUP,
    VCL_16_ROW_PARTIAL_LOAD,
    VCL_16_ROW_CUTOFF,
    VCL_16_ROW_MULTIPLE_LOAD,
    VCL_MULTIROW,
    VCL_16_TRANSPOSE
};

std::string enumString[] = {
    "NAIVE",
    "OPENMP",
    "CONST_VCL16_ROW",
    "CONST_VCL16_TRANSPOSE",
    "VCL_16_ROW",
    "VCL_16_ROW_LOOKUP",
    "VCL_16_ROW_PARTIAL_LOAD",
    "VCL_16_ROW_CUTOFF",
    "VCL_16_ROW_MULTIPLE_LOAD",
    "VCL_MULTIROW",
    "VCL_16_TRANSPOSE"
};

std::ostream &operator<<(std::ostream &o, const Type &t) {
  return o << enumString[t];
}

template <typename F>
double measure(const F &func) {
    constexpr int amortizationCount = 100;
    constexpr int runCount = 20;

    double seconds = 0.0;

    for (int i = 0; i < runCount; ++i) {
    high_resolution_timer timer;
    for (int j = 0; j < amortizationCount; ++j) {
        func();
    }
    seconds += timer.elapsed();
    }

    return seconds;
}

class AbstractGraph {
 public:
  virtual void getWeightedFlow() = 0;
  virtual double getBandWidth(double time_s) = 0;

  double measure_result() {
    auto func = [&]() {getWeightedFlow();};
    (void)measure(func);
    return measure(func);
  }

  virtual ~AbstractGraph() = default;
};

#endif  // PARALLELIZING_GRAPH_ALGORITHMS_ABSTRACTGRAPH_HPP
