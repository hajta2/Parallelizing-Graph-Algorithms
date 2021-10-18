#ifndef PARALLELIZING_GRAPH_ALGORITHMS_ABSTRACTGRAPH_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_ABSTRACTGRAPH_HPP

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/framework/accumulator_set.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/concept_check.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <utility>

#include "timer.h"
#include "util.h"

enum Type {
    NAIVE,
    OPENMP,
    CONST_VCL16_ROW,
    CONST_VCL16_TRANSPOSE,
    VCL_16_ROW,
    VCL_16_TRANSPOSE,
    VCL_MULTIROW
};

std::string enumString[] = {
    "NAIVE",
    "OPENMP", 
    "CONST_VCL16_ROW", 
    "CONST_VCL16_TRANSPOSE",
    "VCL_16_ROW",
    "VCL_16_TRANSPOSE",
    "VCL_MULTIROW"
};

class AbstractGraph {
 public:
  virtual void getWeightedFlow() = 0;
  virtual double getBandWidth(double time_s) = 0;
  virtual float* getResult() = 0;

  std::pair<measurement_result, measurement_result> measure() {
    using namespace boost::accumulators;
    accumulator_set<double,
                    stats<tag::count, tag::mean, tag::median, tag::variance>>
        acc_time;
    accumulator_set<double,
                    stats<tag::count, tag::mean, tag::median, tag::variance>>
        acc_bw;

    getWeightedFlow();
    getWeightedFlow();
    int pilotCount = 10;
    int amortizationCount = 10;
    int sampleSize;
    constexpr double significance_level = 0.05;
    for (int i = 0; i < pilotCount; ++i) {
      high_resolution_timer timer;
      for (int n = 0; n < amortizationCount; ++n) {
        getWeightedFlow();
      }
      double time = timer.elapsed() * 1e3 / amortizationCount;
      acc_time(time);
      acc_bw(getBandWidth(time));
    }
    {
      // estimate required number of samples for confidence intercal with .95
      // confidence
      double Sd = std::sqrt(variance(acc_time) * pilotCount / (pilotCount - 1));
      double Sm = mean(acc_time);
      boost::math::students_t s_dist(pilotCount - 1);
      double z = boost::math::quantile(
          boost::math::complement(s_dist, significance_level / 2));
      double size = z * Sd / ((significance_level / 2) * Sm);
      sampleSize =
          std::max(pilotCount, static_cast<int>(std::ceil(size * size) + 1));
    }
    for (int i = pilotCount; i < sampleSize; ++i) {
      high_resolution_timer timer;
      for (int n = 0; n < amortizationCount; ++n) {
        getWeightedFlow();
      }
      double time = timer.elapsed() * 1e3 / amortizationCount;
      acc_time(time);
      acc_bw(getBandWidth(time));
    }
    boost::math::students_t dist(sampleSize - 1);

    double T = boost::math::quantile(
        boost::math::complement(dist, significance_level / 2));

    double Sd_time =
        std::sqrt(variance(acc_time) * sampleSize / (sampleSize - 1));
    double Sm_time = mean(acc_time);
    double w_time = T * Sd_time / sqrt(double(sampleSize));
    double Sd_bw = std::sqrt(variance(acc_bw) * sampleSize / (sampleSize - 1));
    double Sm_bw = mean(acc_bw);
    double w_bw = T * Sd_bw / sqrt(double(sampleSize));

    return std::make_pair<measurement_result, measurement_result>(
        {Sm_time, median(acc_time), Sd_time, w_time},
        {Sm_bw, median(acc_bw), Sd_bw, w_bw});
  }

  virtual ~AbstractGraph() = default;
};

#endif  // PARALLELIZING_GRAPH_ALGORITHMS_ABSTRACTGRAPH_HPP
