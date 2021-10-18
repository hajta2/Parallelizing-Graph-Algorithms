#ifndef PARALLELIZING_GRAPH_ALGORITHMS_TIMER_H
#define PARALLELIZING_GRAPH_ALGORITHMS_TIMER_H 
#include <chrono>

struct high_resolution_timer {
  using clock      = std::chrono::high_resolution_clock;
  using time_point = clock::time_point;

  high_resolution_timer() : start_time_(take_timestamp()) {}

  double elapsed() const {
    constexpr double nano_to_seconds = 1e-9;
    return nano_to_seconds *
           static_cast<double>(take_timestamp() - start_time_);
  }

  void restart() { start_time_ = take_timestamp(); }


protected:
  static uint64_t take_timestamp() {
    return std::uint64_t(clock::now().time_since_epoch().count());
  }

private:
  uint64_t start_time_;
};


#endif /* ifndef PARALLELIZING_GRAPH_ALGORITHMS_TIMER_H */
