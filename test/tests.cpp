#include <limits>

#include "catch.hpp"
#include "graphCOO.hpp"
#include "graphCSR.hpp"

template <typename REAL>
double abs_tolerance = std::numeric_limits<REAL>::epsilon();
template <typename REAL>
double rel_tolerance = std::numeric_limits<REAL>::epsilon();

template <typename Float>
void require_allclose(const Float *expected, const Float *actual, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    CAPTURE(i);
    CAPTURE(N);
    CAPTURE(expected[i]);
    CAPTURE(actual[i]);
    Float min_val = std::min(std::abs(expected[i]), std::abs(actual[i]));
    const double tolerance =
        abs_tolerance<Float> +
        rel_tolerance<Float> * static_cast<double>(min_val);
    CAPTURE(tolerance);
    const double diff = std::abs(static_cast<double>(expected[i] - actual[i]));
    CAPTURE(diff);
    REQUIRE(diff <= tolerance);
  }
}

TEST_CASE("test csr multirow on small", "[small]") {
  constexpr size_t N = 8;
  constexpr double density = 0.5;
  GraphCOO graphCOO(N, density);
  GraphCSR graphCSR(graphCOO, VCL_MULTIROW);
  GraphCSR graphRef(graphCOO, OPENMP);

  graphCSR.getWeightedFlow();
  graphRef.getWeightedFlow();

  require_allclose(graphRef.getResult(), graphCSR.getResult(), N);
}

TEST_CASE("test csr multirow on small odd", "[small]") {
  constexpr size_t N = 9;
  constexpr double density = 0.5;
  GraphCOO graphCOO(N, density);
  GraphCSR graphCSR(graphCOO, VCL_MULTIROW);
  GraphCSR graphRef(graphCOO, OPENMP);

  graphCSR.getWeightedFlow();
  graphRef.getWeightedFlow();

  require_allclose(graphRef.getResult(), graphCSR.getResult(), N);
}
