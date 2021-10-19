#include <iostream>
#include <random>
#include <vector>

#include "CLI11.hpp"
#include "mmio_cpp.h"

void generate(int N, int n_per_row, std::vector<int> &row,
              std::vector<int> &col, std::vector<double> &vals) {
  // symmetric matrix -> n_per_row / 2
  n_per_row /= 2;
  row.reserve(n_per_row * N);
  col.reserve(n_per_row * N);
  vals.reserve(n_per_row * N);
  std::mt19937_64 gen(42);
  std::uniform_real_distribution dist;

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < n_per_row && (i + j + 1) < N; ++j) {
      row.push_back(i);
      col.push_back(i + j + 1);
      vals.push_back(dist(gen));
    }
  }
}

int main(int argc, const char *argv[]) {
  CLI::App app;
  std::string output_file = "mat.mtx";
  app.add_option("-o,--output", output_file, "Output mtx file");
  int N = 1e6;
  app.add_option("-n", N, "Matrix size for generated matrices");
  int n_per_row = 4;
  app.add_option("-l", n_per_row, "Number of elements in each row");
  CLI11_PARSE(app, argc, argv);

  std::vector<int> row, col;
  std::vector<double> vals;
  generate(N, n_per_row, row, col, vals);
  mm_write_mtx_vec(output_file.c_str(), N, N, static_cast<int>(vals.size()),
                   row, col, vals);
}
