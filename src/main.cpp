#include <filesystem>
#include <fstream>
#include <iostream>

#include "CLI11.hpp"
#include "ellpack.hpp"
#include "graphCOO.hpp"
#include "graphCSR.hpp"
#include "graphDense.hpp"
#include "mkl.h"
#include "mmio_cpp.h"


template <typename Float>
std::vector<value> pack_coo(const std::vector<int> &row,
                            const std::vector<int> &col,
                            const std::vector<Float> &val) {
  std::vector<value> result;
  result.reserve(val.size());
  for (size_t i = 0; i < val.size(); ++i) {
    result.push_back({row[i] - 1, col[i] - 1, static_cast<float>(val[i])});
  }
  std::sort(result.begin(), result.end(), [](const auto &lhs, const auto &rhs) {
    if (lhs.row != rhs.row) return lhs.row < rhs.row;
    return lhs.col < rhs.col;
  });
  return result;
}

std::ostream &operator<<(std::ostream &o, const measurement_result &r) {
  return o << r.mean << ", " << r.confidence_interval_width;
}

int main(int argc, const char *argv[]) {
  std::string input_file = "";
  std::string output_file =
      "/home/hajta2/Parallelizing-Graph-Algorithms/runtimes/matrices.csv";
  int N = 8192;
  float rho = 3e-2f;
  Type t = VCL_16_ROW;

  CLI::App app;
  app.add_option("-o,--output", output_file, "Output file csv");
  app.add_option("-t,--type", t, "Matrix format to measure");
  app.require_subcommand(/* min */ 1, /* max */ 1);

  CLI::App *measure_one_cmd =
      app.add_subcommand("single", "Run measurements on single matrix");
  auto opt_if = measure_one_cmd->add_option("-i,--input", input_file,
                                            "Input mtx file to measure");
  auto opt_n = measure_one_cmd->add_option(
      "-n", N, "Input matrix file for generated matrices");
  auto opt_r = measure_one_cmd->add_option(
      "-r,--rho", rho, "Density of the matrix for generated matrices");

  opt_if->excludes(opt_n);
  opt_if->excludes(opt_r);
  opt_n->excludes(opt_if);
  opt_r->excludes(opt_if);

  CLI::App *scaling_cmd =
      app.add_subcommand("scaling",
                         "Run measurements on a single format with different "
                         "matrix sizes and desinties");

  CLI11_PARSE(app, argc, argv);
  if (t > VCL_MULTIROW) {
    return 1;
  }

  if (measure_one_cmd->parsed()) {
    if (opt_if->count()) {
      int N_x = 0, N_y = 0;
      std::vector<int> row;
      std::vector<int> col;
      std::vector<double> vals;
      mm_read_mtx_crd_vec(input_file.c_str(), &N_x, &N_y, row, col, vals);
      assert(N_x == N_y && "graph class handles squeare matrices only");
      std::vector<value> matrix = pack_coo<double>(row, col, vals);
      std::ofstream myfile(output_file, std::ios_base::app);
      GraphCOO graphCOO(N_x, matrix);
      GraphCSR graphCSR(graphCOO, t);
      // Ellpack ellpack(graphCOO, t, false);
      // Ellpack transposedEllpack(graphCOO, t, true);
      myfile << std::filesystem::path(input_file).stem().string() << ", ";
      {
        auto [time, bw] = graphCSR.measure();
        myfile << time << ", " << bw << ", ";
      }
      myfile << graphCSR.measureMKL() << ", ";
      {
        auto [time, bw] = ellpack.measure();
        myfile << time << ", " << bw << ", ";
      }
      {
        auto [time, bw] = transposedEllpack.measure();
        myfile << time << ", " << bw << ", ";
      }
      myfile << "\n";
    } else {
      GraphCOO coo(N, rho);
      GraphCSR csr(coo, t);
      auto [time, bw] = csr.measure();
      std::cout << "Runtime: " << time << "\n";
      std::cout << "Bandwidth: " << bw "\n";
      std::cout << csr.measureMKL() << "\n";
    }
  } else if (scaling_cmd->parsed()) {
    std::ofstream myfile("../runtimes/" + enumString[t] + ".csv");
    if (t == CONST_VCL16_ROW || t == CONST_VCL16_TRANSPOSE) {
      myfile << "Vertices, CSR w/o MKL, CI w 0.95, BW, CI w 0.95, CSR w/ MKL\n";
      for (int i = 10; i <= 17; ++i) {
        GraphCOO graphCOO(static_cast<int>(std::pow(2, i)));
        GraphCSR graphCSR(graphCOO, t);
        auto [time, bw] = graphCSR.measure();
        myfile << std::pow(2, i) << ", " << time << ", " << bw << ", "
               << graphCSR.measureMKL() << "\n";
      }
    } else {
      myfile << "Vertices,Density,CSR w/o MKL,CSR w/ MKL,Ellpack,Transposed "
                "Ellpack,Bandwidth,Const\n";
      for (int i = 10; i <= 15; ++i) {
        for (float j = 1; j <= 30; j++) {
          GraphCOO graphCOO(static_cast<int>(std::pow(2, i)), j / 1000);
          GraphCSR graphCSR(graphCOO, t);
          Ellpack ellpack(graphCOO, t, false);
          Ellpack transposedEllpack(graphCOO, t, true);
          std::cout << std::pow(2, i) << " " << j / 10 << "\n";
          std::cout << std::pow(2, i) << "," << j / 10 << ",";
          {
            auto [time, bw] = graphCSR.measure();
            std::cout << time << ", " << bw << ", ";
          }
          std::cout << graphCSR.measureMKL() << ",";
          {
            auto [time, bw] = ellpack.measure();
            std::cout << time << ", " << bw << ", ";
          }
          {
            auto [time, bw] = transposedEllpack.measure();
            std::cout << time << ", " << bw << ", ";
          }
          std::cout << "68554.2"
                    << "\n";
        }
      }
    }
  } else {
    assert(false);
    return 1;
  }
  return 0;
}
