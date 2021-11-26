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
  int rowLength = 8;
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
      "-n", N, "Input matrix size for generated matrices");
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

  CLI::App *perf_compare_cmd =
      app.add_subcommand("compare",
                         "Run measurments on multiple formats with same matrix");
  auto opt_cn = perf_compare_cmd->add_option(
      "-c,--size", N, "Input matrix size for generated matrices" 
  );
  auto opt_cl = perf_compare_cmd->add_option(
      "-l,--length", rowLength, "Input matrix row lenght for generated matrices"
  );

  opt_if->excludes(opt_cn);
  opt_if->excludes(opt_cl);
  opt_cn->excludes(opt_if);
  opt_cl->excludes(opt_if);

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
      GraphCOO coo(N_x, matrix);
      GraphCSR csr(coo, VCL_MULTIROW);
      Ellpack ellpack(coo, VCL_16_ROW, false);
      Ellpack transEllpack(coo, VCL_16_ROW, true);
      myfile << std::filesystem::path(input_file).stem().string() << ",";
      double mkl = csr.measureMKL_result();
      double multirow = csr.measure_result();
      double ellp = ellpack.measure_result();
      double tellp = transEllpack.measure_result();
      myfile << mkl << "," << csr.getBandWidth(mkl) << ","
             << multirow << "," << csr.getBandWidth(multirow) << ","
             << ellp << "," << ellpack.getBandWidth(ellp) << ","
             << tellp << "," << transEllpack.getBandWidth(tellp) << "\n";

    } else {
      GraphCOO coo(N, rho);
      GraphCSR csr(coo, t);
      {
        double time = csr.measure_result();
        std::cout << "Runtime: " << time << "\n";
        std::cout << "Bandwidth: " << csr.getBandWidth(time) << "\n";
      }
      {
        double time = csr.measureMKL_result();
        std::cout << "Runtime: " << time << "\n";
        std::cout << "Bandwidth: " << csr.getBandWidth(time) << "\n";
      }
    }
  } else if (scaling_cmd->parsed()) {
    std::ofstream myfile(output_file);
    if (t == CONST_VCL16_ROW || t == CONST_VCL16_TRANSPOSE) {
      myfile << "Vertices, CSR w/o MKL, CI w 0.95, BW, CI w 0.95, CSR w/ MKL\n";
      for (int i = 10; i <= 17; ++i) {
        GraphCOO graphCOO(static_cast<int>(std::pow(2, i)));
        GraphCSR graphCSR(graphCOO, t);
      }
    } else {
      myfile << "Vertices,Density,MKL,MKL_BW,MULTIROW,MULTIROW_BW,ELLPACK,ELLPACK_BW," 
             << "TRANSPOSED_ELLPACK,TRANSPOSED_ELLPACK_BW\n";
      for (int i = 10; i <= 17; ++i) {
        for (float j = 1; j <= 30; j++) {
          GraphCOO coo(static_cast<int>(std::pow(2, i)), j / 1000);
          GraphCSR csr(coo, VCL_MULTIROW);
          Ellpack ellpack(coo, VCL_16_ROW, false);
          Ellpack transEllpack(coo, VCL_16_ROW, true);
          std::cout << std::pow(2, i) << " " << j / 10 << "\n";
          myfile << std::pow(2, i) << "," << j / 10 << ",";
          double mkl = csr.measureMKL_result();
          double multirow = csr.measure_result();
          double ellp = ellpack.measure_result();
          double tellp = transEllpack.measure_result();
          myfile << mkl << "," << csr.getBandWidth(mkl) << ","
                 << multirow << "," << csr.getBandWidth(multirow) << ","
                 << ellp << "," << ellpack.getBandWidth(ellp) << ","
                 << tellp << "," << transEllpack.getBandWidth(tellp) << "\n";
        }
      }
    }
  } else if (perf_compare_cmd->parsed()) {
    // std::ofstream myfile("/home/hajta2/Parallelizing-Graph-Algorithms/runtimes/compare.csv", std::ios_base::app);
    // GraphCOO coo(N, rowLength);
    // GraphCSR csr_row_ld(coo, VCL_16_ROW);
    // GraphCSR csr_row_lu(coo, VCL_16_ROW_LOOKUP);
    // GraphCSR csr_row_pl(coo, VCL_16_ROW_PARTIAL_LOAD);
    // GraphCSR csr_row_co(coo, VCL_16_ROW_CUTOFF);
    // GraphCSR csr_row_ml(coo, VCL_16_ROW_MULTIPLE_LOAD);
    // GraphCSR csr_row_mr(coo, VCL_MULTIROW);
    // myfile << N << "," << rowLength << ",";
    // {
    //         auto [time, bw] = csr_row_ld.measureMKL_and_bw();
    //         myfile << time << ",";
    // }

    // {
    //         auto [time, bw] = csr_row_ld.measure();
    //         myfile << time << ",";
    // }
    // {
    //         auto [time, bw] = csr_row_lu.measure();
    //         myfile << time << ",";
    // }
    // {
    //         auto [time, bw] = csr_row_pl.measure();
    //         myfile << time << ",";
    // }
    // {
    //         auto [time, bw] = csr_row_co.measure();
    //         myfile << time << ",";
    // }
    // {
    //         auto [time, bw] = csr_row_ml.measure();
    //         myfile << time << ",";
    // }
    // {
    //         auto [time, bw] = csr_row_mr.measure();
    //         myfile << time << "\n";
    // }
    std::cout << "Not implemented";
  } else {
    assert(false);
    return 1;
  }
  return 0;
}
