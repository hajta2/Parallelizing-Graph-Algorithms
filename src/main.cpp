#include <filesystem>
#include <fstream>
#include <iostream>

#include "CLI11.hpp"
#include "graphCSR.hpp"
//#include "mmio_cpp.h"

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
      "/home/ri-thajdara/Parallelizing-Graph-Algorithms/runtimes/matrices.csv";
  int N = 8192;
  float rho = 3e-2f;
  Type t = SVE;

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

  CLI11_PARSE(app, argc, argv);
  if (t > SVE) {
    return 1;
  }

  if (measure_one_cmd->parsed()) {
    if (opt_if->count()) {
    //   int N_x = 0, N_y = 0;
    //   std::vector<int> row;
    //   std::vector<int> col;
    //   std::vector<double> vals;
    //   mm_read_mtx_crd_vec(input_file.c_str(), &N_x, &N_y, row, col, vals);
    //   assert(N_x == N_y && "graph class handles squeare matrices only");
    //   std::vector<value> matrix = pack_coo<double>(row, col, vals);
    //   std::ofstream myfile(output_file, std::ios_base::app);
    //   GraphCOO graphCOO(N_x, matrix);
    //   GraphCSR graphCSR(graphCOO, t);
    //   myfile << std::filesystem::path(input_file).stem().string() << ", ";
    //   {
    //     auto [time, bw] = graphCSR.measure();
    //     myfile << t << ", " << time << ", " << bw << ", ";
    //   }
    //   {
    //     auto [time, bw] = graphCSR.measureARM_and_bw();
    //     myfile <<  time << ", " << bw << ", ";
    //   }
    //   myfile << "\n";
    // } else {
    //   GraphCOO coo(N, rho);
    //   GraphCSR csr(coo, t);
    //   {
    //     auto [time, bw] = csr.measure();
    //     std::cout << t << "\n";
    //     std::cout << "Runtime: " << time << "\n";
    //     std::cout << "Bandwidth: " << bw << "\n";
    //     std::cout << "Bandwidth: " << csr.bandWidth() << "\n";
    //   }
    //   {
    //     auto [time, bw] = csr.measureARM_and_bw();
    //     std::cout << "ARM t: " << time << "\n";
    //     std::cout << "ARM b: " << bw << "\n";
    //   }
    std::cout << "Not available\n";
    } else {
      GraphCOO coo(N, rho);
      GraphCSR csr(coo, t);
      {
        auto [time, bw] = csr.measure();
        std::cout << "Time: " << time << "\n";
        std::cout << "BW: " << bw << "\n";
      }
      {
        auto [time, bw] = csr.measureARM_and_bw();
        std::cout << "ARM Time: " << time << "\n";
        std::cout << "ARM BW: " << bw << "\n";
      }
    }
  } else if (scaling_cmd->parsed()) {
    std::ofstream myfile(output_file);
    myfile << "Vertices,Density,MyImplementation,CI w 0.95,BW,CI w 0.95\n";//,ARMPL,CI w 0.95,BW,CI w 0.95\n";
    for (int i = 17; i <= 17; ++i) {
      for (float j = 1; j <= 30; j++) {
        GraphCOO graphCOO(static_cast<int>(std::pow(2, i)), j / 1000);
        GraphCSR graphCSR(graphCOO, t);
        std::cout << std::pow(2, i) << " " << j / 10 << "\n";
        myfile << std::pow(2, i) << "," << j / 10 << ",";
        {
          auto [time, bw] = graphCSR.measure();
          myfile << time << ",";
          myfile << bw << "\n";
          // std::cout << "t: " << time << "\n";
          // std::cout << "bw: " << bw << "\n";
        }
        // {
        //   auto [time, bw] = graphCSR.measureARM_and_bw();
        //   myfile << time << ",";
        //   myfile << bw << "\n";
        //   // std::cout << "arm t: " << time << "\n";
        //   // std::cout << "arm bw: " << bw << "\n";
        // }
      }
    }
  } else {
    assert(false);
    return 1;
  }
  return 0;
}
