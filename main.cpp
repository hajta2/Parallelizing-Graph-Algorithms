#include <fstream>
#include <iostream>

#include "graphCOO.hpp"
#include "graphCSR.hpp"
#include "graphDense.hpp"
#include "mkl.h"
#include "mmio_cpp.h"

template <typename Float>
std::vector<value> pack_coo(const std::vector<int> &row, const vector<int> &col,
                            const vector<Float> &val) {
  std::vector<value> result;
  result.reserve(val.size());
  for (int i = 0; i < val.size(); ++i) {
    result.push_back({row[i], col[i], val[i]});
  }
  std::sort(result.begin(), result.end(), [](const auto &lhs, const auto &rhs) {
    if (lhs.row != rhs.row) return lhs.row < rhs.row;
    return lhs.col < rhs.col;
  });
  return result;
}

int main(int argc, const char *argv[]) {
  // std::ofstream myfile;
  // myfile.open("runtime.txt");

  /*for (int i = 10; i < 12; ++i) {
      GraphCoordinate graphCoordinate(pow(2, i), 0.4);
      GraphDense graphDense(graphCoordinate);
      GraphCompressed graphCompressed(graphCoordinate);
      myfile << "Vertices:" << pow(2, i)
             << "\nRuntimes:\ndense: " << graphDense.measure() << "
  microseconds\n"
             << "coordinate " << graphCoordinate.measure() << " microseconds\n"
             << "compressed " << graphCompressed.measure() << "
  microseconds\n\n";
  }*/

  // GraphCOO graphCOO(pow(2, 8), 0.4);
  // GraphCSR graphCSR(graphCOO);
  // std::cout << graphCSR.measure();
  // myfile.close();

  int N_x = 0, N_y = 0;
  std::vector<int> row;
  std::vector<int> col;
  std::vector<double> vals;
  if (argc > 1) {
    mm_read_mtx_crd_vec(argv[1], &N_x, &N_y, row, col, vals);
    for (int i = 0; i < vals.size(); ++i) {
      std::cout << row[i] << " " << col[i] << " " << vals[i] << "\n";
    }
  }

  // read in the mtx format
  // int NORow, NOCol, NOLines;
  //
  // std::ifstream file("gre_1107.mtx");
  // //ignore comments
  // while (file.peek() == '%') file.ignore(2048, '\n');
  //
  // file >> NORow >> NOCol >> NOLines;
  //
  // std::vector<value> tmpMatrix(NORow * NOCol);
  // value v;
  // for (int i = 0; i < NOLines; i++)
  // {
  //     file >> v.row >> v.col;
  //     file >> v.val;
  //     tmpMatrix.push_back(v);
  // }
  //
  // GraphCOO graphCOO(NORow, tmpMatrix);
  // GraphCSR graphCSR(graphCOO);
  // std::cout << graphCSR.measure();
  // file.close();
  return 0;
}
