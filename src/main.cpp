#include <fstream>
#include <iostream>

#include "graphCOO.hpp"
#include "graphCSR.hpp"
#include "mmio_cpp.h"

template <typename Float>
std::vector<value> pack_coo(const std::vector<int> &row,
                            const std::vector<int> &col,
                            const std::vector<Float> &val) {
  std::vector<value> result;
  result.reserve(val.size());
  for (int i = 0; i < val.size(); ++i) {
    result.push_back({row[i] - 1, col[i] - 1, val[i]});
  }
  std::sort(result.begin(), result.end(), [](const auto &lhs, const auto &rhs) {
    if (lhs.row != rhs.row) return lhs.row < rhs.row;
    return lhs.col < rhs.col;
  });
  return result;
}

int main(int argc, const char *argv[]) {
  
  int N_x = 0, N_y = 0;
  std::vector<int> row;
  std::vector<int> col;
  std::vector<double> vals;
  Type t = OPENMP;
  std::ofstream myfile;
  if (argc > 1) {
    mm_read_mtx_crd_vec(argv[1], &N_x, &N_y, row, col, vals);
    std::vector<value> matrix = pack_coo<double>(row,col,vals);
    myfile.open("/home/hajta2/Parallelizing-Graph-Algorithms/runtimes/matrices.csv", std::ios_base::app);
    GraphCOO graphCOO(N_x, matrix);
    GraphCSR graphCSR(graphCOO, t);
    myfile  << graphCSR.measure() << "\n";
  } else{
    myfile.open("../runtimes/"+enumString[t]+".csv");
    myfile << "Vertices,Density,CSR,Bandwidth,Const\n";
    for(int i = 10; i <= 17; ++i){
      for(float j = 1; j <= 30; j++){
        GraphCOO graphCOO(std::pow(2, i), j/1000); 
        GraphCSR graphCSR(graphCOO, t);
        std::cout << std::pow(2,i) << " " << j/10 << "\n";
        myfile << std::pow(2,i) << ","
                  << j/10 << "," 
                  << graphCSR.measure()<< ","
                  << "68554.2" << "\n";
      }
    }
  }

  myfile.close();

  return 0;
}
