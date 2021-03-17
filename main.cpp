#include <fstream>
#include <iostream>

#include "graphCOO.hpp"
#include "graphCSR.hpp"
#include "graphDense.hpp"
#include "mkl.h"
#include "mmio_cpp.h"

template <typename Float>
    std::vector<value> pack_coo(const std::vector<int> &row, const std::vector<int> &col,
                            const std::vector<Float> &val) {
        std::vector<value> result;
        result.reserve(val.size());
        for (int i = 0; i < val.size(); ++i) {
            result.push_back({row[i]-1, col[i]-1, val[i]});
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
  if (argc > 1) {
    mm_read_mtx_crd_vec(argv[1], &N_x, &N_y, row, col, vals);
    for (int i = 0; i < vals.size(); ++i) {
      std::cout << row[i] << " " << col[i] << " " << vals[i] << "\n";
    }
    std::vector<value> matrix = pack_coo<double>(row,col,vals);
    GraphCOO graphCOO(N_x, matrix);
    GraphCSR graphCSR(graphCOO);
    std::cout<< graphCOO.measure() << std::endl;
    std::cout<< graphCSR.measure() << std::endl;
    std::cout<< graphCSR.measureMKL() << std::endl;
  } else{
    std::ofstream myfile;
    myfile.open("runtime.txt");
    for(int i = 10; i <= 15; ++i){
      GraphCOO graphCOO01(pow(2, i), 0.1);
      GraphCSR graphCSR01(graphCOO01);
      GraphCOO graphCOO02(pow(2, i), 0.2);
      GraphCSR graphCSR02(graphCOO02);
      GraphCOO graphCOO03(pow(2, i), 0.3);
      GraphCSR graphCSR03(graphCOO03);
      GraphCOO graphCOO04(pow(2, i), 0.4);
      GraphCSR graphCSR04(graphCOO04);
      myfile<< "Vertices: " << pow(2,i) << "\n"
            << "Density: 10%\n" 
            //<< "COO: " << graphCOO01.measure() << " microseconds\n"
            << "CSR w/o MKL: " << graphCSR01.measure() << " microseconds\n" 
            << "CSR w/ MKL: " << graphCSR01.measureMKL() << " microseconds\n" 
            << "Density: 20%\n" 
            //<< "COO: " << graphCOO02.measure() << " microseconds\n"
            << "CSR w/o MKL: " << graphCSR02.measure() << " microseconds\n" 
            << "CSR w/ MKL: " << graphCSR02.measureMKL() << " microseconds\n" 
            << "Density: 30%\n" 
           // << "COO: " << graphCOO03.measure() << " microseconds\n"
            << "CSR w/o MKL: " << graphCSR03.measure() << " microseconds\n" 
            << "CSR w/ MKL: " << graphCSR03.measureMKL() << " microseconds\n" 
            << "Density: 40%\n" 
           // << "COO: " << graphCOO04.measure() << " microseconds\n"
            << "CSR w/o MKL: " << graphCSR04.measure() << " microseconds\n" 
            << "CSR w/ MKL: " << graphCSR04.measureMKL() << " microseconds\n\n";
    }
    myfile.close();
  }
  return 0;
}
