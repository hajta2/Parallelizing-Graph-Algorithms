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
  
  // int N_x = 0, N_y = 0;
  // std::vector<int> row;
  // std::vector<int> col;
  // std::vector<double> vals;
  // if (argc > 1) {
  //   mm_read_mtx_crd_vec(argv[1], &N_x, &N_y, row, col, vals);
  //   for (int i = 0; i < vals.size(); ++i) {
  //     std::cout << row[i] << " " << col[i] << " " << vals[i] << "\n";
  //   }
  //   std::vector<value> matrix = pack_coo<double>(row,col,vals);
  //   GraphCOO graphCOO(N_x, matrix);
  //   GraphCSR graphCSR(graphCOO);
  //   std::cout<< graphCOO.measure() << std::endl;
  //   std::cout<< graphCSR.measure() << std::endl;
  //   std::cout<< graphCSR.measureMKL() << std::endl;
  // } else{
  //   std::ofstream myfile;
  //   myfile.open("VectorClassRow16Elements.txt");
  //   for(int i = 10; i <= 15; ++i){
  //     //for(float j = 1; j <= 30; j++){
  //       GraphCOO graphCOO(pow(2, i)); 
  //       GraphCSR graphCSR(graphCOO);
  //       std::cout << pow(2,i) << "\n"; // << j/10 << "\n";
  //       myfile<< "Vertices: " << pow(2,i) << "\n"
  //             //<< "Density:"<< j/10 <<"%\n" 
  //             << "CSR w/o MKL: " << graphCSR.measure() << " microseconds\n" 
  //             << "CSR w/ MKL: " << graphCSR.measureMKL() << " microseconds\n\n";
  //     //}
  //   }
  //   myfile.close();
  // }
  GraphCOO coo(std::pow(2,15));
  GraphCSR csr(coo);

  //coo.print();
 
  std::cout<<csr.measure()<<"\n";
  std::cout << csr.measureMKL() << "\n";
//   std::cout << csr.bandWidth() << "\n";

  return 0;
}
