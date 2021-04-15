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
    GraphCSR graphCSR(graphCOO, CONST_VCL16_TRANSPOSE);
    std::cout<< graphCOO.measure() << std::endl;
    std::cout<< graphCSR.measure() << std::endl;
    std::cout<< graphCSR.measureMKL() << std::endl;
  } else{
    Type t = VCL_16_ROW;
    std::ofstream myfile;
    myfile.open("../runtimes/"+enumString[t]+".csv");
    if (t == CONST_VCL16_ROW || t == CONST_VCL16_TRANSPOSE) {
      myfile << "Vertices, CSR w/o MKL, CSR w/ MKL \n";
      for(int i = 10; i <= 15; ++i){
        GraphCOO graphCOO(std::pow(2, i)); 
        GraphCSR graphCSR(graphCOO, t);
        myfile<< std::pow(2,i) << ", "
              << graphCSR.measure() << ", "
              << graphCSR.measureMKL() << "\n";
      }
    } else {
        myfile << "Vertices, Density, CSR w/o MKL, CSR w/ MKL \n";
        for(int i = 10; i <= 15; ++i){
          for(float j = 1; j <= 30; j++){
            GraphCOO graphCOO(std::pow(2, i)); 
            GraphCSR graphCSR(graphCOO, t);
            myfile<< std::pow(2,i) << ", "
                  << j/10 << ", " 
                  << graphCSR.measure() << ", "
                  << graphCSR.measureMKL() << "\n";
          }
        }
    }
    myfile.close();
  }

//   GraphCOO coo(std::pow(2,15));
//   GraphCSR csr(coo);

//   std::cout<<csr.measure()<<"\n";
//   std::cout << csr.measureMKL() << "\n";
// //   std::cout << csr.bandWidth() << "\n";

  return 0;
}
