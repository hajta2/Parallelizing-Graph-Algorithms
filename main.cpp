#include "graphCompressed.hpp"
#include "graphCoordinate.hpp"
#include "graphDense.hpp"
#include <fstream>
#include <iostream>



int main() {
    std::ofstream myfile;
    myfile.open("runtime.txt");

    for (int i = 1; i < 12; ++i) {
        GraphDense graphDense(pow(2, i));
        GraphCoordinate graphCoordinate(pow(2, i));
        GraphCompressed graphCompressed(pow(2, i));
        myfile << "Vertices:" << pow(2, i)
               << "\nDensity of the graphs:\ndense: " << graphDense.getDensity()
               << "\ncoordinate: " << graphCompressed.getDensity()
               << "\ncompressed: " << graphCompressed.getDensity()
               << "\nRuntimes:\ndense: " << graphDense.measure() << " microseconds\n"
               << "coordinate " << graphCoordinate.measure() << " microseconds\n"
               << "compressed " << graphCompressed.measure() << " microseconds\n\n";
    }

    myfile.close();

    return 0;
}
