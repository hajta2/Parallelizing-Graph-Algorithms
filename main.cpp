#include "graphCompressed.hpp"
#include "graphCoordinate.hpp"
#include "graphDense.hpp"
#include <fstream>
#include <iostream>



int main() {
    std::ofstream myfile;
    myfile.open("runtime.txt");

    for (int i = 10; i < 12; ++i) {
        GraphCoordinate graphCoordinate(pow(2, i), 0.4);
        GraphDense graphDense(graphCoordinate);
        GraphCompressed graphCompressed(graphCoordinate);
        myfile << "Vertices:" << pow(2, i)
               << "\nRuntimes:\ndense: " << graphDense.measure() << " microseconds\n"
               << "coordinate " << graphCoordinate.measure() << " microseconds\n"
               << "compressed " << graphCompressed.measure() << " microseconds\n\n";
    }

    myfile.close();
    return 0;
}
