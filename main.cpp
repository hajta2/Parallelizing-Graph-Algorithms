#include "graphCSR.hpp"
#include "graphCOO.hpp"
#include "graphDense.hpp"
#include <fstream>
#include <iostream>


int main() {

    //std::cout << "Hello World!";
    std::ofstream myfile;
    myfile.open("runtime.txt");

    for (int i = 10; i < 12; ++i) {
        GraphCOO graphCoordinate(pow(2, i), 0.4);
        GraphDense graphDense(graphCoordinate);
        GraphCSR graphCompressed(graphCoordinate);
        myfile << "Vertices:" << pow(2, i)
               << "\nRuntimes:\ndense: " << graphDense.measure() << " microseconds\n"
               << "coordinate " << graphCoordinate.measure() << " microseconds\n"
               << "compressed " << graphCompressed.measure() << " microseconds\n\n";
    }

    //GraphCOO graphCOO(pow(2, 8), 0.4);
    //GraphCSR graphCSR(graphCOO);
    //std::cout << graphCSR.measure();
    //myfile.close();


    //read in the mtx format
    //int NORow, NOCol, NOLines;

 /*   std::ifstream file("gre_1107.mtx");
    //ignore comments
    while (file.peek() == '%') file.ignore(2048, '\n');

    file >> NORow >> NOCol >> NOLines;

    std::vector<value> tmpMatrix(NORow * NOCol);
    value v;
    for (int i = 0; i < NOLines; i++)
    {
        file >> v.row >> v.col;
        file >> v.val;
        tmpMatrix.push_back(v);
    }

    GraphCOO graphCOO(NORow, tmpMatrix);
    GraphCSR graphCSR(graphCOO); */
    myfile.close(); 
    return 0;
}
