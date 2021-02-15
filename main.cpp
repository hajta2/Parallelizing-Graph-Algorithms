#include "graphCompressed.hpp"
#include "graphDense.hpp"
#include <iostream>


int main() {
    Graph g2(512);
    std::cout<< g2.measure();
    GraphCompressed g(8);
    g.print();
    return 0;
}
