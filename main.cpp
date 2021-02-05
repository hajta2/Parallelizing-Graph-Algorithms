#include "graph.hpp"
#include <iostream>


int main() {
    Graph g(18,8);
    g.print();
    std::cout<<std::endl;
    Graph g2(8);
    g2.print();
    return 0;
}
