#ifndef PARALLELIZING_GRAPH_ALGORITHMS_ELLPACK_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_ELLPACK_HPP

#include "graphCOO.hpp"

class Ellpack : public AbstractGraph {
private:
    std::vector<float> weights;
    std::vector<float> neighbourMatrix;
    const int NOVertices;
    int rowLength;
    Type type;
    
    void getWeightedFlow() override {
        std::vector<float> res(NOVertices);
        std::cout << rowLength << std::endl;     
    }

public:
    explicit Ellpack(GraphCOO& graph, Type t) : NOVertices(graph.getNOVertices()), type(t) {
        graph.convertToELLPACK();
        std::vector<value> matrix = graph.getNeighbourMatrix();
        weights = graph.getWeights();
        rowLength = graph.getEllpackRow();
        std::vector<float> tmpNeigbourMatrix(rowLength * NOVertices);
        int actualRow = 0;
        for (value v : matrix){
            if(actualRow != v.row) actualRow++;
            tmpNeigbourMatrix[v.row*rowLength+v.col] = v.val;
        }
        neighbourMatrix=tmpNeigbourMatrix;
    }

};

#endif//PARALLELIZING_GRAPH_ALGORITHMS_ELLPACK_HPP