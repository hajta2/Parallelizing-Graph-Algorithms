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
        if (type == NAIVE) {
            for (int i = 0; i < NOVertices-1; ++i) {
                for (int j = 0; j < rowLength; ++j) {
                    res[i] += neighbourMatrix[i*rowLength+j];
                }
            }
        }
    }

public:
    explicit Ellpack(GraphCOO& graph, Type t) : NOVertices(graph.getNOVertices()), type(t) {
        graph.convertToELLPACK();
        std::vector<value> matrix = graph.getNeighbourMatrix();
        weights = graph.getWeights();
        rowLength = graph.getEllpackRow();
        std::vector<float> tmpNeigbourMatrix(rowLength * NOVertices);
        int actualRow = 0;
        int elementsInRow = 0;
        for (value v : matrix){
            if(actualRow != v.row) {
                actualRow++;
                elementsInRow = 0;
            }
            tmpNeigbourMatrix[v.row*rowLength+elementsInRow] = v.val;
            elementsInRow++;
        }
        neighbourMatrix=tmpNeigbourMatrix;
    }

};

#endif//PARALLELIZING_GRAPH_ALGORITHMS_ELLPACK_HPP