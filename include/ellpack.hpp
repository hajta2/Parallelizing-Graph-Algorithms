#ifndef PARALLELIZING_GRAPH_ALGORITHMS_ELLPACK_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_ELLPACK_HPP

#include "graphCOO.hpp"

class Ellpack : public AbstractGraph {
private:
    std::vector<float> weights;
    std::vector<float> values;
    std::vector<int> columns;
    const int NOVertices;
    int rowLength;
    Type type;
    
    void getWeightedFlow() override {
        std::vector<float> res(NOVertices);
        if (type == NAIVE) {
            for (int i = 0; i < NOVertices-1; ++i) {
                for (int j = 0; j < rowLength; ++j) {
                    res[i] += values[i*rowLength+j] * weights[columns[i*rowLength+j]];
                }
            }
        } else if (type == OPENMP) {
            #pragma omp parallel for
            for (int i = 0; i < NOVertices-1; ++i) {
                for (int j = 0; j < rowLength; ++j) {
                    res[i] += values[i*rowLength+j] * weights[columns[i*rowLength+j]];
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
        std::vector<float> tmpValues(rowLength * NOVertices);
        std::vector<int> tmpColumns(rowLength * NOVertices);
        int actualRow = 0;
        int elementsInRow = 0;
        for (value v : matrix){
            if(actualRow != v.row) {
                actualRow++;
                elementsInRow = 0;
            }
            tmpValues[v.row*rowLength+elementsInRow] = v.val;
            tmpColumns[v.row*rowLength+elementsInRow] = v.col;
            elementsInRow++;
        }
        values=tmpValues;
        columns=tmpColumns;
    }

};

#endif//PARALLELIZING_GRAPH_ALGORITHMS_ELLPACK_HPP