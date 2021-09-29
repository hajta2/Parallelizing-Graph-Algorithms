#ifndef PARALLELIZING_GRAPH_ALGORITHMS_TRANSPOSE_ELLPACK_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_TRANSPOSE_ELLPACK_HPP

#include "graphCOO.hpp"

class TransposeEllpack : public AbstractGraph {
private:
    std::vector<float> weights;
    std::vector<float> values;
    std::vector<int> columns;
    const int NOVertices;
    int rowLength;
    Type type;

public:
    explicit TransposeEllpack(GraphCOO graph, Type t) : NOVertices(graph.getNOVertices()), type(t) {
        graph.convertToTransposedELLPACK();
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

    double getBandWidth(double time_s) override {
      double bytes = sizeof(float) * (weights.size() + 2 * NOVertices) +
                     sizeof(int) * weights.size();
      return bytes / 1000 / time_s;
    }
};



#endif//PARALLELIZING_GRAPH_ALGORITHMS_TRANSPOSE_ELLPACK_HPP
