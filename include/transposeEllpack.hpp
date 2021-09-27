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
    explicit TransposeEllpack(GraphCOO& graph, Type t) : NOVertices(graph.getNOVertices()), type(t) {
        graph.convertToELLPACK();
    }

};



#endif//PARALLELIZING_GRAPH_ALGORITHMS_TRANSPOSE_ELLPACK_HPP