#ifndef PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP

#include "abstractGraph.hpp"
#include "graphCoordinate.hpp"

class GraphCompressed : public AbstractGraph {
private:
    std::vector<int> neighbourMatrixCol;
    std::vector<int> neighbourMatrixValue;
    std::vector<int> rowStart;
    std::vector<int> weights;
    const int NOVertices;


    void getWeightedFlow() override {
        std::vector<int> res(NOVertices);

        for (int i = 0; i < NOVertices; ++i) {
            int start = rowStart[i];
            int end;
            //check if it is the last index
            end = i == NOVertices - 1 ? (int)neighbourMatrixCol.size() - 1 : rowStart[i + 1] - rowStart[i];
            for (int j = start; j < end; ++j) {
                res[i] += neighbourMatrixValue[start] * weights[neighbourMatrixCol[start]];
            }
        }
    }

public:
    explicit GraphCompressed(GraphCoordinate graph) : NOVertices(graph.getNOVertices()){
       std::vector<value> matrix = graph.getNeighbourMatrix();
       weights = graph.getWeights();
       int actualRow = 0;
       rowStart.push_back(0);
       for(value v : matrix){
           if(v.col != actualRow){
               actualRow++;
               rowStart.push_back(matrix.size());
           }
           neighbourMatrixCol.push_back(v.col);
           neighbourMatrixValue.push_back(v.val);
       }
    }
};



#endif//PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP
