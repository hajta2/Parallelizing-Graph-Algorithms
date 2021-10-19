#ifndef PARALLELIZING_GRAPH_ALGORITHMS_GRAPHDENSE_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_GRAPHDENSE_HPP

#include "graphCOO.hpp"


class GraphDense : public AbstractGraph {
private:
    std::vector<float> weights;
    std::vector<int> neighbourMatrix;
    const int NOVertices;
    std::vector<float> flow;

public:
    void getWeightedFlow() override {
        std::vector<float> res(NOVertices);

        for (int i = 0; i < NOVertices; ++i) {
            float sum = 0;
            for (int j = 0; j < NOVertices; ++j) {
                sum += weights[i] * neighbourMatrix[i * NOVertices + j];
            }
            res[i] = sum;
        }
        flow = res;
    }

    explicit GraphDense(GraphCOO& graph) : NOVertices(graph.getNOVertices()){
        std::vector<value> matrix = graph.getNeighbourMatrix();
        weights = graph.getWeights();
        std::vector<int> tmpNeigbourMatrix(NOVertices * NOVertices);
        int actualRow = 0;
        for (value v : matrix){
            if(actualRow != v.row) actualRow++;
            tmpNeigbourMatrix[v.row*NOVertices+v.col] = v.val;
        }
        neighbourMatrix=tmpNeigbourMatrix;
    }

    float *getResult() override {
      return flow.data();
    }

    double getBandWidth(double time_s) override {
      double bytes = sizeof(float) * (weights.size() + neighbourMatrix.size() +
                                      2 * flow.size());
      // Gigabyte per second
      return bytes / 1e9 / time_s;
    }

};

#endif//PARALLELIZING_GRAPH_ALGORITHMS_GRAPHDENSE_HPP
