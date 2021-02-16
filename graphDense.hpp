#ifndef PARALLELIZING_GRAPH_ALGORITHMS_GRAPHDENSE_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_GRAPHDENSE_HPP

#include "abstractGraph.hpp"


class GraphDense : public AbstractGraph {
private:
    std::vector<int> weights;
    std::vector<int> neighbourMatrix;
    const int NOVertices;
    double density;

    void getWeightedFlow() override {
        std::vector<int> res(NOVertices);

        for (int i = 0; i < NOVertices; ++i) {
            int sum = 0;
            for (int j = 0; j < NOVertices; ++j) {
                sum += weights[i] * neighbourMatrix[i * NOVertices + j];
            }
            res[i] = sum;
        }
    }

public:
    GraphDense(int edges, int vertices) : NOVertices(vertices) {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<int> dis(0, vertices - 1);
        std::vector<int> tmpNeighbour(vertices * vertices);
        std::vector<int> tmpWeights(vertices);

        int from;
        int to;
        for (int i = 0; i < edges; ++i) {
            from = dis(gen);
            to = dis(gen);
            while (from == to || tmpNeighbour[from * vertices + to] == 1) {
                from = dis(gen);
                to = dis(gen);
            }
            tmpNeighbour[from * vertices + to] = 1;
        }

        for (int i = 0; i < vertices; ++i) {
            tmpWeights[i] = dis(gen);
        }

        neighbourMatrix = tmpNeighbour;
        weights = tmpWeights;
    }

    explicit GraphDense(int vertices) : NOVertices(vertices) {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<int> dis(0, vertices - 1);
        std::uniform_int_distribution<int> disEdges(0, vertices * (vertices - 1));

        int edges = disEdges(gen);

        std::vector<int> tmpMatrix(vertices * vertices);
        std::vector<int> tmpWeights(vertices);

        int from;
        int to;
        for (int i = 0; i < edges; ++i) {
            from = dis(gen);
            to = dis(gen);
            while (from == to || tmpMatrix[from * vertices + to] == 1) {
                from = dis(gen);
                to = dis(gen);
            }
            tmpMatrix[from * vertices + to] = 1;
        }

        for (int i = 0; i < vertices; ++i) {
            tmpWeights[i] = dis(gen);
        }

        density = (double) edges / ((vertices * (vertices - 1)));
        neighbourMatrix = tmpMatrix;
        weights = tmpWeights;
    }

    double getDensity() override {
        return density;
    }
};

#endif//PARALLELIZING_GRAPH_ALGORITHMS_GRAPHDENSE_HPP
