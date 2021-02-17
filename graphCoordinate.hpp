#ifndef PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOORDINATE_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOORDINATE_HPP

#include "abstractGraph.hpp"

class GraphCoordinate : public AbstractGraph {
private:
    struct value {
        int col, row, val;
    };
    //col,row,val structs in the neighbourmatrix
    std::vector<value> neighbourMatrix;
    std::vector<int> weights;
    const int NOVertices;
    double density;

    void getWeightedFlow() override {
        std::vector<int> res(NOVertices);

        for (value v : neighbourMatrix) {
            res[v.row] = v.val * weights[v.col];
        }
    }

public:
    GraphCoordinate(int edges, int vertices) : NOVertices(vertices) {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<int> dis(0, vertices - 1);
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
        weights = tmpWeights;

        for (int i = 0; i < NOVertices; ++i) {
            for (int j = 0; j < NOVertices; ++j) {
                if (tmpMatrix[i * NOVertices + j] != 0) {
                    value v = {i, j, tmpMatrix[i * NOVertices + j]};
                    neighbourMatrix.push_back(v);
                }
            }
        }
    }

    explicit GraphCoordinate(int vertices) : NOVertices(vertices) {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<int> dis(0, vertices - 1);
        std::uniform_int_distribution<int> dis2(0, vertices * (vertices - 1));
        int edges = dis2(gen);

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
        weights = tmpWeights;
        //compressing the neighbourmatrix
        for (int i = 0; i < NOVertices; ++i) {
            for (int j = 0; j < NOVertices; ++j) {
                if (tmpMatrix[i * NOVertices + j] != 0) {
                    value v = {i, j, tmpMatrix[i * NOVertices + j]};
                    neighbourMatrix.push_back(v);
                }
            }
        }
        density = (double) edges / ((vertices * (vertices - 1)));
    }

    double getDensity() override {
        return density;
    }
};

#endif//PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOORDINATE_HPP
