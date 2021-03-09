#ifndef PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOORDINATE_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOORDINATE_HPP

#include <cmath>

#include "abstractGraph.hpp"

struct value {
    int row, col;
    float val;
};

class GraphCOO : public AbstractGraph {
private:
    std::vector<value> neighbourMatrix;
    std::vector<float> weights;
    const int NOVertices;

    void getWeightedFlow() override{
        std::vector<float> res(NOVertices);

        for (value v : neighbourMatrix) {
            res[v.row] = v.val * weights[v.col];
        }
    }

public:
    GraphCOO(int vertices, std::vector<value> matrix) : NOVertices(vertices), neighbourMatrix(matrix) {}

    GraphCOO(int vertices, float density) : NOVertices(vertices){
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<int> dis(0, vertices - 1);

        std::vector<int> tmpMatrix(vertices*vertices);
        std::vector<float> tmpWeights(vertices);

        int edges = (int)std::floor(density * (float)(vertices * vertices));

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

  
    std::vector<float> getWeights(){
            return weights;
    }

    std::vector<value> getNeighbourMatrix(){
            return neighbourMatrix;
    }

    [[nodiscard]] int getNOVertices() const{
        return NOVertices;
    }
};

#endif//PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOORDINATE_HPP
