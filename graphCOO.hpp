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
    GraphCOO(int vertices, std::vector<value> matrix) : NOVertices(vertices), neighbourMatrix(matrix) {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<int> dis(0, vertices - 1);
        std::vector<float> tmpWeights(vertices);

        for (int i = 0; i < vertices; ++i) {
            tmpWeights[i] = dis(gen);
        }

        weights = tmpWeights;
    }

    GraphCOO(int vertices, float sparsity) : NOVertices(vertices){
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<int> dis(0, vertices - 1);
        std::uniform_real_distribution<float> disVal(0.0,1.0);

        int edges = (int)std::floor(sparsity * (float)(vertices * vertices));
        std::vector<float> tmpWeights(vertices);

        for (int i = 0; i < edges; ++i) {
            int row = dis(gen);
            int col = dis(gen);
            float weight = disVal(gen);
            
            value val = {row, col, weight};
            neighbourMatrix.push_back(val);
        }

        std::sort(neighbourMatrix.begin(), neighbourMatrix.end(), [](const auto &lhs, const auto &rhs) {
            if (lhs.row != rhs.row) return lhs.row < rhs.row;
            return lhs.col < rhs.col;
        });

        auto iter = std::unique(neighbourMatrix.begin(), neighbourMatrix.end(), [](const auto &lhs, const auto &rhs) {
            return lhs.row == rhs.row && lhs.col == rhs.col;
        });

        for (int i = 0; i < vertices; ++i) {
            tmpWeights[i] = dis(gen);
        }
        weights = tmpWeights;

        neighbourMatrix.erase(iter, neighbourMatrix.end()); 
    }

    /*GraphCOO(int vertices, float density) : NOVertices(vertices){
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
    }*/

  
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