#ifndef PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP

#include "abstractGraph.hpp"

class GraphCompressed : public AbstractGraph {
private:
    struct value {
        int col, val;
    };
    //columns and values in the neighbourmatrix
    //the starting indices of the rows in the rowstart matrix
    std::vector<value> neighbourMatrix;
    std::vector<int> rowStart;
    std::vector<int> weights;
    const int NOVertices;
    double density;

    void getWeightedFlow() override {
        std::vector<int> res(NOVertices);

        for (int i = 0; i < NOVertices; ++i) {
            int start = rowStart[i];
            int end;
            //check if it is the last index
            end = i == NOVertices - 1 ? (int)neighbourMatrix.size() - 1 : rowStart[i + 1] - rowStart[i];
            for (int j = start; j < end; ++j) {
                res[i] += neighbourMatrix[start].val * weights[neighbourMatrix[start].col];
            }
        }
    }

public:
    GraphCompressed(int edges, int vertices) : NOVertices(vertices) {
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
                    value v = {j, tmpMatrix[i * NOVertices + j]};
                    neighbourMatrix.push_back(v);
                }
            }
            rowStart.push_back(neighbourMatrix.size());
        }
    }

    explicit GraphCompressed(int vertices) : NOVertices(vertices) {
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
        //converting the dense neighbourmatrix to compressed
        for (int i = 0; i < NOVertices; ++i) {
            for (int j = 0; j < NOVertices; ++j) {
                if (tmpMatrix[i * NOVertices + j] != 0) {
                    value v = {j, tmpMatrix[i * NOVertices + j]};
                    neighbourMatrix.push_back(v);
                }
            }
            rowStart.push_back(neighbourMatrix.size());
        }
        density = (double) edges / ((vertices * (vertices - 1)));
    }
    double getDensity() override {
        return density;
    }
};

#endif//PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP
