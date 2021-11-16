#ifndef PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOORDINATE_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOORDINATE_HPP

#include <cmath>
#include <algorithm>
#include <random>

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
    std::vector<float> flow;

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

        int edges = (int)std::floor(sparsity * vertices * vertices);
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

        neighbourMatrix.erase(iter, neighbourMatrix.end()); 

        for (int i = 0; i < vertices; ++i) {
            tmpWeights[i] = dis(gen);
        }
        weights = tmpWeights;

    }
    
    std::vector<float> getWeights(){
            return weights;
    }

    std::vector<value> getNeighbourMatrix(){
            return neighbourMatrix;
    }

    int getNOVertices() const{
        return NOVertices;
    }

    void print() {
        int row = 0;
        for(int i = 0; i < neighbourMatrix.size(); ++i) {
            if(neighbourMatrix[i].row != row) {
                std::cout << "\n";
                row = neighbourMatrix[i].row;
            }
            std::cout << neighbourMatrix[i].col << " " << neighbourMatrix[i].val << " ";
        }
        std::cout << "\nWeights: ";
        for(int i = 0; i < weights.size(); ++i) {
            std::cout<< weights[i] << " ";
        }
        std::cout<<"\n";
    }

    double getBandWidth(double time_s) override {
        double bytes = sizeof(float) * (weights.size() + 2 * NOVertices) +
                       sizeof(int) * 2 * neighbourMatrix.size();

        return bytes / 1000 / time_s;
    }

    void getWeightedFlow() override{
        std::vector<float> res(NOVertices);

        for (value v : neighbourMatrix) {
            res[v.row] = v.val * weights[v.col];
        }
        flow = res;
    }

    // float *getResult() override {
    //   return flow.data();
    // }

    
};

#endif//PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOORDINATE_HPP
