#ifndef PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOORDINATE_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOORDINATE_HPP

#include <algorithm>
#include <cmath>
#include <random>

#include "abstractGraph.hpp"

struct value {
    int row, col;
    float val;
};

const int VECTOR_SIZE = 16;

class GraphCOO : public AbstractGraph {
private:
    std::vector<value> neighbourMatrix;
    std::vector<float> weights;
    const int NOVertices;
    int ellpackRowLength = 0;
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
    //generate COO with const size
    GraphCOO (int vertices) : NOVertices(vertices) {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<int> dis(0, vertices - 1);
        std::uniform_real_distribution<float> disVal(0.0,1.0);

        std::vector<float> tmpWeights(vertices);
        
        for (int i = 0; i < vertices; ++i) {
            std::vector<int> colIndices(VECTOR_SIZE);
            for (int j = 0; j < VECTOR_SIZE; ++j) {
                int col = dis(gen);
                while (std::find(colIndices.begin(), colIndices.end(), col) != colIndices.end()) {
                    col = dis(gen);
                }
                float weight = disVal(gen);
                value val = {i, col, weight};
                neighbourMatrix.push_back(val);
                colIndices.push_back(col);
            }
        }

        std::sort(neighbourMatrix.begin(), neighbourMatrix.end(), [](const auto &lhs, const auto &rhs) {
            if (lhs.row != rhs.row) return lhs.row < rhs.row;
            return lhs.col < rhs.col;
        });

        for (int i = 0; i < vertices; ++i) {
            tmpWeights[i] = dis(gen);
        }
        weights = tmpWeights;

    }

    void convertToELLPACK() {
        //finding the max length
        int maxLength = 0;
        int actualRow = 0;
        int counter = 0;
        std::vector<int> rowLengths(NOVertices);
        for (value const &v : neighbourMatrix) {
            while (v.row != actualRow) {
                actualRow++;
                counter = 0;
            }
            rowLengths[actualRow]++;
            counter++;
            if (counter > maxLength) {
                maxLength = counter;
            }
        }
        //the new matrix padded filled w/ 0s, row`s length is maxLength
        std::vector<value> ellpack;
        for (int i = 0; i < NOVertices; ++i) {
            for (int j = 0; j < maxLength; ++j) {
                value val = {i, j, 0};
                ellpack.push_back(val);
            }
        }
        int rowStart = 0;
        int rowEnd = 0;
        for (int i = 0; i < NOVertices; ++i) {
            rowEnd += rowLengths[i];
            int offset = 0;
            for (int j = rowStart; j < rowEnd; ++j) {
                ellpack[i * maxLength + offset] = neighbourMatrix[j];
                offset++;
            }
            rowStart += rowLengths[i]; 
        }
        neighbourMatrix = ellpack;
        ellpackRowLength = maxLength;
    }

    void convertToTransposedELLPACK() {
        //finding the max length
        int maxLength = 0;
        int actualRow = 0;
        int counter = 0;
        std::vector<int> rowLengths(NOVertices);
        for (value const &v : neighbourMatrix) {
            while (v.row != actualRow) {
                actualRow++;
                counter = 0;
            }
            rowLengths[actualRow]++;
            counter++;
            if (counter > maxLength) {
                maxLength = counter;
            }
        }
        //the new matrix padded filled w/ 0s, row`s length is maxLength
        std::vector<value> ellpack;
        for (int i = 0; i < NOVertices; ++i) {
            for (int j = 0; j < maxLength; ++j) {
                value val = {i, j, 0};
                ellpack.push_back(val);
            }
        }
        int rowStart = 0;
        int rowEnd = 0;
        for (int i = 0; i < NOVertices; ++i) {
            rowEnd += rowLengths[i];
            int offset = 0;
            for (int j = rowStart; j < rowEnd; ++j) {
                ellpack[i * maxLength + offset] = neighbourMatrix[j];
                offset++;
            }
            rowStart += rowLengths[i]; 
        }

        for (value &v : ellpack) {
            value transpose = {v.col, v.row, v.val};
            v = transpose;
        }
        std::sort(ellpack.begin(), ellpack.end(), [](const auto &lhs, const auto &rhs) {
            if (lhs.row != rhs.row) return lhs.row < rhs.row;
            return lhs.col < rhs.col;
        });

        neighbourMatrix = ellpack;
        ellpackRowLength = maxLength;
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

    int getEllpackRow() {
        return ellpackRowLength;
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

    float *getResult() override {
      return flow.data();
    }

};

#endif//PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOORDINATE_HPP
