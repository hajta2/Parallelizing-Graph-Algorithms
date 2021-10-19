#ifndef PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP

#include "graphCOO.hpp"
#include <omp.h>
#include <cassert>

class GraphCSR : public AbstractGraph {
private:
    std::vector<float> csrVal;
    std::vector<int> csrColInd;
    std::vector<int> csrRowPtr;
    std::vector<float> weights;
    std::vector<float> flow;
    Type type;
    const int NOVertices;

    void naiveFlow() {
        std::vector<float> res(NOVertices);
        for (int i = 0; i < NOVertices-1; ++i) {
            int start = csrRowPtr[i];
            int end = csrRowPtr[i + 1];
            for (int j = start; j < end; ++j) {
                res[i] += csrVal[j] * weights[csrColInd[j]];
            }
        }
    }

    void OPENMPFlow() {
        std::vector<float> res(NOVertices);
        #pragma omp parallel for
        for (int i = 0; i < NOVertices-1; ++i) {
            int start = csrRowPtr[i];
            int end = csrRowPtr[i + 1];
            for (int j = start; j < end; ++j) {
                res[i] += csrVal[j] * weights[csrColInd[j]];
            }
        }
    }

    void getWeightedFlow() override {
        if (type == NAIVE) {
            naiveFlow();
        } else if (type == OPENMP){
            OPENMPFlow();
        }
    }

public:
    explicit GraphCSR(GraphCOO& graph, Type t) : NOVertices(graph.getNOVertices()), type(t) {
        std::vector<value> matrix = graph.getNeighbourMatrix();
        weights = graph.getWeights();
        int actualRow = 0;
        csrRowPtr.push_back(0);
        for(value const &v : matrix){
            while(v.row != actualRow){
                actualRow++;
                csrRowPtr.push_back(csrColInd.size());
            }
            csrColInd.push_back(v.col);
            csrVal.push_back(v.val);
        }
        //NONonZeroElements
        int NONonZeros = csrColInd.size();
        csrRowPtr.push_back(NONonZeros);
    }

    double bandWidth() {

        double time = this -> measure();
        double bytes = 4 * (weights.size() + csrVal.size() + 2 * flow.size());
        //Gigabyte per second
        return (bytes / 1000) / time;
    }

};



#endif//PARALLELIZING_GRAPH_ALGORITHMS_GRAPHCOMPRESSED_HPP
