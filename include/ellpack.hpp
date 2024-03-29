#ifndef PARALLELIZING_GRAPH_ALGORITHMS_ELLPACK_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_ELLPACK_HPP

#include "graphCOO.hpp"
#ifdef USE_VCL_LIB
#include <VCL2/vectorclass.h>
#endif

class Ellpack : public AbstractGraph {
private:
    std::vector<float> weights;
    std::vector<float> values;
    std::vector<int> columns;
    const int NOVertices;
    int rowLength;
    Type type;
    std::vector<float> flow;
    
public:
    void getWeightedFlow() override {
        std::vector<float> res(NOVertices);
        if (type == NAIVE) {
            for (int i = 0; i < NOVertices-1; ++i) {
                for (int j = 0; j < rowLength; ++j) {
                    res[i] += values[i*rowLength+j] * weights[columns[i*rowLength+j]];
                }
            }
        } else if (type == OPENMP) {
            #pragma omp parallel for
            for (int i = 0; i < NOVertices-1; ++i) {
                for (int j = 0; j < rowLength; ++j) {
                    res[i] += values[i*rowLength+j] * weights[columns[i*rowLength+j]];
                }
            }
#ifndef USE_VCL_LIB
        } else {
            assert(false && "App compiled without VCL support");
        }
#else
        } else if (type == VCL_16_ROW) {
            //rounding down to the nearest lower multiple of VECTOR_SIZE
            int regularPart = rowLength & (-VECTOR_SIZE);
            #pragma omp parallel for
            for (int i = 0; i < NOVertices; ++i) {
                Vec16f row, weight, multiplication;
                //if (values[i * rowLength] == 0) continue;
                for (int j = 0; j < regularPart; j += VECTOR_SIZE) {
                    float list[VECTOR_SIZE];
                    float weightList[VECTOR_SIZE];
                    for (int k = 0; k < VECTOR_SIZE; ++k) {
                        //if(values[i*rowLength+j+k] == 0) continue;
                        list[k] = values[i*rowLength+j+k];
                        weightList[k] = weights[columns[i*rowLength+j+k]];
                    }
                    row.load(list);
                    weight.load(weightList);
                    multiplication = row * weight;
                }
                //if (values[i*rowLength + regularPart - 1]) continue;
                for (int j = regularPart - 1; j < rowLength; ++j) {
                    res[i] += values[i*rowLength+j] * weights[columns[i*rowLength+j]];
                }
                //add the multiplication to res[i]
                res[i] += horizontal_add(multiplication);
            }
        }
#endif
      flow = res;
    }

    explicit Ellpack(GraphCOO graph, Type t, bool transposed) : NOVertices(graph.getNOVertices()), type(t) {
        if(transposed) {
            graph.convertToELLPACK();
        } else {
            graph.convertToTransposedELLPACK();
        }
        std::vector<value> matrix = graph.getNeighbourMatrix();
        weights = graph.getWeights();
        rowLength = graph.getEllpackRow();
        std::vector<float> tmpValues(rowLength * NOVertices);
        std::vector<int> tmpColumns(rowLength * NOVertices);
        int actualRow = 0;
        int elementsInRow = 0;
        for (value v : matrix){
            if(actualRow != v.row) {
                actualRow++;
                elementsInRow = 0;
            }
            tmpValues[v.row*rowLength+elementsInRow] = v.val;
            tmpColumns[v.row*rowLength+elementsInRow] = v.col;
            elementsInRow++;
        }
        values=tmpValues;
        columns=tmpColumns;
    }

    double getBandWidth(double time_s) override {
      double bytes = sizeof(float) * (weights.size() + 2 * NOVertices) +
                     sizeof(int) * weights.size();
      return bytes / 1e9 / time_s;
    }
  
  float *getResult() override {
    return flow.data();
  }

};

#endif//PARALLELIZING_GRAPH_ALGORITHMS_ELLPACK_HPP
