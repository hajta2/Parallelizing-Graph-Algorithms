#ifndef PARALLELIZING_GRAPH_ALGORITHMS_GRAPH_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_GRAPH_HPP

#include <iostream>
#include <random>
#include <vector>


class Graph{
private:
    std::vector<int> weights;
    std::vector<int> neighbourMatrix;
public:
    Graph(int edges, int vertices){
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<int> dis(0,vertices-1);
        std::vector<int> tmp(vertices*vertices);

        int from;
        int to;
        for (int i = 0; i < edges; ++i){
            from = dis(gen);
            to = dis(gen);
            while(from==to){
                from = dis(gen);
                to = dis(gen);
            }
            tmp[from*vertices+to] = 1;
        }
        neighbourMatrix=tmp;
    }

    explicit Graph(int vertices){
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<int> dis2(0,vertices*vertices);
        std::uniform_int_distribution<int> dis(0,vertices-1);
        int edges = dis2(gen);

        std::vector<int> tmp(vertices*vertices);

        int from;
        int to;
        for (int i = 0; i < edges; ++i){
            from = dis(gen);
            to = dis(gen);
            while(from==to){
                from = dis(gen);
                to = dis(gen);
            }
            tmp[from*vertices+to] = 1;
        }
        neighbourMatrix=tmp;

    }

    void print(){
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                std::cout<<neighbourMatrix[i*sqrt(neighbourMatrix.size())+j]<<" ";
                if(j==7){
                    std::cout<<std::endl;
                }
            }
        }
    };
};

#endif//PARALLELIZING_GRAPH_ALGORITHMS_GRAPH_HPP
