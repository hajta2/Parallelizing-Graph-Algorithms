#ifndef PARALLELIZING_GRAPH_ALGORITHMS_GRAPH_HPP
#define PARALLELIZING_GRAPH_ALGORITHMS_GRAPH_HPP

#include <chrono>
#include <functional>
#include <iostream>
#include <random>
#include <vector>


class Graph{
private:
    std::vector<int> weights;
    std::vector<int> neighbourMatrix;
    const int NOVertices;
    std::vector<int> weightedFlow;


    void getWeightedFlow(){
        std::vector<int> res(NOVertices);

        for (int i = 0; i < NOVertices; ++i) {
            int sum = 0;
            for (int j = 0; j < NOVertices; ++j) {
                sum += weights[i]*neighbourMatrix[i*NOVertices+j];
            }
            res[i] = sum;
        }
        weightedFlow = res;
    }

public:
    Graph(int edges, int vertices) : NOVertices(vertices) {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<int> dis(0,vertices-1);
        std::vector<int> tmp(vertices*vertices);
        std::vector<int> tmp2(vertices);

        int from;
        int to;
        for (int i = 0; i < edges; ++i){
            from = dis(gen);
            to = dis(gen);
            while(from==to && tmp[from*vertices+to] == 1){
                from = dis(gen);
                to = dis(gen);
            }
            tmp[from*vertices+to] = 1;
        }

        for (int i = 0; i < vertices; ++i) {
            tmp2[i] = dis(gen);
        }

        neighbourMatrix = tmp;
        weights = tmp2;
    }

    explicit Graph(int vertices) : NOVertices(vertices) {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<int> dis(0,vertices-1);
        std::uniform_int_distribution<int> dis2(0,vertices*(vertices-1));
        int edges = dis2(gen);

        std::vector<int> tmp(vertices*vertices);
        std::vector<int> tmp2(vertices);

        int from;
        int to;
        for (int i = 0; i < edges; ++i){
            from = dis(gen);
            to = dis(gen);
            while(from==to && tmp[from*vertices+to] == 1){
                from = dis(gen);
                to = dis(gen);
            }
            tmp[from*vertices+to] = 1;
        }

        for (int i = 0; i < vertices; ++i) {
            tmp2[i] = dis(gen);
        }

        neighbourMatrix = tmp;
        weights = tmp2;

    }

    int measure(){
        auto start = std::chrono::high_resolution_clock::now();
        getWeightedFlow();
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = duration_cast<std::chrono::microseconds>(stop - start);
        return duration.count();
    }

    void print(){
        for (int i = 0; i < NOVertices; ++i) {
            for (int j = 0; j < NOVertices; ++j) {
                std::cout<<neighbourMatrix[i*NOVertices+j]<<" ";
                if(j==7){
                    std::cout<<std::endl;
                }
            }
        }
    };
};

#endif//PARALLELIZING_GRAPH_ALGORITHMS_GRAPH_HPP
