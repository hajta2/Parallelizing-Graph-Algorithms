#!/bin/bash

absolute_path=/home/hajta2/Parallelizing-Graph-Algorithms/

for dir in ../Matrices/*
do
    abs_path=${dir/'../'/$absolute_path}
    file_names=(`ls $abs_path`)
    for i in "${file_names[@]}"
    do
        ../bin/Parallelizing_Graph_Algorithms $i
    done
done
