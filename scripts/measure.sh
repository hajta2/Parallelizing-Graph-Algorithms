#!/bin/bash

absolute_path=/home/hajta2/Parallelizing-Graph-Algorithms/

for dir in ../Matrices/*
do
    abs_path=${dir/'../'/$absolute_path}
    file_names=(`ls $abs_path`)
    for i in "${file_names[@]}"
    do
        OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/Parallelizing_Graph_Algorithms $absolute_path'Matrices/'$dir'/'$i
    done
done
