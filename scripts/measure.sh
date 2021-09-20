#!/bin/bash

absolute_path=/home/hajta2/Parallelizing-Graph-Algorithms/
cd $absolute_path'build'; make;cd $absolute_path'scripts'
for dir in ../Matrices/*
do
    abs_path=${dir/'../'/$absolute_path}
    cd $abs_path
    file_names=(`ls`)
    for i in "${file_names[@]}"
    do
        echo $i
        OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 $absolute_path/bin/Parallelizing_Graph_Algorithms $i
    done
done
