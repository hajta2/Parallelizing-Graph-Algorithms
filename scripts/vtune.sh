#!/bin/bash
module load intel/oneapi
module load gcc/9.2.0

OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/Parallelizing_Graph_Algorithms -t 4  single -n 8192 -r 0.03
