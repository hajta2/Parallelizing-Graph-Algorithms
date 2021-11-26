#!/bin/bash

export MKL_ENABLE_INSTRUCTIONS=AVX512

echo "Run tests that fit in L2 cache (Mem < 1024k)"
OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**12)) -l 8
OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**12)) -l 12
OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**12)) -l 16
OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**12)) -l 20
OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**12)) -l 24
OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**12)) -l 28
OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**12)) -l 32
OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**12)) -l 36
OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**12)) -l 40


# echo "Run tests that fit in L3 cache (1024k < Mem < 22528K)"
# OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**18)) -l 8
# OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**18)) -l 12
# OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**18)) -l 16
# OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**18)) -l 20
# OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**18)) -l 24
# OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**18)) -l 28
# OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**18)) -l 32
# OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**18)) -l 36
# OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**18)) -l 40

# echo "Run tests that do not fit in L3 cache (22528K < Mem)"
# OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**23)) -l 8
# OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**23)) -l 12
# OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**23)) -l 16
# OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**23)) -l 20
# OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**23)) -l 24
# OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**23)) -l 28
# OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**23)) -l 32
# OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**23)) -l 36
# OMP_PROC_BIND=TRUE OMP_NUM_THREADS=32 numactl --cpunodebind=1 ../bin/run_csr -n $((2**23)) -l 40