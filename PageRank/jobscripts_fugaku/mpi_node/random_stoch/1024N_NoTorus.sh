#!/bin/bash
#PJM -N "J1024N_Classic"
#PJM -L  "node=1024"
#PJM -L  "rscgrp=large"
#PJM -L  "elapse=2:00:00"
#PJM --mpi "max-proc-per-node=1"
#PJM -s
#PJM --appname TBSLA
#PJM -j
#PJM -o logs/%n.%j.out
#PJM --spath logs/%n.%j.out.stats

export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
export FLIB_CPU_AFFINITY="12-59:1"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export allocation_policy=simplex
export PLE_MPI_STD_EMPTYFILE="off"


module purge
module load lang/tcsds-1.2.34
export INSTALL_DIR=${HOME}/install
export PATH=$PATH:${INSTALL_DIR}/tbsla/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${INSTALL_DIR}/tbsla/lib

export OMP_NUM_THREADS=48
python tools/run.py --resfile results.json --machine Fugaku --timeout 3600 " mpirun -n 1024 tbsla_perf_page_rank_mpi_omp --numa-init --GR 32 --GC 32 --matrix random_stoch --format CSR --matrix_dim 32000000 --NNZ 3200" --dic "{'op': 'page_rank', 'format': 'CSR', 'matrixtype': 'random_stoch', 'matrixfolder': '.', 'numainit': 'True', 'resfile': 'results.json', 'machine': 'Fugaku', 'timeout': 500, 'dry': 'False', 'nodes': 1024, 'lang': 'MPIOMP', 'walltime': 15, 'threads': 1, 'tpc': 1, 'GC': 32, 'GR': 32, 'NC': 32000000, 'NR': 32000000, 'C': 0, 'NNZ': 3200, 'cores': 49152}" --infile ${PJM_STDOUT_PATH}.1.0

