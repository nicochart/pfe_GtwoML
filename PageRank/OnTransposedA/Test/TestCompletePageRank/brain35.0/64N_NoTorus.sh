#!/bin/bash

#PJM -N "J64N_NoTorus"
#PJM -L "node=64"
#PJM -L "rscgrp=small"
#PJM -L "elapse=60:00"
#PJM --mpi "max-proc-per-node=1"
#PJM -s
#PJM -j
#PJM -o logs/%n.%j.out
#PJM --spath logs/%n.%j.out.stat

mpiexec -n 64 testcom 100000 8 8
