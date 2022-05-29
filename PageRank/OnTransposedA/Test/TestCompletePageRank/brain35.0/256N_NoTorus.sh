#!/bin/bash

#PJM -N "J256N_NoTorus"
#PJM -L "node=256"
#PJM -L "rscgrp=small"
#PJM -L "elapse=60:00"
#PJM --mpi "max-proc-per-node=1"
#PJM -s
#PJM -j
#PJM -o logs/%n.%j.out
#PJM --spath logs/%n.%j.out.stat

mpiexec -n 256 testcom 200000 16 16
