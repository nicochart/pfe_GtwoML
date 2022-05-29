#!/bin/bash

#PJM -N "J1N_NoTorus"
#PJM -L "node=1"
#PJM -L "rscgrp=small"
#PJM -L "elapse=60:00"
#PJM --mpi "max-proc-per-node=1"
#PJM -s
#PJM -j
#PJM -o logs/%n.%j.out
#PJM --spath logs/%n.%j.out.stats

mpiexec -n 1 testcom 12496 1 1
