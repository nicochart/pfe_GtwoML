#!/bin/bash

#PJM -N "J16N_NoTorus"
#PJM -L "node=16"
#PJM -L "rscgrp=small"
#PJM -L "elapse=60:00"
#PJM --mpi "max-proc-per-node=1"
#PJM -s
#PJM -j
#PJM -o logs/%n.%j.out
#PJM --spath logs/%n.%j.out.stat

mpiexec -n 16 pagerank 300000 4 4
