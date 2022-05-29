#!/bin/bash

#PJM -N "J64N"
#PJM -L "node=8x8:torus"
#PJM -L  "rscgrp=small-torus"
#PJM -L "elapse=60:00"
#PJM --mpi "max-proc-per-node=1"
#PJM -s
#PJM -j
#PJM -o logs/%n.%j.out
#PJM --spath logs/%n.%j.out.stats

mpiexec -n 64 pagerank 600000 8 8
