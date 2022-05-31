#!/bin/bash

#PJM -N "J16N"
#PJM -L "node=4x4:torus"
#PJM -L  "rscgrp=small-torus"
#PJM -L "elapse=60:00"
#PJM --mpi "max-proc-per-node=1"
#PJM -s
#PJM -j
#PJM -o logs/%n.%j.out
#PJM --spath logs/%n.%j.out.stats

mpiexec -n 16 pagerank 600000 4 4
