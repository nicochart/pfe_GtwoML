#!/bin/bash

#PJM -N "J4N"
#PJM -L "node=2x2:torus"
#PJM -L  "rscgrp=small-torus"
#PJM -L "elapse=60:00"
#PJM --mpi "max-proc-per-node=1"
#PJM -s
#PJM -j
#PJM -o logs/%n.%j.out
#PJM --spath logs/%n.%j.out.stats

mpiexec -n 4 pagerank 150000 2 2
