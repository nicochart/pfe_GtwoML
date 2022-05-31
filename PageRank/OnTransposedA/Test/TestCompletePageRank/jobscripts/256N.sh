#!/bin/bash

#PJM -N "J256N"
#PJM -L "node=16x25:torus"
#PJM -L "rscgrp=large"
#PJM -L "elapse=60:00"
#PJM --mpi "max-proc-per-node=1"
#PJM -s
#PJM -j
#PJM -o logs/%n.%j.out
#PJM --spath logs/%n.%j.out.stats

mpiexec -n 256 pagerank 2400000 16 16
