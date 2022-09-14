# Scripts for Fugaku

Jobscripts

To use a jobscript, place it in the TBSLA folder and submit it with pjsub command.
Use sub.sh in bash to submit all jobscripts at once.

mpi_node submits jobs with one MPI process per node,
mpi_cmg submits jobs with a maximum of 4 MPI processes per node.

Bash script

"mv.sh" allows you to sort the log files once several jobs have been performed. Put it in the "log" folder (and run it) to use it.

See https://github.com/jgurhem/TBSLA/tree/dev_optimized_coms_pagerank for PageRank on Fugaku code