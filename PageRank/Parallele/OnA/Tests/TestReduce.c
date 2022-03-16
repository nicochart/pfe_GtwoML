/*Test sur le Allreduce de MPI*/
/*
 Il n'est pas possible (avec Allreduce) d'avoir la même adresse d'écriture et de lecture
 Ca aurait pourtant été utile pour éviter de devoir stocker deux vecteurs "vector" de même taille pour pouvoir sommer les "vector_local" dans "vector"
 On aurait pus par exemple sommer les "vector" directement dans "vector"..
*/
/*Nicolas HOCHART*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/*----------
--- Main ---
----------*/

int main(int argc, char **argv)
{
    int my_rank, p, valeur, tag = 0;
    MPI_Status status;

    //Initialisation MPI
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int i,n=10;

    //allocation mémoire pour vector et vector_local
    double *vector,*vector_local;
    vector = (double *)malloc(n * sizeof(double));
    vector_local = (double *)malloc(n * sizeof(double));

    if (my_rank == 0) {printf("vector_local avant Allreduce :");}
    for (i=0;i<n;i++)
    {
        vector_local[i] = i;
        if (my_rank == 0) {printf(" %f",vector_local[i]);}
    }
    if (my_rank == 0) {printf("\n");}

    MPI_Allreduce(vector_local, vector, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); //Reduce du vecteur_local dans le vecteur

    if (my_rank == 0) {printf("vector après Allreduce :");}
    for (i=0;i<n;i++)
    {
        if (my_rank == 0) {printf(" %f",vector[i]);}
    }
    if (my_rank == 0) {printf("\n");}

    free(vector); free(vector_local);
    MPI_Finalize();
    return 0;
}
