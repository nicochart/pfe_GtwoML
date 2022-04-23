/* Tests sur la parallélisation par blocs de ligne et colonnes */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

/*-----------------------------------------------------------
--- Structures pour les blocks (sur les différents cores) ---
-----------------------------------------------------------*/

struct MatrixBlock
{
     int indl; //Indice de ligne du block
     int indc; //Indice de colonne du block
};
typedef struct MatrixBlock MatrixBlock;

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

    if(p != 4)
    {
        if (my_rank == 0) {printf("Ce test est fait pour être testé avec 4 processus MPI, pas %d.\n", p);}
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(1);
    }

    long i,j,k; //pour les boucles
    int q = sqrt(p);
    int nb_blocks_row = q, nb_blocks_column = q; //q est la valeur par défaut du nombre de blocks dans les deux dimensions. q*q = p blocs utilisés
    int my_indl, my_indc; //indice de ligne et colonne du bloc

    my_indl = my_rank / nb_blocks_column;
    my_indc = my_rank % nb_blocks_column;
    struct MatrixBlock myBlock;
    myBlock.indl = my_indl;
    myBlock.indc = my_indc;

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0)
    {
        printf("----------------------\nBilan de votre matrice :\n");
        printf("%i blocs sur les lignes et %i blocs sur les colonnes\n",nb_blocks_row,nb_blocks_column);
        printf("La matrice de blocks ressemble à ceci (numérotés avec leurs my_rank) :\n0 1\n2 3\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int * nombres_row = (int *)malloc(4 * sizeof(int));
    int * nombres_column = (int *)malloc(4 * sizeof(int));
    for (i=0;i<4;i++)
    {
        nombres_row[i] = -1;
        nombres_column[i] = -1;
    }

    int nombre = my_rank;

    MPI_Comm ROW_COMM;
    MPI_Comm_split(MPI_COMM_WORLD, myBlock.indl, myBlock.indc, &ROW_COMM);

    MPI_Comm COLUMN_COMM;
    MPI_Comm_split(MPI_COMM_WORLD, myBlock.indc, myBlock.indl, &COLUMN_COMM);

    MPI_Allgather(&nombre, 1, MPI_INT, nombres_row, 1,  MPI_INT, ROW_COMM);
    MPI_Allgather(&nombre, 1, MPI_INT, nombres_column, 1,  MPI_INT, COLUMN_COMM);

    for (k=0;k<p;k++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (my_rank == k)
        {
            printf("--- Processus %i : ---\nCommunication sur les lignes (vecteur nombres_row) :      ",my_rank);
            for (i=0;i<4;i++)
            {
                printf("%i ",nombres_row[i]);
            }
            printf("\nCommunication sur les colonnes (vecteur nombres_column) : ");
            for (i=0;i<4;i++)
            {
                printf("%i ",nombres_column[i]);
            }
            printf("\n");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
