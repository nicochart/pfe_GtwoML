/* Tests sur la parallélisation par blocs de ligne et colonnes */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

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

    if ((double) sqrt(p) != (int) sqrt(p))
    {
        if (my_rank == 0) {printf("Erreur : le nombre de coeurs disponible (%i) n'a pas de racine entière..\nOn doit pouvoir le passer à la racine et obtenir un nombre entier pour pouvoir diviser en ligne et colonnes.\n",p);}
        exit(1);
    }

    long i,j,k; //pour les boucles
    long n;
    int q = sqrt(p);
    int nb_blocks_row = q, nb_blocks_column = q; //q est la valeur par défaut du nombre de blocks dans les deux dimensions. q*q = p blocs utilisés
    int my_indl, my_indc; //indice de ligne et colonne du bloc
    long long size;

    if (argc < 2)
    {
        if (my_rank == 0) {printf("Veuillez entrer la taille de la matrice après le nom de l'executable : %s n\n", argv[0]);}
        exit(1);
    }
    n = atoll(argv[1]);

    if (argc < 4)
    {
        if (my_rank == 0) {printf("Si vous voulez saisir le nombre de blocs (parallèles), veuillez les préciser dans les deux dimensions : %s n l c\nValeur prise par défaut : sqrt(%i) = %i\n", argv[0], p, q);}
    }
    else if (argc >= 4) //dans le cas où on a des paramètres correspondant au nombre de blocs dans les dimensions
    {
        nb_blocks_row = atoll(argv[2]);
        nb_blocks_column = atoll(argv[3]);
        if (nb_blocks_row * nb_blocks_column != p)
        {
            if (my_rank == 0) {printf("Erreur : %i * %i != %i. Usage : %s n nb_blocks_row nb_blocks_column (les nombres de blocs par ligne/colonne multipliés doit être égale à %i, le nombre de coeurs alloués)\n", nb_blocks_row, nb_blocks_column, p, argv[0], p);}
            exit(1);
        }
    }

    size = n * n; //taille de la matrice
    long nb_ligne = n/nb_blocks_row, nb_colonne = n/nb_blocks_column; //nombre de lignes/colonnes par bloc

    my_indl = my_rank / nb_blocks_column;
    my_indc = my_rank % nb_blocks_column;

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0)
    {
        printf("----------------------\nBilan de votre matrice :\n");
        printf("Taille : %li * %li = %li\n",n,n,size);
        printf("%i blocs sur les lignes (avec %li lignes par bloc) et %i blocs sur les colonnes (avec %li colonnes par bloc)\n",nb_blocks_row,nb_ligne,nb_blocks_column,nb_colonne);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for (k=0;k<p;k++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (my_rank == k)
        {
            printf("Indices de ligne et colonne du bloc my_rank=%i : [%i,%i], indices (de la matrice n*n) contenus dans ce bloc : [%li,%li] jusque [%li,%li]\n",my_rank,my_indl,my_indc,my_indl*nb_ligne,my_indc*nb_colonne,(my_indl+1)*nb_ligne-1,(my_indc+1)*nb_colonne-1);
        }
    }

    MPI_Finalize();
    return 0;
}
