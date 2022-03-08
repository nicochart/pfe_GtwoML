/*Fichier test pour la distribution du vecteur résultat dans les processus*/
/*Nicolas HOCHART*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>

#define NULL ((void *)0)

/*-----------------------------------------------------------
--- Structures pour les blocks (sur les différents cores) ---
-----------------------------------------------------------*/

struct MatrixBlock
{
     int indl; //Indice de ligne du block
     int indc; //Indice de colonne du block
     long dim_l; //nombre de lignes dans le block
     long dim_c; //nombre de colonnes dans le block
     long startRow; //Indice de départ en ligne (inclu)
     long startColumn; //Indice de départ en colonne (inclu)
     long endRow; //Indice de fin en ligne (inclu)
     long endColumn; //Indice de fin en colonne (inclu)

     int pr_result_redistribution_root; //Indice de colonne du block "root" (source) de la communication-redistribution du vecteur résultat
     int result_vector_group; //Indice de groupe de calcul du vecteur résultat
     int indc_in_result_vector_group; //Indice de ligne du block dans le groupe de calcul du vecteur résultat
     long startColumn_in_result_vector_group; //Indice de départ en ligne dans le groupe de calcul du vecteur résultat (inclu)
};
typedef struct MatrixBlock MatrixBlock;

int pgcd(int a, int b)
{
    int tmp;
    while (b!=0)
    {
        tmp = a % b;
        a = b;
        b = tmp;
    }
    return a;
}

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

    long i,j,k; //pour les boucles
    long n;
    int q = sqrt(p);
    int nb_blocks_row = q, nb_blocks_column = q; //q est la valeur par défaut du nombre de blocks dans les deux dimensions. q*q = p blocs utilisés
    int my_indl, my_indc; //indice de ligne et colonne du bloc

    double grid_dim_factor;
    int local_result_vector_size_blocks;
    long local_result_vector_size;

    if (argc < 2)
    {
        if (my_rank==0) {printf("Veuillez entrer la taille de la matrice après le nom de l'executable : %s n\nVous pouvez aussi préciser le nombre de blocks en ligne et en colonne dans la matrice de blocks : %s n nb_blocks_row nb_blocks_column\n", argv[0],argv[0]);}
        exit(1);
    }
    n = atoll(argv[1]);
    if (argc < 4)
    {
        if (my_rank == 0) {printf("Si vous voulez saisir le nombre de blocs (parallèles), veuillez les préciser dans les deux dimensions : %s n nb_blocks_row nb_blocks_column\nValeur prise par défaut : sqrt(%i) = %i\n", argv[0], p, q);}
    }
    else if (argc >= 4) //dans le cas où on a des paramètres correspondant au nombre de blocs dans les dimensions
    {
        nb_blocks_row = atoll(argv[2]);
        nb_blocks_column = atoll(argv[3]);
    }

    if (nb_blocks_row * nb_blocks_column != p)
    {
        if (my_rank == 0)
        {
            printf("Erreur : %i * %i != %i. Usage : %s n nb_blocks_row nb_blocks_column (les nombres de blocs par ligne/colonne multipliés doit être égale à %i, le nombre de coeurs alloués)\n", nb_blocks_row, nb_blocks_column, p, argv[0], p);
            if (argc < 4)
            {
                printf("nb_blocks_row nb_blocks_column n'étant pas précisés, la valeur prise par défaut serait sqrt(%i).. Mais le nombre de coeurs disponible (%i) n'a pas de racine entière..\nNous ne pouvons pas prendre cette valeur par défaut", p, p);
            }
        }
        exit(1);
    }

    long nb_ligne = n/nb_blocks_row, nb_colonne = n/nb_blocks_column; //nombre de lignes/colonnes par bloc

    if (nb_ligne * nb_blocks_row != n)
    {
        if (my_rank ==0) {printf("Erreur : n (%li) n'est pas divisible par le nombre de blocks sur les lignes (%i)\n",n,nb_blocks_row);}
    }
    if (nb_colonne * nb_blocks_column != n)
    {
        if (my_rank ==0) {printf("Erreur : n (%li) n'est pas divisible par le nombre de blocks sur les colonnes (%i)\n",n,nb_blocks_column);}
    }
    if (nb_ligne * nb_blocks_row != n || nb_colonne * nb_blocks_column != n)
    {
        exit(1);
    }

    my_indl = my_rank / nb_blocks_column;
    my_indc = my_rank % nb_blocks_column;
    struct MatrixBlock myBlock;
    myBlock.indl = my_indl;
    myBlock.indc = my_indc;
    myBlock.dim_l = nb_ligne;
    myBlock.dim_c = nb_colonne;
    myBlock.startRow = my_indl*nb_ligne;
    myBlock.endRow = (my_indl+1)*nb_ligne-1;
    myBlock.startColumn = my_indc*nb_colonne;
    myBlock.endColumn = (my_indc+1)*nb_colonne-1;

    /* Communicateurs par ligne et colonne */
    MPI_Comm ROW_COMM;
    MPI_Comm_split(MPI_COMM_WORLD, myBlock.indl, myBlock.indc, &ROW_COMM);

    MPI_Comm COLUMN_COMM;
    MPI_Comm_split(MPI_COMM_WORLD, myBlock.indc, myBlock.indl, &COLUMN_COMM);

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0)
    {
        printf("----------------------\nBilan de votre matrice :\n");
        printf("Taille : %li * %li\n",n,n);
        printf("%i blocs sur les lignes (avec %li lignes par bloc) et %i blocs sur les colonnes (avec %li colonnes par bloc)\n\n",nb_blocks_row,nb_ligne,nb_blocks_column,nb_colonne);
    }

    grid_dim_factor = (double) nb_blocks_row / (double) nb_blocks_column;
    myBlock.pr_result_redistribution_root = (int) myBlock.indl / grid_dim_factor;
    local_result_vector_size_blocks = nb_blocks_column / pgcd(nb_blocks_row, nb_blocks_column);
    local_result_vector_size = local_result_vector_size_blocks * nb_colonne;
    myBlock.result_vector_group = myBlock.indc / local_result_vector_size_blocks;
    myBlock.indc_in_result_vector_group = myBlock.indc % local_result_vector_size_blocks;
    myBlock.startColumn_in_result_vector_group = nb_colonne * myBlock.indc_in_result_vector_group;

    if (my_rank == 0)
    {
        printf("Taille locale du vecteur résultat : %i blocks, soit %li cases mémoire\nFacteur nb_blocks_row/nb_blocks_column = %i/%i = %f\n\n",local_result_vector_size_blocks,local_result_vector_size,nb_blocks_row,nb_blocks_column,grid_dim_factor);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (k=0;k<p;k++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (my_rank == k)
        {
            printf("[my_rank = %i]: Result Vector Group = %i ; IndStartColumn in Result Vector Group : %i (Block), %li (Element) ; Root redistrib Result Vector : %i,%i\n",my_rank,myBlock.result_vector_group,myBlock.indc_in_result_vector_group,myBlock.startColumn_in_result_vector_group,myBlock.indl,myBlock.pr_result_redistribution_root);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
