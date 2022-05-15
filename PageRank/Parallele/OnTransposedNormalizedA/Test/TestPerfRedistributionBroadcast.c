/*Programme permettant de tester la communication "Broadcast" de la redistribution du PageRank appliqué à une matrice d'adjacence transposée.*/
/*Nicolas HOCHART*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>

#include "../includes/matrixstruct.h"

#define NULL ((void *)0)

/*---------------------
--- Mesure de temps ---
---------------------*/

double my_gettimeofday()
{
    struct timeval tmp_time;
    gettimeofday(&tmp_time, NULL);
    return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

/*--------------------------
--- Décision "aléatoire" ---
--------------------------*/

float random_between_0_and_1()
{
    /*Renvoie un nombre aléatoire entre 0 et 1. Permet de faire une décision aléatoire*/
    return (float) rand() / (float) RAND_MAX;
}

/*----------------------
--- Autres fonctions ---
----------------------*/

int pgcd(int a, int b)
{
    int tmp;
    while (b!=0) {tmp = a % b; a = b; b = tmp;}
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

    long i; //pour les boucles
    long n;
    int q = sqrt(p);
    int nb_blocks_row = q, nb_blocks_column = q; //q est la valeur par défaut du nombre de blocks dans les deux dimensions. q*q = p blocs utilisés
    struct MatrixBlock myBlock; //contiendra toutes les informations du block local (processus), indice de ligne/colonne dans la grille 2D, infos pour le PageRank.. voir matrixstruct.h
    long long size;

    double * vector;
    long vector_size;

    double min_time, max_time, sum_time = 0;
    double start_time, total_time;

    int cpt_iteration, nb_iteration_done = 0;

    if (argc < 2) {if (my_rank==0) {printf("Veuillez entrer la taille de la matrice après le nom de l'executable : %s n\nVous pouvez aussi préciser le nombre de blocks en ligne et en colonne dans la matrice de blocks : %s n nb_blocks_row nb_blocks_column\n", argv[0],argv[0]);} exit(1);}
    n = atoll(argv[1]); size = n * n;
    if (argc < 4) {if (my_rank == 0) {printf("Si vous voulez saisir le nombre de blocs (parallèles), veuillez les préciser dans les deux dimensions : %s n nb_blocks_row nb_blocks_column\nValeur prise par défaut : sqrt(%i) = %i\n", argv[0], p, q);}}
    else if (argc >= 4) {nb_blocks_row = atoll(argv[2]); nb_blocks_column = atoll(argv[3]);} //dans le cas où on a des paramètres correspondant au nombre de blocs dans les dimensions

    if (nb_blocks_row * nb_blocks_column != p)
    {
        if (my_rank == 0)
        {
            printf("Erreur : %i * %i != %i. Usage : %s n nb_blocks_row nb_blocks_column (les nombres de blocs par ligne/colonne multipliés doit être égale à %i, le nombre de coeurs alloués)\n", nb_blocks_row, nb_blocks_column, p, argv[0], p);
            if (argc < 4) {printf("nb_blocks_row nb_blocks_column n'étant pas précisés, la valeur prise par défaut serait sqrt(%i).. Mais le nombre de coeurs disponible (%i) n'a pas de racine entière..\nNous ne pouvons pas prendre cette valeur par défaut", p, p);}
        }
        exit(1);
    }
    long nb_ligne = n/nb_blocks_row, nb_colonne = n/nb_blocks_column; //nombre de lignes/colonnes par bloc

    if (nb_ligne * nb_blocks_row != n) {if (my_rank ==0) {printf("Erreur : n (%li) n'est pas divisible par le nombre de blocks sur les lignes (%i)\n",n,nb_blocks_row);}}
    if (nb_colonne * nb_blocks_column != n) {if (my_rank ==0) {printf("Erreur : n (%li) n'est pas divisible par le nombre de blocks sur les colonnes (%i)\n",n,nb_blocks_column);}}
    if (nb_ligne * nb_blocks_row != n || nb_colonne * nb_blocks_column != n) {exit(1);}

    myBlock = fill_matrix_block_info_transposed_adjacency_prv_pagerank(my_rank, nb_blocks_row, nb_blocks_column, n);

    MPI_Comm COLUMN_COMM;
    MPI_Comm_split(MPI_COMM_WORLD, myBlock.indc, myBlock.indl, &COLUMN_COMM);

    vector_size = nb_ligne * nb_blocks_row / pgcd(nb_blocks_row, nb_blocks_column);
    if (my_rank == 0) {printf("Taille du vecteur = %li * %i / %i = %li\n",nb_ligne, nb_blocks_row, pgcd(nb_blocks_row, nb_blocks_column), vector_size);}
    vector = (double *)malloc(vector_size * sizeof(double));

    for (cpt_iteration=0; cpt_iteration<50; cpt_iteration++)
    {
        for (i=0;i<vector_size;i++) {vector[i] = random_between_0_and_1();}

        MPI_Barrier(MPI_COMM_WORLD);
        start_time = my_gettimeofday(); //début mesure de temps

        /********* DEBUT COMMUNICATION BROADCAST DE LA REDISTRIBUTION *********/
        MPI_Bcast(vector, vector_size, MPI_DOUBLE, myBlock.pr_result_redistribution_root, COLUMN_COMM); //chaque processus d'une "colonne de processus" (dans la grille) contient le même vector de taille vector_size
        /********* FIN COMMUNICATION BROADCAST DE LA REDISTRIBUTION *********/

        MPI_Barrier(MPI_COMM_WORLD);
        total_time = my_gettimeofday() - start_time; //fin mesure de temps

        if (my_rank == 0) {printf("[Mesure %i] Temps écoulé : %.5f s\n", cpt_iteration+1, total_time);}
        if (cpt_iteration == 0) {min_time = total_time; max_time = total_time;}
        else {if (min_time > total_time) {min_time = total_time;} if (max_time < total_time) {max_time = total_time;}}
        sum_time += total_time;
        nb_iteration_done++;
    }

    if (my_rank == 0) {printf("Temps moyen écoulé lors d'une itération : %.5f s\nTemps minimum écoulé lors d'une itération : %.5f s\nTemps maximum écoulé lors d'une itération : %.5f s\n", sum_time / nb_iteration_done, min_time, max_time);}

    free(vector);
    MPI_Finalize();
    return 0;
}
