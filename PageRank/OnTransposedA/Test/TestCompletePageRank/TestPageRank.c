/*
PageRank non pondéré parallele utilisant le générateur V4 (Matrice parallèle avec blocks sur les ligne et les colonnes).
Le vecteur résultat n'est pas encore distribué, mais le sera.
*/
/*Nicolas HOCHART*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>

#include "../pagerank_includes.h"
#include "../hardbrain.h"

#define NULL ((void *)0)

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
    int my_rank, p;
    MPI_Status status;

    //Initialisation MPI
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int debug=0; //passer à 1 pour afficher les print de débuggage
    int debug_cerveau=0; //passer à 1 pour avoir les print de débuggage liés aux pourcentages de connexion du cerveau
    long i; //pour les boucles
    long n;
    int q = sqrt(p);
    int nb_blocks_row = q, nb_blocks_column = q; //q est la valeur par défaut du nombre de blocks dans les deux dimensions. q*q = p blocs utilisés
    struct MatrixBlock myBlock; //contiendra toutes les information du block (processus) local
    long long size;
    long nb_non_zeros,nb_non_zeros_local;
    int *neuron_types;

    double min_time, max_time, sum_time;
    double total_time;
    int cpt_iteration, nb_iteration_done;

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

    size = n * n;
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

    /* Remplissage de la structure MatrixBlock : donne les informations sur le block local (processus) + infos bonus pour le PageRank et les communicateurs */
    myBlock = fill_matrix_block_info_transposed_adjacency_prv_pagerank(my_rank, nb_blocks_row, nb_blocks_column, n); //matrixstruct.h

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0)
    {
        printf("----------------------\nBilan de votre matrice :\n");
        printf("Taille : %li * %li = %li\n",n,n,size);
        printf("%i blocs sur les lignes (avec %li lignes par bloc) et %i blocs sur les colonnes (avec %li colonnes par bloc)\n",nb_blocks_row,nb_ligne,nb_blocks_column,nb_colonne);

        printf("Place mémoire totale moyenne utilisée par la matrice : %.2f o (%.2f Go) (Cerveau 35%) ou %.2f o (%.2f Go) (Cerveau 3.5%) ou %.2f o (%.2f Go) (Cerveau 0.35%).\n", (double) 3 * 0.35*size *4, (double) 3 * 0.35*size *4 / 1000000000, (double) 3 * 0.035*size *4, (double) 3 * 0.035*size *4 / 1000000000, (double) 3 * 0.0035*size *4, (double) 3 * 0.0035*size *4 / 1000000000);
        printf("Place mémoire locale moyenne utilisée par la matrice : %.2f o (%.2f Go) (Cerveau 35%) ou %.2f o (%.2f Go) (Cerveau 3.5%) ou %.2f o (%.2f Go) (Cerveau 0.35%).\n", (double) 3 * 0.35*size *4 / p, (double) 3 * 0.35*size *4 / 1000000000 / p, (double) 3 * 0.035*size *4 / p, (double) 3 * 0.035*size *4 / 1000000000 / p, (double) 3 * 0.0035*size *4 / p, (double) 3 * 0.0035*size *4 / 1000000000 / p);
        printf("Place mémoire locale utilisée par le vecteur neuron_types : %i o (%.2f Go)\n", n*4, (double) n*4 / 1000000000);
    }

    //Cerveau écrit en dur
    Brain Cerveau = get_hard_brain(n); //Cerveau défini dans hardbrain.h

    MPI_Barrier(MPI_COMM_WORLD);

    //génération des sous-matrices au format CSR :
    //3 ALLOCATIONS : allocation de mémoire pour CSR_Row, CSR_Column et CSR_Value dans la fonction generate_csr_matrix_for_pagerank()
    struct IntCSRMatrix A_CSR;

    neuron_types = (int *)malloc(n * sizeof(int)); //vecteur contenant les types de tout les neurones

    //choix du type de neurone pour chaque neurone du cerveau
    if (myBlock.indc == 0) //choix des processus qui vont définir les types de neurones
    {
        generate_neuron_types(&Cerveau, myBlock.indl*nb_ligne, nb_ligne, neuron_types + myBlock.indl*nb_ligne);
    }

    for (i=0;i<nb_blocks_row;i++)
    {
        MPI_Bcast(neuron_types + i*nb_ligne, nb_ligne, MPI_INT, i*nb_blocks_column, MPI_COMM_WORLD);
    }

    //Génération de la matrice CSR à partir du cerveau
    struct DebugBrainMatrixInfo MatrixDebugInfo;
    if (debug_cerveau)
    {
        generate_csr_brain_transposed_adjacency_matrix_for_pagerank(&A_CSR, myBlock, &Cerveau, neuron_types, &MatrixDebugInfo);
    }
    else
    {
        generate_csr_brain_transposed_adjacency_matrix_for_pagerank(&A_CSR, myBlock, &Cerveau, neuron_types, NULL);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    long * nb_connections_columns_global;
    if (debug_cerveau)
    {
        nb_connections_columns_global = MatrixDebugInfo.nb_connections;
        nb_non_zeros = MatrixDebugInfo.cpt_values;
    }
    else
    {
        nb_connections_columns_global = get_nnz_columns(&A_CSR, myBlock, n); //obtention du nombre de connexion par neurone (nnz par ligne)
        nb_non_zeros_local = A_CSR.len_values;
        MPI_Allreduce(&nb_non_zeros_local, &nb_non_zeros, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD); //somme MPI_SUM de tout les nb_non_zeros_local dans nb_non_zeros
    }

    int partie, type;
    double pourcentage_espere, sum_pourcentage_espere;
    for(i=0;i<n;i++) // Parcours des neurones
    {
        partie = get_brain_part_ind(i, &Cerveau);
        type = neuron_types[i];
        pourcentage_espere = get_mean_connect_percentage_for_part(&Cerveau, partie, type);
        sum_pourcentage_espere += pourcentage_espere;
    }
    if (my_rank==0)
    {
        printf("\nPourcentage global de valeurs non nulles : %.2f%, pourcentage global espéré : %.2f%\n\n",((double) nb_non_zeros/(double) size) * 100,sum_pourcentage_espere/ (double) n);
    }

    //Libération des éléments utilisés lors de la génération de la matrice
    free_brain(&Cerveau); //free du cerveau, fonction définie dans brainstruct.h
    free(neuron_types);

    //Page Rank
    int maxIter = 10000;
    double epsilon = 0.00000000001;
    double beta = 1;

    sum_time = 0; nb_iteration_done = 0;
    for (cpt_iteration=0; cpt_iteration<50; cpt_iteration++)
    {
        struct PageRankResult PRResult = pagerank_on_transposed(&A_CSR, nb_connections_columns_global, n, myBlock, maxIter, beta, epsilon, debug);

        free(PRResult.result); //on s'en fiche du résultat
        total_time = PRResult.runtime;

        if (my_rank == 0) {printf("[Pagerank %i] Temps écoulé : %.5f s\n", cpt_iteration+1, total_time);}
        if (cpt_iteration == 0) {min_time = total_time; max_time = total_time;}
        else {if (min_time > total_time) {min_time = total_time;} if (max_time < total_time) {max_time = total_time;}}
        sum_time += total_time;
        nb_iteration_done++;
    }

    if (my_rank == 0) {printf("Temps moyen écoulé lors d'un pagerank : %.5f s\nTemps minimum écoulé lors d'un pagerank : %.5f s\nTemps maximum écoulé lors d'un pagerank : %.5f s\n", sum_time / nb_iteration_done, min_time, max_time);}

    MPI_Barrier(MPI_COMM_WORLD);
    free(A_CSR.Row); free(A_CSR.Column); free(A_CSR.Value);
    free(nb_connections_columns_global); //free MatrixDebugInfo.nb_connections si debug_cerveau à 1

    MPI_Finalize();
    return 0;
}
