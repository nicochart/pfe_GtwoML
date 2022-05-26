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
    int my_rank, p, valeur, tag = 0;
    MPI_Status status;

    //Initialisation MPI
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int debug=0; //passer à 1 pour afficher les print de débuggage
    int debug_cerveau=0; //passer à 1 pour avoir les print de débuggage liés aux pourcentages de connexion du cerveau
    int debug_matrix_block=1; //passer à 1 pour afficher les print de débuggage du block de matrice
    int debug_full_pagerank_result=0; //passer à 1 pour allgather et afficher le vecteur résultat complet
    int debug_print_matrix=0; //passer à 1 pour afficher les matrices dans les processus
    long i,j,k; //pour les boucles
    long n;
    int q = sqrt(p);
    int nb_blocks_row = q, nb_blocks_column = q; //q est la valeur par défaut du nombre de blocks dans les deux dimensions. q*q = p blocs utilisés
    struct MatrixBlock myBlock; //contiendra toutes les information du block (processus) local
    long long size;
    long total_memory_allocated_local,nb_zeros,nb_non_zeros,nb_non_zeros_local;
    long *nb_connections_local_tmp,*nb_connections_tmp;
    int *neuron_types;

    double grid_dim_factor; //utilisé seulement si debug_matrix_block = 1
    int local_result_vector_size_row_blocks,local_result_vector_size_column_blocks;

    double start_brain_generation_time, total_brain_generation_time, total_pagerank_time, total_time;

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
    if (debug_matrix_block)
    {
        grid_dim_factor = (double) nb_blocks_column / (double) nb_blocks_row;
    }
    local_result_vector_size_column_blocks = nb_blocks_column / pgcd(nb_blocks_row, nb_blocks_column);
    local_result_vector_size_row_blocks = nb_blocks_row / pgcd(nb_blocks_row, nb_blocks_column);

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0)
    {
        printf("----------------------\nBilan de votre matrice :\n");
        printf("Taille : %li * %li = %li\n",n,n,size);
        printf("%i blocs sur les lignes (avec %li lignes par bloc) et %i blocs sur les colonnes (avec %li colonnes par bloc)\n",nb_blocks_row,nb_ligne,nb_blocks_column,nb_colonne);
    }

    if (my_rank == 0 && debug_matrix_block) //débuggage du vecteur résultat
    {
        printf("Taille locale du vecteur résultat : %i,%i blocks (ligne,colonne), soit %li (=%li) cases mémoire\nFacteur nb_blocks_column/nb_blocks_row = %i/%i = %f, pgcd = %i\n\n",
        local_result_vector_size_row_blocks,local_result_vector_size_column_blocks, //Taille (en blocks)
        myBlock.local_result_vector_size,local_result_vector_size_column_blocks*nb_colonne, //Taille (en cases mémoires : local_result_vector_size_column_blocks*nb_colonne = local_result_vector_row_column_blocks*nb_ligne)
        nb_blocks_column,nb_blocks_row,grid_dim_factor,pgcd(nb_blocks_row, nb_blocks_column)); //Facteur
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (debug_matrix_block) //debuggage des groupes de calcul du vecteur résultat (PageRank)
    {
        for (k=0;k<p;k++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if (my_rank == k)
            {
                 printf(
                "[my_rank = %i]: RV Group = %i ; (IndRow, IndColumn) in RV Group : %i,%i ; (StartRow, StartColumn) in RV Group (Memory): %li,%li ; Inter-RVNeedGroup Communicator rank : %i ; Root redistrib RV : %i,%i\n",
                 my_rank,myBlock.result_vector_calculation_group, //RV Group
                 myBlock.indl_in_result_vector_calculation_group,myBlock.indc_in_result_vector_calculation_group, //IndRow, IndColumn in RV Group
                 myBlock.startRow_in_result_vector_calculation_group, myBlock.startColumn_in_result_vector_calculation_group, //StartRow, StartColumn
                 myBlock.inter_result_vector_need_group_communicaton_group, //Inter-RV Need Communicator rank
                 myBlock.pr_result_redistribution_root,myBlock.indc //Root redistrib RV
                 );
            }
        }
    }

    //Cerveau écrit en dur
    Brain Cerveau = get_hard_brain(n); //Cerveau défini dans hardbrain.h
    if (my_rank == 0 && debug_cerveau) {printf_recap_brain(&Cerveau);}

    MPI_Barrier(MPI_COMM_WORLD);
    start_brain_generation_time = my_gettimeofday(); //début de la mesure de temps de génération de la matrice A transposée

    //génération des sous-matrices au format CSR :
    //3 ALLOCATIONS : allocation de mémoire pour CSR_Row, CSR_Column et CSR_Value dans la fonction generate_csr_matrix_for_pagerank()
    struct IntCSRMatrix A_CSR;

    neuron_types = (int *)malloc(n * sizeof(int)); //vecteur contenant les types de tout les neurones

    //choix du type de neurone pour chaque neurone du cerveau
    if (myBlock.indc == 0) //choix des processus qui vont définir les types de neurones
    {
        if (debug) {printf("Le processus %i répond à l'appel, il va s'occuper des neurones %li à %li\n",my_rank,myBlock.indl*nb_ligne,(myBlock.indl+1)*nb_ligne);}
        generate_neuron_types(&Cerveau, myBlock.indl*nb_ligne, nb_ligne, neuron_types + myBlock.indl*nb_ligne);
    }

    for (i=0;i<nb_blocks_row;i++)
    {
        if (my_rank==0 && debug) {printf("Communication du processus %i vers les autres de %i neurones\n",i*nb_blocks_column,nb_ligne);}
        MPI_Bcast(neuron_types + /*adresse de lecture/ecriture : le ind_block_row dans lequel on est actuellement * nb_ligne*/ i*nb_ligne, nb_ligne, MPI_INT, i*nb_blocks_column, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank==0 && debug)
    {
        for (i=0;i<n;i++)
        {
            printf("%i ",neuron_types[i]);
        }
        printf("\n");
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
    total_brain_generation_time = my_gettimeofday() - start_brain_generation_time; //fin de la mesure de temps de génération de la matrice A transposée

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

    if (debug_cerveau)
    {
        total_memory_allocated_local = MatrixDebugInfo.total_memory_allocated;
        MPI_Allreduce(&total_memory_allocated_local, &(MatrixDebugInfo.total_memory_allocated), 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD); //somme MPI_SUM de tout les total_memory_allocated_local dans MatrixDebugInfo.total_memory_allocated.

        if (my_rank == 0)
        {
            printf("Mémoire totale allouée pour le vecteur Row / le vecteur Column : %li\nNombre de cases mémoires effectivement utilisées : %li\n",MatrixDebugInfo.total_memory_allocated,MatrixDebugInfo.cpt_values);
        }
    }

    //Page Rank
    int maxIter = 10000;
    double epsilon = 0.00000000001;
    double beta = 1;

    if (my_rank == 0) {printf("Running PageRank..\n");}
    struct PageRankResult PRResult = pagerank_on_transposed(&A_CSR, nb_connections_columns_global, n, myBlock, maxIter, beta, epsilon, debug);

    MPI_Barrier(MPI_COMM_WORLD);
    total_pagerank_time = PRResult.runtime;
    total_time = my_gettimeofday() - start_brain_generation_time; //fin de la mesure de temps globale (début génération matrice -> fin pagerank)

    //affichage matrices
    if (debug_print_matrix)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (my_rank == 0) {printf("-------- Matrices:\n");}
        MPI_Barrier(MPI_COMM_WORLD);
        for (k=0;k<p;k++)
        {
            MPI_Barrier(MPI_COMM_WORLD); //des problèmes d'affichage peuvent survenir, MPI_Barrier ne marche pas bien avec les prints..
            if (my_rank == k)
            {
                printf("Matrice du processus %i :\n",my_rank);
                printf_csr_matrix_int_maxdim(&A_CSR, 20);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int partie, type;
    long nbco;
    double pourcentage_espere, sum_pourcentage_espere;
    if (debug_cerveau)
    {
        for(i=0;i<n;i++) // Parcours des neurones
        {
            partie = get_brain_part_ind(i, &Cerveau);
            type = neuron_types[i];
            pourcentage_espere = get_mean_connect_percentage_for_part(&Cerveau, partie, type);
            nbco = MatrixDebugInfo.nb_connections[my_rank*nb_ligne+i];
            if (my_rank == 0)
            {
                printf("neurone %i, type: %i, partie: %i, nbconnections: %li, pourcentage obtenu: %.2f, pourcentage espéré : %.2f\n",i,type,partie,nbco,(double) nbco / (double) n * 100,pourcentage_espere);
            }
            sum_pourcentage_espere += pourcentage_espere;
        }
        if (my_rank==0)
        {
            printf("\nPourcentage global de valeurs non nulles : %.2f%, pourcentage global espéré : %.2f%\n\n",((double) nb_non_zeros/(double) size) * 100,sum_pourcentage_espere/ (double) n);
        }
    }

    double *pagerank_result;
    if (debug_full_pagerank_result)
    {
        /* Communicateur inter groupe de besoin */
        MPI_Comm INTER_RV_NEED_GROUP_COMM; //communicateur externe des groupes de besoin (groupes sur les lignes) ; permet de récupérer le résultat final
        MPI_Comm_split(MPI_COMM_WORLD, myBlock.inter_result_vector_need_group_communicaton_group, my_rank, &INTER_RV_NEED_GROUP_COMM);

        pagerank_result = (double *)malloc(n * sizeof(double));
        MPI_Allgather(PRResult.result, myBlock.local_result_vector_size, MPI_DOUBLE, pagerank_result, myBlock.local_result_vector_size, MPI_DOUBLE, INTER_RV_NEED_GROUP_COMM); //récupération par colonne des morceaux de new_q dans pagerank_result, dans tout les processus
        if (my_rank == 0)
        {
            printf("\nRésultat ");
            for(i=0;i<n;i++) {printf("%.4f ",pagerank_result[i]);}
            printf("obtenu en %i itérations\n",PRResult.nb_iteration_done);
        }
        free(pagerank_result);
    }
    else if (my_rank == 0)
    {
        if ((debug && debug_cerveau) || n <= 64)
        {
            printf("\nMorceau du résultat dans le processus %i : ",my_rank);
            for(i=0;i<nb_colonne;i++) {printf("%.4f ",PRResult.result[i]);}
            printf("obtenu en %i itérations\n",PRResult.nb_iteration_done);
        }
        else
        {
            printf("Morceau du vecteur résultat : %.4f %.4f ... %.4f obtenu en %i itérations\n",PRResult.result[0],PRResult.result[1],PRResult.result[nb_colonne-1],PRResult.nb_iteration_done);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0)
    {
        printf("Temps écoulé lors de la génération : %.1f s\n", total_brain_generation_time);
        printf("Temps écoulé lors de l'application de PageRank : %.1f s\n", total_pagerank_time);
        printf("Temps total écoulé : %.1f s\n", total_time);
    }

    free_brain(&Cerveau); //free du cerveau, fonction définie dans brainstruct.h
    free(PRResult.result);
    free(neuron_types);
    free(A_CSR.Row); free(A_CSR.Column); free(A_CSR.Value);

    free(nb_connections_columns_global); //free MatrixDebugInfo.nb_connections si debug_cerveau à 1

    MPI_Finalize();
    return 0;
}
