/*Brain Inspired Graph Generator - Test : Adjacency matrix generation (untransposed)*/
/*Nicolas HOCHART*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>

#ifndef brainmatrixgenerator
#include "../includes/brainmatrixgenerator.h"
#endif

#ifndef brainstruct
#include "../includes/brainstruct.h"
#endif

#ifndef randomforbrain
#include "../includes/randomforbrain.h"
#endif

#ifndef matrixstruct
#include "../includes/matrixstruct.h"
#endif

#ifndef hardbrain
#include "../includes/hardbrain.h"
#endif

#define NULL ((void *)0)

/*----------------------
--- Time measurement ---
----------------------*/

double my_gettimeofday()
{
    struct timeval tmp_time;
    gettimeofday(&tmp_time, NULL);
    return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
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

    int bonus_debug = 0; //set to 1 to print bonus debug info (for brain and processes grid)
    long i,j,k; //loops
    long n;
    int q = sqrt(p);
    int nb_blocks_row = q, nb_blocks_column = q; //q est la valeur par défaut du nombre de blocks dans les deux dimensions. q*q = p blocs utilisés
    struct MatrixBlock myBlock; //contiendra toutes les informations du block local (processus), indice de ligne/colonne dans la grille 2D, infos pour le PageRank.. voir matrixstruct.h
    long long size;
    long total_memory_allocated_local,nb_zeros;
    long *nb_connections_local_tmp,*nb_connections_tmp;
    int *neuron_types;

    double grid_dim_factor; //utilisé seulement si debug_matrix_block = 1
    int local_result_vector_size_column_blocks;
    long local_result_vector_size_column;

    double start_brain_generation_time, total_brain_generation_time, start_pagerank_time, total_pagerank_time, total_time;

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

    /* Remplissage de la structure MatrixBlock : donne les informations sur le block local (processus) */
    myBlock = fill_matrix_block_info(my_rank, nb_blocks_row, nb_blocks_column, n);

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0 && bonus_debug)
    {
        printf("----------------------\nBilan de votre matrice :\n");
        printf("Taille : %li * %li = %li\n",n,n,size);
        printf("%i blocs sur les lignes (avec %li lignes par bloc) et %i blocs sur les colonnes (avec %li colonnes par bloc)\n",nb_blocks_row,nb_ligne,nb_blocks_column,nb_colonne);
    }

    /*Fetching a raw-coded example brain coded in hardbrain.h*/
    Brain Cerveau = get_hard_brain(n);
    if (my_rank == 0 && bonus_debug) {printf_recap_brain(&Cerveau);}

    MPI_Barrier(MPI_COMM_WORLD);
    start_brain_generation_time = my_gettimeofday(); //start of time measurement for adjacenct matrix (A_CSR) generation

    /*------------------------------------------------------------------------------------------------------------------------------------*/
    /*------------------------------------------- BRAIN MATRIX GENERATION STARTS ---------------------------------------------------------*/
    /*------------------------------------------------------------------------------------------------------------------------------------*/

    //four memory allocations are made during generation: one for the types of neurons, and three for the matrix.

    struct IntCSRMatrix A_CSR;
    neuron_types = (int *)malloc(n * sizeof(int)); //vector which will contain the types of all the neurons

    /* Step 1 : Neuron types random choice */
    //choice of neuron type for each neuron in the brain
    if (myBlock.indc == 0) //choice of processes that will randomly choose the types of neurons
    {
        generate_neuron_types(&Cerveau, myBlock.indl*nb_ligne, nb_ligne, neuron_types + myBlock.indl*nb_ligne);
    }
    //redistribution of selected neuron types to other processes
    for (i=0;i<nb_blocks_row;i++) {MPI_Bcast(neuron_types + /*read/write address : the ind_block_row where we are now * number of rows per block*/ i*nb_ligne, nb_ligne, MPI_INT, i*nb_blocks_column, MPI_COMM_WORLD);}

    /* Step 2 : Matrix generation */
    struct DebugBrainMatrixInfo MatrixDebugInfo;
    generate_csr_brain_transposed_adjacency_matrix_for_pagerank(&A_CSR, myBlock, &Cerveau, neuron_types, &MatrixDebugInfo);

    /*------------------------------------------------------------------------------------------------------------------------------------*/
    /*------------------------------------------- BRAIN MATRIX GENERATION ENDS -----------------------------------------------------------*/
    /*------------------------------------------------------------------------------------------------------------------------------------*/

    MPI_Barrier(MPI_COMM_WORLD);
    total_brain_generation_time = my_gettimeofday() - start_brain_generation_time; //end of time measurement for adjacenct matrix (A_CSR) generation

    //Total allocated memory
    total_memory_allocated_local = MatrixDebugInfo.total_memory_allocated;
    MPI_Allreduce(&total_memory_allocated_local, &(MatrixDebugInfo.total_memory_allocated), 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD); //somme MPI_SUM de tout les total_memory_allocated_local dans MatrixDebugInfo.total_memory_allocated.
    if (my_rank == 0) {printf("Total memory allocated for Row vector / Column vector : %li\nNumber of memory slots actually used : %li\n",MatrixDebugInfo.total_memory_allocated,MatrixDebugInfo.cpt_values);}

    //Matrix prints (only if there is less than 65 neurons)
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0) {printf("\n-------- Matrices:\n");}
    MPI_Barrier(MPI_COMM_WORLD);
    for (k=0;k<p;k++)
    {
        MPI_Barrier(MPI_COMM_WORLD); //display problems may occur, MPI_Barrier does not work well with prints..
        if (my_rank == k)
        {
            printf("Process %i Matrix :\n",my_rank);
            printf_csr_matrix_int_maxdim(&A_CSR, 33);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //Print of part, type and connection information for each neuron
    if (n < 65) //info will be printed for all neurons if there is less than 65 neurons
    {
        if (my_rank==0) {printf("\nInformations for all neurons:\n");}
    }
    else //otherwise, only a few neurons info will be printed
    {
        if (my_rank==0) {printf("\nInformation from a few neurons (impossible to display them all, there are too many): \n");}
    }
    int partie, type;
    long nbco;
    double pourcentage_espere, sum_pourcentage_espere;
    for(i=0;i<n;i++) //Run through neurons
    {
        partie = get_brain_part_ind(i, &Cerveau);
        type = neuron_types[i];
        pourcentage_espere = get_mean_connect_percentage_for_part(&Cerveau, partie, type);

        partie = get_brain_part_ind(i, &Cerveau);
        type = neuron_types[i];
        pourcentage_espere = get_mean_connect_percentage_for_part(&Cerveau, partie, type);
        //if (my_rank == 0) {printf("Ajout de %f au pourcentage éspéré, partie %i neurone de type %i\n",pourcentage_espere,partie,type);} //bug d'affichage print pourcentage global espéré : si on décommente ce print, le bug d'affichage n'est plus.
        if (n < 65) //print info for all neurons only if there is less than 65 neurons
        {
            if (my_rank == 0)
            {
                nbco = MatrixDebugInfo.nb_connections[my_rank*nb_ligne+i];
                printf("neuron %i, type: %i, part: %i, connections: %li, obtained connection percentage: %.2f, expected : %.2f\n",i,type,partie,nbco,(double) nbco / (double) n * 100,pourcentage_espere);
            }
        }
        else if (i % (nb_ligne / 2) == 0) //otherwise, selection of some neurons and print
        {
            if (my_rank == 0)
            {
                nbco = MatrixDebugInfo.nb_connections[my_rank*nb_ligne+i];
                printf("neuron %i, type: %i, part: %i, connections: %li, obtained connection percentage: %.2f, expected : %.2f\n",i,type,partie,nbco,(double) nbco / (double) n * 100,pourcentage_espere);
            }
        }
        sum_pourcentage_espere += pourcentage_espere;
    }
    //Print global information
    if (my_rank==0)
    {
        printf("\nOverall percentage of non-zero values : %.2f%, exptected : %.2f%\n\n",((double) MatrixDebugInfo.cpt_values/(double) size) * 100,sum_pourcentage_espere/ (double) n);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    total_time = my_gettimeofday() - start_brain_generation_time; //end of global time measurement (start of matrix generation -> end of prints)

    if (my_rank == 0)
    {
        printf("Generation time : %.1f s\n", total_brain_generation_time);
        printf("Total time : %.1f s\n", total_time);
    }

    free_brain(&Cerveau); //see brainstruct.h
    free(neuron_types); //<=> free(MatrixDebugInfo.types)
    free(A_CSR.Row); free(A_CSR.Column); free(A_CSR.Value);
    free(MatrixDebugInfo.nb_connections);

    MPI_Finalize();
    return 0;
}
