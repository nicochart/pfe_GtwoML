//! PageRank
/*!
  This file defines the structures and functions for PageRank.
  Nicolas HOCHART
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define pagerank

#ifndef matrixstruct
#include "../../BrainMatrixGenerator/includes/matrixstruct.h"
#endif

#ifndef time_measure
#include "../../BrainMatrixGenerator/includes/time_measure.h"
#endif

//! Structure containing the PageRank results.
/*!
   Structure containing the PageRank results, including : pagerank vector (piece), a runtime measurement, and the number of iterations done.
 */
struct PageRankResult
{
     double * result; //PageRank
     double runtime; //PageRank runtime
     int nb_iteration_done; //number of iterations
};
typedef struct PageRankResult PageRankResult;

/*---------------------------------
--- Opérations sur les vecteurs ---
---------------------------------*/

int one_in_vector(double *vect, int size)
{
    //retourne 1 s'il y a un "1" dans le vecteur (permet de tester un cas particulier du PageRank lorsque beta = 1)
    for (int i=0;i<size;i++)
    {
        if (vect[i] == 1.0) {return 1;}
    }
    return 0;
}

double abs_two_vector_error(double *vect1, double *vect2, int size)
{
    /*Calcul l'erreur entre deux vecteurs de taille "size"*/
    double sum=0;
    for (int i=0;i<size;i++) {sum += fabs(vect1[i] - vect2[i]);}
    return sum;
}

//! Applies the PageRank algorithm on a CSR adjacency matrix
/*!
   Applies the PageRank algorithm on a CSR adjacency matrix, with the parameters passed.
 * @param[in] A_CSR {matrixstruct.h : IntCSRMatrix *} Pointer to a structure corresponding to a CSR adjacency matrix.
 * @param[in] nnz_rows_global {long * [matrix_dim]} Pointer to a vector containing the global nnz (number of non-zero) per rows of the matrix
 * @param[in] matrix_dim {long} matrix global dimension
 * @param[in] BlockInfo {matrixstruct.h : MatrixBlock} structure containing information about the local mpi process ("block")
 * @param[in] maxIter {int} max number of iteration to do
 * @param[in] beta {double} dumping factor (pagerank parameter)
 * @param[in] epsilon {double} max error (to study the convergence)
 * @param[in] debug {boolean} debug option

 * @return PRResult {PageRankResult} PageRank results : pagerank vector (piece), runtime measurement, number of iteration done
 */
PageRankResult pagerank_on_adjacency(IntCSRMatrix *A_CSR, long * nnz_rows_global, long matrix_dim, MatrixBlock BlockInfo, int maxIter, double beta, double epsilon, int debug)
{
    struct PageRankResult PRResult;
    double start_pagerank_time, total_pagerank_time;

    int my_mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_mpi_rank);

    /* Communicateurs par ligne et colonne */
    MPI_Comm ROW_COMM;
    MPI_Comm_split(MPI_COMM_WORLD, BlockInfo.indl, BlockInfo.indc, &ROW_COMM);

    MPI_Comm COLUMN_COMM;
    MPI_Comm_split(MPI_COMM_WORLD, BlockInfo.indc, BlockInfo.indl, &COLUMN_COMM);

    /* Communicateurs par groupe de calcul et de besoin du vecteur résultat (PageRank) */
    MPI_Comm RV_CALC_GROUP_COMM; //communicateur interne des groupes (qui regroupe sur les colonnes les blocks du même groupe de calcul)
    MPI_Comm_split(MPI_COMM_WORLD, BlockInfo.result_vector_calculation_group, BlockInfo.my_result_vector_calculation_group_rank, &RV_CALC_GROUP_COMM);

    MPI_Comm INTER_RV_NEED_GROUP_COMM; //communicateur externe des groupes de besoin (groupes sur les lignes) ; permet de calculer l'erreur et la somme totale du vecteur
    MPI_Comm_split(MPI_COMM_WORLD, BlockInfo.inter_result_vector_need_group_communicaton_group, my_mpi_rank, &INTER_RV_NEED_GROUP_COMM);

    //variables utilisées dans le code
    long i,j; //boucles
    long cpt_iterations;
    double error_vect,error_vect_local;
    double *morceau_new_q,*morceau_new_q_local,*morceau_old_q,*tmp;
    double to_add,sum_totale_old_q,sum_totale_new_q,sum_new_q,tmp_sum,sc;
    int nb_elements_ligne;

    //init variables PageRank
    error_vect=INFINITY; cpt_iterations = 0;

    //allocation mémoire pour old_q et new_q, et initialisation de new_q
    morceau_new_q = (double *)malloc(BlockInfo.local_result_vector_size * sizeof(double));
    morceau_new_q_local = (double *)malloc(BlockInfo.local_result_vector_size * sizeof(double));
    morceau_old_q = (double *)malloc(BlockInfo.local_result_vector_size * sizeof(double));
    for (i=0;i<BlockInfo.local_result_vector_size;i++) {morceau_new_q[i] = (double) 1/matrix_dim;}
    sum_totale_new_q = 1;

    MPI_Barrier(MPI_COMM_WORLD);
    start_pagerank_time = my_gettimeofday(); //Début de la mesure de temps pour le PageRank

    /***************************************************************************************************************/
    /****************************************** DEBUT ALGORITHME PAGERANK ******************************************/
    /***************************************************************************************************************/
    while (error_vect > epsilon && !one_in_vector(morceau_new_q,BlockInfo.local_result_vector_size) && cpt_iterations<maxIter)
    {
        /************ Préparation pour l'itération ************/
        if (my_mpi_rank == 0 && debug) {printf("Itération %i, error = %f\n",cpt_iterations,error_vect);}
        //old_q <=> new_q  &   sum_totale_old_q <=> sum_totale_new_q
        tmp = morceau_new_q;
        morceau_new_q = morceau_old_q;
        morceau_old_q = tmp;
        tmp_sum = sum_totale_new_q;
        sum_totale_new_q = sum_totale_old_q;
        sum_totale_old_q = tmp_sum;
        //les itérations se font sur new_q

        //réinitialisation morceau_new_q_local pour nouvelle itération
        for (i=0; i<BlockInfo.local_result_vector_size; i++)
        {
            morceau_new_q_local[i] = 0;
        }

        /************ Produit matrice-vecteur ************/
        //Produit matrice-vecteur new_q = P * old_q LOCAL
        for(i=0; i<BlockInfo.dim_l; i++)
        {
            nb_elements_ligne = nnz_rows_global[BlockInfo.startRow + i]; //le nombre d'éléments non nulles dans la ligne de la matrice "complète" (pas uniquement local)
            sc = morceau_old_q[BlockInfo.startRow_in_result_vector_calculation_group + i] / (double) nb_elements_ligne; //décalage en lecture prit en compte avec startRow
            for (j=(*A_CSR).Row[i]; j<(*A_CSR).Row[i+1]; j++) //décalage en écriture prit en compte avec startColumn
            {
                morceau_new_q_local[BlockInfo.startColumn_in_result_vector_calculation_group + (*A_CSR).Column[j]] += sc; //Produit matrice-vecteur local
            }
        }

        //Produit matrice-vecteur new_q = P * old_q GLOBAL (Reduce)
        MPI_Allreduce(morceau_new_q_local, morceau_new_q, BlockInfo.local_result_vector_size, MPI_DOUBLE, MPI_SUM, RV_CALC_GROUP_COMM); //Produit matrice_vecteur global : Reduce des morceaux de new_q dans tout les processus du même groupe de calcul
        MPI_Barrier(MPI_COMM_WORLD);

        /************ Amortissement ************/
        //Multiplication du vecteur résultat par le facteur d'amortissement beta et ajout de norme(old_q) * (1-beta) / n
        to_add = sum_totale_old_q * (1-beta)/matrix_dim; //Ce qu'il y a à ajouter au résultat P.olq_q * beta. sum_total_old_q contient déjà la somme des éléments de old_q
        for (i=BlockInfo.startColumn_in_result_vector_calculation_group; i<BlockInfo.startColumn_in_result_vector_calculation_group+BlockInfo.dim_c; i++)
        {
            morceau_new_q_local[i] = morceau_new_q_local[i] * beta + to_add; //au fibal new_q = beta * P.old_q + norme(old_q) * (1-beta) / n    (la partie droite du + étant ajoutée à l'initialisation)
        }

        /************ Redistribution ************/
        MPI_Bcast(morceau_new_q, BlockInfo.local_result_vector_size, MPI_DOUBLE, BlockInfo.pr_result_redistribution_root, ROW_COMM); //chaque processus d'une s"ligne de processus" (dans la grille) contient le même morceau de new_q
        MPI_Barrier(MPI_COMM_WORLD);

        /************ Normalisation du nouveau vecteur résultat ************/
        sum_new_q = 0;
        for (i=0;i<BlockInfo.local_result_vector_size;i++) {sum_new_q += morceau_new_q[i];}
        MPI_Allreduce(&sum_new_q, &sum_totale_new_q, 1, MPI_DOUBLE, MPI_SUM, INTER_RV_NEED_GROUP_COMM); //somme MPI_SUM sur les colonnes de tout les sum_new_q dans sum_totale_new_q, utile pour l'itération suivante
        for (i=0;i<BlockInfo.local_result_vector_size;i++) {morceau_new_q[i] *= 1/sum_totale_new_q;} //normalisation avec sum totale (tout processus confondu)

        /************ Opérations de Fin d'itération ************/
        cpt_iterations++;
        error_vect_local = abs_two_vector_error(morceau_new_q,morceau_old_q,BlockInfo.local_result_vector_size); //calcul de l'erreur local
        MPI_Allreduce(&error_vect_local, &error_vect, 1, MPI_DOUBLE, MPI_SUM, INTER_RV_NEED_GROUP_COMM); //somme MPI_SUM sur les colonnes des erreures locales pour avoir l'erreure totale
        MPI_Barrier(MPI_COMM_WORLD);
    }
    /***************************************************************************************************************/
    /******************************************* FIN ALGORITHME PAGERANK *******************************************/
    /***************************************************************************************************************/
    //cpt_iterations contient le nombre d'itérations faites, morceau_new_q sont les morceaux du vecteur contenant le PageRank

    MPI_Barrier(MPI_COMM_WORLD);
    total_pagerank_time = my_gettimeofday() - start_pagerank_time; //fin de la mesure de temps de calcul pour PageRank

    PRResult.result = morceau_new_q;
    PRResult.runtime = total_pagerank_time;
    PRResult.nb_iteration_done = cpt_iterations;

    free(morceau_new_q_local); free(morceau_old_q);
    MPI_Comm_free(&ROW_COMM); MPI_Comm_free(&COLUMN_COMM); MPI_Comm_free(&RV_CALC_GROUP_COMM); MPI_Comm_free(&INTER_RV_NEED_GROUP_COMM);

    return PRResult;
}

//! Applies the PageRank algorithm on a CSR adjacency matrix
/*!
   Applies the PageRank algorithm on a CSR adjacency matrix, with the parameters passed.
 * @param[in] A_CSR {matrixstruct.h : IntCSRMatrix *} Pointer to a structure corresponding to a CSR adjacency matrix.
 * @param[in] nnz_columns_global {long * [matrix_dim]} Pointer to a vector containing the global nnz (number of non-zero) per rows of the matrix
 * @param[in] matrix_dim {long} matrix global dimension
 * @param[in] BlockInfo {matrixstruct.h : MatrixBlock} structure containing information about the local mpi process ("block")
 * @param[in] maxIter {int} max number of iteration to do
 * @param[in] beta {double} dumping factor (pagerank parameter)
 * @param[in] epsilon {double} max error (to study the convergence)
 * @param[in] debug {boolean} debug option

 * @return PRResult {PageRankResult} PageRank results : pagerank vector (piece), runtime measurement, number of iteration done
 */
PageRankResult pagerank_on_transposed(IntCSRMatrix *A_CSR, long * nnz_columns_global, long matrix_dim, MatrixBlock BlockInfo, int maxIter, double beta, double epsilon, int debug)
{
    struct PageRankResult PRResult;
    double start_pagerank_time, total_pagerank_time;

    int my_mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_mpi_rank);

    /* Communicateurs par ligne et colonne */
    MPI_Comm ROW_COMM;
    MPI_Comm_split(MPI_COMM_WORLD, BlockInfo.indl, BlockInfo.indc, &ROW_COMM);

    MPI_Comm COLUMN_COMM;
    MPI_Comm_split(MPI_COMM_WORLD, BlockInfo.indc, BlockInfo.indl, &COLUMN_COMM);

    /* Communicateurs par groupe de calcul et de besoin du vecteur résultat (PageRank) */
    MPI_Comm RV_CALC_GROUP_COMM; //communicateur interne des groupes (qui regroupe sur les colonnes les blocks du même groupe de calcul)
    MPI_Comm_split(MPI_COMM_WORLD, BlockInfo.result_vector_calculation_group, BlockInfo.my_result_vector_calculation_group_rank, &RV_CALC_GROUP_COMM);

    MPI_Comm INTER_RV_NEED_GROUP_COMM; //communicateur externe des groupes de besoin (groupes sur les lignes) ; permet de calculer l'erreur et la somme totale du vecteur
    MPI_Comm_split(MPI_COMM_WORLD, BlockInfo.inter_result_vector_need_group_communicaton_group, my_mpi_rank, &INTER_RV_NEED_GROUP_COMM);

    //variables utilisées dans le code
    long i,j; //boucles
    long cpt_iterations;
    double error_vect,error_vect_local;
    double *morceau_new_q, *morceau_new_q_local, *morceau_old_q,*tmp;
    double to_add,sum_totale_old_q,sum_totale_new_q,sum_new_q,tmp_sum,sc;
    int nb_elements_colonne;

    //init variables PageRank
    cpt_iterations = 0; error_vect=INFINITY;

    //allocation mémoire pour old_q et new_q, et initialisation de new_q
    morceau_new_q = (double *)malloc(BlockInfo.local_result_vector_size * sizeof(double));
    morceau_new_q_local = (double *)malloc(BlockInfo.local_result_vector_size * sizeof(double));
    morceau_old_q = (double *)malloc(BlockInfo.local_result_vector_size * sizeof(double));
    for (i=0;i<BlockInfo.local_result_vector_size;i++) {morceau_new_q[i] = (double) 1/matrix_dim;}
    sum_totale_new_q = 1;

    MPI_Barrier(MPI_COMM_WORLD);
    start_pagerank_time = my_gettimeofday(); //Début de la mesure de temps pour le PageRank

    /***************************************************************************************************************/
    /****************************************** DEBUT ALGORITHME PAGERANK ******************************************/
    /***************************************************************************************************************/
    while (error_vect > epsilon && !one_in_vector(morceau_new_q,BlockInfo.local_result_vector_size) && cpt_iterations<maxIter)
    {
        /************ Préparation pour l'itération ************/
        //old_q <=> new_q  &   sum_totale_old_q <=> sum_totale_new_q
        tmp = morceau_new_q;
        morceau_new_q = morceau_old_q;
        morceau_old_q = tmp;
        tmp_sum = sum_totale_new_q;
        sum_totale_new_q = sum_totale_old_q;
        sum_totale_old_q = tmp_sum;
        //les itérations se font sur new_q

        //réinitialisation morceau_new_q_local pour nouvelle itération
        for (i=0; i<BlockInfo.local_result_vector_size; i++)
        {
            morceau_new_q_local[i] = 0;
        }

        /************ Produit matrice-vecteur ************/
        //Produit matrice-vecteur new_q = P * old_q LOCAL
        for(i=0; i<BlockInfo.dim_l; i++)
        {
            for (j=(*A_CSR).Row[i]; j<(*A_CSR).Row[i+1]; j++)
            {
                nb_elements_colonne = nnz_columns_global[BlockInfo.startColumn + (*A_CSR).Column[j]]; //le nombre d'éléments non nulles dans la ligne de la matrice "complète" (pas uniquement local)
                sc = morceau_old_q[BlockInfo.startColumn_in_result_vector_calculation_group + (*A_CSR).Column[j]] / (double) nb_elements_colonne; //décalage en lecture prit en compte avec startColumn
                morceau_new_q_local[BlockInfo.startRow_in_result_vector_calculation_group + i /*indice de ligne*/] += sc; //décalage en écriture prit en compte avec startRow
            }
        }

        //Produit matrice-vecteur new_q = P * old_q GLOBAL (Reduce)
        MPI_Allreduce(morceau_new_q_local, morceau_new_q, BlockInfo.local_result_vector_size, MPI_DOUBLE, MPI_SUM, RV_CALC_GROUP_COMM); //Produit matrice_vecteur global : Reduce des morceaux de new_q dans tout les processus du même groupe de calcul
        MPI_Barrier(MPI_COMM_WORLD);

        /************ Amortissement ************/
        //Multiplication du vecteur résultat par le facteur d'amortissement beta et ajout de norme(old_q) * (1-beta) / n
        to_add = sum_totale_old_q * (1-beta)/matrix_dim; //Ce qu'il y a à ajouter au résultat P.olq_q * beta. sum_total_old_q contient déjà la somme des éléments de old_q
        for (i=BlockInfo.startColumn_in_result_vector_calculation_group; i<BlockInfo.startColumn_in_result_vector_calculation_group+BlockInfo.dim_c; i++)
        {
            morceau_new_q_local[i] = morceau_new_q_local[i] * beta + to_add; //au fibal new_q = beta * P.old_q + norme(old_q) * (1-beta) / n    (la partie droite du + étant ajoutée à l'initialisation)
        }

        /************ Redistribution ************/
        MPI_Bcast(morceau_new_q, BlockInfo.local_result_vector_size, MPI_DOUBLE, BlockInfo.pr_result_redistribution_root, COLUMN_COMM); //chaque processus d'une s"ligne de processus" (dans la grille) contient le même morceau de new_q
        MPI_Barrier(MPI_COMM_WORLD);

        /************ Normalisation du nouveau vecteur résultat ************/
        sum_new_q = 0;
        for (i=0;i<BlockInfo.local_result_vector_size;i++) {sum_new_q += morceau_new_q[i];}
        MPI_Allreduce(&sum_new_q, &sum_totale_new_q, 1, MPI_DOUBLE, MPI_SUM, INTER_RV_NEED_GROUP_COMM); //somme MPI_SUM sur les colonnes de tout les sum_new_q dans sum_totale_new_q, utile pour l'itération suivante
        for (i=0;i<BlockInfo.local_result_vector_size;i++) {morceau_new_q[i] *= 1/sum_totale_new_q;} //normalisation avec sum totale (tout processus confondu)

        /************ Opérations de Fin d'itération ************/
        cpt_iterations++;
        error_vect_local = abs_two_vector_error(morceau_new_q,morceau_old_q,BlockInfo.local_result_vector_size); //calcul de l'erreur local
        MPI_Allreduce(&error_vect_local, &error_vect, 1, MPI_DOUBLE, MPI_SUM, INTER_RV_NEED_GROUP_COMM); //somme MPI_SUM sur les colonnes des erreures locales pour avoir l'erreure totale
        MPI_Barrier(MPI_COMM_WORLD);
    }
    /***************************************************************************************************************/
    /******************************************* FIN ALGORITHME PAGERANK *******************************************/
    /***************************************************************************************************************/
    //cpt_iterations contient le nombre d'itérations faites, morceau_new_q sont les morceaux du vecteur contenant le PageRank

    MPI_Barrier(MPI_COMM_WORLD);
    total_pagerank_time = my_gettimeofday() - start_pagerank_time; //fin de la mesure de temps de calcul pour PageRank

    PRResult.result = morceau_new_q;
    PRResult.runtime = total_pagerank_time;
    PRResult.nb_iteration_done = cpt_iterations;

    free(morceau_new_q_local); free(morceau_old_q);
    MPI_Comm_free(&ROW_COMM); MPI_Comm_free(&COLUMN_COMM); MPI_Comm_free(&RV_CALC_GROUP_COMM); MPI_Comm_free(&INTER_RV_NEED_GROUP_COMM);

    return PRResult;
}
