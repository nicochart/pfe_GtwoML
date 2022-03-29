/*PageRank non pondéré parallele utilisant le générateur V3 (Matrice parallèle avec blocks sur les ligne uniquement)*/
/*Nicolas HOCHART*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>

#include "pagerank_includes.h"
#include "../hardbrain.h"

#define NULL ((void *)0)

/*-----------------------------------------------------------
--- Opérations sur les matrices (à déplacer ou supprimer) ---
-----------------------------------------------------------*/

long cpt_nb_zeros_matrix(int *M, long long size)
{
    /*Compte le nombre de 0 dans la matrice M stockée comme un vecteur d'entiers à size elements*/
    long compteur = 0;
    for (int d=0;d<size;d++)
    {
        if (*(M+d) == 0) {compteur++;}
    }
    return compteur;
}

void matrix_column_sum_vector(int *sum_vector, DoubleCSRMatrix * M_CSR)
{
    /*
    Ecrit dans sum_vector (vecteur de taille (*M_CSR).dim_c) la somme des éléments colonne par colonne de la matrice à l'adresse M_CSR.
    Chaque case d'indice i du sum_vector contiendra la somme des éléments de la colonne du même indice i.
    L'allocation mémoire du vecteur sum_vector doit être faite au préalable.
    */
    int i;
    for (i=0;i<(*M_CSR).dim_c;i++) //initialisation du vecteur sum_vector
    {
        *(sum_vector+i) = 0;
    }

    for (i=0;i<(*M_CSR).len_values;i++) //on parcours le vecteur Column et Value, et on ajoute la valeur à la somme de la colonne correspondante
    {
        *(sum_vector + (*M_CSR).Column[i]) += (*M_CSR).Value[i];
    }
}

void normalize_matrix_on_columns(DoubleCSRMatrix * M_CSR)
{
    /*
    Normalise la matrice CSR M_CSR sur les colonnes.
    */
    long i;
    int * sum_vector = (int *)malloc((*M_CSR).dim_c * sizeof(int));
    matrix_column_sum_vector(sum_vector, M_CSR);
    for (i=0;i<(*M_CSR).len_values;i++) //on parcours le vecteur Column et Value, et on divise chaque valeur (de Value) par la somme (dans sum_vector) de la colonne correspondante
    {
        (*M_CSR).Value[i] = (*M_CSR).Value[i] / sum_vector[(*M_CSR).Column[i]];
    }
    free(sum_vector);
}

void matrix_vector_product(double *y, double *A, double *x, int n)
{
    int i,j;
    /* Effectue le produit matrice vecteur y = A.x. A doit être une matrice n*n, y et x doivent être de longueur n*/
    for (i=0;i<n;i++)
    {
        y[i] = 0;
        for (j=0;j<n;j++)
        {
            y[i] += A[i*n+j] * x[j];
        }
    }
}

void csr_matrix_vector_product(double *y, DoubleCSRMatrix *A, double *x)
{
    long i,j;
    /* Effectue le produit matrice vecteur y = A.x. A doit être une matrice stockée au format CSR, x et y doivent être de talle (*A).dim_c*/
    long nb_ligne = (*A).dim_l;
    long nb_col = (*A).dim_c;
    for (i=0;i<nb_ligne;i++)
    {
        y[i] = 0;
        for (j=(*A).Row[i]; j<(*A).Row[i+1]; j++) //for (j=0;j<nb_col;j++)  y[i] += A[i*nb_col+j] * x[j]
        {
            y[i] += (*A).Value[j] * x[(*A).Column[j]];
        }
    }
}

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

double vector_norm(double *vect, int size)
{
    /* somme les éléments du vecteur de doubles à l'adresse vect de taille size, et renvoie le résultat */
    double sum=0;
    for (int i=0;i<size;i++) {sum+=vect[i];}
    return sum;
}

double abs_two_vector_error(double *vect1, double *vect2, int size)
{
    /*Calcul l'erreur entre deux vecteurs de taille "size"*/
    double sum=0;
    for (int i=0;i<size;i++) {sum += fabs(vect1[i] - vect2[i]);}
    return sum;
}

void copy_vector_value(double *vect1, double *vect2, int size)
{
    /*Copie les valeurs du vecteur 1 dans le vecteur 2. Les deux vecteurs doivent être de taille "size".*/
    for (int i=0;i<size;i++) {vect2[i] = vect1[i];}
}

/*---------------------
--- Mesure de temps ---
---------------------*/

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

    int debug=0; //passer à 1 pour afficher les print de débuggage
    int debug_cerveau=0; //passer à 1 pour avoir les print de débuggage liés aux pourcentages de connexion du cerveau
    long i,j,k; //pour les boucles
    long n;
    long long size;
    long total_memory_allocated_local,nb_zeros,nb_non_zeros,nb_non_zeros_local;
    long *nb_connections_local_tmp,*nb_connections_tmp;
    int *neuron_types, *local_types;

    double start_brain_generation_time, total_brain_generation_time, start_pagerank_time, total_pagerank_time, total_time;

    //allocation mémoire et initialisation d'une liste de taille "nombre de processus" contenant les pourcentages de 0 que l'on souhaite pour chaque bloc
    int *zeros_percentages = (int *)malloc(p * sizeof(int)); for (i=0;i<p;i++) {zeros_percentages[i] = 75;}

    if (argc < 2)
    {
        printf("Veuillez entrer la taille de la matrice après le nom de l'executable : %s n\n Vous pouvez aussi indiquer des pourcentages de 0 pour chaque bloc après le n.\n", argv[0]);
        exit(1);
    }
    n = atoll(argv[1]);
    if (argc > 2)
    {
        for (i=2;i<argc && i<p+2;i++)
        {
            *(zeros_percentages+i-2) = 100 - atoll(argv[i]);
        }
    }

    if (debug && my_rank == 0)
    {
        printf("Liste des pourcentages de 0 dans chaque bloc :\n");
        for (i=0;i<p;i++) {printf("%i ",zeros_percentages[i]);}
        printf("\n");
    }

    size = n * n;
    long nb_ligne = n/p; //nombre de lignes par bloc

    //Cerveau écrit en dur
    Brain Cerveau = get_hard_brain(n); //Cerveau défini dans hardbrain.h
    if (my_rank == 0 && debug_cerveau) {printf_recap_brain(&Cerveau);}

    MPI_Barrier(MPI_COMM_WORLD);
    start_brain_generation_time = my_gettimeofday(); //début de la mesure de temps de génération de la matrice A transposée

    //génération des sous-matrices au format CSR :
    //3 ALLOCATIONS : allocation de mémoire pour CSR_Row, CSR_Column et CSR_Value dans la fonction generate_csr_matrix_for_pagerank()
    struct IntCSRMatrix A_CSR;

    local_types = (int *)malloc(nb_ligne * sizeof(int)); //vecteur local (spécifique à chaque processus) contenant les types des neurones my_rank * nb_ligne à (my_rank + 1) * nb_ligne
    neuron_types = (int *)malloc(n * sizeof(int)); //vecteur contenant les types de tout les neurones
    //choix du type de neurone pour chaque neurone du cerveau
    generate_neuron_types(&Cerveau, my_rank*nb_ligne, nb_ligne, local_types);
    MPI_Allgather(local_types, nb_ligne, MPI_INT, neuron_types, nb_ligne,  MPI_INT, MPI_COMM_WORLD); //réunion dans chaque processus des types de tout les neurones (pour génération)

    //Génération de la matrice CSR à partir du cerveau
    struct DebugBrainMatrixInfo MatrixDebugInfo;
    if (debug_cerveau)
    {
        generate_csr_row_brain_matrix_for_pagerank(&A_CSR, my_rank*nb_ligne, &Cerveau, neuron_types, nb_ligne, n, &MatrixDebugInfo);

        /* MatrixDebugInfo.nb_connections contient actuellement (dans chaque processus) le nombre de connexions faites LOCALEMENT par tout les neurones par colonne. */
        nb_connections_local_tmp = MatrixDebugInfo.nb_connections;
        nb_connections_tmp = (long *)malloc(MatrixDebugInfo.dim_c * sizeof(long));
        MatrixDebugInfo.nb_connections = nb_connections_tmp;
        MPI_Allreduce(nb_connections_local_tmp, nb_connections_tmp, MatrixDebugInfo.dim_c, MPI_LONG, MPI_SUM, MPI_COMM_WORLD); //somme MPI_SUM de tout les nombres de connexions local dans nombres de connexions global
        /* MatrixDebugInfo.nb_connections contient maintenant (dans tout les processus) le nombre GLOBAL de connexions faites pour chaque neurone. */
    }
    else
    {
        generate_csr_row_brain_matrix_for_pagerank(&A_CSR, my_rank*nb_ligne, &Cerveau, neuron_types, nb_ligne, n, NULL);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    total_brain_generation_time = my_gettimeofday() - start_brain_generation_time; //fin de la mesure de temps de génération de la matrice A transposée

    nb_non_zeros_local = A_CSR.len_values;
    MPI_Allreduce(&nb_non_zeros_local, &nb_non_zeros, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD); //somme MPI_SUM de tout les nb_non_zeros_local dans nb_non_zeros

    if (debug_cerveau)
    {
        total_memory_allocated_local = MatrixDebugInfo.total_memory_allocated;
        MPI_Allreduce(&total_memory_allocated_local, &(MatrixDebugInfo.total_memory_allocated), 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD); //somme MPI_SUM de tout les total_memory_allocated_local dans MatrixDebugInfo.total_memory_allocated.
        MatrixDebugInfo.cpt_values = nb_non_zeros;

        if (my_rank == 0)
        {
            printf("Mémoire totale allouée pour le vecteur Row / le vecteur Column : %li\nNombre de cases mémoires effectivement utilisées : %li\n",MatrixDebugInfo.total_memory_allocated,MatrixDebugInfo.cpt_values);
        }
    }

    //matrice normalisée format CSR :
    //1 ALLOCATION : allocation mémoire pour le vecteur CSR_Row_Normé (doubles) qui sera différent de CSR_Row (entiers). Le reste est commun.
    struct DoubleCSRMatrix P_CSR;
    P_CSR.len_values = nb_non_zeros_local; //nombre de zéro local
    P_CSR.dim_l = A_CSR.dim_l;
    P_CSR.dim_c = A_CSR.dim_c;
    P_CSR.Value = (double *)malloc(nb_non_zeros_local * sizeof(double));
    P_CSR.Column = A_CSR.Column; P_CSR.Row = A_CSR.Row; //vecteurs Column et Row communs
    //copie du vecteur Value dans NormValue
    for(i=0;i<nb_non_zeros_local;i++) {P_CSR.Value[i] = (double) A_CSR.Value[i];} //P_CSR.Value = A_CSR.Value
    //normalisation de la matrice
    normalize_matrix_on_columns(&P_CSR);

    if (debug && nb_non_zeros <= 256)
    {
        printf("\nVecteur P_CSR.Row dans my_rank=%i:\n",my_rank);
        for(i=0;i<P_CSR.dim_l+1;i++) {printf("%i ",P_CSR.Row[i]);}printf("\n");
        printf("Vecteur P_CSR.Column dans my_rank=%i:\n",my_rank);
        for(i=0;i<P_CSR.len_values;i++) {printf("%i ",P_CSR.Column[i]);}printf("\n");
        printf("Vecteur P_CSR.Value dans my_rank=%i:\n",my_rank);
        for(i=0;i<P_CSR.len_values;i++) {printf("%.2f ",P_CSR.Value[i]);}printf("\n");
    }

    //Page Rank
    double error_vect,beta;
    double *new_q,*old_q,*tmp;
    long cpt_iterations = 0;
    int maxIter = 100000;
    double epsilon = 0.00000000001;

    //variables temporaires pour code parallèle
    double to_add,sum_totale_old_q,sum_totale_new_q,sum_new_q,tmp_sum,sc,morceau_new_q[nb_ligne];

    //init variables PageRank
    beta = 1; error_vect=INFINITY;
    //allocation mémoire pour old_q et new_q, et initialisation de new_q
    new_q = (double *)malloc(n * sizeof(double));
    old_q = (double *)malloc(n * sizeof(double));
    for (i=0;i<n;i++) {new_q[i] = (double) 1/n;}

    MPI_Barrier(MPI_COMM_WORLD);
    start_pagerank_time = my_gettimeofday(); //Début de la mesure de temps pour le PageRank

    while (error_vect > epsilon && !one_in_vector(new_q,n) && cpt_iterations<maxIter)
    {
        //old_q <=> new_q  &   sum_totale_old_q <=> sum_totale_new_q
        tmp = new_q;
        new_q = old_q;
        old_q = tmp;
        tmp_sum = sum_totale_new_q;
        sum_totale_new_q = sum_totale_old_q;
        sum_totale_old_q = tmp_sum;
        //-- itération sur new_q --

        // calcul du produit matrice-vecteur new_q= P * old_q et de la somme des carrés total
        sum_new_q = 0;
        for(i=0; i<nb_ligne; i++)
        {
            sc = 0; //scalaire
            for (j=P_CSR.Row[i]; j<P_CSR.Row[i+1]; j++)
            {
                sc += P_CSR.Value[j] * old_q[P_CSR.Column[j]]; //sc = ligne de P * vecteur old_q
            }
            //étape 1 : new_q = beta * P.old_q
            morceau_new_q[i] = beta * sc; //new_q[i] = beta * ligneP[i] * old_q
            //étape 2 : (chaque element) newq += norme(old_q) * (1-beta) / n
            to_add = sum_totale_old_q * (1-beta)/n; //sum_total_old_q contient déjà la somme des éléments de old_q
            morceau_new_q[i] = morceau_new_q[i] + to_add;
            sum_new_q  += sc;
        }
        MPI_Allreduce(&sum_new_q, &sum_totale_new_q, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); //somme MPI_SUM de tout les sum_new_q dans sum_totale_new_q, utile pour l'itération suivante

        MPI_Allgather(morceau_new_q, nb_ligne, MPI_DOUBLE, new_q, nb_ligne,  MPI_DOUBLE, MPI_COMM_WORLD); //récupération des morceaux de new_q dans new_q, dans tout les processus
        //étape 3 : normalisation de q
        for (i=0;i<n;i++) {new_q[i] *= 1/sum_totale_new_q;}

        //-- fin itération--
        if (debug && my_rank==0)
        {
            printf("--------------- itération %i :\n",cpt_iterations);
            printf("old_q :"); for(i=0;i<n;i++) {printf("%.2f ",old_q[i]);}printf("\nnew_q : "); for(i=0;i<n;i++) {printf("%.2f ",new_q[i]);} printf("\n");
        }
        cpt_iterations++;
        error_vect = abs_two_vector_error(new_q,old_q,n);
    }
    //fin du while : cpt_iterations contient le nombre d'itérations faites, new_q contient la valeur du vecteur PageRank

    MPI_Barrier(MPI_COMM_WORLD);
    total_pagerank_time = my_gettimeofday() - start_pagerank_time; //fin de la mesure de temps de calcul pour PageRank
    total_time = my_gettimeofday() - start_brain_generation_time; //fin de la mesure de temps globale (début génération matrice -> fin pagerank)

    double sum_pourcentage_espere = 0;
    if (debug_cerveau)
    {
        long nbco;
        int partie,type;
        double pourcentage_espere,sum_pourcentage_espere_local = 0;
        MPI_Barrier(MPI_COMM_WORLD);
        if (my_rank == 0) {printf("Matrice A transposée :\n");}
        MPI_Barrier(MPI_COMM_WORLD);
        for (k=0;k<p;k++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if (my_rank == k)
            {
                for (i=0;i<nb_ligne;i++)
                {
                    if (n<=32) //si la dimension de la matrice est inférieur ou égale à 32, on peut l'afficher
                    {
                        for (j=0;j<n;j++)
                        {
                            printf("%i ", get_csr_matrix_value_int(i, j, &A_CSR));
                        }
                    }
                    else if (nb_ligne <= 64) //affichage seulement si le nombre de ligne par processus est inférieur ou égal à 64
                    {
                        nbco = MatrixDebugInfo.nb_connections[i];
                        printf("%03li \"0\" et %03li \"1\" -",n-nbco,nbco);
                    }

                    partie = get_brain_part_ind(my_rank*nb_ligne+i, &Cerveau);
                    type = MatrixDebugInfo.types[my_rank*nb_ligne+i];
                    nbco = MatrixDebugInfo.nb_connections[my_rank*nb_ligne+i];
                    pourcentage_espere = get_mean_connect_percentage_for_part(&Cerveau, partie, type);
                    sum_pourcentage_espere_local += pourcentage_espere;
                    if (nb_ligne <= 64) //affichage seulement si le nombre de ligne par processus est inférieur ou égal à 64
                    {
                        printf(" type: %i, partie: %i, nbconnections: %li, pourcentage obtenu: %.2f, pourcentage espéré : %.2f",type,partie,nbco,(double) nbco / (double) n * 100,pourcentage_espere);
                        printf("\n");
                    }
                }
            }
        }
        MPI_Allreduce(&sum_pourcentage_espere_local, &sum_pourcentage_espere, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (my_rank == 0)
    {
        printf("\nPourcentage global de valeurs non nulles : %.2f%",((double) nb_non_zeros/(double) size) * 100);
        if (debug_cerveau) {printf(", pourcentage global espéré : %.2f%",sum_pourcentage_espere/ (double) n);}
        printf("\n\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (my_rank == 0)
    {
        if (debug || debug_cerveau || n <= 64)
        {
            printf("\nRésultat ");
            for(i=0;i<n;i++) {printf("%.4f ",new_q[i]);}
            printf("obtenu en %i itérations\n",cpt_iterations);
        }
        else
        {
            printf("Résultat %.4f %.4f ... %.4f obtenu en %i itérations\n",new_q[0],new_q[1],new_q[n-1],cpt_iterations);
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
    free(new_q); free(old_q);
    free(local_types); free(neuron_types);
    free(A_CSR.Row); free(A_CSR.Column); free(A_CSR.Value);
    free(P_CSR.Value); //Row et Column communs avec la matrice CSR

    if (debug_cerveau)
    {
        free(MatrixDebugInfo.nb_connections); //MatrixDebugInfo.types est free plus haut : free(neuron_types);
        free(nb_connections_local_tmp);
    }
    MPI_Finalize();
    return 0;
}
