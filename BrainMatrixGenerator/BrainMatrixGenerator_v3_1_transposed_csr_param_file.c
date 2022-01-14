/* Générateur de matrice pour le PageRank correspondant au cerveau. */
/*Nicolas HOCHART*/
/*Hugo AILLERIE*/

/*
 Ce générateur génère directement la matrice AT (matrice d'adjacence A transposée)
 La matrice est directement générée en CSR
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>
#include <omp.h>
#include "param.h"

#define NULL ((void *)0)

/*-------------------------------------------------------------------
--- Structures pour le stockage des matrices au format COO et CSR ---
-------------------------------------------------------------------*/

struct IntCSRMatrix
{
     int * Row; //vecteur de taille "nombre de lignes + 1" (dim_l + 1)
     int * Column; //vecteur de taille len_values = "nombre d'éléments non nuls dans la matrice"
     int * Value; //vecteur de taille len_values = "nombre d'éléments non nuls dans la matrice"
     long dim_l; //nombre de lignes
     long dim_c; //nombre de colonnes
     long len_values;  //taille des vecteurs Column et Value
};
typedef struct IntCSRMatrix IntCSRMatrix;

struct DoubleCSRMatrix
{
     int * Row; //vecteur de taille "nombre de lignes + 1" (dim_l + 1)
     int * Column; //vecteur de taille len_values = "nombre d'éléments non nuls dans la matrice"
     double * Value; //vecteur de taille len_values = "nombre d'éléments non nuls dans la matrice"
     long dim_l; //nombre de lignes
     long dim_c; //nombre de colonnes
     long len_values; //taille des vecteurs Column et Value
};
typedef struct DoubleCSRMatrix DoubleCSRMatrix;


//structure permettant de débugger le générateur de matrice correspondant à un cerveau en COO "generate_csr_brain_matrix_for_pagerank"
struct DebugBrainMatrixInfo
{
     long dim_c; //nombre de neurones "destination" (sur les colonnes de la matrice)
     long dim_l; //nombre de neurones "source" (sur les lignes de la matrice)
     int * types; //vecteur de taille dim_c indiquant le type choisi pour chaque neurones du cerveau
     long * nb_connections; //vecteur de taille dim_c indiquant le nombre de connections qu'a effectué chaque neurone.
     long total_memory_allocated; //memoire totale allouée pour Row (ou pour Column, ce sont les mêmes). Cette mémoire étant allouée dynamiquement, elle peut être plus grande que cpt_values.
     long cpt_values; //nombre de connexions (de 1 dans la matrice générée).
};
typedef struct DebugBrainMatrixInfo DebugBrainMatrixInfo;

/*--------------------------
--- Décision "aléatoire" ---
--------------------------*/

float random_between_0_and_1()
{
    /*Renvoie un nombre aléatoire entre 0 et 1. Permet de faire une décision aléatoire*/
    return (float) rand() / (float) RAND_MAX;
}

/*---------------------------------
--- Opérations sur les cerveaux ---
---------------------------------*/

int get_brain_part_ind(long ind, Brain * brain)
{
    /*
    Renvoie l'indice de la partie du cerveau dans laquelle le neurone "ind" se situe
    Brain est supposé être un cerveau bien formé et ind est supposé être entre 0 et brain.dimension
    */
    long * parts_cerv = (*brain).parties_cerveau;
    if (ind >= parts_cerv[(*brain).nb_part - 1])
    {
        return (*brain).nb_part - 1;
    }
    int i=0;
    while (ind >= parts_cerv[i+1])
    {
        i++;
    }
    return i;
}

int choose_neuron_type(Brain * brain, int part)
{
    /*Choisi de quel type sera le neurone en fonction du cerveau et de la partie auxquels il appartient*/
    if (part >= (*brain).nb_part)
    {
        printf("Erreur dans choose_neuron_type : numéro de partie %i supérieur au nombre de parties dans le cerveau %i.\n",part,(*brain).nb_part);
        exit(1);
    }
    int i=0;
    double * repNCumulee = (*brain).brainPart[part].repartitionNeuronCumulee; //Repartition cumulée des neurones dans les parties
    double decision = random_between_0_and_1();
    while (repNCumulee[i] < decision)
    {
        i++;
    }
    return i;
}

int get_nb_neuron_brain_part(Brain * brain, int part)
{
    /*Renvoie le nombre de neurones dans la partie d'indice part*/
    long n = (*brain).dimension;
    long ind_depart = (*brain).parties_cerveau[part]; //indice de depart auquel commence la partie
    if (part+1 == (*brain).nb_part)
    {
        return n - ind_depart;
    }
    else
    {
        long ind_fin = (*brain).parties_cerveau[part+1];
        return ind_fin - ind_depart;
    }
}

double get_mean_connect_percentage_for_part(Brain * brain, int part, int type)
{
    /*Renvoie le pourcentage (entre 0 et 100) de chances de connection moyen pour un neurone de type donné dans une partie donnée, vers les autres parties*/
    long n,i;
    int nb_part;
    double * probCo = (*brain).brainPart[part].probaConnection; //Proba de connection vers chaque partie
    n = (*brain).dimension;
    nb_part = (*brain).nb_part;

    double sum_proba = 0;
    for (i=0;i<nb_part;i++)
    {
        sum_proba += (double) get_nb_neuron_brain_part(brain,i) * (*brain).brainPart[part].probaConnection[type*nb_part + i];
    }
    return sum_proba/n *100;
}

void generate_neuron_types(Brain * brain, int ind_start_neuron, int nb_neuron, int * types)
{
    /*
     Décide des types des neurones numéro "ind_start_neuron" à "ind_start_neuron + nb_neuron" dans le cerveau "Brain", et les écrit dans "types"
     Un malloc de taille nb_neuron * sizeof(int) doit avoir été fait au préalable pour le pointeur "types".
    */
    long i;
    int ind_part;
    for (i=0;i<nb_neuron;i++) //parcours des lignes
    {
        //récupération de l'indice de la partie source
        ind_part = get_brain_part_ind(ind_start_neuron+i, brain);
        //décision du type de neurone
        types[i] = choose_neuron_type(brain, ind_part);
    }
}

/*---------------------------------
--- Opérations sur les matrices ---
---------------------------------*/

int get_csr_matrix_value_int(long indl, long indc, IntCSRMatrix * M_CSR)
{
    /*
    Renvoie la valeur [indl,indc] de la matrice M_CSR stockée au format CSR.
    Le vecteur à l'adresse (*M_CSR).Value doit être un vecteur d'entiers.
    */
    int *Row,*Column,*Value;
    Row = (*M_CSR).Row; Column = (*M_CSR).Column; Value = (*M_CSR).Value;
    if (indl >= (*M_CSR).dim_l || indc >= (*M_CSR).dim_c)
    {
        perror("ATTENTION : des indices incohérents ont été fournis dans la fonction get_sparce_matrix_value()\n");
        return -1;
    }
    long i;
    long nb_values = Row[indl+1] - Row[indl]; //nombre de valeurs dans la ligne
    for (i=Row[indl];i<Row[indl]+nb_values;i++)
    {
        if (Column[i] == indc) {return Value[i];}
    }
    return 0; //<=> on a parcouru la ligne et on a pas trouvé de valeur dans la colonne
}

/*--------------------------------------------------------------------------------
--- Fonctions pour génération de matrices ou changement de formats de matrices ---
--------------------------------------------------------------------------------*/

void generate_csr_brain_matrix_for_pagerank(IntCSRMatrix *M_CSR, long ind_start_row, Brain * brain, int * neuron_types, long l, long c, DebugBrainMatrixInfo * debugInfo)
{
    /*
    Génère aléatoirement la matrice creuse (pointeur M_CSR, format CSR), pour PageRank, correspondant à un cerveau passé en paramètre.
    l et c sont les nombres de ligne et nombre de colonnes de la matrice, ils seront stockés dans dim_l et dim_c
    neuron_types est un pointeur vers un vecteur d'entiers de taille c correspondant aux types de chaque neurones.
    Attention : on suppose brain.dimension = c
    ind_start_row est, dans le cas où on génère la matrice par morceaux, l'indice de la ligne (dans la matrice complète) où le morceau commence.
    Ce dernier indice permet de remplir la diagonale de la matrice de 0 (pour PageRank : un site ne peut pas être relié à lui même)
    Le pourcentage de valeurs (1 ou 0) dans la matrice est choisi en fonction du cerveau "brain" passé en paramètre.

    debugInfo est un pointeur vers une structure de débuggage.
    Si ce paramètre est à NULL, aucune information de débuggage n'est écrite.
    Si on écrit des informations de débuggage, deux malloc de plus sont fait.

    Attention : La mémoire pour les vecteurs Row, Column et Value est allouée dans la fonction, mais n'est pas libérée dans la fonction.
    */
    long i,j,cpt_values,size=l*c;
    int ind_part_source,ind_part_dest,source_type; double proba_connection,proba_no_connection,random;
    (*M_CSR).dim_l = l; (*M_CSR).dim_c = c;
    if (debugInfo != NULL)
    {
        (*debugInfo).dim_l = l; (*debugInfo).dim_c = c;
        (*debugInfo).types = neuron_types;
        //Attention : ces malloc ne sont pas "free" dans la fonction !
        (*debugInfo).nb_connections = (long *)malloc((*debugInfo).dim_c * sizeof(long));
        for (i=0;i<(*debugInfo).dim_c;i++)
        {
            (*debugInfo).nb_connections[i] = 0;
        }
    }

    //allocations mémoires
    (*M_CSR).Row = (int *)malloc(((*M_CSR).dim_l+1) * sizeof(int));
    //La mémoire allouée pour Column est à la base de 1/10 de la taille de la matrice stockée "normalement". Au besoin, on réalloue de la mémoire dans le code.
    long basic_size = (long) size/10;
    long total_memory_allocated = basic_size; //nombre total de cases mémoires allouées pour 1 vecteur
    (*M_CSR).Column = (int *)malloc(total_memory_allocated * sizeof(int));

    (*M_CSR).Row[0] = 0;
    cpt_values=0;
    for (i=0;i<l;i++) //parcours des lignes
    {
        //récupération de l'indice de la partie (du cerveau) destination
        ind_part_dest = get_brain_part_ind(ind_start_row+i, brain);
        #pragma omp ordered
        for (j=0;j<c;j++) //parcours des colonnes
        {
            //récupération de l'indice de la partie source
            ind_part_source = get_brain_part_ind(j, brain);
            //récupération du type de neurone
            source_type = neuron_types[j];
            //récupération de la probabilité de connexion source -> destination avec le type de neurone donné
            proba_connection = (*brain).brainPart[ind_part_source].probaConnection[source_type*(*brain).nb_part + ind_part_dest];
            proba_no_connection = 1 - proba_connection;
            random = random_between_0_and_1();
            //décision aléatoire, en prenant en compte l'abscence de connexion sur la diagonale de façon brute
            if ( (ind_start_row+i)!=j && random > proba_no_connection) //si on est dans la proba de connexion et qu'on est pas dans la diagonale, alors on place un 1
            {
                if (cpt_values >= total_memory_allocated)
                {
                    total_memory_allocated *= 2;
                    (*M_CSR).Column = (int *) realloc((*M_CSR).Column, total_memory_allocated * sizeof(int));
                    assert((*M_CSR).Column != NULL);
                }
                (*M_CSR).Column[cpt_values] = j;
                if (debugInfo != NULL)
                {
                    (*debugInfo).nb_connections[j] = (*debugInfo).nb_connections[j] + 1;
                }
                cpt_values++;
            }
        }
        (*M_CSR).Row[i+1] = cpt_values;
    }
    //remplissage de la structure de débuggage
    if (debugInfo != NULL)
    {
        (*debugInfo).total_memory_allocated = total_memory_allocated;
        (*debugInfo).cpt_values = cpt_values;
    }
    //remplissage du vecteur Value (avec précisement le nombre de 1 nécéssaire)
    (*M_CSR).Value = (int *)malloc(cpt_values * sizeof(int));
    (*M_CSR).len_values = cpt_values;
    for (i=0; i<cpt_values;i++) {(*M_CSR).Value[i] = 1;}
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

    int debug=1; //passer à 1 pour afficher les print de débuggage
    int debug_cerveau=1; //passer à 1 pour avoir les print de débuggage liés aux pourcentages de connexion du cerveau
    long i,j,k; //pour les boucles
    long n;
    long long size;
    long total_memory_allocated_local,nb_zeros,nb_non_zeros,nb_non_zeros_local,*list_nb_non_zeros_local;
    long *nb_connections_local_tmp,*nb_connections_tmp;
    int *neuron_types, *local_types;

    double start_time, total_time;

    //allocation mémoire pour les nombres de 0 dans chaque sous matrice de chaque processus
    if (debug) {list_nb_non_zeros_local = (long *)malloc(p * sizeof(long));}

    //recuperation du cerveau
    Brain Cerveau;
    paramBrain(&Cerveau,&n);
    if (argc < 2)
    {
        printf("Le n sélectionné correspond a celui du fichier de config\n", argv[0]);
    }
    else if (argc == 2)
    {
        n = atoll(argv[1]);
    }

    size = n * n;
    long nb_ligne = n/p; //nombre de lignes par bloc


    if (my_rank == 0)
    {
      printf("\n#############\nRecap de votre cerveau :\n");

      printf("Taille : %li*%li\nNombre de parties : %i\nIndices auxquelles commencent les parties : [",Cerveau.dimension,Cerveau.dimension,Cerveau.nb_part);
      for (i=0; i<Cerveau.nb_part; i++)
      {
          printf("%li ",Cerveau.parties_cerveau[i]);
      }
      printf("]\n\n");
      for (i=0; i<Cerveau.nb_part; i++)
      {
          printf("\n");
          printf("Partie %li :\n\tNombre de types de neurones : %i\n\t",i,Cerveau.brainPart[i].nbTypeNeuron);
          printf("Probabilités cumulées d'appartenir à chaque type de neurone : [");
          for (j=0;j<Cerveau.brainPart[i].nbTypeNeuron;j++)
          {
              printf("%lf ",Cerveau.brainPart[i].repartitionNeuronCumulee[j]);
          }
          printf("]\n\tConnections :\n\t");
          for (j=0;j<Cerveau.brainPart[i].nbTypeNeuron;j++)
          {
              printf("Connections du type de neurone d'indice %li :\n\t",j);
              for (k=0;k<Cerveau.nb_part;k++)
              {
                  printf("%li -> %li : %lf\n\t",i,k,Cerveau.brainPart[i].probaConnection[j*Cerveau.nb_part + k]);
              }
          }
      }

      printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = my_gettimeofday();

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
        generate_csr_brain_matrix_for_pagerank(&A_CSR, my_rank*nb_ligne, &Cerveau, neuron_types, nb_ligne, n, &MatrixDebugInfo);

        /* MatrixDebugInfo.nb_connections contient actuellement (dans chaque processus) le nombre de connexions faites LOCALEMENT par tout les neurones par colonne. */
        nb_connections_local_tmp = MatrixDebugInfo.nb_connections;
        nb_connections_tmp = (long *)malloc(MatrixDebugInfo.dim_c * sizeof(long));
        MatrixDebugInfo.nb_connections = nb_connections_tmp;
        MPI_Allreduce(nb_connections_local_tmp, nb_connections_tmp, MatrixDebugInfo.dim_c, MPI_LONG, MPI_SUM, MPI_COMM_WORLD); //somme MPI_SUM de tout les nombres de connexions local dans nombres de connexions global
        /* MatrixDebugInfo.nb_connections contient maintenant (dans tout les processus) le nombre GLOBAL de connexions faites pour chaque neurone. */
    }
    else
    {
        generate_csr_brain_matrix_for_pagerank(&A_CSR, my_rank*nb_ligne, &Cerveau, neuron_types, nb_ligne, n, NULL);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    total_time = my_gettimeofday() - start_time;

    nb_non_zeros_local = A_CSR.len_values;
    MPI_Allreduce(&nb_non_zeros_local, &nb_non_zeros, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD); //somme MPI_SUM de tout les nb_non_zeros_local dans nb_non_zeros
    if (debug) {MPI_Allgather(&nb_non_zeros_local, 1, MPI_LONG, list_nb_non_zeros_local, 1,  MPI_LONG, MPI_COMM_WORLD);} //réunion dans chaque processus de tout les nombres de zéros de chaque bloc

    if (debug_cerveau)
    {
        total_memory_allocated_local = MatrixDebugInfo.total_memory_allocated;
        MPI_Allreduce(&total_memory_allocated_local, &(MatrixDebugInfo.total_memory_allocated), 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD); //somme MPI_SUM de tout les total_memory_allocated_local dans MatrixDebugInfo.total_memory_allocated.
        MatrixDebugInfo.cpt_values = nb_non_zeros;

        if (my_rank == 0)
        {
            printf("Mémoire totale allouée pour le vecteur Column : %li\nNombre de cases mémoires effectivement utilisées : %li\n",MatrixDebugInfo.total_memory_allocated,MatrixDebugInfo.cpt_values);
        }
    }

    if (debug && my_rank == 0)
    {
        printf("Liste des nombres de 1 locaux :\n");
        for(i=0;i<p;i++)
        {
            printf("%li ",list_nb_non_zeros_local[i]);
        }
        printf("\n");
    }

    if (debug && nb_non_zeros <= 256)
    {
        printf("\nVecteur A_CSR.Row dans my_rank=%i:\n",my_rank);
        for(i=0;i<A_CSR.dim_l+1;i++) {printf("%i ",A_CSR.Row[i]);}printf("\n");
        printf("Vecteur A_CSR.Column dans my_rank=%i:\n",my_rank);
        for(i=0;i<A_CSR.len_values;i++) {printf("%i ",A_CSR.Column[i]);}printf("\n");
        printf("Vecteur A_CSR.Value dans my_rank=%i:\n",my_rank);
        for(i=0;i<A_CSR.len_values;i++) {printf("%i ",A_CSR.Value[i]);}printf("\n");
    }

    if (debug_cerveau)
    {
        long nbco;
        int partie,type;
        double pourcentage_espere,sum_pourcentage_espere = 0,sum_pourcentage_espere_local = 0;
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
        if (my_rank == 0)
        {
            printf("\nPourcentage global de valeurs non nulles : %.2f%, pourcentage global espéré : %.2f%\n\n",((double) nb_non_zeros/(double) size) * 100,sum_pourcentage_espere/ (double) n);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0) {printf("Temps écoulé lors de la génération : %.1f s\n", total_time);}

    free(local_types); free(neuron_types);
    free(A_CSR.Row); free(A_CSR.Column); free(A_CSR.Value);
    destructeurBrain(&Cerveau);
    if (debug && my_rank == 0)
    {
        free(list_nb_non_zeros_local);
    }
    if (debug_cerveau)
    {
        free(MatrixDebugInfo.nb_connections);
        free(nb_connections_local_tmp);
    }
    MPI_Finalize();
    return 0;
}
