/* Générateur de matrice pour le PageRank correspondant au cerveau */
/*Nicolas HOCHART*/

/*
 Ce générateur génère directement la matrice AT (matrice d'adjacence A transposée)
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>

#define NULL ((void *)0)

/*-------------------------------------------------------------------
--- Structures pour le stockage des matrices au format COO et CSR ---
-------------------------------------------------------------------*/

struct IntCOOMatrix
{
     int * Row; //vecteur de taille len_values = "nombre d'éléments non nuls dans la matrice"
     int * Column; //vecteur de taille len_values = "nombre d'éléments non nuls dans la matrice"
     int * Value; //vecteur de taille len_values = "nombre d'éléments non nuls dans la matrice"
     long dim_l; //nombre de lignes
     long dim_c; //nombre de colonnes
     long len_values; //taille des vecteurs Row, Column et Value
};
typedef struct IntCOOMatrix IntCOOMatrix;

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

/*----------------------------------------------------------------------
--- Structure contenant les informations d'un cerveau et ses parties ---
----------------------------------------------------------------------*/

struct BrainPart
{
     int nbTypeNeuron;
     double * repartitionNeuronCumulee; //taille nbTypeNeuron
     double * probaConnection; //taille (ligne) nbTypeNeuron * (colonne) nb_part
};
typedef struct BrainPart BrainPart;

struct Brain
{
     long dimension; //nombre de neurones total
     int nb_part; //nombre de parties
     long * parties_cerveau; //taille nb_part - indices de 0 à n (dimension de la matrice) auxquels commencent les parties du cerveau
     BrainPart * brainPart; //taille nb_part - adresse d'un vecteur de pointeurs vers des BrainPart.
};
typedef struct Brain Brain;

//structure permettant de débugger le générateur de matrice correspondant à un cerveau en COO "generate_coo_brain_matrix_for_pagerank"
struct DebugBrainMatrixInfo
{
     long dim_c; //nombre de neurones "destination" (sur les colonnes de la matrice)
     long dim_l; //nombre de neurones "source" (sur les lignes de la matrice)
     int * types; //vecteur de taille dim_c indiquant le type choisi pour chaque neurones du cerveau
     long * nb_connections; //vecteur de taille dim_c indiquant le nombre de connections qu'a effectué chaque neurone.
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

/*
-- Nouvelle implémentation de la génération de matrice COO : --
Variables :
n = dimension de la matrice à générer
nb_part = nombre de parties du cerveau qu'on souhaite représenter
p = nombre de coeurs alloués (= nombre de blocs de ligne)

Entrée :
1) vecteur "parties_cerveau" d'entiers de taille nb_part (contenant nb_part valeurs, croissantes, de 0 à n)
---> indique (à l'indice i) la ligne de la matrice à laquelle commence la partie d'indice i du cerveau
Ce vecteur servira pour connaître la partie du cerveau qu'on manipule au niveau des lignes, et aussi au niveau des colonnes.

2) nb_part tableaux "probaConnection" de taille nbTypeNeuron * nb_part associés avec un vecteur "repartitionNeuronCumulee" de taille nbTypeNeuron
---> repartitionNeuronCumulee contient à l'indice i la probabilité (cumulée avec les précédentes) que le neurone (dans la partie du cerveau dans laquelle il apparaît) soit effectivement un neurone de ce type.
---> probaConnection indique à la ligne d'indice i les probabilités, d'indice de colonne j (0 -> nb_part), pour que le type de neurone i se connecte à la partie d'indice j du cerveau
Ces deux données permettront, lors de la génération d'une partie du cerveau, de choisir (pour une ligne) le type de neurone qu'elle représentera,
puis (pour chaque colonne, avec des probas != pour chaque partie du cerveau) choisir si on met une connection ou non.

Algorithme de génération :
Parcours des lignes (indice indl):
    On regarde dans quelle partie du cerveau on est avec "parties_cerveau" : on est à la partie d'indice indp
    On décide de quel type de neuronne (indice indn entre 0 et nbTypeNeuron) sera la ligne avec le "repartitionNeuronCumulee" associé à la partie du cerveau d'indice indp
    On parcours les colonnes (indice indc):
        On regarde à quelle partie du cerveau on essaye de se connecter (indice indpco)
        On prend une décision en fonction de la valeur du tableau "probaConnection" (ligne indn, colonne indpco)
*/

void generate_coo_brain_matrix_for_pagerank(IntCOOMatrix *M_COO, long ind_start_row, Brain * brain, int * neuron_types, long l, long c, DebugBrainMatrixInfo * debugInfo)
{
    /*
    Génère aléatoirement la matrice creuse (*M_COO) (format COO) pour PageRank.
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
    (*M_COO).dim_l = l; (*M_COO).dim_c = c;
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

    //La mémoire allouée est à la base de 1/10 de la taille de la matrice stockée "normalement". Au besoin, on réalloue de la mémoire dans le code.
    long basic_size = (long) size/10;
    long total_memory_allocated = basic_size; //nombre total de cases mémoires allouées pour 1 vecteur
    (*M_COO).Row = (int *)malloc(total_memory_allocated * sizeof(int));
    (*M_COO).Column = (int *)malloc(total_memory_allocated * sizeof(int));

    cpt_values=0;
    for (i=0;i<l;i++) //parcours des lignes
    {
        //récupération de l'indice de la partie (du cerveau) destination
        ind_part_dest = get_brain_part_ind(ind_start_row+i, brain);
        for (j=0;j<c;j++) //parcours des colonnes
        {
            //récupération de l'indice de la partie source
            ind_part_source = get_brain_part_ind(j, brain);
            //récupération du type de neurone
            source_type = neuron_types[j];
            //récupération de la probabilité de connexion source -> destination avec le type de neurone donné
            proba_connection = (*brain).brainPart[ind_part_source].probaConnection[source_type*(*brain).nb_part + ind_part_dest];
            //DEL if (ind_part_dest == 0) {printf("[Neurone %li,%li] Tentative de création d'une connexion de la partie %i à la partie %i (type %i) avec une proba %f\n",i,j,ind_part_source,ind_part_dest,source_type,proba_connection);}
            proba_no_connection = 1 - proba_connection;
            random = random_between_0_and_1();
            //décision aléatoire, en prenant en compte l'abscence de connexion sur la diagonale de façon brute
            if ( (ind_start_row+i)!=j && random > proba_no_connection) //si on est dans la proba de connexion et qu'on est pas dans la diagonale, alors on place un 1
            {
                if (cpt_values >= total_memory_allocated)
                {
                    total_memory_allocated *= 2;
                    (*M_COO).Row = (int *) realloc((*M_COO).Row, total_memory_allocated * sizeof(int));
                    (*M_COO).Column = (int *) realloc((*M_COO).Column, total_memory_allocated * sizeof(int));
                    assert((*M_COO).Row != NULL);
                    assert((*M_COO).Column != NULL);
                }
                (*M_COO).Row[cpt_values] = i;
                (*M_COO).Column[cpt_values] = j;
                if (debugInfo != NULL)
                {
                    (*debugInfo).nb_connections[j] = (*debugInfo).nb_connections[j] + 1;
                }
                cpt_values++;
            }
        }
    }
    //remplissage du vecteur Value (avec précisement le nombre de 1 nécéssaire)
    (*M_COO).Value = (int *)malloc(cpt_values * sizeof(int));
    (*M_COO).len_values = cpt_values;
    for (i=0; i<cpt_values;i++) {(*M_COO).Value[i] = 1;}
}

void coo_to_csr_matrix(IntCOOMatrix * M_COO, IntCSRMatrix * M_CSR)
{
    /*
    Traduit le vecteur Row de la matrice M_COO stockée au format COO en vecteur Row format CSR dans la matrice M_CSR
    A la fin : COO_Column=CSR_Column (adresses), COO_Value=CSR_Value (adresses), et CSR_Row est la traduction en CSR de COO_Row (adresses et valeurs différentes)
    L'allocation mémoire pour CSR_Row (taille dim_l + 1) doit être faite au préalable
    Attention : dim_c, dim_l et len_values ne sont pas modifiés dans le processus
    */
    long i;
    for (i=0;i<(*M_COO).len_values;i++) //on parcours les vecteurs Column et Value de taille "nombre d'éléments non nuls de la matrice" = len_values
    {
        (*M_CSR).Column[i] = (*M_COO).Column[i];
        (*M_CSR).Value[i] = (*M_COO).Value[i];
    }

    int * COO_Row = (*M_COO).Row;
    int * CSR_Row = (*M_CSR).Row;
    long current_indl = 0;
    *(CSR_Row + current_indl) = 0;
    while(COO_Row[0] != current_indl) //cas particulier : première ligne de la matrice remplie de 0 (<=> indice de la première ligne, 0, différent du premier indice de ligne du vecteur Row)
    {
        *(CSR_Row + current_indl) = 0;
        current_indl++;
    }
    for (i=0;i<(*M_COO).len_values;i++)
    {
        if (COO_Row[i] != current_indl)
        {
            *(CSR_Row + current_indl + 1) = i;
            while (COO_Row[i] != current_indl + 1) //cas particulier : ligne de la matrice vide (<=> indice de Row qui passe d'un nombre i à un nombre j supérieur à i+1)
            {
                current_indl++;
                *(CSR_Row + current_indl + 1) = i;
            }
            current_indl = COO_Row[i];
        }
    }
    *(CSR_Row + current_indl + 1) = (*M_COO).len_values;
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
    int debug_cerveau=1; //passer à 1 pour avoir les print de débuggage liés aux pourcentages de connexion du cerveau
    long i,j,k; //pour les boucles
    long n;
    long long size;
    int nb_zeros,nb_non_zeros,nb_non_zeros_local,*list_nb_non_zeros_local;
    long *nb_connections_local_tmp,*nb_connections_tmp;
    int *neuron_types, *local_types;

    //allocation mémoire pour les nombres de 0 dans chaque sous matrice de chaque processus
    if (debug) {list_nb_non_zeros_local = (int *)malloc(p * sizeof(int));}

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

    //Cerveau écrit en brute (pour essayer)
    int nbTypeNeuronIci,nb_part=8;
    BrainPart brainPart[nb_part];
    long part_cerv[nb_part];
    long nb_neurone_par_partie = n / nb_part;
    if (nb_part * nb_neurone_par_partie != n) {printf("Veuillez entrer un n multiple de 8 svp\n"); exit(1);}
    for (i=0; i<nb_part; i++)
    {
        part_cerv[i] = i*nb_neurone_par_partie;
    }
    //partie 1
    nbTypeNeuronIci = 2;
    double repNCumulee1[2] = {0.5, 1};
    double probCo1[16] = {/*type 1*/0.1, 0.4, 0.3, 0.5, 0.4, 0.6, 0.5, 0.4, /*type 2*/0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.1, 0.05};
    brainPart[0].nbTypeNeuron = nbTypeNeuronIci;
    brainPart[0].repartitionNeuronCumulee = repNCumulee1;
    brainPart[0].probaConnection = probCo1;
    //partie 2
    nbTypeNeuronIci = 1;
    double repNCumulee2[1] = {1};
    double probCo2[8] = {/*type 1*/0.4, 0.1, 0.4, 0.5, 0.4, 0.3, 0.5, 0.4};
    brainPart[1].nbTypeNeuron = nbTypeNeuronIci;
    brainPart[1].repartitionNeuronCumulee = repNCumulee2;
    brainPart[1].probaConnection = probCo2;
    //partie 3
    nbTypeNeuronIci = 2;
    double repNCumulee3[2] = {0.7, 1};
    double probCo3[16] = {/*type 1*/0.1, 0.5, 0.4, 0.5, 0.4, 0.4, 0.5, 0.4, /*type 2*/0.6, 0.05, 0.1, 0.1, 0.1, 0.05, 0.1, 0.05};
    brainPart[2].nbTypeNeuron = nbTypeNeuronIci;
    brainPart[2].repartitionNeuronCumulee = repNCumulee3;
    brainPart[2].probaConnection = probCo3;
    //partie 4
    nbTypeNeuronIci = 3;
    double repNCumulee4[3] = {0.5, 0.9, 1};
    double probCo4[24] = {/*type 1*/0.4, 0.5, 0.5, 0.1, 0.4, 0.4, 0.5, 0.4, /*type 2*/0.6, 0.5, 0.2, 0.5, 0.6, 0.5, 0.45, 0.0, /*type 3*/0.6, 0.05, 0.1, 0.1, 0.1, 0.05, 0.1, 0.05};
    brainPart[3].nbTypeNeuron = nbTypeNeuronIci;
    brainPart[3].repartitionNeuronCumulee = repNCumulee4;
    brainPart[3].probaConnection = probCo4;
    //partie 5
    nbTypeNeuronIci = 1;
    double repNCumulee5[1] = {1};
    double probCo5[8] = {/*type 1*/0.4, 0.4, 0.4, 0.5, 0.1, 0.4, 0.5, 0.4};
    brainPart[4].nbTypeNeuron = nbTypeNeuronIci;
    brainPart[4].repartitionNeuronCumulee = repNCumulee5;
    brainPart[4].probaConnection = probCo5;
    //partie 6
    nbTypeNeuronIci = 1;
    double repNCumulee6[1] = {1};
    double probCo6[8] = {/*type 1*/0.4, 0.4, 0.4, 0.55, 0.4, 0.1, 0.6, 0.4};
    brainPart[5].nbTypeNeuron = nbTypeNeuronIci;
    brainPart[5].repartitionNeuronCumulee = repNCumulee6;
    brainPart[5].probaConnection = probCo6;
    //partie 7
    nbTypeNeuronIci = 1;
    double repNCumulee7[1] = {1};
    double probCo7[8] = {/*type 1*/0.4, 0.2, 0.4, 0.6, 0.4, 0.4, 0.05, 0.4};
    brainPart[6].nbTypeNeuron = nbTypeNeuronIci;
    brainPart[6].repartitionNeuronCumulee = repNCumulee7;
    brainPart[6].probaConnection = probCo7;
    //partie 8
    nbTypeNeuronIci = 2;
    double repNCumulee8[2] = {0.3, 1};
    double probCo8[16] = {/*type 1*/0.05, 0.05, 0.1, 0.05, 0.2, 0.1, 0.1, 0.5, /*type 2*/0.4, 0.5, 0.4, 0.6, 0.4, 0.4, 0.05, 0.1};
    brainPart[7].nbTypeNeuron = nbTypeNeuronIci;
    brainPart[7].repartitionNeuronCumulee = repNCumulee8;
    brainPart[7].probaConnection = probCo8;

    Brain Cerveau;
    Cerveau.dimension = n;
    Cerveau.nb_part = nb_part;
    Cerveau.parties_cerveau = part_cerv;
    Cerveau.brainPart = brainPart;

    if (my_rank == 0)
    {
      printf("\n#############\nRecap de votre cerveau :\n");

      printf("Taille : %i*%i\nNombre de parties : %i\nIndices auxquelles commencent les parties : [",Cerveau.dimension,Cerveau.dimension,Cerveau.nb_part);
      for (i=0; i<Cerveau.nb_part; i++)
      {
          printf("%i ",Cerveau.parties_cerveau[i]);
      }
      printf("]\n\n");
      for (i=0; i<Cerveau.nb_part; i++)
      {
          printf("\n");
          printf("Partie %i :\n\tNombre de types de neurones : %i\n\t",i,Cerveau.brainPart[i].nbTypeNeuron);
          printf("Probabilités cumulées d'appartenir à chaque type de neurone : [");
          for (j=0;j<Cerveau.brainPart[i].nbTypeNeuron;j++)
          {
              printf("%lf ",Cerveau.brainPart[i].repartitionNeuronCumulee[j]);
          }
          printf("]\n\tConnections :\n\t");
          for (j=0;j<Cerveau.brainPart[i].nbTypeNeuron;j++)
          {
              printf("Connections du type de neurone d'indice %i :\n\t",j);
              for (k=0;k<Cerveau.nb_part;k++)
              {
                  printf("%i -> %i : %lf\n\t",i,k,Cerveau.brainPart[i].probaConnection[j*Cerveau.nb_part + k]);
              }
          }
      }

      printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //génération des sous-matrices au format COO
    //matrice format COO :
    //3 ALLOCATIONS : allocation de mémoire pour COO_Row, COO_Column et COO_Value dans la fonction generate_coo_matrix_for_pagerank()
    struct IntCOOMatrix A_COO;
    //generate_coo_matrix_for_pagerank(&A_COO, my_rank*nb_ligne, zeros_percentages[my_rank], nb_ligne, n);

    //matrice COO générée à partir du cerveau
    local_types = (int *)malloc(nb_ligne * sizeof(int));
    neuron_types = (int *)malloc(n * sizeof(int));
    generate_neuron_types(&Cerveau, my_rank*nb_ligne, nb_ligne, local_types);
    MPI_Allgather(local_types, nb_ligne, MPI_INT, neuron_types, nb_ligne,  MPI_INT, MPI_COMM_WORLD); //réunion dans chaque processus des types de tout les neurones (pour génération)

    struct DebugBrainMatrixInfo MatrixDebugInfo;
    if (debug_cerveau)
    {
        generate_coo_brain_matrix_for_pagerank(&A_COO, my_rank*nb_ligne, &Cerveau, neuron_types, nb_ligne, n, &MatrixDebugInfo);

        /* MatrixDebugInfo.nb_connections contient actuellement (dans chaque processus) le nombre de connexions faites LOCALEMENT par tout les neurones par colonne. */
        nb_connections_local_tmp = MatrixDebugInfo.nb_connections;
        nb_connections_tmp = (long *)malloc(MatrixDebugInfo.dim_c * sizeof(long));
        MatrixDebugInfo.nb_connections = nb_connections_tmp;
        MPI_Allreduce(nb_connections_local_tmp, nb_connections_tmp, MatrixDebugInfo.dim_c, MPI_LONG, MPI_SUM, MPI_COMM_WORLD); //somme MPI_SUM de tout les nombres de connexions local dans nombres de connexions global
        /* MatrixDebugInfo.nb_connections contient maintenant (dans tout les processus) le nombre GLOBAL de connexions faites pour chaque neurone. */
    }
    else
    {
        generate_coo_brain_matrix_for_pagerank(&A_COO, my_rank*nb_ligne, &Cerveau, neuron_types, nb_ligne, n, NULL);
    }

    nb_non_zeros_local = A_COO.len_values;
    MPI_Allreduce(&nb_non_zeros_local, &nb_non_zeros, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); //somme MPI_SUM de tout les nb_non_zeros_local dans nb_non_zeros
    if (debug) {MPI_Allgather(&nb_non_zeros_local, 1, MPI_INT, list_nb_non_zeros_local, 1,  MPI_INT, MPI_COMM_WORLD);} //réunion dans chaque processus de tout les nombres de zéros de chaque bloc

    if ((debug || debug_cerveau) && my_rank == 0)
    {
        printf("nb_non_zeros total = %i\n",nb_non_zeros);
        printf("Pourcentage de valeurs non nulles : 100 * %i / %i = %.2f%\n", nb_non_zeros, size, (double) 100 * (double) nb_non_zeros / (double) size);
    }

    if (debug && my_rank == 0)
    {
        printf("Liste des nombres de 1 locaux :\n");
        for(i=0;i<p;i++)
        {
            printf("%i ",list_nb_non_zeros_local[i]);
        }
        printf("\n");
    }

    if (debug)
    {
        printf("\nVecteur A_COO.Row dans my_rank=%i:\n",my_rank);
        for(i=0;i<A_COO.len_values;i++) {printf("%i ",A_COO.Row[i]);}printf("\n");
        printf("Vecteur A_COO.Column dans my_rank=%i:\n",my_rank);
        for(i=0;i<A_COO.len_values;i++) {printf("%i ",A_COO.Column[i]);}printf("\n");
        printf("Vecteur A_COO.Value dans my_rank=%i:\n",my_rank);
        for(i=0;i<A_COO.len_values;i++) {printf("%i ",A_COO.Value[i]);}printf("\n");
    }

    //convertion de la matrice COO au format CSR :
    //1 ALLOCATION : allocation de mémoire pour CSR_Row qui sera différent de COO_Row. Les vecteurs Column et Value sont communs
    struct IntCSRMatrix A_CSR;
    A_CSR.dim_l = A_COO.dim_l;
    A_CSR.dim_c = A_COO.dim_c;
    A_CSR.len_values = nb_non_zeros_local;
    A_CSR.Row = (int *)malloc((n+1) * sizeof(int));
    A_CSR.Column = A_COO.Column; A_CSR.Value = A_COO.Value; //Vecteurs Column et Value communs
    coo_to_csr_matrix(&A_COO, &A_CSR);

    if (debug)
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
        if (my_rank == 0) {printf("Matrice A :\n");}
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
                    else
                    {
                        nbco = MatrixDebugInfo.nb_connections[i];
                        printf("%03li \"0\" et %03li \"1\" -",n-nbco,nbco);
                    }

                    partie = get_brain_part_ind(my_rank*nb_ligne+i, &Cerveau);
                    type = MatrixDebugInfo.types[my_rank*nb_ligne+i];
                    nbco = MatrixDebugInfo.nb_connections[my_rank*nb_ligne+i];
                    pourcentage_espere = get_mean_connect_percentage_for_part(&Cerveau, partie, type);
                    sum_pourcentage_espere_local += pourcentage_espere;
                    printf(" type: %i, partie: %i, nbconnections: %li, pourcentage obtenu: %.2f, pourcentage espéré : %.2f",type,partie,nbco,(double) nbco / (double) n * 100,pourcentage_espere);
                    printf("\n");
                }
            }
        }
        MPI_Allreduce(&sum_pourcentage_espere_local, &sum_pourcentage_espere, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (my_rank == 0)
        {
            printf("\nPourcentage global : %.2f, pourcentage global espéré : %.2f\n\n",((double) nb_non_zeros/(double) size) * 100,sum_pourcentage_espere/ (double) n);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    free(local_types); free(neuron_types);
    free(A_COO.Row); free(A_COO.Column); free(A_COO.Value);
    free(A_CSR.Row); //Column et Value sont communs avec la matrice COO

    if (debug && my_rank == 0)
    {
        free(list_nb_non_zeros_local);
    }
    if (debug_cerveau)
    {
        free(MatrixDebugInfo.nb_connections);
    }
    MPI_Finalize();
    return 0;
}
