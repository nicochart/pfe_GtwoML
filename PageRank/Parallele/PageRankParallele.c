/*Travail sur PageRank non pondéré parallele*/
/*Nicolas HOCHART*/

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
     int * types; //vecteur de taille dimension_l indiquant le type choisi pour chaque neurones
     long * nb_connections; //vecteur de taille dimension_l indiquant le nombre de connections qu'a effectué chaque neurone
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

/*---------------------------------
--- Opérations sur les matrices ---
---------------------------------*/

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

double get_csr_matrix_value_double(long indl, long indc, DoubleCSRMatrix * M_CSR)
{
    /*
    Renvoie la valeur [indl,indc] de la matrice M_CSR stockée au format CSR.
    Le vecteur à l'adresse (*M_CSR).Value doit être un vecteur de doubles.
    */
    int *Row,*Column; double *Value;
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

/*--------------------------------------------------------------------------------
--- Fonctions pour génération de matrices ou changement de formats de matrices ---
--------------------------------------------------------------------------------*/

void init_row_dense_matrix(int *M, long i, long n, int zero_percentage)
{
    /*
    Rempli n éléments de la ligne i de la matrice M stockée comme un vecteur d'entiers.
    Il y a zero_percentage % de chances que le nombre soit 0.
    Statistiquement, zero_percentage % de la matrice sont des 0 et (100 - zero_percentage) % sont des 1
    */
    long j;

    for (j=0;j<n;j++)
    {
        if (random_between_0_and_1() < zero_percentage/100.0) {*(M + i*n+j) = 0;} //zero_percentage % de chances de mettre un 0
        else {*(M + i*n+j) = 1;}
    }
}

void generate_coo_matrix_for_pagerank(IntCOOMatrix *M_COO, long ind_start_row, int zero_percentage, long l, long c)
{
    /*
    Génère aléatoirement la matrice creuse (*M_COO) (format COO) pour PageRank.
    l et c sont les nombres de ligne et nombre de colonnes de la matrice, ils seront stockés dans dim_l et dim_c
    ind_start_row est, dans le cas où on génère la matrice par morceaux, l'indice de la ligne (dans la matrice complète) où le morceau commence.
    Ce dernier indice permet de remplir la diagonale de la matrice de 0 (pour PageRank : un site ne peut pas être relié à lui même)
    Statistiquement, il y a zero_percentage % de 0 dans la matrice l*c.
    Environs zero_percentage % de la matrice dense correspondante sont des 0 et (100 - zero_percentage) % sont des 1.
    (Ce n'est pas exact, car un test est effectué avec ind_start_row pour remplir la diagonale de 0. Ce problème sera corrigé plus tard)
    */
    long i,j,cpt_values,size=l*c;
    long mean_nb_non_zeros = (int) size * (100 - zero_percentage) / 100; //nombre moyen de 1 dans la matrice
    (*M_COO).dim_l = l; (*M_COO).dim_c = c;
    //Attention : La mémoire pour les vecteurs Row, Column et Value est allouée dans la fonction, mais n'est pas libérée dans la fonction.
    //La mémoire allouée est (statistiquement) plus grande que la mémoire qui sera utilisée en pratique. On ne peut pas savoir à l'avance exactement combien de valeurs aura la matrice.
    (*M_COO).Row = (int *)malloc(mean_nb_non_zeros * sizeof(int));
    (*M_COO).Column = (int *)malloc(mean_nb_non_zeros * sizeof(int));
    (*M_COO).Value = (int *)malloc(mean_nb_non_zeros * sizeof(int));

    cpt_values=0;
    for (i=0;i<l;i++) //parcours des lignes
    {
        for (j=0;j<c;j++) //parcours des colonnes
        {
            if ( (ind_start_row+i)!=j && random_between_0_and_1() > zero_percentage/100.0) //si on est dans le pourcentage de non zero et qu'on est pas dans la diagonale, alors on place un 1
            {
                if (cpt_values < mean_nb_non_zeros)
                {
                    (*M_COO).Row[cpt_values] = i;
                    (*M_COO).Column[cpt_values] = j;
                    (*M_COO).Value[cpt_values] = 1;
                    cpt_values++;
                }
            }
        }
    }
    (*M_COO).len_values = cpt_values;
}

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

void generate_coo_brain_matrix_for_pagerank(IntCOOMatrix *M_COO, long ind_start_row, Brain * brain, long l, long c, DebugBrainMatrixInfo * debugInfo)
{
    /*
    Génère aléatoirement la matrice creuse (*M_COO) (format COO) pour PageRank.
    l et c sont les nombres de ligne et nombre de colonnes de la matrice, ils seront stockés dans dim_l et dim_c
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
    int ind_part_source,ind_part_dest,i_type; double proba_connection,proba_no_connection,random;
    (*M_COO).dim_l = l; (*M_COO).dim_c = c;
    if (debugInfo != NULL)
    {
        (*debugInfo).dim_l = l; (*debugInfo).dim_c = c;
        //Attention : ces malloc ne sont pas "free" dans la fonction !
        (*debugInfo).types = (int *)malloc((*debugInfo).dim_l * sizeof(int));
        (*debugInfo).nb_connections = (long *)malloc((*debugInfo).dim_l * sizeof(long));
    }

    //La mémoire allouée est à la base de 1/10 de la taille de la matrice stockée "normalement". Au besoin, on réalloue de la mémoire dans le code.
    long basic_size = (long) size/10;
    long total_memory_allocated = basic_size; //nombre total de cases mémoires allouées pour 1 vecteur
    (*M_COO).Row = (int *)malloc(total_memory_allocated * sizeof(int));
    (*M_COO).Column = (int *)malloc(total_memory_allocated * sizeof(int));

    cpt_values=0;
    for (i=0;i<l;i++) //parcours des lignes
    {
        //récupération de l'indice de la partie source
        ind_part_source = get_brain_part_ind(ind_start_row+i, brain);
        //décision du type de neurone
        i_type = choose_neuron_type(brain, ind_part_source);
        if (debugInfo != NULL)
        {
            (*debugInfo).types[i] = i_type;
            (*debugInfo).nb_connections[i] = 0;
        }
        for (j=0;j<c;j++) //parcours des colonnes
        {
            //récupération de l'indice de la partie destination
            ind_part_dest = get_brain_part_ind(j, brain);
            //récupération de la probabilité de connexion source -> destination avec le type de neurone donné
            proba_connection = (*brain).brainPart[ind_part_source].probaConnection[i_type*(*brain).nb_part + ind_part_dest];
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
                    (*debugInfo).nb_connections[i] = (*debugInfo).nb_connections[i] + 1;
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

void dense_to_coo_matrix(int *M, IntCOOMatrix * M_COO)
{
    /*
    Traduit la matrice stockée normalement dans M en matrice stockée en format COO dans M_COO.
    Les vecteurs Row, Column et Value sont de taille "nombre d'éléments non nulles dans la matrice".
    Les dimensions de la matrice (dim_l,dim_c) = (nombre de lignes, nombre de colonnes) doivent déjà être définis dans M_COO. Les allocations mémoires doivent aussi être fait au préalable.
    */
    long i, j, nb = 0;
    for (i=0;i<(*M_COO).dim_l;i++)
    {
        for (j=0;j<(*M_COO).dim_c;j++)
        {
            if (*(M + i*(*M_COO).dim_c+j) != 0)
            {
                (*M_COO).Row[nb] = i; (*M_COO).Column[nb] = j;
                (*M_COO).Value[nb] = *(M+i*(*M_COO).dim_c+j);
                nb++;
            }
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
    long nb_zeros,nb_non_zeros,nb_non_zeros_local,*list_nb_non_zeros_local;

    //allocation mémoire pour les nombres de 0 dans chaque sous matrice de chaque processus
    if (debug) {list_nb_non_zeros_local = (long *)malloc(p * sizeof(long));}

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
    double probCo1[16] = {/*type 1*/0.1, 0.4, 0.4, 0.5, 0.4, 0.4, 0.5, 0.4, /*type 2*/0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.1, 0.05};
    brainPart[0].nbTypeNeuron = nbTypeNeuronIci;
    brainPart[0].repartitionNeuronCumulee = repNCumulee1;
    brainPart[0].probaConnection = probCo1;
    //partie 2
    nbTypeNeuronIci = 1;
    double repNCumulee2[1] = {1};
    double probCo2[8] = {/*type 1*/0.4, 0.1, 0.4, 0.5, 0.4, 0.4, 0.5, 0.4};
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
    struct DebugBrainMatrixInfo MatrixDebugInfo;
    if (debug_cerveau)
    {
        generate_coo_brain_matrix_for_pagerank(&A_COO, my_rank*nb_ligne, &Cerveau, nb_ligne, n, &MatrixDebugInfo);
    }
    else
    {
        generate_coo_brain_matrix_for_pagerank(&A_COO, my_rank*nb_ligne, &Cerveau, nb_ligne, n, NULL);
    }

    nb_non_zeros_local = A_COO.len_values;
    MPI_Allreduce(&nb_non_zeros_local, &nb_non_zeros, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD); //somme MPI_SUM de tout les nb_non_zeros_local dans nb_non_zeros
    if (debug) {MPI_Allgather(&nb_non_zeros_local, 1, MPI_LONG, list_nb_non_zeros_local, 1,  MPI_LONG, MPI_COMM_WORLD);} //réunion dans chaque processus de tout les nombres de zéros de chaque bloc

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

    if (debug)
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
                    if (n<=64) //si la dimension de la matrice est inférieur ou égal à 64, on peut l'afficher
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
                    type = MatrixDebugInfo.types[i];
                    nbco = MatrixDebugInfo.nb_connections[i];
                    pourcentage_espere = get_mean_connect_percentage_for_part(&Cerveau, partie, type);
                    sum_pourcentage_espere_local += pourcentage_espere;
                    printf(" type: %i, partie: %i, nbconnections: %li, pourcentage: %.2f, pourcentage espéré : %.2f",type,partie,nbco,(double) nbco / (double) n * 100,pourcentage_espere);
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

    if (my_rank == 0)
    {
        printf("\nRésultat ");
        for(i=0;i<n;i++) {printf("%.4f ",new_q[i]);}
        printf("obtenu en %i itérations\n",cpt_iterations);
    }

    free(new_q); free(old_q);
    free(A_COO.Row); free(A_COO.Column); free(A_COO.Value);
    free(A_CSR.Row); //Column et Value sont communs avec la matrice COO
    free(P_CSR.Value); //Row et Column communs avec la matrice CSR

    if (debug && my_rank == 0)
    {
        free(list_nb_non_zeros_local);
    }
    if (debug_cerveau)
    {
        free(MatrixDebugInfo.types); free(MatrixDebugInfo.nb_connections);
    }
    MPI_Finalize();
    return 0;
}
