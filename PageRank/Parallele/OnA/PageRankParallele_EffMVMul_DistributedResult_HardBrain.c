/*PageRank non pondéré (plus optimité) parallele utilisant le générateur V5 (Matrice d'adjacence générée parallèle avec blocks sur les ligne et les colonnes)*/
/*Le PageRank est appliqué directement avec la matrice générée. (Pas besoin de la normaliser)*/
/*Le vecteur résultat du PageRank est réparti sur les processus*/
/*Nicolas HOCHART*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
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
     int indc_in_result_vector_calculation_group; //Indice de colonne du block dans le groupe de calcul du vecteur résultat (groupes de blocks colonnes)
     int inter_result_vector_need_group_communicaton_group; //Indice du Groupe de communication inter-groupe de besoin (utile pour récupérer le résultat final)
     long startColumn_in_result_vector_group; //Indice de départ en ligne dans le groupe de calcul du vecteur résultat (inclu)
     int my_result_vector_group_rank; //my_rank dans le groupe de calcul du vecteur résultat
};
typedef struct MatrixBlock MatrixBlock;

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
     long * nb_connections; //vecteur de taille dim_c en sortie du générateur, et de taille n (dimension totale de la matrice) après communications, indiquant le nombre de connections qu'a effectué chaque neurone.
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

void normalize_matrix_on_rows(DoubleCSRMatrix * M_CSR, MatrixBlock BlockInfo, long * row_sum_vector)
{
    /*
    Normalise la matrice parallèle CSR M_CSR, stockée en parallèle, sur les lignes.
    Les lignes dans la matrice (complète) doivent être de longueur matrix_dim_c
    Le vecteur row_sum_vector doit contenir le nombre d'éléments sur chaque ligne (vecteur de longueur matrix_dim_l).
    */
    long i,j;
    for(i=0; i<(*M_CSR).dim_l; i++) //parcours du vecteur row
    {
        for (j=(*M_CSR).Row[i]; j<(*M_CSR).Row[i+1]; j++) //parcours du vecteur column
        {
            (*M_CSR).Value[j] = 1.0/row_sum_vector[BlockInfo.startRow+i];
        }
    }
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

void generate_csr_brain_adjacency_matrix_for_pagerank(IntCSRMatrix *M_CSR, MatrixBlock BlockInfo, Brain * brain, int * neuron_types, DebugBrainMatrixInfo * debugInfo)
{
    /*
    Génère aléatoirement la matrice creuse (pointeur M_CSR, format CSR), pour PageRank, correspondant à un cerveau passé en paramètre.
    Les nombres de ligne et nombre de colonnes de la matrice sont passés en paramètre dans BlockInfo.dim_l et BlockInfo.dim_c, ils seront stockés dans dim_l et dim_c
    neuron_types est un pointeur vers un vecteur d'entiers de taille c correspondant aux types de chaque neurones.
    Attention : on suppose brain.dimension > BlockInfo.dim_c
    (Le pourcentage de 0 global n'est pas tout à fait respécté, car un test est effectué avec BlockInfo.startRow et BlockInfo.startColumn pour remplir la diagonale de 0. Ce problème sera corrigé plus tard)

    debugInfo est un pointeur vers une structure de débuggage.
    Si ce paramètre est à NULL, aucune information de débuggage n'est écrite.
    Si on écrit des informations de débuggage, deux malloc de plus sont fait.

    Attention : La mémoire pour les vecteurs Row, Column et Value est allouée dans la fonction, mais n'est pas libérée dans la fonction.
    */
    long i,j,cpt_values,size = BlockInfo.dim_l * BlockInfo.dim_c;
    int ind_part_source,ind_part_dest,source_type; double proba_connection,proba_no_connection,random;
    (*M_CSR).dim_l = BlockInfo.dim_l; (*M_CSR).dim_c = BlockInfo.dim_c;
    if (debugInfo != NULL)
    {
        (*debugInfo).dim_l = BlockInfo.dim_l; (*debugInfo).dim_c = BlockInfo.dim_c;
        (*debugInfo).types = neuron_types;
        //Attention : ces malloc ne sont pas "free" dans la fonction !
        (*debugInfo).nb_connections = (long *)malloc((*debugInfo).dim_l * sizeof(long));
        for (i=0;i<(*debugInfo).dim_l;i++)
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
    for (i=0;i<BlockInfo.dim_l;i++) //parcours des lignes
    {
        //récupération de l'indice de la partie (du cerveau) source
        ind_part_source = get_brain_part_ind(BlockInfo.startRow+i, brain);
        //récupération du type de neurone source
        source_type = neuron_types[BlockInfo.startRow+i];
        for (j=0;j<BlockInfo.dim_c;j++) //parcours des colonnes
        {
            //récupération de l'indice de la partie destination
            ind_part_dest = get_brain_part_ind(BlockInfo.startColumn+j, brain);
            //récupération de la probabilité de connexion source -> destination avec le type de neurone donné
            proba_connection = (*brain).brainPart[ind_part_source].probaConnection[source_type*(*brain).nb_part + ind_part_dest];
            proba_no_connection = 1 - proba_connection;
            random = random_between_0_and_1();
            //décision aléatoire, en prenant en compte l'abscence de connexion sur la diagonale de façon brute
            if ( (BlockInfo.startRow+i)!=(BlockInfo.startColumn+j) && random > proba_no_connection) //si on est dans la proba de connexion et qu'on est pas dans la diagonale, alors on place un 1
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
                    (*debugInfo).nb_connections[i] = (*debugInfo).nb_connections[i] + 1;
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

void coo_to_csr_matrix(IntCOOMatrix * M_COO, IntCSRMatrix * M_CSR)
{
    /*
    Traduit le vecteur Row de la matrice M_COO stockée au format COO en vecteur Row format CSR dans la matrice M_CSR
    A la fin : COO_Column=CSR_Column (adresses), COO_Value=CSR_Value (adresses), et CSR_Row est la traduction en CSR de COO_Row (adresses et valeurs différentes)
    L'allocation mémoire pour CSR_Row (taille dim_l + 1) doit être faite au préalable
    Attention : dim_c, dim_l et len_values ne sont pas modifiés dans le processus, et doivent être remplis au préalable.
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
    if (current_indl < (*M_CSR).dim_l) //cas particulier : dernière(s) ligne(s) remplie(s) de 0
    {
        current_indl++;
        *(CSR_Row + current_indl + 1) = i;
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
    int debug_matrix_block=1; //passer à 1 pour afficher les print de débuggage du block de matrice
    int debug_cerveau=0; //passer à 1 pour avoir les print de débuggage liés aux pourcentages de connexion du cerveau
    int debug_print_matrix=1; //passer à 1 pour afficher les matrices dans les processus
    int debug_pagerank=0; //passer à 1 pour afficher les débugs du pagerank
    int debug_print_full_pagerank_result=1; //passer à 1 pour allgather et afficher le vecteur résultat complet
    long i,j,k; //pour les boucles
    long n;
    int q = sqrt(p);
    int nb_blocks_row = q, nb_blocks_column = q; //q est la valeur par défaut du nombre de blocks dans les deux dimensions. q*q = p blocs utilisés
    int my_indl, my_indc; //indice de ligne et colonne du bloc
    long long size;
    long total_memory_allocated_local,nb_zeros,nb_non_zeros,nb_non_zeros_local;
    long *nb_connections_local_tmp,*nb_connections_tmp;
    int *neuron_types;

    double grid_dim_factor;
    int local_result_vector_size_column_blocks;
    int local_result_vector_size_row_blocks;
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

    grid_dim_factor = (double) nb_blocks_row / (double) nb_blocks_column;
    myBlock.pr_result_redistribution_root = (int) myBlock.indl / grid_dim_factor;
    local_result_vector_size_column_blocks = nb_blocks_column / pgcd(nb_blocks_row, nb_blocks_column);
    local_result_vector_size_row_blocks = nb_blocks_row / pgcd(nb_blocks_row, nb_blocks_column);
    local_result_vector_size_column = local_result_vector_size_column_blocks * nb_colonne;
    myBlock.result_vector_group = myBlock.indc / local_result_vector_size_column_blocks;
    myBlock.indc_in_result_vector_calculation_group = myBlock.indc % local_result_vector_size_column_blocks;
    myBlock.inter_result_vector_need_group_communicaton_group = (myBlock.indl % local_result_vector_size_row_blocks) * nb_blocks_column + myBlock.indc;
    myBlock.startColumn_in_result_vector_group = nb_colonne * myBlock.indc_in_result_vector_calculation_group;
    myBlock.my_result_vector_group_rank = myBlock.indc_in_result_vector_calculation_group + myBlock.indl * local_result_vector_size_column_blocks;

    /* Communicateurs par ligne et colonne */
    MPI_Comm ROW_COMM;
    MPI_Comm_split(MPI_COMM_WORLD, myBlock.indl, myBlock.indc, &ROW_COMM);

    MPI_Comm COLUMN_COMM;
    MPI_Comm_split(MPI_COMM_WORLD, myBlock.indc, myBlock.indl, &COLUMN_COMM);

    /* Communicateurs par groupe de calcul et de besoin du vecteur résultat (PageRank) */
    MPI_Comm RV_CALC_GROUP_COMM; //communicateur interne des groupes (qui regroupe sur les colonnes les blocks du même groupe de calcul)
    MPI_Comm_split(MPI_COMM_WORLD, myBlock.result_vector_group, myBlock.my_result_vector_group_rank, &RV_CALC_GROUP_COMM);

    MPI_Comm INTER_RV_NEED_GROUP_COMM; //communicateur externe des groupes de besoin (groupes sur les lignes) ; permet de récupérer le résultat final
    MPI_Comm_split(MPI_COMM_WORLD, myBlock.inter_result_vector_need_group_communicaton_group, my_rank, &INTER_RV_NEED_GROUP_COMM);

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0)
    {
        printf("----------------------\nBilan de votre matrice :\n");
        printf("Taille : %li * %li = %li\n",n,n,size);
        printf("%i blocs sur les lignes (avec %li lignes par bloc) et %i blocs sur les colonnes (avec %li colonnes par bloc)\n",nb_blocks_row,nb_ligne,nb_blocks_column,nb_colonne);
    }

    if (my_rank == 0 && debug_matrix_block) //débuggage du vecteur résultat
    {
        printf("Taille locale du vecteur résultat : %i blocks colonne, soit %li cases mémoire\nFacteur nb_blocks_row/nb_blocks_column = %i/%i = %f, pgcd = %i\n\n",local_result_vector_size_column_blocks,local_result_vector_size_column,nb_blocks_row,nb_blocks_column,grid_dim_factor,pgcd(nb_blocks_row, nb_blocks_column));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (debug_matrix_block) //debuggage des groupes de calcul du vecteur résultat (PageRank)
    {
        for (k=0;k<p;k++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if (my_rank == k)
                {printf("[my_rank = %i]: Result Vector Group = %i ; IndStartColumn in Result Vector Group : %i (Block), %li (Element) ; Inter-RVNeedGroup Communicator rank : %i ; Root redistrib Result Vector : %i,%i\n",my_rank,myBlock.result_vector_group,myBlock.indc_in_result_vector_calculation_group,myBlock.startColumn_in_result_vector_group,myBlock.inter_result_vector_need_group_communicaton_group,myBlock.indl,myBlock.pr_result_redistribution_root);}
        }
    }

    //Cerveau écrit en brute (pour essayer)
    int nbTypeNeuronIci,nb_part=8;
    BrainPart brainPart[nb_part];
    long part_cerv[nb_part];
    long nb_neurone_par_partie = n / nb_part;
    if (nb_part * nb_neurone_par_partie != n) {printf("Veuillez entrer un n multiple de %i svp\n",nb_part); exit(1);}
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

    if (my_rank == 0 && debug_cerveau)
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
    start_brain_generation_time = my_gettimeofday(); //début de la mesure de temps de génération de la matrice A transposée

    //génération des sous-matrices au format CSR :
    //3 ALLOCATIONS : allocation de mémoire pour CSR_Row, CSR_Column et CSR_Value dans la fonction generate_csr_matrix_for_pagerank()
    struct IntCSRMatrix A_CSR;

    neuron_types = (int *)malloc(n * sizeof(int)); //vecteur contenant les types de tout les neurones

    //choix du type de neurone pour chaque neurone du cerveau
    if (myBlock.indc == 0) //choix des processus qui vont définir les types de neurones
    {
        if (debug_cerveau) {printf("Le processus %i répond à l'appel de définition des types de neurone, il va s'occuper des neurones %li à %li\n",my_rank,myBlock.indl*nb_ligne,(myBlock.indl+1)*nb_ligne);}
        generate_neuron_types(&Cerveau, myBlock.indl*nb_ligne, nb_ligne, neuron_types + myBlock.indl*nb_ligne);
    }

    for (i=0;i<nb_blocks_row;i++)
    {
        if (my_rank==0 && debug_cerveau) {printf("Communication du processus %i vers les autres de %i neurones\n",i*nb_blocks_column,nb_ligne);}
        MPI_Bcast(neuron_types + /*adresse de lecture/ecriture : le ind_block_row dans lequel on est actuellement * nb_ligne*/ i*nb_ligne, nb_ligne, MPI_INT, i*nb_blocks_column, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank==0 && debug_cerveau)
    {
        for (i=0;i<n;i++)
        {
            printf("%i ",neuron_types[i]);
        }
        printf("\n");
    }

    //Génération de la matrice CSR à partir du cerveau
    struct DebugBrainMatrixInfo MatrixDebugInfo;
    generate_csr_brain_adjacency_matrix_for_pagerank(&A_CSR, myBlock, &Cerveau, neuron_types, &MatrixDebugInfo);

    /* MatrixDebugInfo.nb_connections contient actuellement (dans chaque processus) le nombre de connexions faites LOCALEMENT par tout les neurones par colonne. */
    nb_connections_local_tmp = (long *)malloc(n * sizeof(long)); //réecriture des informations de débug sur le nombre de connexion dans un vecteur de taille n (dimension de la matrice) aux indices correspondants, pour allreduce
    for (i=0;i<n;i++) {nb_connections_local_tmp[i] = 0;} //initialisation à 0
    for (i=myBlock.startRow;i<=myBlock.endRow;i++)
    {
        nb_connections_local_tmp[i] = MatrixDebugInfo.nb_connections[i - myBlock.startRow]; //prise en compte du décalage ligne (pour écrire aux indices qui correspondent aux neurones dans la matrice globale)
    }
    nb_connections_tmp = (long *)malloc(n * sizeof(long));
    MPI_Allreduce(nb_connections_local_tmp, nb_connections_tmp, n, MPI_LONG, MPI_SUM, MPI_COMM_WORLD); //somme MPI_SUM de tout les nb_non_zeros_local dans nb_non_zeros
    free(nb_connections_local_tmp);
    MatrixDebugInfo.nb_connections = nb_connections_tmp;
    /* MatrixDebugInfo.nb_connections contient maintenant (dans tout les processus) le nombre GLOBAL de connexions faites pour chaque neurone. */

    MPI_Barrier(MPI_COMM_WORLD);
    total_brain_generation_time = my_gettimeofday() - start_brain_generation_time; //fin de la mesure de temps de génération de la matrice A transposée

    nb_non_zeros_local = A_CSR.len_values;
    MPI_Allreduce(&nb_non_zeros_local, &nb_non_zeros, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD); //somme MPI_SUM de tout les nb_non_zeros_local dans nb_non_zeros

    //debug
    total_memory_allocated_local = MatrixDebugInfo.total_memory_allocated;
    MPI_Allreduce(&total_memory_allocated_local, &(MatrixDebugInfo.total_memory_allocated), 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD); //somme MPI_SUM de tout les total_memory_allocated_local dans MatrixDebugInfo.total_memory_allocated.
    MatrixDebugInfo.cpt_values = nb_non_zeros;

    if (debug_cerveau && my_rank == 0)
    {
        printf("Mémoire totale allouée pour le vecteur Row / le vecteur Column : %li\nNombre de cases mémoires effectivement utilisées : %li\n",MatrixDebugInfo.total_memory_allocated,MatrixDebugInfo.cpt_values);
    }

    //Page Rank
    double error_vect,error_vect_local,beta;
    double *morceau_new_q,*morceau_new_q_local,*morceau_old_q,*tmp;
    long cpt_iterations = 0;
    int maxIter = 10000;
    double epsilon = 0.00000000001;

    //variables temporaires pour code parallèle
    double to_add,sum_totale_old_q,sum_totale_new_q,sum_new_q,tmp_sum,sc;
    int nb_elements_ligne;

    //utilisé uniquement si debug_pagerank est activé
    double *q_global;
    if (debug_pagerank)
    {
        q_global = (double *)malloc(n * sizeof(double));
    }

    //init variables PageRank
    beta = 1; error_vect=INFINITY;
    //allocation mémoire pour old_q et new_q, et initialisation de new_q
    morceau_new_q = (double *)malloc(local_result_vector_size_column * sizeof(double));
    morceau_new_q_local = (double *)malloc(local_result_vector_size_column * sizeof(double));
    morceau_old_q = (double *)malloc(local_result_vector_size_column * sizeof(double));
    for (i=0;i<local_result_vector_size_column;i++) {morceau_new_q[i] = (double) 1/n;}
    sum_totale_new_q = n;

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0) {printf("Running PageRank..\n");}
    start_pagerank_time = my_gettimeofday(); //Début de la mesure de temps pour le PageRank

    while (error_vect > epsilon && !one_in_vector(morceau_new_q,local_result_vector_size_column) && cpt_iterations<maxIter)
    {
        if (my_rank == 0 /*&& debug*/) {printf("Itération %i, error = %f\n",cpt_iterations,error_vect);}
        //old_q <=> new_q  &   sum_totale_old_q <=> sum_totale_new_q
        tmp = morceau_new_q;
        morceau_new_q = morceau_old_q;
        morceau_old_q = tmp;
        tmp_sum = sum_totale_new_q;
        sum_totale_new_q = sum_totale_old_q;
        sum_totale_old_q = tmp_sum;
        //-- itération sur new_q --

        //réinitialisation morceau_new_q_local pour nouvelle ittération
        to_add = sum_totale_old_q * (1-beta)/n; //Ce qu'il y a à ajouter au résultat P.olq_q * beta. sum_total_old_q contient déjà la somme des éléments de old_q
        for (i=0; i<local_result_vector_size_column; i++)
        {
            morceau_new_q_local[i] = 0;
        }

        // calcul du produit matrice-vecteur new_q= P * old_q et de la somme des carrés total
        sum_new_q = 0;
        for(i=0; i<nb_ligne; i++)
        {
            nb_elements_ligne = MatrixDebugInfo.nb_connections[myBlock.startRow + i]; //le nombre d'éléments non nulles dans la ligne de la matrice "complète" (pas uniquement local)
            sc = morceau_old_q[myBlock.startColumn_in_result_vector_group + i] / (double) nb_elements_ligne;
            for (j=A_CSR.Row[i]; j<A_CSR.Row[i+1]; j++)
            {
                morceau_new_q_local[myBlock.startColumn_in_result_vector_group + A_CSR.Column[j]] += sc; //Produit matrice-vecteur local
            }

            for (k=myBlock.startColumn_in_result_vector_group; k<myBlock.startColumn_in_result_vector_group+nb_colonne; k++)
            {
                morceau_new_q_local[k] = morceau_new_q_local[k] * beta + to_add; //au fibal new_q = beta * P.old_q + norme(old_q) * (1-beta) / n    (la partie droite du + étant ajoutée à l'initialisation)
            }
        }

        if (debug_pagerank && my_rank % nb_blocks_column == 0)
        {
            printf("(%i,%i) rank %i, morceau q local avant reduce : ",myBlock.indl,myBlock.indc,my_rank);
            for(k=0;k<local_result_vector_size_column;k++) {printf("%.2f ",morceau_new_q_local[k]);}
            printf(", le to_add était de %f\n",to_add);
        }

        MPI_Allreduce(morceau_new_q_local, morceau_new_q, local_result_vector_size_column, MPI_DOUBLE, MPI_SUM, RV_CALC_GROUP_COMM); //Produit matrice_vecteur global : Reduce des morceaux de new_q dans tout les processus du même groupe de calcul
        MPI_Barrier(MPI_COMM_WORLD);

        if (debug_pagerank && my_rank % nb_blocks_column == 0)
        {
            printf("(%i,%i) rank %i, morceau q local après reduce : ",myBlock.indl,myBlock.indc,my_rank);
            for(k=0;k<local_result_vector_size_column;k++) {printf("%.2f ",morceau_new_q[k]);}
            printf("\n");
        }

        //if (debug_pagerank && myBlock.indc == myBlock.pr_result_redistribution_root) {printf("Communication du processus root=%i (indice %i dans le communicateur) vers les autres de la même ligne de %i éléments du nouveau vecteur q\n", myBlock.indl * nb_blocks_column + myBlock.indl,myBlock.pr_result_redistribution_root,local_result_vector_size_column);}
        MPI_Bcast(morceau_new_q, local_result_vector_size_column, MPI_DOUBLE, myBlock.pr_result_redistribution_root, ROW_COMM); //chaque processus d'une s"ligne de processus" (dans la grille) contient le même morceau de new_q
        MPI_Barrier(MPI_COMM_WORLD);

        //étape 3 : normalisation de q
        for (i=0;i<local_result_vector_size_column;i++) {sum_new_q += morceau_new_q[i];}
        MPI_Allreduce(&sum_new_q, &sum_totale_new_q, 1, MPI_DOUBLE, MPI_SUM, INTER_RV_NEED_GROUP_COMM); //somme MPI_SUM sur les colonnes de tout les sum_new_q dans sum_totale_new_q, utile pour l'itération suivante
        for (i=0;i<local_result_vector_size_column;i++) {morceau_new_q[i] *= 1/sum_totale_new_q;} //normalisation avec sum totale (tout processus confondu)

        //-- fin itération--
        cpt_iterations++;
        error_vect_local = abs_two_vector_error(morceau_new_q,morceau_old_q,nb_colonne); //calcul de l'erreur local
        MPI_Allreduce(&error_vect_local, &error_vect, 1, MPI_DOUBLE, MPI_SUM, INTER_RV_NEED_GROUP_COMM); //somme MPI_SUM sur les colonnes des erreures locales pour avoir l'erreure totale
        MPI_Barrier(MPI_COMM_WORLD);

        //debug
        if (debug_pagerank)
        {
            MPI_Allgather(morceau_new_q, local_result_vector_size_column, MPI_DOUBLE, q_global, local_result_vector_size_column, MPI_DOUBLE, INTER_RV_NEED_GROUP_COMM); //récupération par colonne des morceaux de new_q dans q_global, dans tout les processus
            if (my_rank == 0)
            {
                printf("q actuel : ");
                for(i=0;i<n;i++) {printf("%.2f ",q_global[i]);}
                printf("\n");
            }
        }
    }
    //fin du while : cpt_iterations contient le nombre d'itérations faites, new_q contient la valeur du vecteur PageRank

    MPI_Barrier(MPI_COMM_WORLD);
    total_pagerank_time = my_gettimeofday() - start_pagerank_time; //fin de la mesure de temps de calcul pour PageRank
    total_time = my_gettimeofday() - start_brain_generation_time; //fin de la mesure de temps globale (début génération matrice -> fin pagerank)

    double *pagerank_result;
    if (debug_print_full_pagerank_result)
    {
        pagerank_result = (double *)malloc(n * sizeof(double));
        MPI_Allgather(morceau_new_q, local_result_vector_size_column, MPI_DOUBLE, pagerank_result, local_result_vector_size_column, MPI_DOUBLE, INTER_RV_NEED_GROUP_COMM); //récupération par colonne des morceaux de new_q dans pagerank_result, dans tout les processus
    }

    //affichage matrices
    if (debug_print_matrix)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (my_rank == 0) {printf("-------- Matrices:\n");}
        MPI_Barrier(MPI_COMM_WORLD);
        for (k=0;k<p;k++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if (my_rank == k)
            {
                printf("Matrice du processus %i :\n",my_rank);
                for (i=0;i<myBlock.dim_l;i++)
                {
                    if (nb_ligne<=20 && nb_colonne <=20) //si la dimension de la matrice est inférieur ou égale à 32, on peut l'afficher
                    {
                        for (j=0;j<myBlock.dim_c;j++)
                        {
                            printf("%i ", get_csr_matrix_value_int(i, j, &A_CSR));
                        }
                        printf("\n");
                    }
                }
                if (nb_ligne>20 || nb_colonne>20)
                {
                    printf("trop grande pour être affichée\n");
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
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

    if (my_rank == 0 && debug_print_full_pagerank_result)
    {
        printf("\nRésultat ");
        for(i=0;i<n;i++) {printf("%.4f ",pagerank_result[i]);}
        printf("obtenu en %i itérations\n",cpt_iterations);
    }
    else if (my_rank == 0)
    {
        if ((debug && debug_cerveau) || n <= 64)
        {
            printf("\nMorceau du résultat dans le processus %i : ",my_rank);
            for(i=0;i<nb_colonne;i++) {printf("%.4f ",morceau_new_q[i]);}
            printf("obtenu en %i itérations\n",cpt_iterations);
        }
        else
        {
            printf("Morceau du vecteur résultat : %.4f %.4f ... %.4f obtenu en %i itérations\n",morceau_new_q[0],morceau_new_q[1],morceau_new_q[nb_colonne-1],cpt_iterations);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0)
    {
        printf("Temps écoulé lors de la génération : %.1f s\n", total_brain_generation_time);
        printf("Temps écoulé lors de l'application de PageRank : %.1f s\n", total_pagerank_time);
        printf("Temps total écoulé : %.1f s\n", total_time);
    }

    if (debug_pagerank) {free(q_global);}
    if (debug_print_full_pagerank_result) {free(pagerank_result);}
    free(morceau_new_q); free(morceau_new_q_local); free(morceau_old_q);
    free(neuron_types);
    free(A_CSR.Row); free(A_CSR.Column); free(A_CSR.Value);

    if (debug_cerveau)
    {
        free(MatrixDebugInfo.nb_connections); //MatrixDebugInfo.types est free plus haut : free(neuron_types);
    }
    MPI_Finalize();
    return 0;
}
