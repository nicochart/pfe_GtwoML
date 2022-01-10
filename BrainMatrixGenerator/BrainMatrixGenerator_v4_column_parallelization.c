/* Travail sur un générateur de matrice parallèle avec des blocs de ligne et colonnes. */
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

/*--------------------------------------------------------------------------------
--- Fonctions pour génération de matrices ou changement de formats de matrices ---
--------------------------------------------------------------------------------*/

void generate_coo_matrix_for_pagerank(IntCOOMatrix *M_COO, MatrixBlock BlockInfo, int zero_percentage, long l, long c)
{
    /*
    Génère complètement aléatoirement (ne correspondant pas à un cerveau) la matrice creuse (*M_COO) (format COO) pour PageRank.
    l et c sont les nombres de ligne et nombre de colonnes de la matrice, ils seront stockés dans dim_l et dim_c
    ind_start_row est, dans le cas où on génère la matrice par morceaux, l'indice de la ligne (dans la matrice complète) où le morceau commence.
    Ce dernier indice permet de remplir la diagonale de la matrice de 0 (pour PageRank : un site ne peut pas être relié à lui même)
    Statistiquement, il y a zero_percentage % de 0 dans la matrice l*c.
    Environs zero_percentage % de la matrice dense correspondante sont des 0 et (100 - zero_percentage) % sont des 1.
    (Ce n'est pas exact, car un test est effectué avec ind_start_row pour remplir la diagonale de 0. Ce problème sera corrigé plus tard)
    */
    long i = 0, j = 0, cpt_values, size = BlockInfo.dim_l * BlockInfo.dim_c;
    long mean_nb_non_zeros = (int) size * (100 - zero_percentage) / 100; //nombre moyen de 1 dans la matrice
    (*M_COO).dim_l = l; (*M_COO).dim_c = c;
    //Attention : La mémoire pour les vecteurs Row, Column et Value est allouée dans la fonction, mais n'est pas libérée dans la fonction.
    //La mémoire allouée est (statistiquement) plus grande que la mémoire qui sera utilisée en pratique. On ne peut pas savoir à l'avance exactement combien de valeurs aura la matrice.
    (*M_COO).Row = (int *)malloc(mean_nb_non_zeros * sizeof(int));
    (*M_COO).Column = (int *)malloc(mean_nb_non_zeros * sizeof(int));
    (*M_COO).Value = (int *)malloc(mean_nb_non_zeros * sizeof(int));

    cpt_values=0;
    while (i <= BlockInfo.dim_l) //parcours des lignes
    {
        j = 0;
        while (j <= BlockInfo.dim_c) //parcours des colonnes
        {
            if ( (BlockInfo.startRow + i != BlockInfo.startColumn + j) && random_between_0_and_1() > zero_percentage/100.0) //si on est dans le pourcentage de non zero et qu'on est pas dans la diagonale, alors on place un 1
            {
                if (cpt_values < mean_nb_non_zeros)
                {
                    (*M_COO).Row[cpt_values] = i;
                    (*M_COO).Column[cpt_values] = j;
                    (*M_COO).Value[cpt_values] = 1;
                    cpt_values++;
                }
            }
            j++;
        }
        i++;
    }
    (*M_COO).len_values = cpt_values;
}

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

    if ((double) sqrt(p) != (int) sqrt(p))
    {
        if (my_rank == 0) {printf("Erreur : le nombre de coeurs disponible (%i) n'a pas de racine entière..\nOn doit pouvoir le passer à la racine et obtenir un nombre entier pour pouvoir diviser en ligne et colonnes.\n",p);}
        exit(1);
    }

    int debug=1; //passer à 1 pour afficher les print de débuggage
    int debug_cerveau=1; //passer à 1 pour avoir les print de débuggage liés aux pourcentages de connexion du cerveau
    long i,j,k; //pour les boucles
    long n;
    int q = sqrt(p);
    int nb_blocks_row = q, nb_blocks_column = q; //q est la valeur par défaut du nombre de blocks dans les deux dimensions. q*q = p blocs utilisés
    int my_indl, my_indc; //indice de ligne et colonne du bloc
    long long size;
    long total_memory_allocated_local,nb_zeros,nb_non_zeros,nb_non_zeros_local;
    long *nb_connections_local_tmp,*nb_connections_tmp;
    int *neuron_types, *local_types;

    double start_time, total_time;

    if (argc < 2)
    {
        printf("Veuillez entrer la taille de la matrice après le nom de l'executable : %s n\n Vous pouvez aussi indiquer des pourcentages de 0 pour chaque bloc après le n.\n", argv[0]);
        exit(1);
    }
    n = atoll(argv[1]);
    if (argc < 4)
    {
        if (my_rank == 0) {printf("Si vous voulez saisir le nombre de blocs (parallèles), veuillez les préciser dans les deux dimensions : %s n l c\nValeur prise par défaut : sqrt(%i) = %i\n", argv[0], p, q);}
    }
    else if (argc >= 4) //dans le cas où on a des paramètres correspondant au nombre de blocs dans les dimensions
    {
        nb_blocks_row = atoll(argv[2]);
        nb_blocks_column = atoll(argv[3]);
        if (nb_blocks_row * nb_blocks_column != p)
        {
            if (my_rank == 0) {printf("Erreur : %i * %i != %i. Usage : %s n nb_blocks_row nb_blocks_column (les nombres de blocs par ligne/colonne multipliés doit être égale à %i, le nombre de coeurs alloués)\n", nb_blocks_row, nb_blocks_column, p, argv[0], p);}
            exit(1);
        }
    }

    size = n * n; //taille de la matrice
    long nb_ligne = n/nb_blocks_row, nb_colonne = n/nb_blocks_column; //nombre de lignes/colonnes par bloc

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

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0)
    {
        printf("----------------------\nBilan de votre matrice :\n");
        printf("Taille : %li * %li = %li\n",n,n,size);
        printf("%i blocs sur les lignes (avec %li lignes par bloc) et %i blocs sur les colonnes (avec %li colonnes par bloc)\n",nb_blocks_row,nb_ligne,nb_blocks_column,nb_colonne);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank==0) {printf("Debut génération matrice\n");}
    start_time = my_gettimeofday();

    //génération des sous-matrices au format COO
    //matrice format COO :
    //3 ALLOCATIONS : allocation de mémoire pour COO_Row, COO_Column et COO_Value dans la fonction generate_coo_matrix_for_pagerank()
    struct IntCOOMatrix A_COO;
    generate_coo_matrix_for_pagerank(&A_COO, myBlock, 50, nb_ligne, n);

    nb_non_zeros_local = A_COO.len_values;
    printf("Nombre de non zero local : %i\n",nb_non_zeros_local);//DEL
    MPI_Allreduce(&nb_non_zeros_local, &nb_non_zeros, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); //somme MPI_SUM de tout les nb_non_zeros_local dans nb_non_zeros

    if (debug)
    {
        printf("\nVecteur A_COO.Row dans my_rank=%i:\n",my_rank);
        for(i=0;i<A_COO.len_values;i++) {printf("%i ",A_COO.Row[i]);}printf("\n");
        printf("Vecteur A_COO.Column dans my_rank=%i:\n",my_rank);
        for(i=0;i<A_COO.len_values;i++) {printf("%i ",A_COO.Column[i]);}printf("\n");
        printf("Vecteur A_COO.Value dans my_rank=%i:\n",my_rank);
        for(i=0;i<A_COO.len_values;i++) {printf("%i ",A_COO.Value[i]);}printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

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
        MPI_Barrier(MPI_COMM_WORLD);
        if (my_rank == 0) {printf("-------- Matrices:\n");}
        MPI_Barrier(MPI_COMM_WORLD);
        for (k=0;k<p;k++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            if (my_rank == k)
            {
                printf("Matrices du processus %i :\n",my_rank);
                for (i=0;i<myBlock.dim_l;i++)
                {
                    if (n<=32) //si la dimension de la matrice est inférieur ou égale à 32, on peut l'afficher
                    {
                        for (j=0;j<myBlock.dim_c;j++)
                        {
                            printf("%i ", get_csr_matrix_value_int(i, j, &A_CSR));
                        }
                        printf("\n");
                    }
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0) {printf("Temps écoulé lors de la génération : %.1f s\n", total_time);}

    free(A_COO.Row); free(A_COO.Column); free(A_COO.Value);
    free(A_CSR.Row);

    MPI_Finalize();
    return 0;
}
