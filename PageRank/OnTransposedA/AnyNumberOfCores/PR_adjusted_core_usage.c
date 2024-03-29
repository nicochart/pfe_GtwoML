/*Travail pour rendre indépendant le code du nombre de coeurs / dimensions en utilisant un nombre de coeurs adapté parmis ceux alloués.
PAS ENCORE FAIT : le code est le même que PageRankParallele.c*/
/*PR non pondéré parallele*/
/*Nicolas HOCHART*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

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

/*--------------------------
--- Décision "aléatoire" ---
--------------------------*/

float random_between_0_and_1()
{
    /*Renvoie un nombre aléatoire entre 0 et 1. Permet de faire une décision aléatoire*/
    return (float) rand() / (float) RAND_MAX;
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
        for (j=0;j<c;j++) //parcours des lignes
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
    int my_rank, nb_allocated_cores, valeur, tag = 0;
    MPI_Status status;

    //Initialisation MPI
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nb_allocated_cores);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int p; //nombre de coeurs utilisés parmis ceux alloués
    int debug=1; //passer à 1 pour afficher les print de débuggage
    long i,j,k; //pour les boucles
    long n;
    long long size;
    int nb_zeros,nb_non_zeros,nb_non_zeros_local,*list_nb_non_zeros_local;

    if (argc < 2)
    {
        printf("Veuillez entrer la taille de la matrice après le nom de l'executable : %s n\n Vous pouvez aussi indiquer des pourcentages de 0 pour chaque bloc après le n.\n", argv[0]);
        exit(1);
    }
    n = atoll(argv[1]);
    size = n * n;

    p = nb_allocated_cores;
    while (n%p !=0)
    {
        p--; //on décrémente le nombre de coeurs utilisés jusqu'à ce que p divise n.
    }

    if (my_rank == 0) {printf("Nombre de coeurs choisis = %i",p);}

    if (my_rank <=p)
    {
    long nb_ligne = n/p; //nombre de lignes par bloc

    //allocation mémoire pour les nombres de 0 dans chaque sous matrice de chaque processus
    if (debug) {list_nb_non_zeros_local = (int *)malloc(p * sizeof(int));}

    //allocation mémoire et initialisation d'une liste de taille "nombre de processus" contenant les pourcentages de 0 que l'on souhaite pour chaque bloc
    int *zeros_percentages = (int *)malloc(p * sizeof(int)); for (i=0;i<p;i++) {zeros_percentages[i] = 75;}

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

    //génération des sous-matrices au format COO
    //matrice format COO :
    //3 ALLOCATIONS : allocation de mémoire pour COO_Row, COO_Column et COO_Value dans la fonction generate_coo_matrix_for_pagerank()
    struct IntCOOMatrix A_COO;
    generate_coo_matrix_for_pagerank(&A_COO, my_rank*nb_ligne, zeros_percentages[my_rank], nb_ligne, n);

    nb_non_zeros_local = A_COO.len_values;
    MPI_Allreduce(&nb_non_zeros_local, &nb_non_zeros, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); //somme MPI_SUM de tout les nb_non_zeros_local dans nb_non_zeros
    if (debug) {MPI_Allgather(&nb_non_zeros_local, 1, MPI_INT, list_nb_non_zeros_local, 1,  MPI_INT, MPI_COMM_WORLD);} //réunion dans chaque processus de tout les nombres de zéros de chaque bloc

    if (debug && my_rank == 0)
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
        //ca coince : Allgather et Allreduce attendent des valeurs de tout les processus...... y compris ceux non utilisés
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

    if (my_rank == 0) {printf("Matrice A :\n");}
    MPI_Barrier(MPI_COMM_WORLD);
    for (k=0;k<p;k++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (my_rank == k)
        {
            for (i=0;i<nb_ligne;i++)
            {
                for (j=0;j<n;j++)
                {
                    printf("%i ", get_csr_matrix_value_int(i, j, &A_CSR));
                }
                printf("\n");
            }
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
    }
    MPI_Finalize();
    return 0;
}
