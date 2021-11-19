/*Travail sur PageRank non pondéré sequentiel*/
/*Nicolas HOCHART*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

struct IntCOOMatrix
{
     int * Row; //vecteur de taille "nombre d'éléments non nuls dans la matrice"
     int * Column; //vecteur de taille "nombre d'éléments non nuls dans la matrice"
     int * Value; //vecteur de taille "nombre d'éléments non nuls dans la matrice"
     long dim_l; //nombre de lignes
     long dim_c; //nombre de colonnes
     long len_values; //taille des vecteurs Row, Column et Value
};
typedef struct IntCOOMatrix IntCOOMatrix;

struct IntCSRMatrix
{
     int * Row; //vecteur de taille "nombre de lignes + 1" (dim_l + 1)
     int * Column; //vecteur de taille "nombre d'éléments non nuls dans la matrice"
     int * Value; //vecteur de taille "nombre d'éléments non nuls dans la matrice"
     long dim_l; //nombre de lignes
     long dim_c; //nombre de colonnes
     long len_values;  //taille des vecteurs Column et Value
};
typedef struct IntCSRMatrix IntCSRMatrix;

struct DoubleCSRMatrix
{
     int * Row; //vecteur de taille "nombre de lignes + 1" (dim_l + 1)
     int * Column; //vecteur de taille "nombre d'éléments non nuls dans la matrice"
     double * Value; //vecteur de taille "nombre d'éléments non nuls dans la matrice"
     long dim_l; //nombre de lignes
     long dim_c; //nombre de colonnes
     long len_values; //taille des vecteurs Column et Value
};
typedef struct DoubleCSRMatrix DoubleCSRMatrix;

float random_between_0_and_1()
{
    /*Renvoie un nombre aléatoire entre 0 et 1. Permet de faire une décision aléatoire*/
    return (float) rand() / (float) RAND_MAX;
}

void init_row_dense_matrix(int *M, long i, long n, int zero_percentage)
{
    /*
    Rempli n éléments de la ligne i de la matrice M.
    Il y a zero_percentage % de chances que le nombre soit 0.
    Statistiquement, zero_percentage % de la matrice sont des 0 et (100 - zero_percentage) % sont des 1
    */
    long j;

    for (j=0;j<n;j++)
    {
        if (random_between_0_and_1() < zero_percentage/100.0) //zero_percentage % de chances de mettre un 0
        {
            *(M + i*n+j) = 0;
        }
        else
        {
            *(M + i*n+j) = 1;
        }
    }
}

long cpt_nb_zeros_matrix(int *M, long long size)
{
    /*Compte le nombre de 0 dans la matrice M à size elements*/
    long compteur = 0;
    for (int d=0;d<size;d++)
    {
        if (*(M+d) == 0)
        {
            compteur++;
        }
    }
    return compteur;
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

void coo_to_csr_matrix(IntCOOMatrix * M_COO, IntCSRMatrix * M_CSR)
{
    /*
    Traduit la matrice M_COO stockée au format COO en matrice stockée en format CSR dans M_CSR
    A la fin COO_Column=CSR_Column, COO_Value=CSR_Value, et CSR_Row est la traduction en CSR de COO_Row
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
    Renvoie la valeur [indl,indc] de la matrice CSR stockée dans Row,Column,Value. len_values est la longueur du vecteur Value.
    l le nombre de lignes de la matrice (longueur du vecteur Row - 1) et c le nombre de colonnes.
    Le vecteur Value doit être un vecteur d'entiers.
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
        if (Column[i] == indc)
        {
            return Value[i];
        }
    }
    return 0;
}

double get_csr_matrix_value_double(long indl, long indc, DoubleCSRMatrix * M_CSR)
{
    /*
    Renvoie la valeur [indl,indc] de la matrice CSR stockée dans Row,Column,Value. len_values est la longueur du vecteur Value.
    l le nombre de lignes de la matrice (longueur du vecteur Row - 1) et c le nombre de colonnes.
    Le vecteur Value doit être un vecteur de doubles.
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
        if (Column[i] == indc)
        {
            return Value[i];
        }
    }
    return 0;
}

void fill_matrix_column_sum_vector(int *sum_vector, DoubleCSRMatrix * M_CSR)
{
    /*
    Ecrit dans sum_vector (vecteur de taille c) la somme des éléments de chaque colonnes d'une matrice au format CSR (Row (ici non utilisé),Column,Value).
    Chaque case d'indice i du sum_vector contiendra la somme des éléments de la colonne du même indice i.
    len_values est la longueur du vecteur Value, et c le nombre de colonnes de la matrice.
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

void normalize_matrix(DoubleCSRMatrix * M_CSR)
{
    /*
    Normalise la matrice CSR M_CSR en utilisant le vecteur sum_vector (contenant déjà la somme des éléments colonne par colonne)
    */
    long i;
    int * sum_vector = (int *)malloc((*M_CSR).dim_c * sizeof(int));
    fill_matrix_column_sum_vector(sum_vector, M_CSR);

    for (i=0;i<(*M_CSR).len_values;i++) //on parcours le vecteur Column et Value, et on divise chaque valeur (de Value) par la somme (dans sum_vector) de la colonne correspondante
    {
        (*M_CSR).Value[i] = (*M_CSR).Value[i] / sum_vector[(*M_CSR).Column[i]];
    }
    free(sum_vector);
}

/*Fonctions pour PageRank*/

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
    double sum=0;
    for (int i=0;i<size;i++)
    {
        sum+=vect[i];
    }
    return sum;
}

double abs_two_vector_error(double *vect1, double *vect2, int size)
{
    /*Calcul l'erreur entre deux vecteurs de taille "size"*/
    double sum=0;
    for (int i=0;i<size;i++)
    {
        sum += fabs(vect1[i] - vect2[i]);
        //printf("%f - %f ; sum = %f\n",vect1[i],vect2[i],sum);
    }
    return sum;
}

void copy_vector_value(double *vect1, double *vect2, int size)
{
    /*Copie les valeurs du vecteur 1 dans le vecteur 2. Les deux vecteurs doivent être de taille "size".*/
    for (int i=0;i<size;i++) {vect2[i] = vect1[i];}
}

void iterationMP(DoubleCSRMatrix *P, double *new_q, double *old_q, int n, double beta)
{
    /*
    Fait une itération de la méthode de la puissance
    P est la matrice de passage, old_q le vecteur pagerank q précédent, et n la dimension de la matrice
    */
    int i;
    double norme_old_q,norme_new_q,to_add;
    //étape 1 : new_q = beta * P.old_q
    csr_matrix_vector_product(new_q,P,old_q);
    for (i=0;i<n;i++) {new_q[i] *= beta;}
    //étape 2 : (chaque element) newq += norme(old_q) * (1-beta) / n
    norme_old_q = vector_norm(old_q,n);
    to_add = norme_old_q * (1-beta)/n;
    for (i=0;i<n;i++) {new_q[i] += to_add;}
    //étape 3 : normalisation de q
    norme_new_q = vector_norm(new_q,n);
    for (i=0;i<n;i++) {new_q[i] *= 1/norme_new_q;}
}

int methodeDeLaPuissance(DoubleCSRMatrix *P, double *q_init, double *q_end, double beta, double epsilon, int maxIter)
{
    /*
    Applique la méthode de la puissance au vecteur initial q_init passé en paramètre, avec la matrice de passage P passée en paramètre
    */
    long n=(*P).dim_c,i,cpt = 0;
    double *new_q = (double *)malloc(n * sizeof(double));
    double *old_q = (double *)malloc(n * sizeof(double));
    double *tmp;
    copy_vector_value(q_init,old_q,n); //old_q = q_init
    copy_vector_value(q_init,new_q,n); //new_q = q_init
    for (i=0;i<n;i++) {old_q[i] *= 1000;} //init pour avoir une différence
    while (abs_two_vector_error(new_q,old_q,n) > epsilon && !one_in_vector(new_q,n) && cpt<maxIter)
    {
        /*old_q = new_q <=> copy_vector_value(new_q,old_q,n)*/
        tmp = new_q;
        new_q = old_q;
        old_q = tmp;
        /*itération sur new_q*/
        iterationMP(P, new_q, old_q, n, beta);
        cpt++;
    }
    copy_vector_value(new_q,q_end,n); //q_end = new_q
    free(new_q);free(old_q);
    return cpt;
}

void csr_to_dense_matrix(double *M, DoubleCSRMatrix * M_CSR)
{
    /*Fonction temporaire pour faire un pagerank avec une matrice stockée normalement*/
    int i,j;
    for (i=0;i<(*M_CSR).dim_l;i++)
    {
        for (j=0;j<(*M_CSR).dim_c;j++)
        {
            *(M+i*(*M_CSR).dim_c+j) = get_csr_matrix_value_double(i, j, M_CSR);
        }
    }
}

int main(int argc, char **argv)
{
    int debug=1; //passer à 1 pour avoir plus de print
    long i,j; //pour les boucles
    long n;
    long long size;
    int nb_zeros,nb_non_zeros;
    int *A;

    //matrices au format COO et CSR
    struct IntCOOMatrix A_COO;
    struct IntCSRMatrix A_CSR;
    struct DoubleCSRMatrix norm_A_CSR;

    //variables pour la normalisation de la matrice
    double *NormValue;

    if (argc < 2)
    {
        printf("Veuillez entrer la taille de la matrice après le nom de l'executable : %s n\n", argv[0]);
        exit(1);
    }

    n = atoll(argv[1]);
    size = n * n;
    A = (int *)malloc(size * sizeof(int));

    for (i=0;i<n;i++)
    {
        init_row_dense_matrix(A, i, n, 75);
    }
    for (i=0;i<n;i++) //remplissage de la diagonale de 0
    {
        A[i*(n+1)] = 0;
    }

    printf("Matrice stockée \"dense\" :\n");
    for (i=0;i<n;i++){for (j=0;j<n;j++){printf("%i ",A[i*n+j]);} printf("\n");}

    nb_zeros = cpt_nb_zeros_matrix(A, size);
    nb_non_zeros = size - nb_zeros;

    A_COO.len_values = A_CSR.len_values = norm_A_CSR.len_values = nb_non_zeros;
    A_CSR.dim_l = norm_A_CSR.dim_l = n;
    A_COO.dim_l = A_COO.dim_c = A_CSR.dim_c = norm_A_CSR.dim_c = n;
    A_COO.Row = (int *)malloc(nb_non_zeros * sizeof(int));
    A_CSR.Row = norm_A_CSR.Row = (int *)malloc((n+1) * sizeof(int));
    A_COO.Column = A_CSR.Column = norm_A_CSR.Column = (int *)malloc(nb_non_zeros * sizeof(int));
    A_COO.Value = A_CSR.Value = (int *)malloc(nb_non_zeros * sizeof(int));
    norm_A_CSR.Value = (double *)malloc(nb_non_zeros * sizeof(double));

    dense_to_coo_matrix(A, &A_COO);
    coo_to_csr_matrix(&A_COO, &A_CSR);

    //copie du vecteur Value dans NormValue
    for(i=0;i<nb_non_zeros;i++) {norm_A_CSR.Value[i] = (double) A_CSR.Value[i];} //norm_A_CSR.Value = A_CSR.Value

    //normalisation de la matrice (Row, Column, Value) dans (Row, Column, NormValue)
    normalize_matrix(&norm_A_CSR);

    if (debug)
    {
        printf("Nombre de zeros : %i\n",nb_zeros);
        printf("Nombre de valeurs non nulles : %i\n",nb_non_zeros);

        printf("\nVecteur Row :\n");
        for(i=0;i<n+1;i++) {printf("%i ",norm_A_CSR.Row[i]);}
        printf("\nVecteur Column :\n");
        for(i=0;i<nb_non_zeros;i++) {printf("%i ",norm_A_CSR.Column[i]);}
        printf("\nVecteur Value :\n");
        for(i=0;i<nb_non_zeros;i++) {printf("%f ",norm_A_CSR.Value[i]);} printf("\n");

        printf("\nMatrice creuse stockée en format CSR:\n");
        for (i=0;i<n;i++){for (j=0;j<n;j++){printf("%i ",get_csr_matrix_value_int(i, j, &A_CSR));} printf("\n");}

        printf("\nMatrice normalisée sur les colonnes (stockée en format CSR):\n");
        for (i=0;i<n;i++){for (j=0;j<n;j++){printf("%.2f ",get_csr_matrix_value_double(i, j, &norm_A_CSR));} printf("\n");}
        printf("\n");
    }

    /*Page Rank*/
    double beta;
    double *q;
    int maxIter = 100000,nb_iterations_faites;
    double epsilon = 0.00000000001;

    beta = 1;
    q = (double *)malloc(n * sizeof(double));
    for (i=0;i<n;i++) {q[i] = (double) 1/n;}

    nb_iterations_faites = methodeDeLaPuissance(&norm_A_CSR, q, q, beta, epsilon, maxIter);
    printf("\nrésultat ");
    for(i=0;i<n;i++) {printf("%f ",q[i]);}
    printf("obtenu en %i itérations\n",nb_iterations_faites);

    free(q);
    free(A_COO.Row); free(A_COO.Column); free(A_COO.Value);
    free(A_CSR.Row); //Column et Value sont communs avec la matrice COO
    free(norm_A_CSR.Value); //Row et Column communs avec la matrice CSR
    return 0;
}
