/*Travail sur PageRank non pondéré sequentiel*/
/*Nicolas HOCHART*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

void dense_to_coo_matrix(int *M, int *Row, int *Column, int *Value, long l, long c)
{
    /*
    Traduit la matrice stockée normalement dans M (de taille l*c) en matrice stockée en format COO (Row, Column, Value)
    Les vecteurs Row, Column et Value sont de taille "nombre d'éléments non nulles dans la matrice".
    */
    long i, j, nb = 0;
    for (i=0;i<l;i++)
    {
        for (j=0;j<c;j++)
        {
            if (*(M + i*c+j) != 0)
            {
                *(Row + nb) = i; *(Column + nb) = j;
                *(Value + nb) = *(M+i*c+j);
                nb++;
            }
        }
    }
}

void coo_to_csr_matrix(int *COO_Row, int *CSR_Row, long len_coo_row)
{
    /*
    Traduit la matrice stockée au format COO dans (COO_Row, COO_Column, COO_Value) en matrice stockée en format csr (CSR_Row, CSR_Column, CSR_Value)
    COO_Column=CSR_Column, COO_Value=CSR_Value donc on le nes passe pas en paramètre.
    Le vecteur COO_Row est de longueur len_coo_row, et le vecteur CSR_Row est de longueur "nombre de ligne de la matrice + 1" (l+1).
    */
    long i,current_indl = COO_Row[0];
    printf("len_coo_row = %i\n",len_coo_row);
    for (i=0;i<len_coo_row;i++)
    {
        if (COO_Row[i] != current_indl)
        {
            *(CSR_Row + current_indl + 1) = i;
            current_indl = COO_Row[i];
        }
    }
    *(CSR_Row + current_indl + 1) = len_coo_row;
}

int get_csr_matrix_value_int(long indl, long indc, int *Row, int *Column, int *Value, long l, long c)
{
    /*
    Renvoie la valeur [indl,indc] de la matrice CSR stockée dans Row,Column,Value. len_values est la longueur du vecteur Value.
    l le nombre de lignes de la matrice (longueur du vecteur Row - 1) et c le nombre de colonnes.
    Le vecteur Value doit être un vecteur d'entiers.
    */
    if (indl >= l || indc >= c)
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

double get_csr_matrix_value_double(long indl, long indc, int *Row, int *Column, double *Value, long l, long c)
{
    /*
    Renvoie la valeur [indl,indc] de la matrice CSR stockée dans Row,Column,Value. len_values est la longueur du vecteur Value.
    l le nombre de lignes de la matrice (longueur du vecteur Row - 1) et c le nombre de colonnes.
    Le vecteur Value doit être un vecteur de doubles.
    */
    if (indl >= l || indc >= c)
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

void fill_matrix_column_sum_vector(int *sum_vector, int *Column, int *Value, long len_values, long c)
{
    /*
    Ecrit dans sum_vector (vecteur de taille c) la somme des éléments de chaque colonnes d'une matrice au format CSR (Row (ici non utilisé),Column,Value).
    Chaque case d'indice i du sum_vector contiendra la somme des éléments de la colonne du même indice i.
    len_values est la longueur du vecteur Value, et c le nombre de colonnes de la matrice.
    */
    int i;
    for (i=0;i<c;i++) //initialisation du vecteur sum_vector
    {
        *(sum_vector+i) = 0;
    }
    
    for (i=0;i<len_values;i++) //on parcours le vecteur Column et Value, et on ajoute la valeur à la somme de la colonne correspondante
    {
        *(sum_vector + Column[i]) += Value[i];
    }
}

void normalize_matrix(int *sum_vector, int *Column, double *Value, long len_values, long c)
{
    /*
    Normalise la matrice CSR (Row (ici non utilisé),Column,Value) en utilisant le vecteur sum_vector (contenant déjà la somme des éléments colonne par colonne)
    len_values est la longueur du vecteur Value, et c le nombre de colonnes de la matrice.
    Attention : le vecteur Value doit être un vecteur de doubles.
    */
    long i;
    for (i=0;i<len_values;i++) //on parcours le vecteur Column et Value, et on divise chaque valeur (de Value) par la somme (dans sum_vector) de la colonne correspondante
    {
        Value[i] = Value[i] / sum_vector[Column[i]];
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
    
    //variables de la matrice creuse COO et CSR
    //COO et CSR
    int *Value,*Column;
    //COO
    int *COO_Row;
    //CSR
    int *Row;
    
    //variables pour la normalisation de la matrice
    int *VectSum_Column;
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
    
    nb_zeros = cpt_nb_zeros_matrix(A, size);
    nb_non_zeros = size - nb_zeros;
    
    COO_Row = (int *)malloc(nb_non_zeros * sizeof(int));
    Row = (int *)malloc((n+1) * sizeof(int));
    Column = (int *)malloc(nb_non_zeros * sizeof(int));
    Value = (int *)malloc(nb_non_zeros * sizeof(int));

    dense_to_coo_matrix(A, COO_Row, Column, Value, n, n);
    coo_to_csr_matrix(COO_Row, Row, nb_non_zeros);
    //Row[1] = 4;
    
    VectSum_Column = (int *)malloc(n * sizeof(int));
    fill_matrix_column_sum_vector(VectSum_Column, Column, Value, nb_non_zeros, n);
    
    NormValue = (double *)malloc(nb_non_zeros * sizeof(double));
    //copie du vecteur Value dans NormValue
    for(i=0;i<nb_non_zeros;i++) {NormValue[i] = (double) Value[i];}
    
    //normalisation de la matrice (Row, Column, Value) dans (Row, Column, NormValue)
    normalize_matrix(VectSum_Column, Column, NormValue, nb_non_zeros, n);
    
    if (debug)
    {
        printf("Nombre de zeros : %i\n",nb_zeros);
        printf("Nombre de valeurs non nulles : %i\n",nb_non_zeros);
        
        printf("\nVecteur COO_Row :\n");
        for(i=0;i<nb_non_zeros;i++) {printf("%i ",COO_Row[i]);}
        printf("\nVecteur Row :\n");
        for(i=0;i<n+1;i++) {printf("%i ",Row[i]);}
        printf("\nVecteur Column :\n");
        for(i=0;i<nb_non_zeros;i++) {printf("%i ",Column[i]);}
        printf("\nVecteur Value :\n");
        for(i=0;i<nb_non_zeros;i++) {printf("%i ",Value[i]);} printf("\n");
        printf("\nMatrice creuse stockée en format CSR:\n");
        for (i=0;i<n;i++){for (j=0;j<n;j++){printf("%i ",get_csr_matrix_value_int(i, j, Row, Column, Value, n, n));} printf("\n");}
        
        printf("\nVecteur VectSum_Column (somme des éléments colonne par colonne:\n");
        for(i=0;i<n;i++){printf("%i ",VectSum_Column[i]);}printf("\n");
        
        printf("\nMatrice normalisée sur les colonnes (stockée en format CSR):\n");
        for (i=0;i<n;i++){for (j=0;j<n;j++){printf("%.2f ",get_csr_matrix_value_double(i, j, Row, Column, NormValue, n, n));} printf("\n");}
    }
    
    /*Page Rank*/
    
    
}
