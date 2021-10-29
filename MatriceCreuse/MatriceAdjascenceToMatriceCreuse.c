/*Changement de stockage d'une matrice carrée à l*c elements remplie de 0 et de 1 (stockée normalement) en stockage pour matrice creuse (on ne stock pas les 0)*/
/*La matrice initiale est générée aléatoirement, c'est une matrice qui pourrait être utilisée pour faire un PageRank non pondéré*/
/*Nicolas HOCHART*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float random_between_0_and_1()
{
    /*Renvoie un nombre aléatoire entre 0 et 1. Permet de faire une décision aléatoire*/
    return (float) rand() / (float) RAND_MAX;
}

void init_matrix(int *M, long l, long c, int zero_percentage)
{
    /*
    Rempli la matrice M de taille l*c de nombres. Il y a zero_percentage % de chances que le nombre soit 0.
    Statistiquement, zero_percentage % de la matrice sont des 0 et (100 - zero_percentage) % sont des 1
    */
    long i,j;
    
    for (i=0;i<l;i++)
    {
        for (j=0;j<c;j++)
        {
            if (random_between_0_and_1() < zero_percentage/100.0) //zero_percentage % de chances de mettre un 0
            {
                *(M + i*c+j) = 0;
            }
            else
            {
                *(M + i*c+j) = 1;
            }
        }
    }
}

void init_row_matrix(int *M, long i, long n, int zero_percentage)
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

void fill_sparce_matrix(int *M, int *Row, int *Column, int *Value, long l, long c)
{
    /*
    Traduit la matrice stockée dans M (de taille l*c) en matrice creuse dans les vecteurs Row, Column et Value
    Le vecteur Row est de taille n+1 (nombre de lignes + 1)
    Les vecteurs Column (indices de colonne) et Value (valeur) sont de taille "nombre d'éléments non nulles dans la matrice".
    */
    long i,j,nb = 0;
    for (i=0;i<l;i++)
    {
        *(Row+i) = nb;
        for (j=0;j<c;j++)
        {
            if (*(M + i*c+j) != 0)
            {
                *(Column+nb) = j;
                *(Value+nb) = *(M+i*c+j);
                nb++;
            }
        }
    }
    *(Row+l) = nb;
}

int get_sparce_matrix_value_int(long indl, long indc, int *Row, int *Column, int *Value, long len_values, long l, long c)
{
    /*
    Renvoie la valeur [indl,indc] de la matrice creuse stockée dans Row,Column,Value. len_values est la longueur du vecteur Value.
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

double get_sparce_matrix_value_double(long indl, long indc, int *Row, int *Column, double *Value, long len_values, long l, long c)
{
    /*
    Renvoie la valeur [indl,indc] de la matrice creuse stockée dans Row,Column,Value. len_values est la longueur du vecteur Value.
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
    int i;
    for (i=0;i<len_values;i++) //on parcours le vecteur Column et Value, et on divise chaque valeur (de Value) par la somme (dans sum_vector) de la colonne correspondante
    {
        Value[i] = Value[i] / sum_vector[Column[i]];
    }
}

int main(int argc, char **argv)
{
    long i,j; //pour les boucles
    long l,c;
    long long size;
    int nb_zeros;
    int *A;
    
    //variables de la matrice creuse
    int *Row,*Column;
    int *Value;
    
    //variables pour la normalisation de la matrice
    int *VectSum_Column;
    double *NormValue;
    
    if (argc < 3)
    {
        printf("Veuillez entrer la taille de la matrice après le nom de l'executable : %s [l] [c]\n", argv[0]);
        exit(1);
    }
    
    l = atoll(argv[1]);
    c = atoll(argv[2]);
    size = l * c;
    A = (int *)malloc(size * sizeof(int));
    
    for (i=0;i<l;i++)
    {
        init_row_matrix(A, i, c, 75);
    }
    
    printf("\nMatrice creuse stockée \"normalement\":\n");
    for (i=0;i<l;i++)
    {
        for (j=0;j<c;j++)
        {
            printf("%i ",A[i*c+j]);
        }
        printf("\n");
    }
    
    nb_zeros = cpt_nb_zeros_matrix(A, size);
    printf("Nombre de zeros : %i\n",nb_zeros); //devrait être environs égal à n*n / 2
    printf("Nombre de valeurs non nulles : %i\n",size - nb_zeros);
    
    Row = (int *)malloc((l+1) * sizeof(int));
    Column = (int *)malloc((size - nb_zeros) * sizeof(int));
    Value = (int *)malloc((size - nb_zeros) * sizeof(int));
    fill_sparce_matrix(A, Row, Column, Value, l, c);
    
    printf("\nVecteur Row :\n");
    for(i=0;i<l+1;i++)
    {
        printf("%i ",Row[i]);
    }
    printf("\nVecteur Column :\n");
    for(i=0;i<(size - nb_zeros);i++)
    {
        printf("%i ",Column[i]);
    }
    printf("\nVecteur Value :\n");
    for(i=0;i<(size - nb_zeros);i++)
    {
        printf("%i ",Value[i]);
    }
    printf("\n");
    printf("\nMatrice creuse stockée en format CSR:\n");
    for (i=0;i<l;i++)
    {
        for (j=0;j<c;j++)
        {
            printf("%i ",get_sparce_matrix_value_int(i, j, Row, Column, Value, size - nb_zeros, l, c));
        }
        printf("\n");
    }
    
    VectSum_Column = (int *)malloc(c * sizeof(int));
    fill_matrix_column_sum_vector(VectSum_Column, Column, Value, size - nb_zeros, c);
    printf("\nVecteur VectSum_Column (somme des éléments colonne par colonne:\n");
    for(i=0;i<c;i++)
    {
        printf("%i ",VectSum_Column[i]);
    }
    printf("\n");
    
    NormValue = (double *)malloc((size - nb_zeros) * sizeof(double));
    //copie du vecteur Value dans NormValue
    for(i=0;i<(size - nb_zeros);i++)
    {
        NormValue[i] = (double) Value[i];
    }
    
    //normalisation de la matrice (Row, Column, Value) dans (Row, Column, NormValue)
    normalize_matrix(VectSum_Column, Column, NormValue, size - nb_zeros, c);
    
    printf("\nMatrice normalisée (stockée en format CSR):\n");
    for (i=0;i<l;i++)
    {
        for (j=0;j<c;j++)
        {
            printf("%.2f ",get_sparce_matrix_value_double(i, j, Row, Column, NormValue, size - nb_zeros, l, c));
        }
        printf("\n");
    }
}
