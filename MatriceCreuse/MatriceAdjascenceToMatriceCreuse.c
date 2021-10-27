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

void init_matrix(int *M, int l, int c)
{
    /*Rempli la matrice M de taille l*c de nombres. Il y a une chance sur deux que le nombre soit 0. Statistiquement, la moitier de la matrice contient des 0*/
    int i,j;
    
    for (i=0;i<l;i++)
    {
        for (j=0;j<c;j++)
        {
            if (random_between_0_and_1() > 0.5) //une chance sur deux de mettre un 0
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

void init_row_matrix(int *M, int i, int n)
{
    /*Rempli n éléments de la ligne i de la matrice M. Il y a une chance sur deux que le nombre soit 0. Statistiquement, la moitier de la ligne sont des 0*/
    int j;

    for (j=0;j<n;j++)
    {
        if (random_between_0_and_1() > 0.5) //une chance sur deux de mettre un 0
        {
            *(M + i*n+j) = 0;
        }
        else
        {
            *(M + i*n+j) = 1;
        }
    }
}

int cpt_nb_zeros_matrix(int *M, int size)
{
    /*Compte le nombre de 0 dans la matrice M à size elements*/
    int compteur = 0;
    for (int d=0;d<size;d++)
    {
        if (*(M+d) == 0)
        {
            compteur++;
        }
    }
    return compteur;
}

void fill_sparce_matrix(int *M, int *Row, int *Column, int *Value, int l, int c)
{
    /*
    Traduit la matrice stockée dans M (de taille l*c) en matrice creuse dans les vecteurs Row, Column et Value
    Le vecteur Row est de taille n+1 (nombre de lignes + 1)
    Les vecteurs Column (indices de colonne) et Value (valeur) sont de taille "nombre d'éléments non nulles dans la matrice".
    */
    int i,j,nb = 0;
    for (i=0;i<l;i++)
    {
        *(Row+i) = nb;
        for (j=0;j<c;j++)
        {
            //printf("i,j = %i,%i : ",i,j);
            if (*(M + i*c+j) != 0)
            {
                //printf("Valeur trouvée à l'indice %i*%i+%i = %i : %i écrit à l'indice %i\n",i,l,j,i*l+j,*(M+i*c+j),nb);
                *(Column+nb) = j;
                *(Value+nb) = *(M+i*c+j);
                nb++;
            }
        }
    }
    *(Row+l) = nb;
}

int get_sparce_matrix_value(int indl, int indc, int *Row, int *Column, int *Value, int len_values, int l, int c)
{
    /*Renvoie la valeur [indl,indc] de la matrice creuse stockée dans Row,Column,Value. len_values est la longueur du vecteur Value. l le nombre de lignes de la matrice (longueur du vecteur Row - 1) et c le nombre de colonnes.*/
    if (indl >= l || indc >= c)
    {
        perror("ATTENTION : des indices incohérents ont été fournis dans la fonction get_sparce_matrix_value()\n");
        return -1;
    }
    int i;
    int nb_values = Row[indl+1] - Row[indl]; //nombre de valeurs dans la ligne
    for (i=Row[indl];i<Row[indl]+nb_values;i++)
    {
        if (Column[i] == indc)
        {
            return Value[i];
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    int i,j; //pour les boucles
    int l,c,size,nb_zeros;
    int *A;
    
    //variables de la matrice creuse
    int *Row,*Column;
    int *Value;
    
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
        init_row_matrix(A, i, c);
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
    printf("\nMatrice creuse stockée de manière économique:\n");
    for (i=0;i<l;i++)
    {
        for (j=0;j<c;j++)
        {
            printf("%i ",get_sparce_matrix_value(i, j, Row, Column, Value, size - nb_zeros, l, c));
        }
        printf("\n");
    }
    printf("\n");
}
