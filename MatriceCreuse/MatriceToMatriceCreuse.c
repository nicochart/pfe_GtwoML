/*Changement de stockage d'une matrice carrée à n elements en (stockée normalement) en stockage pour matrice creuse*/
/*Nicolas HOCHART*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float random_between_0_and_1()
{
    return (float) rand() / (float) RAND_MAX;
}

void init_matrix(double *M, int n)
{
    /*Rempli la matrice M de taille n*n de nombres. Il y a une chance sur deux que le nombre soit 0. Statistiquement, la moitier de la matrice contient des 0*/
    int i,j,tmp;
    
    for (i=0;i<n;i++)
    {
        for (j=0;j<n;j++)
        {
            if (random_between_0_and_1() > 0.5) //une chance sur deux de mettre un 0
            {
                *(M + i*n+j) = 0;
            }
            else
            {
                tmp = random_between_0_and_1() * 1000;
                *(M + i*n+j) = (float)tmp/10;
            }
        }
    }
}

int cpt_nb_zeros_matrix(double *M, int size)
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

void fill_sparce_matrix(double *M, int *Row, int *Column, double *Value, int n)
{
    int i,j,nb = 0;
    for (i=0;i<n;i++)
    {
        *(Row+i) = nb;
        for (j=0;j<n;j++)
        {
            if (*(M + i*n+j) != 0)
            {
                *(Column+nb) = j;
                *(Value+nb) = *(M+i*n+j);
                nb++;
            }
        }
    }
}

int main(int argc, char **argv)
{
    int i,j; //pour les boucles
    int n,size,nb_zeros;
    double *A;
    
    //variables de la matrice creuse
    int *Row,*Column;
    double *Value;
    
    if (argc < 2)
    {
        printf("Veuillez entrer la taille de la matrice après le nom de l'executable : %s [n]\n", argv[0]);
        exit(1);
    }
    
    n = atoll(argv[1]);
    size = n * n;
    A = (double *)malloc(size * sizeof(double));
    
    random_between_0_and_1();
    init_matrix(A, n);
    
    for (i=0;i<n;i++)
    {
        for (j=0;j<n;j++)
        {
            printf("%f ",A[i*n+j]);
        }
        printf("\n");
    }
    
    nb_zeros = cpt_nb_zeros_matrix(A, size);
    printf("Nombre de zeros : %i\n",nb_zeros); //devrait être environs égal à n*n / 2
    printf("Nombre de valeurs non nulles : %i\n",size - nb_zeros);
    
    Row = (int *)malloc(n * sizeof(int));
    Column = (int *)malloc((size - nb_zeros) * sizeof(int));
    Value = (double *)malloc((size - nb_zeros) * sizeof(double));
    fill_sparce_matrix(A, Row, Column, Value, n);
    
    printf("Vecteur Row :\n");
    for(i=0;i<n;i++)
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
        printf("%f ",Value[i]);
    }
    printf("\n");
}
