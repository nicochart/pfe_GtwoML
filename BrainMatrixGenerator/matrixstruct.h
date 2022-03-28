/* Structures permettant de stocker les matrices et les informations sur les matrices, en COO et CSR, et fonctions permettant de faire des opérations sur les matrices */
/* à compléter */
/*Nicolas HOCHART*/

#define matrixstruct

#include <stdio.h>
#include <stdlib.h>
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
     int result_vector_calculation_group; //Indice de groupe de calcul du vecteur résultat
     int indc_in_result_vector_calculation_group; //Indice de colonne du block dans le groupe de calcul du vecteur résultat (groupes de blocks colonnes)
     int inter_result_vector_need_group_communicaton_group; //Indice du Groupe de communication inter-groupe de besoin (utile pour récupérer le résultat final)
     long startColumn_in_result_vector_calculation_group; //Indice de départ en colonne dans le groupe de calcul du vecteur résultat (inclu)
     long startRow_in_result_vector_calculation_group; //Indice de départ en ligne dans le groupe de calcul du vecteur résultat (inclu), utile dans le PageRank pour aller chercher des valeurs dans le vecteur q
     int my_result_vector_calculation_group_rank; //my_rank dans le groupe de calcul du vecteur résultat
};
typedef struct MatrixBlock MatrixBlock;

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
