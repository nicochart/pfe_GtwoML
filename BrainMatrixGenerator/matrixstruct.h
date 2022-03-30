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

void printf_csr_matrix_int_maxdim(IntCSRMatrix * M, int max_dim)
{
    /*
    Imprime la matrice d'entiers stockée en format CSR passée en paramètre
    Si une des dimension de la matrice dépasse max_dim, le message "trop grande pour être affichée" est affiché
    */
    long i,j;

    if ((*M).dim_l <= max_dim && (*M).dim_c <=max_dim) //si l'une des dimensions de la matrice est supérieure à max_dim, on n'affiche pas la matrice
    {
        for (i=0;i<(*M).dim_l;i++)
        {
            for (j=0;j<(*M).dim_c;j++)
            {
                printf("%i ", get_csr_matrix_value_int(i, j, M));
            }
            printf("\n");
        }
    }
    else
    {
        printf("Matrice trop grande pour être affichée..\n");
    }
}

void printf_csr_matrix_int(IntCSRMatrix * M)
{
    /*
    Imprime la matrice d'entiers stockée en format CSR passée en paramètre
    Si une des dimension de la matrice dépasse 32, le message "trop grande pour être affichée" est affiché
    */
    printf_csr_matrix_int_maxdim(M, 32);
}

MatrixBlock fill_matrix_block_info(int rank, int nb_blocks_row, int nb_blocks_column, long n)
{
    /*
    Rempli une structure MatrixBlock en fonction des paramètres reçu, et la retourne
    Les informations remplies sont uniquement les informations basiques
    */
    struct MatrixBlock MBlock;
    MBlock.indl = rank / nb_blocks_column; //indice de ligne dans la grille 2D de processus
    MBlock.indc = rank % nb_blocks_column; //indice de colonne dans la grille 2D de processus
    MBlock.dim_l = n/nb_blocks_row; //nombre de lignes dans un block
    MBlock.dim_c = n/nb_blocks_column; //nombre de colonnes dans un block
    MBlock.startRow = MBlock.indl*MBlock.dim_l;
    MBlock.endRow = (MBlock.indl+1)*MBlock.dim_l-1;
    MBlock.startColumn = MBlock.indc*MBlock.dim_c;
    MBlock.endColumn = (MBlock.indc+1)*MBlock.dim_c-1;
    return MBlock;
}

MatrixBlock fill_matrix_block_info_adjacency_prv_pagerank(int rank, int nb_blocks_row, int nb_blocks_column, long n)
{
    /*
    Rempli une structure MatrixBlock en fonction des paramètres reçu, et la retourne
    La structure est remplie pour un PageRank avec vecteur résultat réparti, appliqué à une matrice d'adjacence
    Les explications sur ce remplissage sont disponible dans le pdf "explication du PageRank version 6"
    */
    int pgcd_nbr_nbc, local_result_vector_size_row_blocks, local_result_vector_size_column_blocks;
    long local_result_vector_size_column;
    double grid_dim_factor;

    int tmp_r = nb_blocks_row, tmp_c = nb_blocks_column;
    while (tmp_c!=0) {pgcd_nbr_nbc = tmp_r % tmp_c; tmp_r = tmp_c; tmp_c = pgcd_nbr_nbc;}
    pgcd_nbr_nbc = tmp_r;

    struct MatrixBlock MBlock = fill_matrix_block_info(rank, nb_blocks_row, nb_blocks_column, n);

    grid_dim_factor = (double) nb_blocks_row / (double) nb_blocks_column;
    MBlock.pr_result_redistribution_root = (int) MBlock.indl / grid_dim_factor;
    local_result_vector_size_column_blocks = nb_blocks_column /pgcd_nbr_nbc;
    local_result_vector_size_row_blocks = nb_blocks_row / pgcd_nbr_nbc;
    local_result_vector_size_column = local_result_vector_size_column_blocks * MBlock.dim_c;
    MBlock.result_vector_calculation_group = MBlock.indc / local_result_vector_size_column_blocks;
    MBlock.indc_in_result_vector_calculation_group = MBlock.indc % local_result_vector_size_column_blocks;
    MBlock.inter_result_vector_need_group_communicaton_group = (MBlock.indl % local_result_vector_size_row_blocks) * nb_blocks_column + MBlock.indc;
    MBlock.startColumn_in_result_vector_calculation_group = MBlock.dim_c * MBlock.indc_in_result_vector_calculation_group;
    MBlock.startRow_in_result_vector_calculation_group = MBlock.dim_l * MBlock.indl % local_result_vector_size_row_blocks;
    MBlock.my_result_vector_calculation_group_rank = MBlock.indc_in_result_vector_calculation_group + MBlock.indl * local_result_vector_size_column_blocks;
    return MBlock;
}
