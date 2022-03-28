/*
Générateur de matrice-cerveau parallèle A (matrice d'adjacence, non transposée) avec des blocs de ligne et colonnes.
Contiendra toutes les versions utilisées du générateur
*/
/*Nicolas HOCHART*/

#define brainmatrixgenerator

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>

#ifndef brainstruct
#include "brainstruct.h"
#endif

#ifndef randomforbrain
#include "randomforbrain.h"
#endif

#ifndef matrixstruct
#include "matrixstruct.h"
#endif

#define NULL ((void *)0)

/*--------------------------------------
--- Génération de la matrice-cerveau ---
--------------------------------------*/

/*V5*/
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

/*V4*/
void generate_csr_brain_transposed_adjacency_matrix_for_pagerank(IntCSRMatrix *M_CSR, MatrixBlock BlockInfo, Brain * brain, int * neuron_types, DebugBrainMatrixInfo * debugInfo)
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
    for (i=0;i<BlockInfo.dim_l;i++) //parcours des lignes
    {
        //récupération de l'indice de la partie (du cerveau) destination
        ind_part_dest = get_brain_part_ind(BlockInfo.startRow+i, brain);
        for (j=0;j<BlockInfo.dim_c;j++) //parcours des colonnes
        {
            //récupération de l'indice de la partie source
            ind_part_source = get_brain_part_ind(BlockInfo.startColumn+j, brain);
            //récupération du type de neurone
            source_type = neuron_types[BlockInfo.startColumn+j];
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
