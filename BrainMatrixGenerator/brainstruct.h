//! Brain structures
/*!
  This file defines the structures and functions related to the Brains used by the brain-matrix generator
  Nicolas HOCHART
*/

#define brainstruct

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>

#ifndef randomforbrain
#include "randomforbrain.h"
#endif

#define NULL ((void *)0)

/*----------------------------------------------------------------------
--- Structure contenant les informations d'un cerveau et ses parties ---
----------------------------------------------------------------------*/

//! Structure containing the information of a part of the brain
/*!
   Structure containing the information of a part of the brain. It contains :
   The number of neuron types that can be encountered in the part, the cumulative distribution of neurons in the neuron types, and the probability of connection (for each type) to other parts of the brain.
   This structure depends on the brain to which it belongs.
 */
struct BrainPart
{
     int nbTypeNeuron;
     double * repartitionNeuronCumulee; //taille nbTypeNeuron
     double * probaConnection; //taille (ligne) nbTypeNeuron * (colonne) nb_part
};
typedef struct BrainPart BrainPart;

//! Structure containing the information of a brain
/*!
   Structure containing the information of a brain. It contains :
   The total number of neurons, the number of parts in the brain, the indices (neurons) at which parts start, the brain parts (see BrainPart structure).
 */
struct Brain
{
     long dimension; //nombre de neurones total
     int nb_part; //nombre de parties
     long * parties_cerveau; //taille nb_part - indices de 0 à n (dimension de la matrice) auxquels commencent les parties du cerveau
     BrainPart * brainPart; //taille nb_part - adresse d'un vecteur de pointeurs vers des BrainPart.
};
typedef struct Brain Brain;

/*---------------------------------
--- Opérations sur les cerveaux ---
---------------------------------*/

//! Function that returns the index of the part of the brain in which a neuron is located
/*!
   Function that returns the index of the part of the brain "brain" in which the neuron of index "ind" is located
 * @param[in] ind {long} index of the neuron
 * @param[in] brain {Brain *} pointer to a brain
 * @return i {int} index of the brain part
   Condition : "brain" is a well formed brain and "ind" is assumed to be between 0 and brain.dimension
 */
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

//! Function that returns the number of neurons in a specific brain part
/*!
   Function that returns the number of neurons in the brain part if index "part", in the brain "brain"
 * @param[in] ind {long} index of the neuron
 * @param[in] brain {Brain *} pointer to a brain
 * @return {long} number of neurons in the brain part
 */
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

//! Function that returns the average percentage of connection chances for a specific neuron to the other parts
/*!
   Function that returns the average percentage (between 0 and 100) of connection chances for a neuron of a given type in a given part, to the other parts
 * @param[in] brain {Brain *} pointer to a brain
 * @param[in] part {int} brain part index
 * @param[in] type {int} neuron index
 * @return {double} average percentage of connection chance
 */
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

//! Choose and returns the neuron type for a neuron in a specitic brain part
/*!
   Choose and returns a neuron type for a neuron in the brain part of index "part", in the brain "brain"
 * @param[in] brain {Brain *} pointer to a brain
 * @param[in] part {int} brain part index
 * @return {int} neuron type
 */
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

//! Function that generates (chooses) the types of multiple neurons and writes them in "types"
/*!
   Generates the types of neurons from index "ind_start_neuron" to "ind_start_neuron + nb_neuron" in the brain "brain", and writes them in "types"
 * @param[in] brain {Brain *} pointer to a brain
 * @param[in] part {int} brain part index
 * @return {int} neuron type
   Condition : The memory allocation (malloc of size nb_neuron * sizeof(int)) for "types" must be done beforehand.
 */
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

//! Brain print
/*!
   Function that displays a summary of the brain passed as a parameter (useful for debugging a brain)
 * @param[in] brain {Brain *} pointer to a brain
 */
void printf_recap_brain(Brain * brain)
{
    int i,j,k;
    /*affiche un récapitulatif du cerveau passé en paramètre.*/
    printf("\n#############\nRecap de votre cerveau :\n");

    printf("Taille : %i*%i\nNombre de parties : %i\nIndices auxquelles commencent les parties : [",(*brain).dimension,(*brain).dimension,(*brain).nb_part);
    for (i=0; i<(*brain).nb_part; i++)
    {
        printf("%i ",(*brain).parties_cerveau[i]);
    }
    printf("]\n\n");
    for (i=0; i<(*brain).nb_part; i++)
    {
        printf("\n");
        printf("Partie %i :\n\tNombre de types de neurones : %i\n\t",i,(*brain).brainPart[i].nbTypeNeuron);
        printf("Probabilités cumulées d'appartenir à chaque type de neurone : [");
        for (j=0;j<(*brain).brainPart[i].nbTypeNeuron;j++)
        {
            printf("%lf ",(*brain).brainPart[i].repartitionNeuronCumulee[j]);
        }
        printf("]\n\tConnexions :\n\t");
        for (j=0;j<(*brain).brainPart[i].nbTypeNeuron;j++)
        {
            printf("Connexions du type de neurone d'indice %i :\n\t",j);
            for (k=0;k<(*brain).nb_part;k++)
            {
                printf("%i -> %i : %lf\n\t",i,k,(*brain).brainPart[i].probaConnection[j*(*brain).nb_part + k]);
            }
        }
    }
    printf("\n");
}

//! Brain destructor
/*!
   Function that frees a brain
 * @param[in] brain {Brain *} pointer to a brain
 */
void free_brain(Brain * brain)
{
    /*libère les ressources allouées dans un cerveau. Ne libère/détruit pas le cerveau.*/
    for (int i=0; i<(*brain).nb_part; i++)
    {
        free((*brain).brainPart[i].repartitionNeuronCumulee);
        free((*brain).brainPart[i].probaConnection);
    }
    free((*brain).brainPart);
    free((*brain).parties_cerveau);
}
