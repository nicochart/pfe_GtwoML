/*Démonstration d'utilisation des structures représentant un Cerveau*/
/*Nicolas HOCHART*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/*----------------------------------------------------------------------
--- Structure contenant les informations d'un cerveau et ses parties ---
----------------------------------------------------------------------*/

struct BrainPart
{
     int nbTypeNeuron;
     double * repartitionNeuronCumulee; //taille nbTypeNeuron
     double * probaConnection; //taille (ligne) nbTypeNeuron * (colonne) nb_part
};
typedef struct BrainPart BrainPart;

struct Brain
{
     int dimension; //nombre de neurones total
     int nb_part; //nombre de parties
     int * parties_cerveau; //taille nb_part - indices de 0 à n (dimension de la matrice) auxquels commencent les parties du cerveau
     BrainPart * brainPart; //taille nb_part - adresse d'un vecteur de pointeurs vers des BrainPart.
};
typedef struct Brain Brain;

int main(int argc, char **argv)
{
    int i,j,k; //boucles
    int nb_part;
    int n=16;
    printf("Entrez le nombre de parties que contiendra votre cerveau : ",i);
    scanf("%d", &nb_part);

    int * part_cerv = (int *)malloc(nb_part * sizeof(int));
    int ind_brainpart;
    printf("Entrez les indices auxquels chaque partie du cerveau vont commencer (0 à n=%i croissants):\n",n);
    for (i=0;i<nb_part;i++)
    {
        printf("Partie d'indice %i : ",i);
        scanf("%d", &ind_brainpart);
        part_cerv[i] = ind_brainpart;
    }
    int nbTypeNeuron;
    double probacumulee;

    BrainPart brainPart[nb_part];

    for (i=0; i<nb_part; i++)
    {
        printf("------------------\nPartie d'indice %i du cerveau :\n",i);
        printf("Veuillez entrer le nombre de types de neurones pour cette partie : ",i);
        scanf("%d", &nbTypeNeuron);

        double * repNCumulee = (double *)malloc(nbTypeNeuron * sizeof(double));
        double * probCo = (double *)malloc(nbTypeNeuron * nb_part * sizeof(double));
        printf("Entrez la proba cumulée d'appartenir à un type de neurone :\n");
        for (j=0;j<nbTypeNeuron;j++)
        {
            printf("type d'indice %i : ",j);
            scanf("%lf", &probacumulee);
            repNCumulee[j] = probacumulee;
        }
        for (j=0;j<nbTypeNeuron;j++)
        {
            printf("Connections du type de neurone d'indice %i :\n",j);
            for (k=0;k<nb_part;k++)
            {
                printf("Entrez la proba d'avoir une connection %i -> %i avec neurone de type %i : ",i,k,j);
                scanf("%lf", &probacumulee);
                probCo[j*nb_part + k] = probacumulee;
            }
        }

        brainPart[i].nbTypeNeuron = nbTypeNeuron;
        brainPart[i].repartitionNeuronCumulee = repNCumulee;
        brainPart[i].probaConnection = probCo;
    }

    Brain Cerveau;
    Cerveau.dimension = n;
    Cerveau.nb_part = nb_part;
    Cerveau.parties_cerveau = part_cerv;
    Cerveau.brainPart = brainPart;

    printf("\n#############\nRecap de votre cerveau :\n");

    printf("Taille : %i*%i\nNombre de parties : %i\nIndices auxquelles commencent les parties : [",Cerveau.dimension,Cerveau.dimension,Cerveau.nb_part);
    for (i=0; i<Cerveau.nb_part; i++)
    {
        printf("%i ",Cerveau.parties_cerveau[i]);
    }
    printf("]\n\n");
    for (i=0; i<Cerveau.nb_part; i++)
    {
        printf("\n");
        printf("Partie %i :\n\tNombre de types de neurones : %i\n\t",i,Cerveau.brainPart[i].nbTypeNeuron);
        printf("Probabilités cumulées d'appartenir à chaque type de neurone : [");
        for (j=0;j<Cerveau.brainPart[i].nbTypeNeuron;j++)
        {
            printf("%lf ",Cerveau.brainPart[i].repartitionNeuronCumulee[j]);
        }
        printf("]\n\tConnections :\n\t");
        for (j=0;j<Cerveau.brainPart[i].nbTypeNeuron;j++)
        {
            printf("Connections du type de neurone d'indice %i :\n\t",j);
            for (k=0;k<Cerveau.nb_part;k++)
            {
                printf("%i -> %i : %lf\n\t",i,k,Cerveau.brainPart[i].probaConnection[j*Cerveau.nb_part + k]);
            }
        }
    }

    printf("\n");
    return 0;
}
