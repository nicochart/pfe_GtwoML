/*Hard-coded brain, to test the generator or to apply PageRank on a matrix generated with it*/
/*Nicolas HOCHART*/

#define hardbrain

#ifndef brainstruct
#include "../../BrainMatrixGenerator/brainstruct.h"
#endif

//Hard-coded brain (number of parts = 8, equal number of neurons in each part)
Brain get_hard_brain(long n)
{
    int i,nb_part=8;
    BrainPart * brainPart = (BrainPart *)malloc(nb_part * sizeof(BrainPart));
    long * part_cerv = (long *)malloc(nb_part * sizeof(long));
    long nb_neurone_par_partie = n / nb_part;
    if (nb_part * nb_neurone_par_partie != n) {printf("Veuillez entrer un n multiple de %i svp\n",nb_part); exit(1);}
    for (i=0; i<nb_part; i++)
    {
        part_cerv[i] = i*nb_neurone_par_partie;
    }
    //partie 1
    double tmp_repNCumulee1[2] = {0.5, 1};
    double * repNCumulee1 = (double *)malloc(2 * sizeof(double)); for (i=0;i<2;i++) {repNCumulee1[i] = tmp_repNCumulee1[i];}
    double tmp_probCo1[16] = {/*type 1*/0.1, 0.4, 0.4, 0.5, 0.4, 0.4, 0.5, 0.4, /*type 2*/0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.1, 0.05};
    double * probCo1 = (double *)malloc(16 * sizeof(double)); for (i=0;i<16;i++) {probCo1[i] = tmp_probCo1[i];}
    brainPart[0].nbTypeNeuron = 2;
    brainPart[0].repartitionNeuronCumulee = repNCumulee1;
    brainPart[0].probaConnection = probCo1;
    //partie 2
    double tmp_repNCumulee2[1] = {1};
    double * repNCumulee2 = (double *)malloc(1 * sizeof(double)); repNCumulee2[0] = tmp_repNCumulee2[0];
    double tmp_probCo2[8] = {/*type 1*/0.4, 0.1, 0.4, 0.5, 0.4, 0.4, 0.5, 0.4};
    double * probCo2 = (double *)malloc(8 * sizeof(double)); for (i=0;i<8;i++) {probCo2[i] = tmp_probCo2[i];}
    brainPart[1].nbTypeNeuron = 1;
    brainPart[1].repartitionNeuronCumulee = repNCumulee2;
    brainPart[1].probaConnection = probCo2;
    //partie 3
    double tmp_repNCumulee3[2] = {0.7, 1};
    double * repNCumulee3 = (double *)malloc(2 * sizeof(double)); for (i=0;i<2;i++) {repNCumulee3[i] = tmp_repNCumulee3[i];}
    double tmp_probCo3[16] = {/*type 1*/0.1, 0.5, 0.4, 0.5, 0.4, 0.4, 0.5, 0.4, /*type 2*/0.6, 0.05, 0.1, 0.1, 0.1, 0.05, 0.1, 0.05};
    double * probCo3 = (double *)malloc(16 * sizeof(double)); for (i=0;i<16;i++) {probCo3[i] = tmp_probCo3[i];}
    brainPart[2].nbTypeNeuron = 2;
    brainPart[2].repartitionNeuronCumulee = repNCumulee3;
    brainPart[2].probaConnection = probCo3;
    //partie 4
    double tmp_repNCumulee4[3] = {0.5, 0.9, 1};
    double * repNCumulee4 = (double *)malloc(3 * sizeof(double)); for (i=0;i<3;i++) {repNCumulee4[i] = tmp_repNCumulee4[i];}
    double tmp_probCo4[24] = {/*type 1*/0.4, 0.5, 0.5, 0.1, 0.4, 0.4, 0.5, 0.4, /*type 2*/0.6, 0.5, 0.2, 0.5, 0.6, 0.5, 0.45, 0.0, /*type 3*/0.6, 0.05, 0.1, 0.1, 0.1, 0.05, 0.1, 0.05};
    double * probCo4 = (double *)malloc(24 * sizeof(double)); for (i=0;i<24;i++) {probCo4[i] = tmp_probCo4[i];}
    brainPart[3].nbTypeNeuron = 3;
    brainPart[3].repartitionNeuronCumulee = repNCumulee4;
    brainPart[3].probaConnection = probCo4;
    //partie 5
    double tmp_repNCumulee5[1] = {1};
    double * repNCumulee5 = (double *)malloc(1 * sizeof(double)); repNCumulee5[0] = tmp_repNCumulee5[0];
    double tmp_probCo5[8] = {/*type 1*/0.4, 0.4, 0.4, 0.5, 0.1, 0.4, 0.5, 0.4};
    double * probCo5 = (double *)malloc(8 * sizeof(double)); for (i=0;i<8;i++) {probCo5[i] = tmp_probCo5[i];}
    brainPart[4].nbTypeNeuron = 1;
    brainPart[4].repartitionNeuronCumulee = repNCumulee5;
    brainPart[4].probaConnection = probCo5;
    //partie 6
    double tmp_repNCumulee6[1] = {1};
    double * repNCumulee6 = (double *)malloc(1 * sizeof(double)); repNCumulee6[0] = tmp_repNCumulee6[0];
    double tmp_probCo6[8] = {/*type 1*/0.4, 0.4, 0.4, 0.55, 0.4, 0.1, 0.6, 0.4};
    double * probCo6 = (double *)malloc(8 * sizeof(double)); for (i=0;i<8;i++) {probCo6[i] = tmp_probCo6[i];}
    brainPart[5].nbTypeNeuron = 1;
    brainPart[5].repartitionNeuronCumulee = repNCumulee6;
    brainPart[5].probaConnection = probCo6;
    //partie 7
    double tmp_repNCumulee7[1] = {1};
    double * repNCumulee7 = (double *)malloc(1 * sizeof(double)); repNCumulee7[0] = tmp_repNCumulee7[0];
    double tmp_probCo7[8] = {/*type 1*/0.4, 0.2, 0.4, 0.6, 0.4, 0.4, 0.05, 0.4};
    double * probCo7 = (double *)malloc(8 * sizeof(double)); for (i=0;i<8;i++) {probCo7[i] = tmp_probCo7[i];}
    brainPart[6].nbTypeNeuron = 1;
    brainPart[6].repartitionNeuronCumulee = repNCumulee7;
    brainPart[6].probaConnection = probCo7;
    //partie 8
    double tmp_repNCumulee8[2] = {0.3, 1};
    double * repNCumulee8 = (double *)malloc(2 * sizeof(double)); for (i=0;i<2;i++) {repNCumulee8[i] = tmp_repNCumulee8[i];}
    double tmp_probCo8[16] = {/*type 1*/0.05, 0.05, 0.1, 0.05, 0.2, 0.1, 0.1, 0.5, /*type 2*/0.4, 0.5, 0.4, 0.6, 0.4, 0.4, 0.05, 0.1};
    double * probCo8 = (double *)malloc(16 * sizeof(double)); for (i=0;i<16;i++) {probCo8[i] = tmp_probCo8[i];}
    brainPart[7].nbTypeNeuron = 2;
    brainPart[7].repartitionNeuronCumulee = repNCumulee8;
    brainPart[7].probaConnection = probCo8;

    Brain Cerveau;
    Cerveau.dimension = n;
    Cerveau.nb_part = nb_part;
    Cerveau.parties_cerveau = part_cerv;
    Cerveau.brainPart = brainPart;

    return Cerveau;
}
