struct BrainPart{
	int nbTypeNeuron;
	double * repartitionNeuronCumulee;
	double * probaConnection;
};
typedef struct BrainPart BrainPart;

struct Brain{
	long dimension;
	int nb_part;
	long * parties_cerveau;
	BrainPart * brainPart;
};
typedef struct Brain Brain;

int get_nb_part(){
	return 4;
}

void destructeurBrain(Brain *Cerveau){
	free(Cerveau->brainPart[0].repartitionNeuronCumulee);
	free(Cerveau->brainPart[0].probaConnection);
	free(Cerveau->brainPart[1].repartitionNeuronCumulee);
	free(Cerveau->brainPart[1].probaConnection);
	free(Cerveau->brainPart[2].repartitionNeuronCumulee);
	free(Cerveau->brainPart[2].probaConnection);
	free(Cerveau->brainPart[3].repartitionNeuronCumulee);
	free(Cerveau->brainPart[3].probaConnection);
	free(Cerveau->parties_cerveau);
	free(Cerveau->brainPart);
}

void paramBrain(Brain *Cerveau, long *n){
	int nbTypeNeuronIci,nb_part=4;
	*n=40;
	BrainPart *brainPart = malloc(sizeof(BrainPart)*nb_part);
	long *part_cerv = malloc(sizeof(long)*nb_part);
	long nb_neurone_par_partie = *n / nb_part;
	if (nb_part * nb_neurone_par_partie != *n) {printf("Erreur nbPart*nb_neurone_par_partie!=n(numberTotalofNeuron)\n"); exit(1);}
	for (int i=0; i<nb_part; i++){part_cerv[i] = i*nb_neurone_par_partie;}

//partie 0
	brainPart[0].nbTypeNeuron = 2;
	brainPart[0].repartitionNeuronCumulee = malloc(sizeof(double)*2);
	brainPart[0].repartitionNeuronCumulee[0] = 0.7;
	brainPart[0].repartitionNeuronCumulee[1] = 1.0;
	brainPart[0].probaConnection = malloc(sizeof(double)*8);
	brainPart[0].probaConnection[0] = 0.12800000000000003;
	brainPart[0].probaConnection[1] = 0.03200000000000001;
	brainPart[0].probaConnection[2] = 0.03200000000000001;
	brainPart[0].probaConnection[3] = 0.008000000000000002;
	brainPart[0].probaConnection[4] = 0.32000000000000006;
	brainPart[0].probaConnection[5] = 0.08000000000000002;
	brainPart[0].probaConnection[6] = 0.08000000000000002;
	brainPart[0].probaConnection[7] = 0.020000000000000004;
//partie 1
	brainPart[1].nbTypeNeuron = 1;
	brainPart[1].repartitionNeuronCumulee = malloc(sizeof(double)*1);
	brainPart[1].repartitionNeuronCumulee[0] = 1.0;
	brainPart[1].probaConnection = malloc(sizeof(double)*4);
	brainPart[1].probaConnection[0] = 0.24000000000000005;
	brainPart[1].probaConnection[1] = 0.7200000000000001;
	brainPart[1].probaConnection[2] = 0.06000000000000001;
	brainPart[1].probaConnection[3] = 0.18000000000000002;
//partie 2
	brainPart[2].nbTypeNeuron = 2;
	brainPart[2].repartitionNeuronCumulee = malloc(sizeof(double)*2);
	brainPart[2].repartitionNeuronCumulee[0] = 0.7;
	brainPart[2].repartitionNeuronCumulee[1] = 1.0;
	brainPart[2].probaConnection = malloc(sizeof(double)*8);
	brainPart[2].probaConnection[0] = 0.03200000000000001;
	brainPart[2].probaConnection[1] = 0.008000000000000002;
	brainPart[2].probaConnection[2] = 0.12800000000000003;
	brainPart[2].probaConnection[3] = 0.03200000000000001;
	brainPart[2].probaConnection[4] = 0.08000000000000002;
	brainPart[2].probaConnection[5] = 0.020000000000000004;
	brainPart[2].probaConnection[6] = 0.32000000000000006;
	brainPart[2].probaConnection[7] = 0.08000000000000002;
//partie 3
	brainPart[3].nbTypeNeuron = 1;
	brainPart[3].repartitionNeuronCumulee = malloc(sizeof(double)*1);
	brainPart[3].repartitionNeuronCumulee[0] = 1.0;
	brainPart[3].probaConnection = malloc(sizeof(double)*4);
	brainPart[3].probaConnection[0] = 0.06000000000000001;
	brainPart[3].probaConnection[1] = 0.18000000000000002;
	brainPart[3].probaConnection[2] = 0.24000000000000005;
	brainPart[3].probaConnection[3] = 0.7200000000000001;
	Cerveau->dimension = *n;
	Cerveau->nb_part = nb_part;
	Cerveau->parties_cerveau = part_cerv;
	Cerveau->brainPart = brainPart;
}