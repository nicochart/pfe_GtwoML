/* Méthode de la puissance appliquée à une matrice d'adjascence générée aléatoirement */
/* N. Hochart */ 
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <sys/time.h>

#define PRNG_MAX 0x0007FFFFFFFll
#define PRNG_1   0x00473EE661Dll
#define PRNG_2   0x024719D0275ll
#define RANGE    101

double my_gettimeofday()
{
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

float random_between_0_and_1()
{
    /*Renvoie un nombre aléatoire entre 0 et 1. Permet de faire une décision aléatoire*/
    return (float) rand() / (float) RAND_MAX;
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

void fill_sparce_matrix(int *M, int *Row, int *Column, int *Value, int n)
{
    /*
    Traduit la matrice stockée dans M (de taille n*n) en matrice creuse dans les recteurs Row, Column et Value
    Le vecteur Row est de taille n (nombre de lignes)
    Les vecteurs Column (indices de colonne) et Value (valeur) sont de taille "nombre d'éléments non nulles dans la matrice".
    */
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
    int my_rank, p, valeur, tag = 0;
    MPI_Status status;
  
    /* Initialisation MPI */
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  
    int debug = 0; //mettre à 1 pour un affichage plus détaillé qui permet de voir si les Bcast fonctionnent correctement
    long i, j, n;
    long long size;
    double norm, error, start_time, total_time, delta;
    int *morceauA;
    double *X, *Y, *tmp;
    int n_iterations;
    FILE *output;

    if (argc < 2)
    {
        printf("USAGE: %s [n]\n", argv[0]);
        exit(1);
    }

    n = atoll(argv[1]);
    size = n * n * sizeof(int);
    if (my_rank==0)
    {
        printf("taille totale de la matrice : %.3f G\n", size / 1073741824.);
    }
    int nb_ligne = n/p; //nombre de lignes par bloc
    int count = nb_ligne*n; //nombre d’éléments par bloc
    printf("taille de la matrice dans le processus %i : %.3f G\n", my_rank, count * sizeof(int)/ 1073741824.);
    
    double somme_carres,somme_carres_total,sc,norm2Ax,inv_norm2Ax;; //variables utilisées dans le code
    double morceau_Ax[nb_ligne];

    /*** allocation de la matrice et des vecteurs ***/
    morceauA = (int *)malloc(count * sizeof(int));
    if (morceauA == NULL)
    {
        fprintf(stderr,"impossible d'allouer le morceau de matrice sur le processus %i",my_rank);
        perror("");
        exit(1);
    }
    
    X = malloc(n * sizeof(double));
    Y = malloc(n * sizeof(double));
    
    if (X == NULL || Y == NULL)
    {
        fprintf(stderr,"impossible d'allouer le vecteur X ou Y sur le processus %i",my_rank);
        exit(1);
    }

    /*** initialisation de x ***/
    
    double nombre_initial = 1.0 / n;
    for (i = 0; i < n; i++)
    {
        X[i] = nombre_initial;
    }
    
    if (debug) {printf("X départ :\n"); for(i=0;i<10;i++) {printf("%f ",X[i]);} printf("...\n");}
    
    
    /* Initialisation de la matrice A dans le processus 0 */
    for (i = my_rank * nb_ligne; i < (my_rank + 1) * nb_ligne; i++)
    {
        init_row_matrix(morceauA - (my_rank * nb_ligne * n), i, n);
    }
    
    if (debug)
    {
        /*Vérification de la matrice complète A*/
        int *A;
        if (my_rank==0)
        {
            A = (int *)malloc(n*n * sizeof(int));
        }
        MPI_Gather(morceauA, count, MPI_INT, A, count,  MPI_INT, 0, MPI_COMM_WORLD);
        if (my_rank==0)
        {
            for (i=0;i<n;i++)
            {
                for (j=0;j<n;j++)
                {
                    printf("%i ",A[i*n+j]);
                }
                printf("\n");
            }
        }
        if (my_rank==0)
        {
            free(A);
        }
    }

    start_time = my_gettimeofday();
    error = INFINITY;
    n_iterations = 0;
    
    while (error > 1e-9)
    {
        if (my_rank == 0)
        {	
            if (debug) {printf("---------------------\n");}
            printf("iteration %4d, erreur actuelle %g\n", n_iterations, error);
            somme_carres = 0;
            for(i=0; i<nb_ligne; i++)
            {
                sc = 0; //scalaire
                for (j=0; j<n; j++)
                {
                    sc += morceauA[i*n+j] * X[j];
                }
                morceau_Ax[i] = sc;
                somme_carres  += sc * sc;
            }
        }
        else
        {
            somme_carres = 0;
            for(i=0; i<nb_ligne; i++)
            {
                sc = 0; //scalaire
                for (j=0; j<n; j++)
                {
                    sc += morceauA[i*n+j] * X[j];
                }
                morceau_Ax[i] = sc;
                somme_carres  += sc * sc;
            }
        }
        
        MPI_Allreduce(&somme_carres, &somme_carres_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); //somme MPI_SUM de tout les somme_carres dans somme_carres_total
        MPI_Allgather(morceau_Ax, nb_ligne, MPI_DOUBLE, Y, nb_ligne,  MPI_DOUBLE, MPI_COMM_WORLD);

        norm2Ax = sqrt(somme_carres_total);
        inv_norm2Ax = 1.0 / norm2Ax;
            
            
        for (i = 0; i < n; i++)
        {
            Y[i] *= inv_norm2Ax;
        }
            
        //error <--- ||x - y||
        error = 0;
        for (i = 0; i < n; i++)
        {
            delta = X[i] - Y[i];
            error += delta * delta;
        }
        error = sqrt(error);
            
        // x <--> y
        tmp = X; X = Y; Y = tmp; 
            
        n_iterations++;
        
        //Debug : Affichage des variables dans le processus 0 et le processus 1
        if (debug)
        {
            if (my_rank == 0 || my_rank == 1)
            {
                printf("X actuel dans le processus %i : ",my_rank);
                for(i=0;i<10;i++) {printf("%g ",X[i]);}
                printf(" ... \n");
                
                printf("Erreur dans le processus %i : %g\n",my_rank,error);
            }
        }
    }
    //fin du while : on a trouvé la valeur propre et l'erreur finale

    
    if (my_rank == 0)
    {
        total_time = my_gettimeofday() - start_time;
        printf("---------------------\nerreur finale après %4d iterations: %g (|VP| = %g)\n", n_iterations, error, norm2Ax);
        printf("time : %.1f s      MFlops : %.1f \n", total_time, (2.0 * n * n + 7.0 * n) * n_iterations / 1048576. / total_time);

        // stocke le vecteur propre dans un fichier
        output = fopen("result.out", "w");
        if (output == NULL)
        {
            perror("impossible d'ouvrir result.out en écriture");
            exit(1);
        }
        fprintf(output, "%ld\n", n);
        for (i = 0; i < n; i++)
        {
            fprintf(output, "%.17g\n", X[i]);
        }
        fclose(output);
    }

    free(X);
    if (my_rank == 0)
    {
        free(Y);
    }
    free(morceauA);
    
    MPI_Finalize();
    return 0;
}
