/* indent -nfbs -i4 -nip -npsl -di0 -nut iterated_seq.c */
/* Auteur: C. Bouillaguet et P. Fortin (Univ. Lille) + N. Hochart (Polytech) code Parallèle */ 
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <sys/time.h>

#define PRNG_MAX 0x0007FFFFFFFll
#define PRNG_1   0x00473EE661Dll
#define PRNG_2   0x024719D0275ll
#define RANGE    101

double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

// Initialise la ligne N°i, à n éléments, de la matrice A : 
void init_ligne(double *A, long i, long n)
{
  for (long j = 0; j < n; j++)
  {
    A[i*n+j] = (((double)((i * i * PRNG_1 + j * j * PRNG_2) & PRNG_MAX)) / PRNG_MAX) / n;
  }
  for (long k = 1; k < n; k *= 2)
  {
    if (i + k < n)
    {
      A[i*n + i+k] = ((i - k) * PRNG_2 + i * PRNG_1) % RANGE;
    }
    if (i - k >= 0)
    {
      A[i*n +i-k] = ((i + k) * PRNG_2 + i * PRNG_1) % RANGE;
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
    double *morceauA, *X, *Y;
    int n_iterations;
    FILE *output;

    if (argc < 2)
    {
        printf("USAGE: %s [n]\n", argv[0]);
        exit(1);
    }
    n = atoll(argv[1]);
    size = n * n * sizeof(double);
    if (my_rank == 0) {printf("taille totale de la matrice : %.3f G\n", size / 1073741824.);}
    long nb_ligne = n/p; //nombre de lignes par bloc
    long count = nb_ligne*n; //nombre d’éléments par bloc

    /*** allocation de la matrice et des vecteurs ***/
    morceauA = (double *)malloc(count * sizeof(double));
    if (morceauA == NULL)
    {
        fprintf(stderr,"impossible d'allouer le morceau de matrice sur le processus %i",my_rank);
        perror("");
        exit(1);
    }
    
    X = malloc(n * sizeof(double));
    
    if (X == NULL)
    {
        perror("impossible d'allouer le vecteur X");
        exit(1);
    }
    
    if (my_rank==0)
    {        
        Y = malloc(n * sizeof(double));
        
        if (Y == NULL)
        {
            perror("impossible d'allouer le vecteur Y");
            exit(1);
        }
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
        init_ligne(morceauA - (my_rank * nb_ligne * n), i, n);
    }
    
    if (my_rank==0)
    {
        printf("A[0] = %g |",morceauA[0]);
        printf("A[1] = %g \n",morceauA[1]);
    }
    if (my_rank==1)
    {
        printf("nbligne = %i\n",nb_ligne);
        printf("A[1048576] = %g |",morceauA[0]);
        printf("A[1048577] = %g\n",morceauA[1]);
    }
    
    double somme_carres,somme_carres_total,sc,inv_norm2Ax;
    double morceau_Ax[nb_ligne];
    
    start_time = my_gettimeofday();
    error = INFINITY;
    n_iterations = 0;
    
    while (error > 1e-9)
    {
        if (my_rank == 0)
        {	
            if (debug) {printf("---------------------\n");}
            printf("iteration %4d, erreur actuelle %g\n", n_iterations, error);
        }
        
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
        
        MPI_Reduce(&somme_carres, &somme_carres_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); //somme MPI_SUM de tout les somme_carres dans somme_carres_total
        MPI_Gather(morceau_Ax, nb_ligne, MPI_DOUBLE, Y, nb_ligne,  MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (my_rank==0)
        {
            inv_norm2Ax = 1.0 / sqrt(somme_carres_total);
            
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
            double * tmp = X; X = Y; Y = tmp; 
            
            n_iterations++;
        }

        //communications pour être prêt pour l'itération suivante
        MPI_Bcast(X, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /*Debug : Affichage des variables dans le processus 0 et le processus 1*/
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
        printf("---------------------\nerreur finale après %4d iterations: %g (|VP| = %g)\n", n_iterations, error, sqrt(somme_carres_total));
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
