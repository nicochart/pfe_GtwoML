/* indent -nfbs -i4 -nip -npsl -di0 -nut iterated_seq.c */
/* Auteur: C. Bouillaguet et P. Fortin (Univ. Lille) */ 
#include <stdio.h>
#include <stdlib.h>
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
void init_ligne(double *A, long i, long n){
  for (long j = 0; j < n; j++) {
    A[i*n+j] = (((double)((i * i * PRNG_1 + j * j * PRNG_2) & PRNG_MAX)) / PRNG_MAX) / n;
  }
  for (long k = 1; k < n; k *= 2) {
    if (i + k < n) {
      A[i*n + i+k] = ((i - k) * PRNG_2 + i * PRNG_1) % RANGE;
    }
    if (i - k >= 0) {
      A[i*n +i-k] = ((i + k) * PRNG_2 + i * PRNG_1) % RANGE;
    }
  }
}


int main(int argc, char **argv){
    long i, j, n;
    long long size;
    double norm, inv_norm, error, start_time, total_time, delta;
    double *A, *X, *Y;
    int n_iterations;
    FILE *output;

    if (argc < 2) {
        printf("USAGE: %s [n]\n", argv[0]);
        exit(1);
    }
    n = atoll(argv[1]);
    size = n * n * sizeof(double);
    printf("taille de la matrice : %.3f G\n", size / 1073741824.);

    /*** allocation de la matrice et des vecteurs ***/
    A = (double *)malloc(size);
    if (A == NULL) {
        perror("impossible d'allouer la matrice");
        exit(1);
    }
    X = malloc(n * sizeof(double));
    Y = malloc(n * sizeof(double));
    if ((X == NULL) || (Y == NULL)) {
        perror("impossible d'allouer les vecteur");
        exit(1);
    }
    /*** initialisation de la matrice et de x ***/
    for (i = 0; i < n; i++)
    {
        init_ligne(A, i, n);
    }
    
    printf("A[0] = %g |",A[0]);
    printf("A[1] = %g\n",A[1]);

    printf("VRAI A[1048576] = %g |",A[1048576]);
    printf("A[1048577] = %g\n",A[1048577]);

    for (i = 0; i < n; i++) {
        X[i] = 1.0 / n;
    }

    start_time = my_gettimeofday();
    error = INFINITY;
    n_iterations = 0;
    while (error > 1e-9) {
        printf("iteration %4d, erreur actuelle %g\n", n_iterations, error);

        /*** y <--- A.x ***/
        for (i = 0; i < n; i++) {
            Y[i] = 0;
            for (j = 0; j < n; j++) {
                Y[i] += A[i*n+j] * X[j];
            }
        }

        /*** norm <--- ||y|| ***/
        norm = 0;
        for (i = 0; i < n; i++) {
            norm += Y[i] * Y[i];
            if (i==0 || i==1) {printf("sc * sc = %g\n",Y[i] * Y[i]);}
        }
        printf("somme des carrés total : %g\n",norm);
        norm = sqrt(norm);
        printf("norm = %g\n",norm);
        
        /*** y <--- y / ||y|| ***/
        inv_norm = 1.0 / norm;
        for (i = 0; i < n; i++) {
            Y[i] *= inv_norm;
        }
        
        printf("Y[0] = %g , Y[1] = %g\n",Y[0],Y[1]);
        
        /*** error <--- ||x - y|| ***/
        error = 0;
        for (i = 0; i < n; i++) {
            delta = X[i] - Y[i];
            error += delta * delta;
        }
        error = sqrt(error);

        /*** x <--> y ***/
        double *tmp = X; X = Y ; Y = tmp; 
        
        printf("X actuel : ");
        for(i=0;i<10;i++) {printf("%g ",X[i]);}
        printf(" ... \n");
        
        n_iterations++;
    }

    total_time = my_gettimeofday() - start_time;
    printf("erreur finale après %4d iterations: %g (|VP| = %g)\n", n_iterations, error, norm);
    printf("time : %.1f s      MFlops : %.1f \n", total_time, (2.0 * n * n + 7.0 * n) * n_iterations / 1048576. / total_time);

    /*** stocke le vecteur propre dans un fichier ***/
    output = fopen("result.out", "w");
    if (output == NULL) {
        perror("impossible d'ouvrir result.out en écriture");
        exit(1);
    }
    fprintf(output, "%ld\n", n);
    for (i = 0; i < n; i++) {
        fprintf(output, "%.17g\n", X[i]);
    }
    fclose(output);

    free(A);
    free(X);
    free(Y);
}
