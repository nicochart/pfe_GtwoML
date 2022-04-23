/* indent -nfbs -i4 -nip -npsl -di0 -nut checker.c */
/* Auteur: C. Bouillaguet et P. Fortin (Univ. Lille) */ 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

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
    long i, j, k, n;
    double norm, inv_norm, error, delta, start_time;
    double *A, *X, *Y;
    FILE *input;

    if (argc < 2) {
        printf("USAGE: %s [file]\n", argv[0]);
        exit(1);
    }
    input = fopen(argv[1], "r");
    if (input == NULL) {
        fprintf(stderr, "impossible d'ouvrir %s en lecture", argv[1]);
        exit(1);
    }
    fscanf(input, "%ld", &n);
    printf("lu : n = %ld\n", n);


    X = (double *)malloc(n * sizeof(double));
    Y = (double *)malloc(n * sizeof(double));
    if ((X == NULL) || (Y == NULL)) {
        perror("impossible d'allouer la mémoire");
        exit(1);
    }
    for (i = 0; i < n; i++) {
        fscanf(input, "%lf\n", X + i);
    }
    fclose(input);

    start_time = my_gettimeofday();
    error = 0;

    A = (double *)malloc(n * n * sizeof(double));
    if (A == NULL) {
      perror("impossible d'allouer la mémoire");
      exit(1);
    }
    
    for (i = 0; i < n; i++) {
      init_ligne(A, i, n);
    }

    for (i = 0; i < n; i++) {
      Y[i] = 0;
      for (j = 0; j < n; j++) {
	Y[i] += A[i*n+j] * X[j];
      }
    }
    
    free(A);
    
    /*** norm <--- ||y|| ***/
    norm = 0;
    for (i = 0; i < n; i++) {
        norm += Y[i] * Y[i];
    }
    norm = sqrt(norm);

    /*** y <--- y / ||y|| ***/
    inv_norm = 1.0 / norm;
    for (i = 0; i < n; i++) {
        Y[i] *= inv_norm;
    }

    /*** error <--- ||x - y|| ***/
    error = 0;
    for (i = 0; i < n; i++) {
        delta = X[i] - Y[i];
        error += delta * delta;
    }
    error = sqrt(error);

    printf("erreur : %g (|VP| = %g)\n", error, norm);
    printf("time : %.1f s\n", my_gettimeofday() - start_time);

    free(X);
    free(Y);
}
