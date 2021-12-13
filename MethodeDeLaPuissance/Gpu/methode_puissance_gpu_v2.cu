/* Méthode de la puissance GPU */
/* Auteur: C. Bouillaguet et P. Fortin (Univ. Lille) code séquentiel | D.Leroye & N.Hochart code GPU */
/* Compilation : nvcc nom.cu -o nom --generate-code arch=compute_61,code=sm_61 -O3 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "defs.h"

//Taille des blocs : donne des blocs de 256 threads
#define TAILLE_BLOC_X 1
#define TAILLE_BLOC_Y 256

// Initialise la ligne N°i, à n éléments, de la matrice A :
__host__ __device__ void init_ligne_gpu(REAL_T *A, long i, long n)
{
  for (long j = 0; j < n; j++)
	{
    A[i*n+j] = (((REAL_T)((i * i * PRNG_1 + j * j * PRNG_2) & PRNG_MAX)) / PRNG_MAX) / n;
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

/*kernels*/

//Kernel 1 : calcule le produit matrice vecteur
__global__ void prodmatvectKernel(REAL_T *d_A, REAL_T *d_X, REAL_T *d_Y, int n)
{
	unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
  long j;

	if (i < n)
	{
		REAL_T temp=0;
		for(j=0;j<n;j++)
		{
			temp += d_A[i*n+j] * d_X[j];
		}
		d_Y[i] = temp;
    }
}

//Kernel 2 : Somme des éléments d'un vecteur pour calcul de norme par réduction. k permet de savoir à quelle étape on est.
__global__ void somme_elements_vecteur(REAL_T *d_vect, int k, int n, REAL_T *d_norm)
{
		unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
		if (i<n)
		{
			if (i % k == 0)
			{
				d_vect[i] = fabs(d_vect[i]) + fabs(d_vect[i + k/2]);
			}
		}
    *d_norm = d_vect[0]; //à chaque fois, on copie la valeur à l'indice 0 dans un autre vecteur. Si on est pas à la dernière étape de la reduction, cette valeur n'a pas de sens.
}

//Kernel 3 : Divise l'ensemble des éléments d'un vecteur par un reel et calcule le vecteur erreur contenant les erreurs locales
__global__ void normAndError(REAL_T *d_Y, REAL_T *d_normY,  REAL_T *d_X, REAL_T *d_Err, int n)
{
	unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
	if (i < n)
	{
		d_Y[i] = d_Y[i] / *d_normY;
    d_Err[i] = fabs(d_X[i] - d_Y[i]);
  }
}

//Kernel 4 : Génération de morceaux de matrice directement sur GPU
__global__ void matrixInit(REAL_T *d_A, int n)
{
	unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
	if (i < n)
	{
		init_ligne_gpu(d_A, i, n); //initialisation de la ligne i de d_A (gpu)
  }
}

//Kernel 5 : Initialisation du vecteur X à 1/n sur GPU
__global__ void vectorInit(REAL_T *d_X, int n)
{
	unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
	if (i < n)
	{
		d_X[i] = 1.0 / (double) n;
	}
}

//Kernel 6 : Copie de vect2 dans vect1
__global__ void vectorCopy(REAL_T *d_vect1, REAL_T *d_vect2, int n)
{
	unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
	if (i < n)
	{
		d_vect1[i] = d_vect2[i];
	}
}

int main(int argc, char **argv)
{
    int i,k,n;
    long long size, size_vector;
    REAL_T norm, error;
    REAL_T *d_error, *d_norm, *tmp, *X, *d_A, *d_X, *d_Y, *d_Y_tmpsum, *d_Err;
    double start_time, total_time;
    int n_iterations;
    FILE *output;

    if (argc < 2)
		{
        printf("USAGE: %s [n]\n", argv[0]);
        exit(1);
    }
    n = atoi(argv[1]);
    size = (long long) n * n * sizeof(REAL_T);
    size_vector = (long long) n * sizeof(REAL_T);
    printf("taille de la matrice : %.1f G\n", size / 1073741824.);

    // Allocation CPU
    X = (REAL_T *)malloc(n * sizeof(REAL_T));
    if (X == NULL)
    {
        perror("impossible d'allouer le vecteur X");
        exit(1);
    }

    // Allocation GPU
    cudaMalloc((void **) &d_A, size);
    cudaMalloc((void **) &d_X, size_vector);
    cudaMalloc((void **) &d_Y, size_vector);
		cudaMalloc((void **) &d_Y_tmpsum, size_vector);
    cudaMalloc((void **) &d_Err, size_vector);
    cudaMalloc((void **) &d_norm, sizeof(REAL_T));
    cudaMalloc((void **) &d_error, sizeof(REAL_T));

    //Initialisation des kernels
    dim3 threadsParBloc(TAILLE_BLOC_Y);
    dim3 tailleGrille(TAILLE_BLOC_Y);

		/*** initialisation de la matrice ***/
		//Lancement de Kernel 4 : génération de la matrice
		matrixInit<<<tailleGrille,threadsParBloc>>>(d_A, n);
		/*** initialisation de x ***/
		//Lancement de Kernel 5 : initialisation de X
		vectorInit<<<tailleGrille,threadsParBloc>>>(d_X, n);

    start_time = my_gettimeofday();
    error = 10;
    n_iterations = 0;
    while(error > ERROR_THRESHOLD)
    {
        printf("iteration %4d, erreur actuelle %g\n", n_iterations, error);

        /*** y <--- A.x ***/
        //Lancement de Kernel 1 (asynchrone) (calcul vecteur y)
        prodmatvectKernel<<<tailleGrille,threadsParBloc>>>(d_A,d_X,d_Y,n);

				//Lancement de Kernel 6 (copie de d_Y dans d_Y_tmpsum)
				vectorCopy<<<tailleGrille,threadsParBloc>>>(d_Y_tmpsum, d_Y, n);

				for (k=2;k<=n;k*=2)
				{
        	//Plusieurs lancements du Kernel 2 (étapes de réduction pour calcul de norme)
        	somme_elements_vecteur<<<tailleGrille,threadsParBloc>>>(d_Y_tmpsum,2,n,d_norm);
				}

        /*** y <--- y / ||y||   &   vecteur erreur ***/
        //Lancement de Kernel 3 (normalisation + calul du vecteur erreur)
        normAndError<<<tailleGrille,threadsParBloc>>>(d_Y,d_norm,d_X,d_Err,n);

        /*** error <--- ||x - y|| ***/
				for (k=2;k<=n;k*=2)
				{
					//Plusieurs lancements du Kernel 2 (étapes de réduction pour calcul de norme)
        	somme_elements_vecteur<<<tailleGrille,threadsParBloc>>>(d_Err,k,n,d_error);
				}

        //transfert GPU -> CPU de d_error vers error
        cudaMemcpy(&error, d_error, sizeof(REAL_T), cudaMemcpyDeviceToHost);

        /*** x <--> y ***/
        tmp = d_X; d_X = d_Y ; d_Y = tmp;

        n_iterations++;
    }

    //transfert GPU -> CPU de d_X vers X et d_norm vers norm
    cudaMemcpy(X, d_X, size_vector, cudaMemcpyDeviceToHost);
    cudaMemcpy(&norm, d_norm, sizeof(REAL_T), cudaMemcpyDeviceToHost);

    total_time = my_gettimeofday() - start_time;
    printf("erreur finale après %4d iterations : %g (|VP| = %g)\n", n_iterations, error, norm);
    printf("temps : %.1f s      Mflop/s : %.1f \n", total_time, (2.0 * n * n + 7.0 * n) * n_iterations / 1048576. / total_time);

    /*** stocke le vecteur propre dans un fichier ***/
    output = fopen("result.out", "w");
    if (output == NULL)
    {
        perror("impossible d'ouvrir result.out en écriture");
        exit(1);
    }
    fprintf(output, "%d\n", n);
    for (i = 0; i < n; i++)
    {
        fprintf(output, "%.17g\n", X[i]);
    }
    fclose(output);

    cudaFree(d_A);
    cudaFree(d_X);
    cudaFree(d_Y);
		cudaFree(d_Y_tmpsum);
    cudaFree(d_Err);
    cudaFree(d_norm);
    cudaFree(d_error);
    free(X);
}
