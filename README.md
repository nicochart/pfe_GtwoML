# pfe_GtwoML
Projet de Fin d'Etudes - Polytech IS5

PageRank :
- Sequentiel : Code séquentiel en C
- Parallele :
	- DemoCerveau : Essais et démonstration des structures (représentant le cerveau) que nous utilisons
	- PageRankParallele : Programme générant une matrice COO représentant un cerveau (paramétrable), avec ses parties et ses connexions.
		La matrice est ensuite convertie en CSR puis normalisée sur les colonnes, et un PageRank non pondéré parallèle est appliqué.
	- AnyNumberOfCores : Tentative pour régler le cas (problème) où le nombre de processus ne divise pas la dimension n de la matrice.

MethodeDeLaPuissance :
- Sequential : Code séquentiel
- Parallel_1 : Code parallèle sans Spread-With-Add, sans OpenMP
- Parallel_2 : Code parallèle avec Spread-With-Add, sans OpenMP
- FromAdjascencyMatrix : Code parallèle basé sur une matrice creuse au format CSR type matrice de passage P (0 ou 1) pour le PageRank.
- Gpu : Code cuda 3 kernels (premier essai)

OpenMP :
- MP_Parallel_2_OMP : Code parallèle de la Méthode de la puissance, avec Spread-With-Add, avec du OpenMP

MatriceCreuse : Essais sur la génération, le stockage et l'accès aux valeurs de matrices creuses
- DenseToCSR_double : Premier essai - traduction d'une matrice de doubles (stockée normalement) en CSR
- DenseToCooToCsr : Génération aléatoire d'une matrice stockée normalement, conversion en COO, puis conversion COO -> CSR.
- CSRMatrixForPageRank : Préparation d'une matrice CSR pour PageRank