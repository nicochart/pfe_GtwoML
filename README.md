# pfe_GtwoML
Projet de Fin d'Etudes - Polytech IS5

BrainMatrixGenerator :
- v1 : Génère la matrice d'adjacence d'un cerveau en COO (Parallélisée en blocks de ligne)
- v2 : Génère la matrice d'adjacence transposée d'un cerveau en COO (Parallélisée en blocks de ligne)
- v3 : Génère la matrice d'adjacence transposée d'un cerveau en CSR (Parallélisée en blocks de ligne)
- v4 : Génère la matrice d'adjacence transposée d'un cerveau en CSR (Parallélisée en blocks de ligne et colonne, dans une grille de processus)
- v5 : Génère la matrice d'adjacence d'un cerveau en CSR (Parallélisée en blocks de ligne et colonne, dans une grille de processus)

Traducteur :
- convertisseur : Permet de traduire un fichier json contenant les informations sur un cerveau en fichier .h contenant une structure "Cerveau" pouvant être passée en paramètre au générateur.
- Configs : Contient les fichiers json testés

PageRank :
- Sequentiel : Code séquentiel en C
- Parallele :
	- DemoCerveau : Essais et démonstration des structures (représentant le cerveau) que nous utilisons
	- On transposed and normalized A (matrice générée transposée directement, générateur v4)
		- PageRankParallele : Programme générant une matrice CSR représentant un cerveau (paramétrable), avec ses parties et ses connexions, parallélisée avec une grille de processus (ligne et colonne).
		L'algorithme du PageRank non pondéré parallèle est ensuite appliqué sur la matrice générée.
		- AnyNumberOfCores : Tentative pour régler le cas (problème) où le nombre de processus ne divise pas la dimension n de la matrice.
		- EfficientMatrixVectorMul : travail sur un algorithme du PageRank plus optimisé, en travaillent directement sur la matrice d'adjascence A.
	- On A (utilisation du générateur V5, A non transposé)
		- PageRankParallele_OnNormalizedA : Algorithme du PageRank appliqué à la matrice A normalisée sur les lignes
		- PageRankParallele_EffMVMul : futur algorithme plus optimisé, sans normalisation nécéssaire

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

TestsMPI :
- tests_communicateurs : tests sur les communicateurs par ligne et colonne dans une grille de processus 2D MPI. (aide à la compréhension des communicateurs)
- tests_parallelisation : tests sur la parallélision dans une grille de processus (ligne et colonne) de la génération de la matrice. (aide à la compréhension)