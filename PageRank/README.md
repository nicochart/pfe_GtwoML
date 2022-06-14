# PageRank parallèle

OnTransposedA (PageRank sur AT, matrice générée transposée directement) :
- PR_DistributedRV_HardBrain : Programme générant une matrice d'adjacence transposée en CSR inspiré d'un cerveau (paramétrable), parallélisée avec une grille de processus (ligne et colonne). Le PageRank est appliqué à cette matrice.
OnA (PageRank sur A, une matrice d'adjacence non transposé) :
- PR_DistributedRV_HardBrain : algorithme plus optimisé, appliqué à la matrice A directement, sans multiplications, avec vecteur résultat réparti sur les processus
- PR_DistributedRV_HardBrain_OpenMP : Même chose avec OpenMP
- Explication de l'algorithme dans des fichiers .pdf

pagerank_includes.h est le fichier utilisé pour inclure les méthodes nécéssaires à la génération / application du PageRank.

hardbrain.h est un cerveau écrit en dur, souvent utilisé pour faire des tests.