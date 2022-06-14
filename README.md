# pfe_GtwoML
Projet de Fin d'Etudes - Polytech IS5

BTIDG2 :
- Contient le Brain Topology Inspired Distributed Graph Generator (BTIDG2). Les includes (sources), le traducteur json vers .h, et des exemples d'utilisation.

PageRank (Parallèle) :
- On transposed and normalized A (matrice générée transposée directement)
	- PR_DistributedRV_HardBrain : Programme générant une matrice d'adjacence transposée en CSR inspiré d'un cerveau (paramétrable), parallélisée avec une grille de processus (ligne et colonne). Le PageRank est appliqué à cette matrice.
- On A (A est une matrice d'adjacence non transposé)
	- PR_DistributedRV_HardBrain : algorithme plus optimisé, appliqué à la matrice A directement, sans multiplications, avec vecteur résultat réparti sur les processus
	- PR_DistributedRV_HardBrain_OpenMP : Même chose avec OpenMP

Autre :
- Contient d'autres codes / tests effectués pour expérimenter de nouvelles choses et avancer dans le code. La plupart de ces codes sont outdated.
	- MethodeDeLaPuissance : Essais sur la Méthode de la Puissance (utilisée dans le PageRank)
	- PageRankSequentiel : Algorithmes du PageRank en séquentiel
	- TestsGenerator : Tests pour le générateur BTIDG2
	- TestsMatriceCreuse : Essais sur la génération, le stockage et l'accès aux valeurs de matrices creuses
	- TestsMPI : Divers essais en MPI, sur les communicateurs et autre.
	- TestsOpenMP : Divers essais en OpenMP.
