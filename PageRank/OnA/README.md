# PageRank Parallèle

OnA (A est une matrice d'adjacence non transposé) :
- PR_DistributedRV_HardBrain : algorithme plus optimisé, appliqué à la matrice A directement, sans multiplications, avec vecteur résultat réparti sur les processus
- PR_DistributedRV_HardBrain_OpenMP : Même chose avec OpenMP
- PR_AlgoComplet_A.pdf explique dans le détail comment fonctionne l'algorithme utilisé dans PR_DistributedRV_HardBrain.
- PR_AlgoResume_A.pdf résume en une page l'algorithme utilisé dans PR_DistributedRV_HardBrain.

Le dossier "Test" contient des fichiers avec des prints permettant de mieux comprendre comment est réparti le vecteur résultat.


Dossier OldVersions, plus anciennes versions :
- PR_OnNormalizedA_HardBrain : Algorithme du PageRank appliqué à la matrice A normalisée sur les lignes
- PR_EffMVMul_HardBrain : algorithme plus optimisé, sans normalisation nécéssaire