# pfe_GtwoML : BrainMatrixGenerator

- v1 : Génère la matrice d'adjascence d'un cerveau en COO (Parallélisée en blocks de ligne)
- v2 : Génère la matrice d'adjascence transposée d'un cerveau en COO (Parallélisée en blocks de ligne)
- v3 : Génère la matrice d'adjascence transposée d'un cerveau en CSR (Parallélisée en blocks de ligne)
- v4 : Génère la matrice d'adjascence transposée d'un cerveau en CSR (Parallélisée en blocks de ligne et colonne, dans une grille de processus)
- v5 : Génère la matrice d'adjascence d'un cerveau en CSR (Parallélisée en blocks de ligne et colonne, dans une grille de processus)

La version utilisée pour les derniers tests est actuellement la v4. La v5 sera utilisée avec un algorithme du PageRank plus optimisé pour de futurs tests.
Utilisé avec une structure de Cerveau traduite d'un fichier json avec le traducteur, permet de générer une matrice qui ressemble à un cerveau, avec ses parties et ses connexions internes/externes.