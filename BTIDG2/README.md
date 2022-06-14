# Brain Topology Inspired Distributed Graph Generator (BTIDG2)
https://github.com/SMG2S/BTIDG2


- includes, sources du générateur, il existe plusieurs versions du générateur (définies dans brainmatrixgenerator.h) :
	- generate_coo_row_transposed_adjacency_brain_matrix_for_pagerank : Génère la matrice d'adjacence transposée d'un cerveau en COO (Parallélisée en blocks de ligne)
	- generate_csr_row_transposed_adjacency_brain_matrix_for_pagerank : Génère la matrice d'adjacence transposée d'un cerveau en CSR (Parallélisée en blocks de ligne)
	- generate_csr_brain_transposed_adjacency_matrix_for_pagerank : Génère la matrice d'adjacence transposée d'un cerveau en CSR (Parallélisée en blocks de ligne et colonne, dans une grille de processus)
	- generate_csr_brain_adjacency_matrix_for_pagerank : Génère la matrice d'adjacence d'un cerveau en CSR (Parallélisée en blocks de ligne et colonne, dans une grille de processus)

- exemples :
	- exemples d'utilisation du générateur

- translator :
	- convertisseur : Permet de traduire un fichier json contenant les informations sur un cerveau en fichier .h contenant une structure "Cerveau" pouvant être passée en paramètre au générateur.
	- Configs : Contient les fichiers json testés