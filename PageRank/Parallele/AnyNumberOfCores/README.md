# PageRank with any number of cores

Jusqu'à maintenant, nous avons supposé que le nombre de coeurs (processus MPI) utilisés devait diviser la dimension de la matrice.
Nous tentons ici de se débarasser de cette dépendance.

-PR_nb_ligne_local.c est une première tentative (non fonctionnelle) : elle modifie la V1 du PageRank Parallèle pour générer les morceaux de matrice avec un nombre de ligne LOCAL, pour pouvoir prendre des lignes bonus s'il y en a.
Cette première tentative ne fonctionne pas car le MPI_Allgather a besoin du paramètre "nb_ligne" (nombre de ligne par processus) qui doit être le même dans chaque processus.. Ca coince.

-PR_adjusted_core_usage.c sera une tentative dans laquelle on s'arrangera pour que le nombre de coeurs utilisés divise la dimension de la matrice. Pour ceci, nous laisserons de côté les coeurs en trop.