# PageRank with any number of cores

Jusqu'à maintenant, nous avons supposé que le nombre de coeurs (processus MPI) utilisés devait diviser la dimension de la matrice.
Nous tentons ici de se débarasser de cette dépendance.

-PR_nb_ligne_local.c est une première tentative (non fonctionnelle) : elle modifie la V1 du PageRank Parallèle pour générer les morceaux de matrice avec un nombre de ligne LOCAL, pour pouvoir prendre des lignes bonus s'il y en a.
Cette première tentative ne fonctionne pas car le MPI_Allgather a besoin du paramètre "nb_ligne" (nombre de ligne par processus) qui doit être le même dans chaque processus.. Ca coince.

-PR_adjusted_core_usage.c est une tentative dans laquelle on s'arrange pour que le nombre de coeurs utilisés divise la dimension de la matrice. Pour ceci, nous laissons de côté les coeurs en trop.
Ceci ne fonctionne pas non plus, à peu prêt pour les même raisons : le Allgather ainsi que le Allreduce attendent des valeurs de tout les processus (y compris de ceux non utilisés).