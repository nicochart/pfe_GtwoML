/* Fonctions utilisées pour faire de l'aléatoire dans la générationd de cerveau / de matrice-cerveau */
/*Nicolas HOCHART*/

#define randomforbrain

#include <stdlib.h>

/*--------------------------
--- Décision "aléatoire" ---
--------------------------*/

float random_between_0_and_1()
{
    /*Renvoie un nombre aléatoire entre 0 et 1. Permet de faire une décision aléatoire*/
    return (float) rand() / (float) RAND_MAX;
}
