# Composition des fichiers de config json

## Description des parties 

 - Distribution : un tableau qui contient les probabilités de distribution vers les autres parties. A l'indice 0 connection vers la partie de l'indice 0. Le tableau contient les connections interne entre une meme partie
 - connectionOpposite : proba de connection vers le côté opposé du cerveau (gauche/droite).
 - typeNeuron : un tableau de dictionnaire correspondant au neuronne present dans la partie du cerveau.
 - nbNeuron : le nombre de neuronnes.
 - nbConnection :   le nombre de connection d'un neuronne.
 - id, namePart, name : sont des champs pour l'humain, ils ne servent qu'a se repérer.

Il est très simple aussi de rajouter des informations comme les neurotransmeteurs

