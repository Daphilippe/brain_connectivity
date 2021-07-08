Étude du barycentre
# main_barycentre.py
Détermination du barycentre itératif

Attention aux différents chemins, cela peut générer des erreurs lors de l'appel des fichiers.

Paramètres les plus importants : 
* itermax donner le nombre de sujets calculé dans le barycentre inséré en X_init
* Nmax : nombre de sujets au total (comprenant le nombre de sujets déjà calculé au sein du barycentre précédent)
Remarque : On peut calculer les barycentres avec un support d'initialisation de 500. 
Cela donne les mêmes résultats après passage d'un noyau gaussien.
# test_barycenter_exp1.py
Test de robustesse du barycentre : support d'initialisation
# test_barycenter_exp2.py
Test de robustesse du barycentre : ordre présentation des sujets
# test_max_exp1.py
Étude comparative des barycentres avec détermination des maximaux locaux pour un sujet
# test_max_exp2.py
Étude comparative des barycentres avec détermination des maximaux locaux pour l'ensemble des sujets dans un dossier