# Projet de clustering des immages Digit pour ET4 info Polytech Paris Saclay

### step 1 : téléchargement des données et installation des packages
    - a. installer les requierements : "pip install -r requierements.txt"

### step 2 : configuration du chemin vers les donnés
    - a. dans le dossier src/constant.py, modifier la variable "PATH_DATA" par le chemin vers le dossier contenant les données à clusteriser.

### step 3 :  run de la pipeline clustering
    - a. aller dans le dossier src
    - c. exécutez la commande : "python pipeline.py"
    
### step 4 : lancement du dashboard
    - a. aller dans le dossier src 
    - b. exécutez la commande : "streamlit run dashboard_clustering.py"

### step 5 : Créer un docker
    - a. installer docker
    - b. compléter le dockerfile pour lancer votre dashboard depuis le docker
    - c. pour build une image docker : docker build -t mon-app-python .
    - d. pour run l'image :  docker run -d -p 8000:8000 mon-app-python 