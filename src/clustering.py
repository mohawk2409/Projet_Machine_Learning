from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import numpy as np
from sklearn import metrics

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        """
        Initialise un objet KMeans.

        Entrées:
        - n_clusters (int): Le nombre de clusters à former (par défaut 8).
        - max_iter (int): Le nombre maximum d'itérations pour l'algorithme (par défaut 300).
        - random_state (int ou None): La graine pour initialiser le générateur de nombres aléatoires (par défaut None).
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def initialize_centers(self, X):
        """
        Initialise les centres de clusters avec n_clusters points choisis aléatoirement à partir des données X.

        Entrée:
        - X (np.array): Les données d'entrée.

        Sortie:
        - Aucune sortie directe, mais les centres de clusters sont stockés dans self.cluster_centers_.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Sélectionner n_clusters indices aléatoires sans remplacement
        indices = np.random.choice(len(X), size=self.n_clusters, replace=False)
        self.cluster_centers_ = X[indices].copy()   

        pass

    def nearest_cluster(self, X):
        """
        Calcule la distance euclidienne entre chaque point de X et les centres de clusters,
        puis retourne l'indice du cluster le plus proche pour chaque point.

        Entrée:
        - X (np.array): Les données d'entrée.

        Sortie:
        - np.array: Un tableau d'indices représentant le cluster le plus proche pour chaque point.
        """

        distances = np.sqrt(((X[:, np.newaxis] - self.cluster_centers_) ** 2).sum(axis=2))
        
        # Retourner l'indice du centre le plus proche pour chaque point
        return np.argmin(distances, axis=1)

        pass

    def fit(self, X):
        """
        Exécute l'algorithme K-means sur les données X.

        Entrée:
        - X (np.array): Les données d'entrée.

        Sortie:
        - Aucune sortie directe, mais les centres de clusters sont stockés dans self.cluster_centers_.
        """
        self.initialize_centers(X)


        for iteration in range(self.max_iter):
            # Sauvegarder les anciens centres
            old_centers = self.cluster_centers_.copy()
            
            # Assigner chaque point au cluster le plus proche
            self.labels_ = self.nearest_cluster(X)
            
            # Recalculer les centres comme la moyenne des points de chaque cluster
            for i in range(self.n_clusters):
                points_in_cluster = X[self.labels_ == i]
                if len(points_in_cluster) > 0:
                    self.cluster_centers_[i] = points_in_cluster.mean(axis=0)
            
            # Vérifier la convergence
            if np.allclose(old_centers, self.cluster_centers_):
                break

        

    def predict(self, X):
        """
        Prédit l'appartenance aux clusters pour les données X en utilisant les centres de clusters appris pendant l'entraînement.

        Entrée:
        - X (np.array): Les données d'entrée.

        Sortie:
        - np.array: Un tableau d'indices représentant le cluster prédit pour chaque point.
        """
        return self.nearest_cluster(X)




    

def show_metric(labels_true, labels_pred, descriptors, bool_return=False, name_descriptor="", name_model="kmeans", bool_show=True):
    """
    Fonction d'affichage et création des métriques pour le clustering.
    Compatible avec KMeans (labels >= 0) et DBSCAN (labels avec -1 = bruit).

    Input :
    - labels_true : étiquettes réelles des données
    - labels_pred : étiquettes prédites (label = -1 pour les points de bruit DBSCAN)
    - descriptors : descripteurs utilisés pour le clustering
    - bool_return : retourner le dictionnaire de métriques
    - name_descriptor : nom du descripteur
    - name_model : nom du modèle
    - bool_show : afficher les métriques

    Output :
    - dictionnaire contenant les métriques d'évaluation des clusters
    """
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    descriptors = np.array(descriptors)

    # Filtrer les points de bruit DBSCAN (label = -1)
    # Pour KMeans, mask = all True (aucun effet)
    mask = labels_pred != -1
    n_noise = int(np.sum(~mask))
    lt = labels_true[mask]
    lp = labels_pred[mask]
    desc = descriptors[mask]
    n_clusters = len(np.unique(lp))

    # Cas limite DBSCAN : tous les points sont du bruit (-1)
    # Évite des scores trompeurs calculés sur des tableaux vides.
    if len(lp) == 0:
        if bool_show:
            print(f"########## Métrique descripteur : {name_descriptor} | modèle : {name_model}")
            print("Aucun point assigné à un cluster (tous les points sont du bruit).")
            print(f"Clusters détectés : 0  |  Points de bruit : {n_noise}")
        if bool_return:
            return {
                "ami": float('nan'),
                "ari": float('nan'),
                "silhouette": float('nan'),
                "homogeneity": float('nan'),
                "completeness": float('nan'),
                "v_measure": float('nan'),
                "jaccard": float('nan'),
                "descriptor": name_descriptor,
                "name_model": name_model,
                "n_clusters": 0,
                "n_noise": n_noise,
            }
        return

    if n_noise > 0 and bool_show:
        print(f"  → {n_noise} points de bruit (label=-1) exclus des métriques")

    homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(lt, lp)
    ami = metrics.adjusted_mutual_info_score(lt, lp)
    ari = adjusted_rand_score(lt, lp)

    # jaccard_score peut échouer si les ensembles de labels ne correspondent pas
    try:
        jaccard = metrics.jaccard_score(lt, lp, average='macro')
    except Exception:
        jaccard = float('nan')

    # silhouette nécessite au moins 2 clusters et plus d'échantillons que de clusters
    try:
        silhouette = silhouette_score(desc, lp) if n_clusters > 1 and len(lp) > n_clusters else float('nan')
    except Exception:
        silhouette = float('nan')

    if bool_show:
        print(f"########## Métrique descripteur : {name_descriptor} | modèle : {name_model}")
        print(f"Clusters détectés : {n_clusters}  |  Points de bruit : {n_noise}")
        print(f"Adjusted Rand Index: {ari}")
        print(f"Jaccard Index: {jaccard}")
        print(f"Homogeneity: {homogeneity}")
        print(f"Completeness: {completeness}")
        print(f"V-measure: {v_measure}")
        print(f"Silhouette Score: {silhouette}")
        print(f"Adjusted Mutual Information: {ami}")

    if bool_return:
        return {"ami": ami,
                "ari": ari,
                "silhouette": silhouette,
                "homogeneity": homogeneity,
                "completeness": completeness,
                "v_measure": v_measure,
                "jaccard": jaccard,
                "descriptor": name_descriptor,
                "name_model": name_model,
                "n_clusters": n_clusters,
                "n_noise": n_noise}
