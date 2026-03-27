from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import numpy as np

try:
    from .features import *
    from .clustering import *
    from .utils import *
    from .constant import PATH_OUTPUT, MODEL_CLUSTERING, CLASS_NAMES
except ImportError:
    from features import *
    from clustering import *
    from utils import *
    from constant import PATH_OUTPUT, MODEL_CLUSTERING, CLASS_NAMES
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans as SklearnKMeans

tab_score_hog = []

"""Calcule Hog et Hist KMeans"""
def pipeline():
    """Pipeline de base HOG/HIST avec KMeans et exports dashboard."""

    print("\n\n ##### Chargement des données Snacks + split train/test ######")
    (
        images_train, labels_train, paths_train,
        images_test,  labels_test,  paths_test, 
    ) = load_train_test_split()
    print(f"- {len(CLASS_NAMES)} classes")

    print("\n\n ##### Extraction de Features ######")
    # --- Train ---
    print("- calcul features hog (train)...")
    desc_hog_train   = compute_hog_descriptors(images_train)
    print("- calcul features Histogram gris (train)...")
    desc_hist_train  = compute_gray_histograms(images_train)
    # --- Test ---
    print("- calcul features hog (test)...")
    desc_hog_test   = compute_hog_descriptors(images_test)
    print("- calcul features Histogram gris (test)...")
    desc_hist_test  = compute_gray_histograms(images_test)

    print("\n\n ##### Normalisation (fit sur train, transform sur test) ######")
    # Le scaler est ajusté UNIQUEMENT sur le train pour éviter toute fuite
    # d'information vers le test (data leakage).
    scaler_hog   = StandardScaler().fit(desc_hog_train)
    scaler_hist  = StandardScaler().fit(desc_hist_train)

    desc_hog_train_n   = scaler_hog.transform(desc_hog_train)
    desc_hist_train_n  = scaler_hist.transform(desc_hist_train)
    desc_hist_test_n   = scaler_hist.transform(desc_hist_test)


    desc_hog_test_n    = scaler_hog.transform(desc_hog_test)
    pca = PCA(n_components=50, random_state=42)
    desc_hog_train_pca = pca.fit_transform(desc_hog_train_n)
    desc_hog_test_pca  = pca.transform(desc_hog_test_n)

    print("\n\n ##### Clustering (fit sur train, predict sur test) ######")
    number_cluster = len(CLASS_NAMES)

    print("- KMeans HOG...")
    km_hog = SklearnKMeans(n_clusters=number_cluster, random_state=42, n_init=30)
    km_hog.fit(desc_hog_train_pca)
    pred_hog_test = km_hog.predict(desc_hog_test_pca)

    print("- KMeans Histogram gris...")
    km_hist = SklearnKMeans(n_clusters=number_cluster, random_state=42, n_init=30)
    km_hist.fit(desc_hist_train_n)
    pred_hist_test = km_hist.predict(desc_hist_test_n)

    print("\n\n ##### Résultats (métriques sur l'ensemble de TEST) ######")
    metric_hist  = show_metric(labels_test, pred_hist_test,  desc_hist_test_n,
                               bool_show=True, name_descriptor="HISTOGRAM",  bool_return=True)
    metric_hog   = show_metric(labels_test, pred_hog_test,   desc_hog_test_n,
                               bool_show=True, name_descriptor="HOG",         bool_return=True)

    print("\n- Export des données test vers le dashboard")
    df_metric = pd.DataFrame([metric_hist, metric_hog])

    x_3d_hist  = conversion_3d(desc_hist_test_n)
    x_3d_hog   = conversion_3d(desc_hog_test_n)

    df_hist  = create_df_to_export(x_3d_hist,  labels_test, pred_hist_test,  paths_test)
    df_hog   = create_df_to_export(x_3d_hog,   labels_test, pred_hog_test,   paths_test)

    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    df_hist.to_excel(PATH_OUTPUT  + "/save_clustering_hist_kmeans.xlsx")
    df_hog.to_excel(PATH_OUTPUT   + "/save_clustering_hog_kmeans.xlsx")
    df_metric.to_excel(PATH_OUTPUT + "/save_metric.xlsx")
    print("Fin. \n\n Pour avoir la visualisation dashboard : streamlit run dashboard_clustering.py")

def hog_kmeans():
    """Exécute uniquement HOG + KMeans et stocke un score simplifié."""
    print("\n\n ##### Chargement des données Snacks + split train/test ######")
    (
        images_train, labels_train, paths_train,
        images_test,  labels_test,  paths_test
    ) = load_train_test_split()
    print(f"- {len(CLASS_NAMES)} classes")
    print("- calcul features hog (train)...")
    desc_hog_train   = compute_hog_descriptors(images_train)
    print("- calcul features hog (test)...")
    desc_hog_test   = compute_hog_descriptors(images_test)
    number_cluster = len(CLASS_NAMES)
    scaler_hog   = StandardScaler().fit(desc_hog_train)
    desc_hog_train_n   = scaler_hog.transform(desc_hog_train)
    desc_hog_test_n    = scaler_hog.transform(desc_hog_test)
    km_hog = SklearnKMeans(n_clusters=number_cluster, random_state=42, n_init=30)
    km_hog.fit(desc_hog_train_n)
    pred_hog_test = km_hog.predict(desc_hog_test_n)
    metric_hog   = show_metric(labels_test, pred_hog_test,   desc_hog_test_n,
                               bool_show=True, name_descriptor="HOG",         bool_return=True)

    x_3d_hog   = conversion_3d(desc_hog_test_n)
    df_hog   = create_df_to_export(x_3d_hog,   labels_test, pred_hog_test,   paths_test)
    tab = np.array([float(metric_hog["ami"]), float(metric_hog["silhouette"])], dtype=float)
    tab_score_hog.append(tab)

def hist_dbscan_kmeans_pipeline():
    """
    Pipeline HIST KMeans / DBSCAN
    ======================================
    On les compare avec deux modèles de clustering :
      - KMeans  : nombre de clusters fixé = nombre de classes
      - DBSCAN  : découverte automatique des clusters par densité
                  (eps et min_samples à ajuster selon le dataset)
    """
    from sklearn.cluster import KMeans as SklearnKMeans, DBSCAN as SklearnDBSCAN
    from sklearn.preprocessing import normalize
    from sklearn.metrics import pairwise_distances

    print("\n\n ##### [Hist] Chargement des données Snacks + split train/test ######")
    (
        images_train, labels_train, paths_train, _,
        images_test,  labels_test,  paths_test,  _
    ) = load_train_test_split()

    print("\n\n ##### [Hist] Extraction des descripteurs ######")
    hist_train = compute_gray_histograms(images_train)
    hist_test  = compute_gray_histograms(images_test)
    print(f"- Train : {hist_train.shape}  |  Test : {hist_test.shape}  (256 bins/image)")

    print("\n\n ##### [Hist] Normalisation (fit sur train) ######")
    scaler_hist = StandardScaler().fit(hist_train)
    hist_train_n = scaler_hist.transform(hist_train)
    hist_test_n  = scaler_hist.transform(hist_test)

    # Espace DBSCAN : histogrammes L2-normalisés.
    hist_train_db = normalize(hist_train, norm='l2')
    hist_test_db = normalize(hist_test, norm='l2')
    print(hist_train_n.shape)
    print(hist_test_n.shape)

    pca = PCA(n_components=30, random_state=42)
    hist_train_pca = pca.fit_transform(hist_train_n)
    hist_test_pca  = pca.transform(hist_test_n)
    hist_train_db_pca = pca.fit_transform(hist_train_db)
    hist_test_db_pca  = pca.transform(hist_test_db)
    print(hist_train_pca.shape)
    print(hist_test_pca.shape)
    print("\n\n ##### [Hist] Clustering ######")
    number_cluster = len(CLASS_NAMES)

    # KMeans : fit train, predict test.
    print(f"- KMeans ({number_cluster} clusters)...")
    km_hist = SklearnKMeans(n_clusters=number_cluster, random_state=42, n_init=15)
    km_hist.fit(hist_train_pca)
    pred_km_test = km_hist.predict(hist_test_pca)

    # DBSCAN : fit train, puis assignation test par centre le plus proche.
    dbscan_metric = "cosine"
    dbscan_eps = 0.15
    dbscan_min_samples = 3
    print(f"- DBSCAN (eps={dbscan_eps}, min_samples={dbscan_min_samples}, metric='{dbscan_metric}')...")
    dbscan = SklearnDBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric=dbscan_metric)
    dbscan.fit(hist_train_db)
    n_db = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    n_noise = int(np.sum(dbscan.labels_ == -1))
    print(f"  → {n_db} clusters sur train, {n_noise} points de bruit")

    if n_db > 0:
        cluster_ids = [c for c in set(dbscan.labels_) if c != -1]
        centers = np.array([hist_train_db[dbscan.labels_ == c].mean(axis=0) for c in cluster_ids])
        dists = pairwise_distances(hist_test_db, centers, metric=dbscan_metric)
        pred_db_test = np.array([cluster_ids[i] for i in np.argmin(dists, axis=1)])
    else:
        # Cas limite : aucun cluster détecté (labels bruit).
        pred_db_test = np.full(len(labels_test), -1, dtype=int)

    print("\n\n ##### [Hist] Résultats (métriques sur TEST) ######")
    metric_hist_kmeans = show_metric(labels_test, pred_km_test, hist_test_pca,
                                   bool_show=True, name_descriptor="HIST_KMEANS",
                                   name_model="kmeans", bool_return=True)
    metric_hist_dbscan = show_metric(labels_test, pred_db_test, hist_test_db,
                                   bool_show=True, name_descriptor="HIST_DBSCAN",
                                   name_model="dbscan", bool_return=True)

    print("\n- Export des données TEST vers le dashboard")
    x_3d_hist = conversion_3d(hist_test_n)

    df_hist_kmeans = create_df_to_export(x_3d_hist, labels_test, pred_km_test, paths_test)
    df_hist_dbscan = create_df_to_export(x_3d_hist, labels_test, pred_db_test, paths_test)
    df_metric_hist = pd.DataFrame([metric_hist_kmeans, metric_hist_dbscan])

    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    df_hist_kmeans.to_excel(PATH_OUTPUT + "/save_clustering_hist_kmeans.xlsx")
    df_hist_dbscan.to_excel(PATH_OUTPUT + "/save_clustering_hist_dbscan.xlsx")
    df_metric_hist.to_excel(PATH_OUTPUT + "/save_metric_hist.xlsx")
    print("Fin. \n\n Pour visualiser, lancer : streamlit run dashboard_clustering.py")

def hog_dbscan_kmeans_pipeline():
    """
    Pipeline HOG KMeans / DBSCAN
    ======================================
    On les compare avec deux modèles de clustering :
      - KMeans  : nombre de clusters fixé = nombre de classes
      - DBSCAN  : découverte automatique des clusters par densité
                  (eps et min_samples à ajuster selon le dataset)
    """
    from sklearn.cluster import KMeans as SklearnKMeans, DBSCAN as SklearnDBSCAN
    from sklearn.preprocessing import normalize
    from sklearn.metrics import pairwise_distances

    print("\n\n ##### [HOG] Chargement des données Snacks + split train/test ######")
    (
        images_train, labels_train, paths_train, _,
        images_test,  labels_test,  paths_test,  _
    ) = load_train_test_split()

    print("\n\n ##### [HOG] Extraction des descripteurs ######")
    hog_train = compute_hog_descriptors(images_train)
    hog_test  = compute_hog_descriptors(images_test)
    print(f"- Train : {hog_train.shape}  |  Test : {hog_test.shape}  (256 bins/image)")

    print("\n\n ##### [HOG] Normalisation (fit sur train) ######")
    scaler_hog = StandardScaler().fit(hog_train)
    hog_train_n = scaler_hog.transform(hog_train)
    hog_test_n  = scaler_hog.transform(hog_test)

    # Espace DBSCAN : normalisation L2.
    hog_train_db = normalize(hog_train, norm='l2')
    hog_test_db = normalize(hog_test, norm='l2')
    print(hog_train_n.shape)
    print(hog_test_n.shape)

    pca = PCA(n_components=30, random_state=42)
    hog_train_pca = pca.fit_transform(hog_train_n)
    hog_test_pca  = pca.transform(hog_test_n)
    hog_train_db_pca = pca.fit_transform(hog_train_db)
    hog_test_db_pca  = pca.transform(hog_test_db)
    print(hog_train_pca.shape)
    print(hog_test_pca.shape)
    print("\n\n ##### [HOG] Clustering ######")
    number_cluster = len(CLASS_NAMES)

    # KMeans : versions sans PCA et avec PCA.
    print(f"- KMeans ({number_cluster} clusters)...")
    km_hog = SklearnKMeans(n_clusters=number_cluster, random_state=42, n_init=15)
    km_hog.fit(hog_train_n)
    pred_km_test = km_hog.predict(hog_test_n)

    km_hog_pca = SklearnKMeans(n_clusters=number_cluster, random_state=42, n_init=15)
    km_hog_pca.fit(hog_train_pca)
    pred_km_test_pca = km_hog_pca.predict(hog_test_pca)

    # DBSCAN : fit train, puis assignation test via centres.
    dbscan_metric = "euclidean"
    dbscan_eps = 0.5
    dbscan_min_samples = 3
    print(f"- DBSCAN (eps={dbscan_eps}, min_samples={dbscan_min_samples}, metric='{dbscan_metric}')...")
    dbscan = SklearnDBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric=dbscan_metric)
    dbscan.fit(hog_train_db)
    n_db = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    n_noise = int(np.sum(dbscan.labels_ == -1))
    print(f"{n_db} clusters sur train, {n_noise} points de bruit")

    if n_db > 0:
        cluster_ids = [c for c in set(dbscan.labels_) if c != -1]
        centers = np.array([hog_train_db_pca[dbscan.labels_ == c].mean(axis=0) for c in cluster_ids])
        dists = pairwise_distances(hog_test_db_pca, centers, metric=dbscan_metric)
        pred_db_test_pca = np.array([cluster_ids[i] for i in np.argmin(dists, axis=1)])

        # Variante sans PCA pour comparaison.

        cluster_ids = [c for c in set(dbscan.labels_) if c != -1]
        centers = np.array([hog_train_db[dbscan.labels_ == c].mean(axis=0) for c in cluster_ids])
        dists = pairwise_distances(hog_test_db, centers, metric=dbscan_metric)
        pred_db_test = np.array([cluster_ids[i] for i in np.argmin(dists, axis=1)])
    else:
        # Cas limite : aucun cluster détecté (labels bruit).
        pred_db_test_pca = np.full(len(labels_test), -1, dtype=int)
        pred_db_test = np.full(len(labels_test), -1, dtype=int)

    print("\n\n ##### [HOG] Résultats (métriques sur TEST) ######")
    print("Avec PCA :\n")
    metric_hog_kmeans_pca = show_metric(labels_test, pred_km_test_pca, hog_test_pca,
                                   bool_show=True, name_descriptor="HOG_KMEANS_PCA",
                                   name_model="kmeans", bool_return=True)
    print("Sans PCA :\n")
    metric_hog_kmeans = show_metric(labels_test, pred_km_test, hog_test_n,
                                   bool_show=True, name_descriptor="HOG_KMEANS",
                                   name_model="kmeans", bool_return=True)
    print("Avec PCA :\n")
    metric_hog_dbscan_pca = show_metric(labels_test, pred_db_test_pca, hog_test_db_pca,
                                   bool_show=True, name_descriptor="HOG_DBSCAN_PCA",
                                   name_model="dbscan", bool_return=True)
    print("Sans PCA :\n")
    metric_hog_dbscan = show_metric(labels_test, pred_db_test, hog_test_db,
                                   bool_show=True, name_descriptor="HOG_DBSCAN",
                                   name_model="dbscan", bool_return=True)

    print("\n- Export des données TEST vers le dashboard")
    x_3d_hist = conversion_3d(hog_test_n)

    df_hog_kmeans = create_df_to_export(x_3d_hist, labels_test, pred_km_test_pca, paths_test)
    df_hog_dbscan = create_df_to_export(x_3d_hist, labels_test, pred_db_test_pca, paths_test)
    df_metric_hog = pd.DataFrame([
        metric_hog_kmeans_pca,
        metric_hog_kmeans,
        metric_hog_dbscan_pca,
        metric_hog_dbscan
    ])

    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    df_hog_kmeans.to_excel(PATH_OUTPUT + "/save_clustering_hog_kmeans.xlsx")
    df_hog_dbscan.to_excel(PATH_OUTPUT + "/save_clustering_hog_dbscan.xlsx")
    df_metric_hog.to_excel(PATH_OUTPUT + "/save_metric_hog.xlsx")
    print("Fin. \n\n Pour visualiser, lancer : streamlit run dashboard_clustering.py")
if __name__ == "__main__":
    hog_dbscan_kmeans_pipeline()