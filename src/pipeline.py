from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import numpy as np

from features import *
from clustering import *
from utils import *
from constant import PATH_OUTPUT, MODEL_CLUSTERING, CLASS_NAMES
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans as SklearnKMeans

tab_score_hog = []

def pipeline():

    print("\n\n ##### Chargement des données Snacks + split train/test ######")
    (
        images_train, labels_train, paths_train, images_color_train,
        images_test,  labels_test,  paths_test,  images_color_test
    ) = load_train_test_split()
    print(f"- {len(CLASS_NAMES)} classes")

    print("\n\n ##### Extraction de Features ######")
    # --- Train ---
    print("- calcul features hog (train)...")
    desc_hog_train   = compute_hog_descriptors(images_train)
    print("- calcul features Histogram gris (train)...")
    desc_hist_train  = compute_gray_histograms(images_train)
    print("- calcul features Histogramme couleur HSV (train)...")
    desc_color_train = compute_hsv_histograms(images_color_train)
    # --- Test ---
    print("- calcul features hog (test)...")
    desc_hog_test   = compute_hog_descriptors(images_test)
    print("- calcul features Histogram gris (test)...")
    desc_hist_test  = compute_gray_histograms(images_test)
    print("- calcul features Histogramme couleur HSV (test)...")
    desc_color_test = compute_hsv_histograms(images_color_test)

    print("\n\n ##### Normalisation (fit sur train, transform sur test) ######")
    # Le scaler est ajusté UNIQUEMENT sur le train pour éviter toute fuite
    # d'information vers le test (data leakage).
    scaler_hog   = StandardScaler().fit(desc_hog_train)
    scaler_hist  = StandardScaler().fit(desc_hist_train)
    scaler_color = StandardScaler().fit(desc_color_train)

    desc_hog_train_n   = scaler_hog.transform(desc_hog_train)
    desc_hist_train_n  = scaler_hist.transform(desc_hist_train)
    desc_color_train_n = scaler_color.transform(desc_color_train)
    desc_hist_test_n   = scaler_hist.transform(desc_hist_test)
    desc_color_test_n  = scaler_color.transform(desc_color_test)


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

    print("- KMeans Couleur HSV...")
    km_color = SklearnKMeans(n_clusters=number_cluster, random_state=42, n_init=30)
    km_color.fit(desc_color_train_n)
    pred_color_test = km_color.predict(desc_color_test_n)

    print("\n\n ##### Résultats (métriques sur l'ensemble de TEST) ######")
    metric_hist  = show_metric(labels_test, pred_hist_test,  desc_hist_test_n,
                               bool_show=True, name_descriptor="HISTOGRAM",  bool_return=True)
    metric_hog   = show_metric(labels_test, pred_hog_test,   desc_hog_test_n,
                               bool_show=True, name_descriptor="HOG",         bool_return=True)
    metric_color = show_metric(labels_test, pred_color_test, desc_color_test_n,
                               bool_show=True, name_descriptor="COLOR_HSV",   bool_return=True)

    print("\n- Export des données test vers le dashboard")
    df_metric = pd.DataFrame([metric_hist, metric_hog, metric_color])

    x_3d_hist  = conversion_3d(desc_hist_test_n)
    x_3d_hog   = conversion_3d(desc_hog_test_n)
    x_3d_color = conversion_3d(desc_color_test_n)

    df_hist  = create_df_to_export(x_3d_hist,  labels_test, pred_hist_test,  paths_test)
    df_hog   = create_df_to_export(x_3d_hog,   labels_test, pred_hog_test,   paths_test)
    df_color = create_df_to_export(x_3d_color, labels_test, pred_color_test, paths_test)

    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    df_hist.to_excel(PATH_OUTPUT  + "/save_clustering_hist_kmeans.xlsx")
    df_hog.to_excel(PATH_OUTPUT   + "/save_clustering_hog_kmeans.xlsx")
    df_color.to_excel(PATH_OUTPUT + "/save_clustering_color_kmeans.xlsx")
    df_metric.to_excel(PATH_OUTPUT + "/save_metric.xlsx")
    print("Fin. \n\n Pour avoir la visualisation dashboard : streamlit run dashboard_clustering.py")

def hog_kmeans():
    print("\n\n ##### Chargement des données Snacks + split train/test ######")
    (
        images_train, labels_train, paths_train, images_color_train,
        images_test,  labels_test,  paths_test,  images_color_test
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


def hog_hsv_kmeans():
    from sklearn.preprocessing import normalize

    print("\n\n ##### Chargement des données Snacks + split train/test ######")
    (
        images_train, labels_train, paths_train, images_color_train,
        images_test,  labels_test,  paths_test,  images_color_test
    ) = load_train_test_split()
    print(f"- {len(CLASS_NAMES)} classes")

    print("- calcul features hog (train)...")
    desc_hog_train  = compute_hog_descriptors(images_train)
    print("- calcul features hog (test)...")
    desc_hog_test   = compute_hog_descriptors(images_test)
    print("- calcul features HSV (train)...")
    desc_hsv_train  = compute_hsv_histograms(images_color_train)
    print("- calcul features HSV (test)...")
    desc_hsv_test   = compute_hsv_histograms(images_color_test)

    number_cluster = len(CLASS_NAMES)

    # --- HOG : StandardScaler puis L2 norm + PCA ---
    # StandardScaler centre les features HOG avant PCA (nécessaire pour PCA)
    scaler_hog = StandardScaler().fit(desc_hog_train)
    desc_hog_train_n = scaler_hog.transform(desc_hog_train)
    desc_hog_test_n  = scaler_hog.transform(desc_hog_test)

    # PCA avec plus de composantes pour capturer plus de variance HOG
    pca = PCA(n_components=100, random_state=42)
    desc_hog_train_pca = pca.fit_transform(desc_hog_train_n)
    desc_hog_test_pca  = pca.transform(desc_hog_test_n)

    # L2 normalisation sur HOG réduit (meilleure géométrie pour KMeans)
    desc_hog_train_pca = normalize(desc_hog_train_pca, norm='l2')
    desc_hog_test_pca  = normalize(desc_hog_test_pca,  norm='l2')

    # --- HSV : déjà L1-normalisé dans compute_hsv_histograms ---
    # On applique juste une L2 norm pour cohérence avec HOG
    desc_hsv_train_n = normalize(desc_hsv_train, norm='l2')
    desc_hsv_test_n  = normalize(desc_hsv_test,  norm='l2')

    # --- Concaténation avec poids HSV × 2 (couleur plus discriminante pour les aliments) ---
    HSV_WEIGHT = 3.0
    desc_combined_train = np.concatenate(
        [desc_hog_train_pca, HSV_WEIGHT * desc_hsv_train_n], axis=1
    )
    desc_combined_test = np.concatenate(
        [desc_hog_test_pca,  HSV_WEIGHT * desc_hsv_test_n],  axis=1
    )

    # L2 normalisation finale sur le vecteur combiné
    desc_combined_train = normalize(desc_combined_train, norm='l2')
    desc_combined_test  = normalize(desc_combined_test,  norm='l2')

    print(f"- Descripteur combiné : {desc_combined_train.shape[1]} dimensions "
          f"({desc_hog_train_pca.shape[1]} HOG-PCA + {desc_hsv_train_n.shape[1]} HSV×{HSV_WEIGHT})")

    # --- KMeans sur le descripteur combiné ---
    km = SklearnKMeans(n_clusters=number_cluster, random_state=42, n_init=50)
    km.fit(desc_combined_train)
    pred_test = km.predict(desc_combined_test)

    metric = show_metric(labels_test, pred_test, desc_combined_test,
                         bool_show=True, name_descriptor="HOG+HSV", bool_return=True)

    x_3d = conversion_3d(desc_combined_test)
    df   = create_df_to_export(x_3d, labels_test, pred_test, paths_test)

    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    df.to_excel(PATH_OUTPUT + "/save_clustering_hog_hsv_kmeans.xlsx")
    pd.DataFrame([metric]).to_excel(PATH_OUTPUT + "/save_metric_hog_hsv.xlsx")
    tab = np.array([float(metric["ami"]), float(metric["silhouette"])], dtype=float)
    tab_score_hog.append(tab)
    print("Fin hog_hsv_kmeans.")


def hu_moments_pipeline():
    """
    Pipeline Hu Moments + KMeans / DBSCAN
    ======================================
    Les moments de Hu sont 7 invariants de forme (translation, rotation, échelle).
    On les compare avec deux modèles de clustering :
      - KMeans  : nombre de clusters fixé = nombre de classes
      - DBSCAN  : découverte automatique des clusters par densité
                  (eps et min_samples à ajuster selon le dataset)
    """
    from sklearn.cluster import KMeans as SklearnKMeans, DBSCAN as SklearnDBSCAN

    print("\n\n ##### [Hu Moments] Chargement des données Snacks + split train/test ######")
    (
        images_train, labels_train, paths_train, _,
        images_test,  labels_test,  paths_test,  _
    ) = load_train_test_split()

    print("\n\n ##### [Hu Moments] Extraction des descripteurs ######")
    hu_train = compute_hu_moments(images_train)
    hu_test  = compute_hu_moments(images_test)
    print(f"- Train : {hu_train.shape}  |  Test : {hu_test.shape}  (7 invariants/image)")

    print("\n\n ##### [Hu Moments] Normalisation (fit sur train) ######")
    scaler_hu = StandardScaler().fit(hu_train)
    hu_train_n = scaler_hu.transform(hu_train)
    hu_test_n  = scaler_hu.transform(hu_test)

    print("\n\n ##### [Hu Moments] Clustering ######")
    number_cluster = len(CLASS_NAMES)

    # --- KMeans : fit sur train, predict sur test ---
    print(f"- KMeans ({number_cluster} clusters)...")
    km_hu = SklearnKMeans(n_clusters=number_cluster, random_state=42, n_init=10)
    km_hu.fit(hu_train_n)
    pred_km_test = km_hu.predict(hu_test_n)

    # --- DBSCAN : fit sur train, assignation test au barycentre le plus proche ---
    print("- DBSCAN (eps=0.6, min_samples=5)...")
    dbscan = SklearnDBSCAN(eps=0.6, min_samples=15)
    dbscan.fit(hu_train_n)
    n_db = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    n_noise = int(np.sum(dbscan.labels_ == -1))
    print(f"  → {n_db} clusters sur train, {n_noise} points de bruit")

    if n_db > 0:
        cluster_ids = [c for c in set(dbscan.labels_) if c != -1]
        centers = np.array([hu_train_n[dbscan.labels_ == c].mean(axis=0) for c in cluster_ids])
        dists = np.sqrt(((hu_test_n[:, np.newaxis] - centers) ** 2).sum(axis=2))
        pred_db_test = np.array([cluster_ids[i] for i in np.argmin(dists, axis=1)])
    else:
        pred_db_test = np.zeros(len(labels_test), dtype=int)

    print("\n\n ##### [Hu Moments] Résultats (métriques sur TEST) ######")
    metric_hu_kmeans = show_metric(labels_test, pred_km_test, hu_test_n,
                                   bool_show=True, name_descriptor="HU_KMEANS",
                                   name_model="kmeans", bool_return=True)
    metric_hu_dbscan = show_metric(labels_test, pred_db_test, hu_test_n,
                                   bool_show=True, name_descriptor="HU_DBSCAN",
                                   name_model="dbscan", bool_return=True)

    print("\n- Export des données TEST vers le dashboard")
    x_3d_hu = conversion_3d(hu_test_n)

    df_hu_kmeans = create_df_to_export(x_3d_hu, labels_test, pred_km_test, paths_test)
    df_hu_dbscan = create_df_to_export(x_3d_hu, labels_test, pred_db_test, paths_test)
    df_metric_hu = pd.DataFrame([metric_hu_kmeans, metric_hu_dbscan])

    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    df_hu_kmeans.to_excel(PATH_OUTPUT + "/save_clustering_hu_kmeans.xlsx")
    df_hu_dbscan.to_excel(PATH_OUTPUT + "/save_clustering_hu_dbscan.xlsx")
    df_metric_hu.to_excel(PATH_OUTPUT + "/save_metric_hu.xlsx")
    print("Fin. \n\n Pour visualiser, lancer : streamlit run dashboard_clustering.py")


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

    # Représentation dédiée DBSCAN : histogrammes L2-normalisés
    # (plus stable que z-score + PCA pour une distance de similarité entre histogrammes)
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

    # --- KMeans : fit sur train, predict sur test ---
    print(f"- KMeans ({number_cluster} clusters)...")
    km_hist = SklearnKMeans(n_clusters=number_cluster, random_state=42, n_init=15)
    km_hist.fit(hist_train_pca)
    pred_km_test = km_hist.predict(hist_test_pca)

    # --- DBSCAN : fit sur train, assignation test au barycentre le plus proche ---
    dbscan_metric = "cosine"
    dbscan_eps = 0.15
    dbscan_min_samples = 3
    print(f"- DBSCAN (eps={dbscan_eps}, min_samples={dbscan_min_samples}, metric='{dbscan_metric}')...")
    dbscan = SklearnDBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric=dbscan_metric)
    dbscan.fit(hist_train_db_pca)
    n_db = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    n_noise = int(np.sum(dbscan.labels_ == -1))
    print(f"  → {n_db} clusters sur train, {n_noise} points de bruit")

    if n_db > 0:
        cluster_ids = [c for c in set(dbscan.labels_) if c != -1]
        centers = np.array([hist_train_db_pca[dbscan.labels_ == c].mean(axis=0) for c in cluster_ids])
        dists = pairwise_distances(hist_test_db_pca, centers, metric=dbscan_metric)
        pred_db_test = np.array([cluster_ids[i] for i in np.argmin(dists, axis=1)])
    else:
        # Aucun cluster trouvé sur train : on conserve explicitement "bruit"
        pred_db_test = np.full(len(labels_test), -1, dtype=int)

    print("\n\n ##### [Hist] Résultats (métriques sur TEST) ######")
    metric_hist_kmeans = show_metric(labels_test, pred_km_test, hist_test_pca,
                                   bool_show=True, name_descriptor="HIST_KMEANS",
                                   name_model="kmeans", bool_return=True)
    metric_hist_dbscan = show_metric(labels_test, pred_db_test, hist_test_db_pca,
                                   bool_show=True, name_descriptor="HIST_DBSCAN",
                                   name_model="dbscan", bool_return=True)

    print("\n- Export des données TEST vers le dashboard")
    x_3d_hu = conversion_3d(hist_test_pca)

    df_hist_kmeans = create_df_to_export(x_3d_hu, labels_test, pred_km_test, paths_test)
    df_hist_dbscan = create_df_to_export(x_3d_hu, labels_test, pred_db_test, paths_test)
    df_metric_hist = pd.DataFrame([metric_hist_kmeans, metric_hist_dbscan])

    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    df_hist_kmeans.to_excel(PATH_OUTPUT + "/save_clustering_hist_kmeans.xlsx")
    df_hist_dbscan.to_excel(PATH_OUTPUT + "/save_clustering_hist_dbscan.xlsx")
    df_metric_hist.to_excel(PATH_OUTPUT + "/save_metric_hist.xlsx")
    print("Fin. \n\n Pour visualiser, lancer : streamlit run dashboard_clustering.py")
if __name__ == "__main__":
    # pipeline()             # HISTOGRAM + HOG + COLOR_HSV avec KMeans
    # hu_moments_pipeline()  # Hu Moments avec KMeans + DBSCAN
    # hog_kmeans()
    # hog_hsv_kmeans()
    hist_dbscan_kmeans_pipeline()