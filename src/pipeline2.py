"""
Pipeline 2 — Descripteurs combinés + BoVW + SIFT
=================================================

Ce pipeline explore deux approches :

1. BoVW + SIFT (Bag of Visual Words)
   → Encode la texture locale via un vocabulaire de patches SIFT.
   → Gris uniquement : bon pour les textures, mais ignore la couleur.

2. COMBINED : HSV + HOG + LBP  ← le plus performant
   → HSV histogram : capture la distribution des couleurs (très discriminant
     pour la nourriture : orange vs carotte, fraise vs pomme...)
   → HOG : capture les gradients/contours (forme générale du produit)
   → LBP : capture la texture locale (granuleux, lisse, fibreux...)
   → Combinaison : KMeans sklearn (init k-means++) + DBSCAN avec PCA
"""

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import pandas as pd
import cv2
import numpy as np

from features import *
from clustering import KMeans, show_metric
from utils import conversion_3d, create_df_to_export
from constant import PATH_OUTPUT, CLASS_NAMES, DATA_PATH


def BovW():

    print("\n\n ##### [BoVW] Chargement des données Snacks + split train/test ######")
    (
        images_train, labels_train, paths_train, _,
        images_test,  labels_test,  paths_test,  _
    ) = load_train_test_split()

    # ------------------------------------------------------------------
    # ÉTAPE 1 : Extraction SIFT
    # Le vocabulaire est construit sur le TRAIN uniquement.
    # ------------------------------------------------------------------
    print("\n\n ##### [BoVW] Étape 1 – Extraction SIFT ######")
    sift = cv2.SIFT_create()

    all_desc_train = []
    desc_per_image_train = []
    desc_per_image_test  = []

    for img in images_train:
        _, des = sift.detectAndCompute(img, None)
        if des is not None:
            all_desc_train.append(des)
            desc_per_image_train.append(des)
        else:
            desc_per_image_train.append(None)

    for img in images_test:
        _, des = sift.detectAndCompute(img, None)
        desc_per_image_test.append(des)

    all_descriptors_stacked = np.vstack(all_desc_train)
    print(f"- {all_descriptors_stacked.shape[0]} descripteurs SIFT extraits (train)")

    # ------------------------------------------------------------------
    # ÉTAPE 2 : Vocabulaire visuel (sur TRAIN uniquement)
    # ------------------------------------------------------------------
    VOCAB_SIZE = 150
    print(f"\n\n ##### [BoVW] Étape 2 – Vocabulaire visuel (K={VOCAB_SIZE}, sur train) ######")

    from sklearn.cluster import KMeans as SklearnKMeans
    vocab_km = SklearnKMeans(n_clusters=VOCAB_SIZE, random_state=42, n_init=3, max_iter=300)
    vocab_km.fit(all_descriptors_stacked)

    class _VocabWrapper:
        def __init__(self, sk_model):
            self._model = sk_model
        def nearest_cluster(self, X):
            return self._model.predict(X)

    vocab = _VocabWrapper(vocab_km)
    print(f"- Vocabulaire construit : {VOCAB_SIZE} mots visuels")

    # ------------------------------------------------------------------
    # ÉTAPE 3 : Encodage BoVW (train ET test)
    # ------------------------------------------------------------------
    def encode_bovw(desc_per_image):
        features = []
        for des in desc_per_image:
            if des is not None:
                words = vocab.nearest_cluster(des)
                hist, _ = np.histogram(words, bins=np.arange(VOCAB_SIZE + 1))
                hist = hist.astype(float)
                if hist.sum() > 0:
                    hist /= hist.sum()
            else:
                hist = np.zeros(VOCAB_SIZE, dtype=float)
            features.append(hist)
        return np.array(features)

    print("\n\n ##### [BoVW] Étape 3 – Encodage BoVW ######")
    bovw_train = encode_bovw(desc_per_image_train)
    bovw_test  = encode_bovw(desc_per_image_test)
    print(f"- Train : {bovw_train.shape}  |  Test : {bovw_test.shape}")

    # ------------------------------------------------------------------
    # CLUSTERING : fit sur train, predict sur test
    # ------------------------------------------------------------------
    number_cluster = len(CLASS_NAMES)
    print(f"\n\n ##### [BoVW] Clustering KMeans ({number_cluster} clusters) – fit train ######")
    km = SklearnKMeans(n_clusters=number_cluster, random_state=42, n_init=10)
    km.fit(bovw_train)
    pred_test = km.predict(bovw_test)
    print("- Prédiction sur le test : OK")

    print("\n\n ##### [BoVW] Résultats (métriques sur TEST) ######")
    metric_bovw = show_metric(
        labels_test, pred_test, bovw_test,
        bool_show=True, name_descriptor="BoVW+SIFT", bool_return=True
    )

    print("\n- Export des données TEST vers le dashboard")
    scaler = StandardScaler().fit(bovw_train)
    bovw_test_norm = scaler.transform(bovw_test)
    x_3d_bovw = conversion_3d(bovw_test_norm)

    df_bovw = create_df_to_export(x_3d_bovw, labels_test, pred_test, paths_test)
    df_metric = pd.DataFrame([metric_bovw])

    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    df_bovw.to_excel(PATH_OUTPUT + "/save_clustering_bovw_sift_kmeans.xlsx")
    df_metric.to_excel(PATH_OUTPUT + "/save_metric_bovw.xlsx")
    print("Fin BoVW.")


def combined_pipeline():
    """
    Pipeline avec descripteur combiné : HSV + HOG + LBP
    =====================================================
    Combine trois descripteurs complémentaires :
      - HSV histogram (34 dims)  : distribution des couleurs
      - HOG           (1176 dims) : gradients / forme
      - LBP           (10 dims)  : texture locale
    Total : ~1220 dimensions par image.

    Méthodologie train/test :
      - Descripteurs calculés sur train ET test.
      - Scaler ajusté sur train, appliqué sur test (pas de data leakage).
      - sklearn KMeans : fit sur train, predict sur test.
      - DBSCAN : fit sur train (PCA), assign test au cluster le plus proche.
    """
    from sklearn.cluster import KMeans as SklearnKMeans, DBSCAN as SklearnDBSCAN
    from sklearn.decomposition import PCA

    print("\n\n ##### [COMBINED] Chargement des données Snacks + split train/test ######")
    (
        images_train, labels_train, paths_train, images_color_train,
        images_test,  labels_test,  paths_test,  images_color_test
    ) = load_train_test_split()

    # ------------------------------------------------------------------
    # EXTRACTION
    # ------------------------------------------------------------------
    print("\n\n ##### [COMBINED] Extraction des descripteurs ######")
    print("- HSV (train + test)...")
    hsv_train  = compute_hsv_histograms(images_color_train)
    hsv_test   = compute_hsv_histograms(images_color_test)
    print("- HOG (train + test)...")
    hog_train  = compute_hog_descriptors(images_train)
    hog_test   = compute_hog_descriptors(images_test)
    print("- LBP (train + test)...")
    lbp_train  = compute_lbp_descriptor(images_train)
    lbp_test   = compute_lbp_descriptor(images_test)

    combined_train = np.concatenate([hsv_train, hog_train, lbp_train], axis=1)
    combined_test  = np.concatenate([hsv_test,  hog_test,  lbp_test],  axis=1)
    print(f"- Vecteur combiné : train {combined_train.shape}  |  test {combined_test.shape}")

    # ------------------------------------------------------------------
    # NORMALISATION : fit sur train uniquement
    # ------------------------------------------------------------------
    print("\n\n ##### [COMBINED] Normalisation ######")
    scaler = StandardScaler().fit(combined_train)
    train_norm = scaler.transform(combined_train)
    test_norm  = scaler.transform(combined_test)

    # ------------------------------------------------------------------
    # CLUSTERING 1 : KMeans – fit sur train, predict sur test
    # ------------------------------------------------------------------
    number_cluster = len(CLASS_NAMES)
    print(f"\n\n ##### [COMBINED] KMeans ({number_cluster} clusters) ######")
    km = SklearnKMeans(n_clusters=number_cluster, random_state=42, n_init=10)
    km.fit(train_norm)
    pred_km_test = km.predict(test_norm)
    print("- Prédiction KMeans sur test : OK")

    # ------------------------------------------------------------------
    # CLUSTERING 2 : DBSCAN sur PCA
    # DBSCAN n'a pas de predict() natif — on assigne chaque point test
    # au cluster dont le centre (barycentre) est le plus proche.
    # ------------------------------------------------------------------
    print("\n\n ##### [COMBINED] DBSCAN (PCA 50 comp.) ######")
    pca = PCA(n_components=50, random_state=42)
    train_pca = pca.fit_transform(train_norm)   # fit PCA sur train
    test_pca  = pca.transform(test_norm)         # transform test
    print(f"- Variance expliquée : {pca.explained_variance_ratio_.sum():.1%}")

    dbscan = SklearnDBSCAN(eps=3.0, min_samples=5)
    dbscan.fit(train_pca)
    n_dbscan_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    n_noise_train = int(np.sum(dbscan.labels_ == -1))
    print(f"  → {n_dbscan_clusters} clusters détectés sur train, {n_noise_train} points de bruit")

    # Assignation du test : chaque point test → barycentre le plus proche
    if n_dbscan_clusters > 0:
        cluster_ids = [c for c in set(dbscan.labels_) if c != -1]
        centers = np.array([
            train_pca[dbscan.labels_ == c].mean(axis=0) for c in cluster_ids
        ])
        dists = np.sqrt(((test_pca[:, np.newaxis] - centers) ** 2).sum(axis=2))
        pred_db_test = np.array([cluster_ids[i] for i in np.argmin(dists, axis=1)])
    else:
        pred_db_test = np.zeros(len(labels_test), dtype=int)
    print(f"- Assignation DBSCAN sur test : OK")

    # ------------------------------------------------------------------
    # MÉTRIQUES (sur TEST)
    # ------------------------------------------------------------------
    print("\n\n ##### [COMBINED] Résultats (métriques sur TEST) ######")
    metric_kmeans = show_metric(
        labels_test, pred_km_test, test_norm,
        bool_show=True, name_descriptor="COMBINED_KMEANS",
        name_model="kmeans", bool_return=True
    )
    metric_dbscan = show_metric(
        labels_test, pred_db_test, test_pca,
        bool_show=True, name_descriptor="COMBINED_DBSCAN",
        name_model="dbscan", bool_return=True
    )

    # ------------------------------------------------------------------
    # EXPORT (données TEST uniquement dans le dashboard)
    # ------------------------------------------------------------------
    print("\n- Export des données TEST vers le dashboard")
    x_3d = conversion_3d(test_norm)

    df_km = create_df_to_export(x_3d, labels_test, pred_km_test, paths_test)
    df_db = create_df_to_export(x_3d, labels_test, pred_db_test, paths_test)
    df_metric_combined = pd.DataFrame([metric_kmeans, metric_dbscan])

    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    df_km.to_excel(PATH_OUTPUT + "/save_clustering_combined_kmeans.xlsx")
    df_db.to_excel(PATH_OUTPUT + "/save_clustering_combined_dbscan.xlsx")
    df_metric_combined.to_excel(PATH_OUTPUT + "/save_metric_combined.xlsx")
    print("Fin COMBINED.")

if __name__ == "__main__":
    combined_pipeline()   # HSV + HOG + LBP  ← meilleur descripteur
    BovW()                # BoVW + SIFT       ← descripteur de texture

