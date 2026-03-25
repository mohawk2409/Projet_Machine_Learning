"""
Pipeline dédié au clustering hiérarchique sur 4 descripteurs :
- HOG
- Histogramme niveaux de gris
- LBP
- BoVW (SIFT)
"""

import os
import cv2
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    pairwise_distances,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

from features import (
    load_train_test_split,
    compute_hog_descriptors,
    compute_gray_histograms,
    compute_lbp_descriptor,
)
from utils import conversion_3d, create_df_to_export
from constant import PATH_OUTPUT, CLASS_NAMES


def _safe_silhouette(x, y_pred):
    unique_labels = np.unique(y_pred)
    if len(unique_labels) < 2:
        return float("nan")
    return float(silhouette_score(x, y_pred))


def _safe_silhouette_with_metric(x, y_pred, metric_name):
    unique_labels = np.unique(y_pred)
    if len(unique_labels) < 2:
        return float("-inf")
    try:
        if metric_name == "euclidean":
            return float(silhouette_score(x, y_pred))
        return float(silhouette_score(x, y_pred, metric=metric_name))
    except Exception:
        return float("-inf")


def show_metric(
    y_true,
    y_pred,
    x=None,
    bool_show=True,
    name_descriptor="DESCRIPTOR",
    name_model="hierarchical",
    bool_return=False,
):
    metric = {
        "descriptor": name_descriptor,
        "model": name_model,
        "ami": float(adjusted_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "homogeneity": float(homogeneity_score(y_true, y_pred)),
        "completeness": float(completeness_score(y_true, y_pred)),
        "v_measure": float(v_measure_score(y_true, y_pred)),
    }

    if x is not None:
        try:
            metric["silhouette"] = _safe_silhouette(x, y_pred)
        except Exception:
            metric["silhouette"] = float("nan")
    else:
        metric["silhouette"] = float("nan")

    if bool_show:
        print(f"\n--- Metrics {name_descriptor} ({name_model}) ---")
        print(f"AMI          : {metric['ami']:.4f}")
        print(f"ARI          : {metric['ari']:.4f}")
        print(f"Homogeneity  : {metric['homogeneity']:.4f}")
        print(f"Completeness : {metric['completeness']:.4f}")
        print(f"V-measure    : {metric['v_measure']:.4f}")
        if np.isnan(metric["silhouette"]):
            print("Silhouette   : NaN")
        else:
            print(f"Silhouette   : {metric['silhouette']:.4f}")

    if bool_return:
        return metric
    return None


def _compute_bovw_train_test(images_train, images_test, vocab_size=150):
    sift = cv2.SIFT_create()
    desc_per_image_train = []
    desc_per_image_test = []
    all_desc_train = []

    for img in images_train:
        _, des = sift.detectAndCompute(img, None)
        desc_per_image_train.append(des)
        if des is not None and len(des) > 0:
            all_desc_train.append(des)

    for img in images_test:
        _, des = sift.detectAndCompute(img, None)
        desc_per_image_test.append(des)

    if len(all_desc_train) == 0:
        raise ValueError("Aucun descripteur SIFT trouvé sur l'ensemble train pour construire BoVW.")

    all_descriptors_stacked = np.vstack(all_desc_train)
    vocab_size = min(vocab_size, all_descriptors_stacked.shape[0])
    vocab_size = max(vocab_size, 2)

    vocab_km = SklearnKMeans(n_clusters=vocab_size, random_state=42, n_init=3, max_iter=300)
    vocab_km.fit(all_descriptors_stacked)

    def encode_bovw(desc_per_image):
        features = []
        for des in desc_per_image:
            if des is None or len(des) == 0:
                hist = np.zeros(vocab_size, dtype=float)
            else:
                words = vocab_km.predict(des)
                hist, _ = np.histogram(words, bins=np.arange(vocab_size + 1))
                hist = hist.astype(float)
                if hist.sum() > 0:
                    hist /= hist.sum()
            features.append(hist)
        return np.array(features)

    bovw_train = encode_bovw(desc_per_image_train)
    bovw_test = encode_bovw(desc_per_image_test)
    return bovw_train, bovw_test


def _fit_agglomerative(x_train, n_clusters, linkage_name, metric_name):
    if linkage_name == "ward":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    else:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_name,
            metric=metric_name,
        )
    labels = model.fit_predict(x_train)
    return model, labels


def _cluster_centers(x_train, cluster_labels, n_clusters):
    return np.array([x_train[cluster_labels == c].mean(axis=0) for c in range(n_clusters)])


def _predict_from_centers(x_test, centers, metric_name):
    dists = pairwise_distances(x_test, centers, metric=metric_name)
    return np.argmin(dists, axis=1)


def _transform_histogram_like(x_train, x_test):
    # Histogram-friendly transform: L1 norm -> sqrt (Hellinger) -> L2 norm.
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    x_train = np.clip(x_train, a_min=0.0, a_max=None)
    x_test = np.clip(x_test, a_min=0.0, a_max=None)
    x_train = normalize(x_train, norm="l1")
    x_test = normalize(x_test, norm="l1")
    x_train = np.sqrt(x_train)
    x_test = np.sqrt(x_test)
    x_train = normalize(x_train, norm="l2")
    x_test = normalize(x_test, norm="l2")
    return x_train, x_test


def _select_best_hierarchical_config(x_train, y_train, n_clusters, candidates):
    best = None
    best_score = (float("-inf"), float("-inf"))
    best_train_sil = float("-inf")
    best_train_ami = float("-inf")

    for cfg in candidates:
        linkage_name = cfg["linkage"]
        metric_name = cfg["metric"]

        _, train_clusters = _fit_agglomerative(x_train, n_clusters, linkage_name, metric_name)
        train_sil = _safe_silhouette_with_metric(x_train, train_clusters, metric_name)
        train_ami = float(adjusted_mutual_info_score(y_train, train_clusters))

        # Priorite a l'alignement labels/cluster (AMI), puis silhouette en tie-break.
        score = (train_ami, train_sil)

        if score > best_score:
            best_score = score
            best = cfg
            best_train_sil = train_sil
            best_train_ami = train_ami

    return best, best_train_ami, best_train_sil


def hierarchical_pipeline():
    print("\n\n ##### [HIERARCHICAL] Chargement des données Snacks + split train/test ######")
    (
        images_train, labels_train, paths_train, _,
        images_test, labels_test, paths_test, _
    ) = load_train_test_split()

    number_cluster = len(CLASS_NAMES)

    print("\n\n ##### [HIERARCHICAL] Extraction des descripteurs ######")
    print("- HOG...")
    hog_train = compute_hog_descriptors(images_train)
    hog_test = compute_hog_descriptors(images_test)

    print("- Histogramme gris...")
    hist_train = compute_gray_histograms(images_train)
    hist_test = compute_gray_histograms(images_test)

    print("- LBP...")
    lbp_train = compute_lbp_descriptor(images_train)
    lbp_test = compute_lbp_descriptor(images_test)

    print("- BoVW (SIFT)...")
    bovw_train, bovw_test = _compute_bovw_train_test(images_train, images_test, vocab_size=150)

    print("\n\n ##### [HIERARCHICAL] Pre-traitement par descripteur ######")
    scaler_hog = StandardScaler().fit(hog_train)
    hog_train_n = scaler_hog.transform(hog_train)
    hog_test_n = scaler_hog.transform(hog_test)
    pca_hog_dim = min(40, hog_train_n.shape[1], hog_train_n.shape[0])
    pca_hog = PCA(n_components=pca_hog_dim, random_state=42, whiten=False)
    hog_train_f = pca_hog.fit_transform(hog_train_n)
    hog_test_f = pca_hog.transform(hog_test_n)
    hog_train_f = normalize(hog_train_f, norm="l2")
    hog_test_f = normalize(hog_test_f, norm="l2")

    scaler_hist = StandardScaler().fit(hist_train)
    hist_train_n = scaler_hist.transform(hist_train)
    hist_test_n = scaler_hist.transform(hist_test)

    # LBP tuned: StandardScaler + PCA(3)
    scaler_lbp_h = StandardScaler().fit(lbp_train)
    lbp_train_s = scaler_lbp_h.transform(lbp_train)
    lbp_test_s = scaler_lbp_h.transform(lbp_test)
    pca_lbp_h = PCA(n_components=min(3, lbp_train_s.shape[1], lbp_train_s.shape[0]), random_state=42, whiten=True)
    lbp_train_n = pca_lbp_h.fit_transform(lbp_train_s)
    lbp_test_n = pca_lbp_h.transform(lbp_test_s)

    # BoVW tuned: StandardScaler only
    scaler_bovw_h = StandardScaler().fit(bovw_train)
    bovw_train_n = scaler_bovw_h.transform(bovw_train)
    bovw_test_n = scaler_bovw_h.transform(bovw_test)

    descriptor_data = {
        "HOG_HIERARCHICAL": {
            "x_train": hog_train_f,
            "x_test": hog_test_f,
            "candidates": [
                {"linkage": "average", "metric": "euclidean"},
            ],
        },
        "HIST_HIERARCHICAL": {
            "x_train": hist_train_n,
            "x_test": hist_test_n,
            "candidates": [
                {"linkage": "ward", "metric": "euclidean"},
            ],
        },
        "LBP_HIERARCHICAL": {
            "x_train": lbp_train_n,
            "x_test": lbp_test_n,
            "candidates": [
                {"linkage": "ward", "metric": "euclidean"},
            ],
        },
        "BOVW_HIERARCHICAL": {
            "x_train": bovw_train_n,
            "x_test": bovw_test_n,
            "candidates": [
                {"linkage": "average", "metric": "cosine"},
            ],
        },
    }

    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    metric_rows = []

    print("\n\n ##### [HIERARCHICAL] Clustering et metriques ######")
    for descriptor_name, cfg in descriptor_data.items():
        print(f"- {descriptor_name}...")

        x_train = cfg["x_train"]
        x_test = cfg["x_test"]
        best_cfg, best_train_ami, best_train_sil = _select_best_hierarchical_config(
            x_train,
            labels_train,
            number_cluster,
            cfg["candidates"],
        )
        print(
            f"  -> best linkage={best_cfg['linkage']}, metric={best_cfg['metric']}, "
            f"train AMI={best_train_ami:.4f}, train silhouette={best_train_sil:.4f}"
        )

        _, train_clusters = _fit_agglomerative(
            x_train,
            number_cluster,
            best_cfg["linkage"],
            best_cfg["metric"],
        )
        centers = _cluster_centers(x_train, train_clusters, number_cluster)
        pred_test = _predict_from_centers(x_test, centers, best_cfg["metric"])

        metric = show_metric(
            labels_test,
            pred_test,
            x_test,
            bool_show=True,
            name_descriptor=descriptor_name,
            name_model=f"hierarchical_{best_cfg['linkage']}_{best_cfg['metric']}",
            bool_return=True,
        )
        metric_rows.append(metric)

        x_3d = conversion_3d(x_test)
        df_export = create_df_to_export(x_3d, labels_test, pred_test, paths_test)
        df_export.to_excel(PATH_OUTPUT + f"/save_clustering_{descriptor_name.lower()}.xlsx")

    df_metric = pd.DataFrame(metric_rows)
    df_metric.to_excel(PATH_OUTPUT + "/save_metric_hierarchical.xlsx")

    print("Fin HIERARCHICAL.")


def lbp_dbscan_kmeans_pipeline():
    """
    Pipeline LBP avec KMeans et DBSCAN.
    Exporte les fichiers attendus par le dashboard Streamlit.
    """
    print("\n\n ##### [LBP] Chargement des données Snacks + split train/test ######")
    (
        images_train, labels_train, paths_train, _,
        images_test, labels_test, paths_test, _
    ) = load_train_test_split()

    print("\n\n ##### [LBP] Extraction des descripteurs ######")
    lbp_train = compute_lbp_descriptor(images_train)
    lbp_test = compute_lbp_descriptor(images_test)
    print(f"- Train : {lbp_train.shape}  |  Test : {lbp_test.shape}")

    print("\n\n ##### [LBP] Pré-traitement ######")
    scaler_lbp = StandardScaler().fit(lbp_train)
    lbp_train_n = scaler_lbp.transform(lbp_train)
    lbp_test_n = scaler_lbp.transform(lbp_test)

    pca_dim = min(3, lbp_train_n.shape[1], lbp_train_n.shape[0])
    pca = PCA(n_components=pca_dim, random_state=42, whiten=True)
    lbp_train_pca = pca.fit_transform(lbp_train_n)
    lbp_test_pca = pca.transform(lbp_test_n)

    # DBSCAN tuned space: z-score + L2
    lbp_train_db = normalize(lbp_train_n, norm="l2")
    lbp_test_db = normalize(lbp_test_n, norm="l2")

    number_cluster = len(CLASS_NAMES)
    print("\n\n ##### [LBP] Clustering ######")

    print(f"- KMeans ({number_cluster} clusters)...")
    km_lbp = SklearnKMeans(n_clusters=number_cluster, random_state=42, n_init=40)
    km_lbp.fit(lbp_train_pca)
    pred_km_test = km_lbp.predict(lbp_test_pca)

    dbscan_metric = "euclidean"
    dbscan_eps = 0.15
    dbscan_min_samples = 3
    print(f"- DBSCAN (eps={dbscan_eps}, min_samples={dbscan_min_samples}, metric='{dbscan_metric}')...")
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric=dbscan_metric)
    dbscan.fit(lbp_train_db)
    n_db = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    n_noise = int(np.sum(dbscan.labels_ == -1))
    print(f"  -> {n_db} clusters sur train, {n_noise} points de bruit")

    if n_db > 0:
        cluster_ids = [c for c in set(dbscan.labels_) if c != -1]
        centers = np.array([lbp_train_db[dbscan.labels_ == c].mean(axis=0) for c in cluster_ids])
        pred_db_test = _predict_from_centers(lbp_test_db, centers, dbscan_metric)
        pred_db_test = np.array([cluster_ids[i] for i in pred_db_test])
    else:
        pred_db_test = np.full(len(labels_test), -1, dtype=int)

    print("\n\n ##### [LBP] Résultats (métriques sur TEST) ######")
    metric_lbp_kmeans = show_metric(
        labels_test,
        pred_km_test,
        lbp_test_pca,
        bool_show=True,
        name_descriptor="LBP_KMEANS",
        name_model="kmeans",
        bool_return=True,
    )
    metric_lbp_dbscan = show_metric(
        labels_test,
        pred_db_test,
        lbp_test_db,
        bool_show=True,
        name_descriptor="LBP_DBSCAN",
        name_model="dbscan",
        bool_return=True,
    )

    print("\n- Export des données TEST vers le dashboard")
    x_3d_lbp = conversion_3d(lbp_test_n)
    df_lbp_kmeans = create_df_to_export(x_3d_lbp, labels_test, pred_km_test, paths_test)
    df_lbp_dbscan = create_df_to_export(x_3d_lbp, labels_test, pred_db_test, paths_test)
    df_metric_lbp = pd.DataFrame([metric_lbp_kmeans, metric_lbp_dbscan])

    if not os.path.exists(PATH_OUTPUT):
        os.makedirs(PATH_OUTPUT)

    df_lbp_kmeans.to_excel(PATH_OUTPUT + "/save_clustering_lbp_kmeans.xlsx")
    df_lbp_dbscan.to_excel(PATH_OUTPUT + "/save_clustering_lbp_dbscan.xlsx")
    df_metric_lbp.to_excel(PATH_OUTPUT + "/save_metric_lbp.xlsx")
    print("Fin LBP KMeans + DBSCAN.")


def run_all_pipelines_for_dashboard():
    """
    Lance les pipelines nécessaires au dashboard dédié et crée
    un fichier métrique consolidé.
    """
    hierarchical_pipeline()
    lbp_dbscan_kmeans_pipeline()

    metric_paths = [
        PATH_OUTPUT + "/save_metric_hierarchical.xlsx",
        PATH_OUTPUT + "/save_metric_lbp.xlsx",
    ]
    metric_dfs = []
    for path in metric_paths:
        if os.path.exists(path):
            df = pd.read_excel(path)
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
            metric_dfs.append(df)

    if len(metric_dfs) > 0:
        df_all = pd.concat(metric_dfs, ignore_index=True)
        df_all.to_excel(PATH_OUTPUT + "/save_metric_hierarchical_all.xlsx")
        print("Métriques consolidées exportées : output/save_metric_hierarchical_all.xlsx")


if __name__ == "__main__":
    run_all_pipelines_for_dashboard()
