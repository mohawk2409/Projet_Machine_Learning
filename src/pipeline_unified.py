import os

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import StandardScaler, normalize

try:
    from .constant import PATH_OUTPUT
    from .pipeline import hist_dbscan_kmeans_pipeline, hog_dbscan_kmeans_pipeline
    from .pipeline2 import BovW
    from .pipeline_hierarchical import hierarchical_pipeline, lbp_dbscan_kmeans_pipeline
    from .pipeline_hierarchical import _compute_bovw_train_test, _fit_agglomerative, _cluster_centers, _predict_from_centers
    from .features import load_train_test_split, compute_hog_descriptors, compute_gray_histograms, compute_lbp_descriptor
except ImportError:
    from constant import PATH_OUTPUT
    from pipeline import hist_dbscan_kmeans_pipeline, hog_dbscan_kmeans_pipeline
    from pipeline2 import BovW
    from pipeline_hierarchical import hierarchical_pipeline, lbp_dbscan_kmeans_pipeline
    from pipeline_hierarchical import _compute_bovw_train_test, _fit_agglomerative, _cluster_centers, _predict_from_centers
    from features import load_train_test_split, compute_hog_descriptors, compute_gray_histograms, compute_lbp_descriptor


OUTPUT_DIR = PATH_OUTPUT
# Fichiers de clustering conservés après nettoyage.
ALLOWED_CLUSTERING_FILES = {
    "save_clustering_hist_kmeans.xlsx",
    "save_clustering_hist_dbscan.xlsx",
    "save_clustering_hog_kmeans.xlsx",
    "save_clustering_hog_dbscan.xlsx",
    "save_clustering_lbp_kmeans.xlsx",
    "save_clustering_lbp_dbscan.xlsx",
    "save_clustering_bovw_kmeans.xlsx",
    "save_clustering_bovw_dbscan.xlsx",
    "save_clustering_hog_hierarchical.xlsx",
    "save_clustering_hist_hierarchical.xlsx",
    "save_clustering_lbp_hierarchical.xlsx",
    "save_clustering_bovw_hierarchical.xlsx",
}
# Fichiers de métriques conservés après nettoyage.
ALLOWED_METRIC_FILES = {
    "save_metric_kmeans.xlsx",
    "save_metric_dbscan.xlsx",
    "save_metric_hierarchical.xlsx",
    "save_metric_all.xlsx",
    "save_metric_silhouette_curve.xlsx",
}


def _safe_silhouette_euclidean(x, labels):
    """Retourne la silhouette (euclidienne) ou NaN si non calculable"""
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")
    try:
        return float(silhouette_score(x, labels))
    except Exception:
        return float("nan")


def generate_silhouette_curves(k_values=(5, 10, 15, 20, 25)):
    """Génère les courbes de silhouette pour KMeans et hiérarchique"""
    print("Computing silhouette curves (KMeans + Hierarchical) from pipelines...")

    (
        images_train, labels_train, _, _,
        images_test, _, _, _,
    ) = load_train_test_split()

    hog_train = compute_hog_descriptors(images_train)
    hog_test = compute_hog_descriptors(images_test)

    hist_train = compute_gray_histograms(images_train)
    hist_test = compute_gray_histograms(images_test)

    lbp_train = compute_lbp_descriptor(images_train)
    lbp_test = compute_lbp_descriptor(images_test)

    bovw_train, bovw_test = _compute_bovw_train_test(images_train, images_test, vocab_size=150)

    # Espaces de features utilisés par les pipelines KMeans.
    scaler_hist_k = StandardScaler().fit(hist_train)
    hist_train_n_k = scaler_hist_k.transform(hist_train)
    hist_test_n_k = scaler_hist_k.transform(hist_test)
    pca_hist_k = PCA(n_components=min(30, hist_train_n_k.shape[1], hist_train_n_k.shape[0]), random_state=42)
    hist_train_kmeans = pca_hist_k.fit_transform(hist_train_n_k)
    hist_test_kmeans = pca_hist_k.transform(hist_test_n_k)

    scaler_hog_k = StandardScaler().fit(hog_train)
    hog_train_n_k = scaler_hog_k.transform(hog_train)
    hog_test_n_k = scaler_hog_k.transform(hog_test)
    pca_hog_k = PCA(n_components=min(30, hog_train_n_k.shape[1], hog_train_n_k.shape[0]), random_state=42)
    hog_train_kmeans = pca_hog_k.fit_transform(hog_train_n_k)
    hog_test_kmeans = pca_hog_k.transform(hog_test_n_k)

    scaler_lbp_k = StandardScaler().fit(lbp_train)
    lbp_train_n_k = scaler_lbp_k.transform(lbp_train)
    lbp_test_n_k = scaler_lbp_k.transform(lbp_test)
    pca_lbp_k = PCA(n_components=min(3, lbp_train_n_k.shape[1], lbp_train_n_k.shape[0]), random_state=42, whiten=True)
    lbp_train_kmeans = pca_lbp_k.fit_transform(lbp_train_n_k)
    lbp_test_kmeans = pca_lbp_k.transform(lbp_test_n_k)

    bovw_train_kmeans = bovw_train
    bovw_test_kmeans = bovw_test

    # Espaces de features utilisés par le pipeline hiérarchique
    scaler_hog_h = StandardScaler().fit(hog_train)
    hog_train_n_h = scaler_hog_h.transform(hog_train)
    hog_test_n_h = scaler_hog_h.transform(hog_test)
    pca_hog_h = PCA(n_components=min(40, hog_train_n_h.shape[1], hog_train_n_h.shape[0]), random_state=42, whiten=False)
    hog_train_h = normalize(pca_hog_h.fit_transform(hog_train_n_h), norm="l2")
    hog_test_h = normalize(pca_hog_h.transform(hog_test_n_h), norm="l2")

    scaler_hist_h = StandardScaler().fit(hist_train)
    hist_train_h = scaler_hist_h.transform(hist_train)
    hist_test_h = scaler_hist_h.transform(hist_test)

    scaler_lbp_h = StandardScaler().fit(lbp_train)
    lbp_train_s_h = scaler_lbp_h.transform(lbp_train)
    lbp_test_s_h = scaler_lbp_h.transform(lbp_test)
    pca_lbp_h = PCA(n_components=min(3, lbp_train_s_h.shape[1], lbp_train_s_h.shape[0]), random_state=42, whiten=True)
    lbp_train_h = pca_lbp_h.fit_transform(lbp_train_s_h)
    lbp_test_h = pca_lbp_h.transform(lbp_test_s_h)

    scaler_bovw_h = StandardScaler().fit(bovw_train)
    bovw_train_h = scaler_bovw_h.transform(bovw_train)
    bovw_test_h = scaler_bovw_h.transform(bovw_test)

    kmeans_data = {
        "hist": {"x_train": hist_train_kmeans, "x_test": hist_test_kmeans, "n_init": 15},
        "hog": {"x_train": hog_train_kmeans, "x_test": hog_test_kmeans, "n_init": 15},
        "lbp": {"x_train": lbp_train_kmeans, "x_test": lbp_test_kmeans, "n_init": 40},
        "bovw": {"x_train": bovw_train_kmeans, "x_test": bovw_test_kmeans, "n_init": 10},
    }

    hierarchical_data = {
        "hog": {"x_train": hog_train_h, "x_test": hog_test_h, "linkage": "average", "metric": "euclidean"},
        "hist": {"x_train": hist_train_h, "x_test": hist_test_h, "linkage": "ward", "metric": "euclidean"},
        "lbp": {"x_train": lbp_train_h, "x_test": lbp_test_h, "linkage": "ward", "metric": "euclidean"},
        "bovw": {"x_train": bovw_train_h, "x_test": bovw_test_h, "linkage": "average", "metric": "cosine"},
    }

    rows = []  # Lignes du fichier save_metric_silhouette_curve.xlsx

    for descriptor, cfg in kmeans_data.items():
        x_train = cfg["x_train"]
        x_test = cfg["x_test"]
        for k in k_values:
            if k < 2 or k >= x_train.shape[0]:
                rows.append({"descriptor": descriptor, "model": "kmeans", "k": int(k), "silhouette": float("nan")})
                continue
            km = SklearnKMeans(n_clusters=int(k), random_state=42, n_init=cfg["n_init"])
            km.fit(x_train)
            pred_test = km.predict(x_test)
            sil = _safe_silhouette_euclidean(x_test, pred_test)
            rows.append({"descriptor": descriptor, "model": "kmeans", "k": int(k), "silhouette": sil})

    for descriptor, cfg in hierarchical_data.items():
        x_train = cfg["x_train"]
        x_test = cfg["x_test"]
        for k in k_values:
            if k < 2 or k >= x_train.shape[0]:
                rows.append({"descriptor": descriptor, "model": "hierarchical", "k": int(k), "silhouette": float("nan")})
                continue
            _, train_clusters = _fit_agglomerative(x_train, int(k), cfg["linkage"], cfg["metric"])
            centers = _cluster_centers(x_train, train_clusters, int(k))
            pred_test = _predict_from_centers(x_test, centers, cfg["metric"])
            sil = _safe_silhouette_euclidean(x_test, pred_test)
            rows.append({"descriptor": descriptor, "model": "hierarchical", "k": int(k), "silhouette": sil})

    df_sil = pd.DataFrame(rows)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    df_sil.to_excel(os.path.join(OUTPUT_DIR, "save_metric_silhouette_curve.xlsx"), index=False)


def _safe_read_metric(path):
    """Lit un fichier métrique Excel et supprime l'index parasite si présent"""
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_excel(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def cleanup_combined_outputs():
    """Supprime les exports Excel legacy contenant 'combined'"""
    if not os.path.isdir(OUTPUT_DIR):
        return
    for name in os.listdir(OUTPUT_DIR):
        lower = name.lower()
        if "combined" in lower and lower.endswith(".xlsx"):
            try:
                os.remove(os.path.join(OUTPUT_DIR, name))
            except OSError:
                pass


def normalize_and_prune_outputs():
    """Normalise les noms legacy BoVW puis supprime les exports non autorisés"""
    legacy_bovw_kmeans = os.path.join(OUTPUT_DIR, "save_clustering_bovw_sift_kmeans.xlsx")
    canonical_bovw_kmeans = os.path.join(OUTPUT_DIR, "save_clustering_bovw_kmeans.xlsx")
    if os.path.exists(legacy_bovw_kmeans):
        try:
            os.replace(legacy_bovw_kmeans, canonical_bovw_kmeans)
        except OSError:
            pass

    legacy_bovw_dbscan = os.path.join(OUTPUT_DIR, "save_clustering_bovw_sift_dbscan.xlsx")
    canonical_bovw_dbscan = os.path.join(OUTPUT_DIR, "save_clustering_bovw_dbscan.xlsx")
    if os.path.exists(legacy_bovw_dbscan):
        try:
            os.replace(legacy_bovw_dbscan, canonical_bovw_dbscan)
        except OSError:
            pass

    if not os.path.isdir(OUTPUT_DIR):
        return

    allowed = ALLOWED_CLUSTERING_FILES | ALLOWED_METRIC_FILES
    for name in os.listdir(OUTPUT_DIR):
        if not name.endswith(".xlsx"):
            continue
        if name.startswith("save_clustering_") or name.startswith("save_metric"):
            if name not in allowed:
                try:
                    os.remove(os.path.join(OUTPUT_DIR, name))
                except OSError:
                    pass


def _extract_metric_row(df, descriptor_predicate):
    """Extrait la ligne métrique correspondant au descripteur demandé"""
    if df.empty or "descriptor" not in df.columns:
        return None

    candidates = df[df["descriptor"].astype(str).str.lower().apply(descriptor_predicate)]
    if candidates.empty:
        return None

    # Privilégie la ligne non-PCA quand les deux existent.
    non_pca = candidates[~candidates["descriptor"].astype(str).str.lower().str.contains("pca")]
    row = non_pca.iloc[0] if not non_pca.empty else candidates.iloc[0]
    return row


def _metric_to_record(row, descriptor, model):
    """Mappe une ligne métrique brute vers un format standardisé"""
    wanted = ["ami", "ari", "homogeneity", "completeness", "v_measure", "silhouette"]
    out = {"descriptor": descriptor, "model": model}
    for key in wanted:
        out[key] = row.get(key, float("nan"))
    return out


def consolidate_metrics():
    """Consolide les métriques par modèle puis génère les fichiers finaux"""
    metric_hist = _safe_read_metric(os.path.join(OUTPUT_DIR, "save_metric_hist.xlsx"))
    metric_hog = _safe_read_metric(os.path.join(OUTPUT_DIR, "save_metric_hog.xlsx"))
    metric_lbp = _safe_read_metric(os.path.join(OUTPUT_DIR, "save_metric_lbp.xlsx"))
    metric_bovw = _safe_read_metric(os.path.join(OUTPUT_DIR, "save_metric_bovw.xlsx"))
    metric_hier = _safe_read_metric(os.path.join(OUTPUT_DIR, "save_metric_hierarchical.xlsx"))

    kmeans_rows = []
    dbscan_rows = []
    hierarchical_rows = []

    # Règles de sélection de la meilleure ligne KMeans par descripteur.
    mapping_kmeans = [
        (metric_hist, "hist", lambda v: "hist_kmeans" in v),
        (metric_hog, "hog", lambda v: "hog_kmeans" in v and "pca" not in v),
        (metric_lbp, "lbp", lambda v: "lbp_kmeans" in v),
        (metric_bovw, "bovw", lambda v: "bovw+sift" in v and "dbscan" not in v and "pca" not in v),
    ]
    for df, descriptor, pred in mapping_kmeans:
        row = _extract_metric_row(df, pred)
        if row is not None:
            kmeans_rows.append(_metric_to_record(row, descriptor, "kmeans"))

    # Règles de sélection de la meilleure ligne DBSCAN par descripteur.
    mapping_dbscan = [
        (metric_hist, "hist", lambda v: "hist_dbscan" in v),
        (metric_hog, "hog", lambda v: "hog_dbscan" in v and "pca" not in v),
        (metric_lbp, "lbp", lambda v: "lbp_dbscan" in v),
        (metric_bovw, "bovw", lambda v: "bovw+sift_dbscan" in v and "pca" not in v),
    ]
    for df, descriptor, pred in mapping_dbscan:
        row = _extract_metric_row(df, pred)
        if row is not None:
            dbscan_rows.append(_metric_to_record(row, descriptor, "dbscan"))

    # Règles de sélection de la ligne hiérarchique par descripteur.
    mapping_hier = [
        ("hog", lambda v: "hog_hierarchical" in v or v == "hog"),
        ("hist", lambda v: "hist_hierarchical" in v or v == "hist"),
        ("lbp", lambda v: "lbp_hierarchical" in v or v == "lbp"),
        ("bovw", lambda v: "bovw_hierarchical" in v or v == "bovw"),
    ]
    for descriptor, pred in mapping_hier:
        row = _extract_metric_row(metric_hier, pred)
        if row is not None:
            hierarchical_rows.append(_metric_to_record(row, descriptor, "hierarchical"))

    metric_kmeans_out = pd.DataFrame(kmeans_rows)
    metric_dbscan_out = pd.DataFrame(dbscan_rows)
    metric_hier_out = pd.DataFrame(hierarchical_rows)

    all_parts = [df for df in [metric_kmeans_out, metric_dbscan_out, metric_hier_out] if not df.empty]
    metric_all = pd.concat(all_parts, ignore_index=True) if len(all_parts) > 0 else pd.DataFrame()

    if not metric_kmeans_out.empty:
        metric_kmeans_out.to_excel(os.path.join(OUTPUT_DIR, "save_metric_kmeans.xlsx"), index=False)
    if not metric_dbscan_out.empty:
        metric_dbscan_out.to_excel(os.path.join(OUTPUT_DIR, "save_metric_dbscan.xlsx"), index=False)
    if not metric_hier_out.empty:
        metric_hier_out.to_excel(os.path.join(OUTPUT_DIR, "save_metric_hierarchical.xlsx"), index=False)
    if not metric_all.empty:
        metric_all.to_excel(os.path.join(OUTPUT_DIR, "save_metric_all.xlsx"), index=False)


def run_pipeline():
    """Exécute tous les pipelines, puis nettoie et consolide les sorties"""
    print("Running HIST KMeans + DBSCAN pipeline...")
    hist_dbscan_kmeans_pipeline()

    print("Running HOG KMeans + DBSCAN pipeline...")
    hog_dbscan_kmeans_pipeline()

    print("Running BoVW KMeans + DBSCAN pipeline...")
    BovW()

    print("Running hierarchical pipeline (HOG, HIST, LBP, BOVW)...")
    hierarchical_pipeline()

    print("Running LBP KMeans + DBSCAN pipeline...")
    lbp_dbscan_kmeans_pipeline()

    cleanup_combined_outputs()
    consolidate_metrics()
    generate_silhouette_curves(k_values=(5, 10, 15, 20, 25))
    normalize_and_prune_outputs()

    print("Unified pipeline done.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_pipeline()
