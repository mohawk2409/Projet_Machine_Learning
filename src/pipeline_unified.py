import os

import pandas as pd

try:
    from .constant import PATH_OUTPUT
    from .pipeline import hist_dbscan_kmeans_pipeline, hog_dbscan_kmeans_pipeline
    from .pipeline2 import BovW
    from .pipeline_hierarchical import hierarchical_pipeline, lbp_dbscan_kmeans_pipeline
except ImportError:
    from constant import PATH_OUTPUT
    from pipeline import hist_dbscan_kmeans_pipeline, hog_dbscan_kmeans_pipeline
    from pipeline2 import BovW
    from pipeline_hierarchical import hierarchical_pipeline, lbp_dbscan_kmeans_pipeline


OUTPUT_DIR = PATH_OUTPUT
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
ALLOWED_METRIC_FILES = {
    "save_metric_kmeans.xlsx",
    "save_metric_dbscan.xlsx",
    "save_metric_hierarchical.xlsx",
    "save_metric_all.xlsx",
}


def _safe_read_metric(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_excel(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def cleanup_combined_outputs():
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
    if df.empty or "descriptor" not in df.columns:
        return None

    candidates = df[df["descriptor"].astype(str).str.lower().apply(descriptor_predicate)]
    if candidates.empty:
        return None

    # Prefer non-PCA row when both are present.
    non_pca = candidates[~candidates["descriptor"].astype(str).str.lower().str.contains("pca")]
    row = non_pca.iloc[0] if not non_pca.empty else candidates.iloc[0]
    return row


def _metric_to_record(row, descriptor, model):
    wanted = ["ami", "ari", "homogeneity", "completeness", "v_measure", "silhouette"]
    out = {"descriptor": descriptor, "model": model}
    for key in wanted:
        out[key] = row.get(key, float("nan"))
    return out


def consolidate_metrics():
    metric_hist = _safe_read_metric(os.path.join(OUTPUT_DIR, "save_metric_hist.xlsx"))
    metric_hog = _safe_read_metric(os.path.join(OUTPUT_DIR, "save_metric_hog.xlsx"))
    metric_lbp = _safe_read_metric(os.path.join(OUTPUT_DIR, "save_metric_lbp.xlsx"))
    metric_bovw = _safe_read_metric(os.path.join(OUTPUT_DIR, "save_metric_bovw.xlsx"))
    metric_hier = _safe_read_metric(os.path.join(OUTPUT_DIR, "save_metric_hierarchical.xlsx"))

    kmeans_rows = []
    dbscan_rows = []
    hierarchical_rows = []

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
    normalize_and_prune_outputs()

    print("Unified pipeline done.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_pipeline()
