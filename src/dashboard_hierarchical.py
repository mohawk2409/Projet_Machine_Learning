import os
import colorsys
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from constant import CLASS_NAMES, DATA_PATH

_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.abspath(DATA_PATH)
_OUTPUT_CANDIDATES = [
    os.path.join(_DIR, "output"),
    os.path.abspath(os.path.join(_DIR, "..", "output")),
]


def _find_output_dir():
    for output_dir in _OUTPUT_CANDIDATES:
        if os.path.isdir(output_dir):
            return output_dir
    return _OUTPUT_CANDIDATES[-1]


_OUTPUT_DIR = _find_output_dir()


def _load_excel_optional(filename):
    path = os.path.join(_OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_excel(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def _fix_image_path(path):
    if isinstance(path, str) and os.path.exists(path):
        return path

    if not isinstance(path, str):
        return path

    norm = path.replace("\\", "/")
    marker = "images/data/test/"
    idx = norm.find(marker)
    if idx != -1:
        relative = norm[idx + len(marker):]
        candidate = os.path.join(_DATA_PATH, relative)
        if os.path.exists(candidate):
            return candidate
    return path


@st.cache_data
def _scatter_3d(df_plot, selected_cluster):
    unique_clusters = sorted(df_plot["cluster"].unique())
    n_clusters = len(unique_clusters)

    color_map = {}
    for i, cluster_id in enumerate(unique_clusters):
        hue = i / max(n_clusters, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
        color_map[str(cluster_id)] = f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"

    tmp = df_plot.copy()
    tmp["cluster"] = tmp["cluster"].astype(str)
    fig = px.scatter_3d(
        tmp,
        x="x",
        y="y",
        z="z",
        color="cluster",
        color_discrete_map=color_map,
        title="Projection 3D des clusters",
    )

    highlight = df_plot[df_plot["cluster"] == selected_cluster]
    fig.add_scatter3d(
        x=highlight["x"],
        y=highlight["y"],
        z=highlight["z"],
        mode="markers",
        marker=dict(color="red", size=6),
        name=f"Cluster {selected_cluster}",
    )
    return fig


@st.cache_data
def _bar_ami(df_metrics):
    fig = px.bar(
        df_metrics,
        x="descriptor",
        y="ami",
        color="descriptor",
        title="AMI par descripteur",
        labels={"descriptor": "Descripteur", "ami": "Adjusted Mutual Information"},
    )
    return fig


def _dominant_class(df_desc, cluster_id):
    subset = df_desc[df_desc["cluster"] == cluster_id]
    if len(subset) == 0:
        return "(vide)"
    label = int(subset["label"].mode().iloc[0])
    return CLASS_NAMES[label]


def main():
    st.set_page_config(page_title="Dashboard Clustering Hierarchical", layout="wide")
    st.title("Dashboard Clustering - Pipeline Hierarchical")
    st.caption(f"Dossier des résultats: {_OUTPUT_DIR}")

    descriptor_files = {
        "HOG_HIERARCHICAL": "save_clustering_hog_hierarchical.xlsx",
        "HIST_HIERARCHICAL": "save_clustering_hist_hierarchical.xlsx",
        "LBP_HIERARCHICAL": "save_clustering_lbp_hierarchical.xlsx",
        "BOVW_HIERARCHICAL": "save_clustering_bovw_hierarchical.xlsx",
        "LBP_KMEANS": "save_clustering_lbp_kmeans.xlsx",
        "LBP_DBSCAN": "save_clustering_lbp_dbscan.xlsx",
    }

    descriptor_data = {}
    for name, filename in descriptor_files.items():
        df = _load_excel_optional(filename)
        if df is not None:
            descriptor_data[name] = df

    metrics_all = _load_excel_optional("save_metric_hierarchical_all.xlsx")
    if metrics_all is None:
        metrics_h = _load_excel_optional("save_metric_hierarchical.xlsx")
        metrics_lbp = _load_excel_optional("save_metric_lbp.xlsx")
        metric_frames = [d for d in [metrics_h, metrics_lbp] if d is not None]
        metrics_all = pd.concat(metric_frames, ignore_index=True) if len(metric_frames) > 0 else pd.DataFrame()

    if len(descriptor_data) == 0:
        st.warning("Aucun fichier de clustering trouvé. Lance d'abord: python src/pipeline_hierarchical.py")
        st.stop()

    tab_desc, tab_metrics = st.tabs(["Analyse par descripteur", "Analyse globale"])

    with tab_desc:
        names = sorted(descriptor_data.keys())
        selected_descriptor = st.sidebar.selectbox("Descripteur", names)
        df_selected = descriptor_data[selected_descriptor]

        clusters = sorted([int(c) for c in df_selected["cluster"].unique() if int(c) >= 0])
        n_noise = int((df_selected["cluster"] == -1).sum())
        if n_noise > 0:
            st.sidebar.warning(f"DBSCAN: {n_noise} points de bruit (-1) non affichés")

        cluster_labels = [f"Cluster {c} - {_dominant_class(df_selected, c)}" for c in clusters]
        selected_cluster_label = st.sidebar.selectbox("Cluster", cluster_labels)
        selected_cluster = int(selected_cluster_label.split()[1])

        filtered = df_selected[df_selected["cluster"] == selected_cluster]

        st.subheader(f"Descripteur: {selected_descriptor}")
        st.write(
            f"Cluster {selected_cluster} | {len(filtered)} images | classe dominante: {_dominant_class(df_selected, selected_cluster)}"
        )

        fig = _scatter_3d(df_selected, selected_cluster)
        st.plotly_chart(fig, width="stretch")

        composition = (
            filtered["label"]
            .value_counts()
            .rename(index=lambda i: CLASS_NAMES[int(i)])
            .sort_values(ascending=False)
            .reset_index()
        )
        composition.columns = ["Classe", "Nombre"]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(composition, hide_index=True, width="stretch")
        with col2:
            fig_comp = px.bar(composition, x="Classe", y="Nombre", color="Classe", title="Composition du cluster")
            fig_comp.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig_comp, width="stretch")

        if "image_path" in filtered.columns and len(filtered) > 0:
            st.subheader("Aperçu images")
            n_display = min(15, len(filtered))
            shown = filtered.sample(n=n_display, random_state=42)
            cols = st.columns(5)
            for i, (_, row) in enumerate(shown.iterrows()):
                path = _fix_image_path(row["image_path"])
                try:
                    img = Image.open(path)
                    cols[i % 5].image(img, caption=CLASS_NAMES[int(row["label"])], width="stretch")
                except Exception as exc:
                    cols[i % 5].write(f"Erreur image: {exc}")

    with tab_metrics:
        st.subheader("Métriques globales")
        if metrics_all.empty:
            st.info("Aucune métrique disponible.")
        else:
            if "descriptor" in metrics_all.columns:
                st.plotly_chart(_bar_ami(metrics_all), width="stretch")
            st.dataframe(metrics_all, width="stretch")


if __name__ == "__main__":
    main()
