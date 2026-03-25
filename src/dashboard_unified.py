import colorsys
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

try:
    from .constant import CLASS_NAMES, DATA_PATH
except ImportError:
    from constant import CLASS_NAMES, DATA_PATH


_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.abspath(DATA_PATH)
_OUTPUT_CANDIDATES = [
    os.path.join(_DIR, "output"),
    os.path.abspath(os.path.join(_DIR, "..", "output")),
]
ALLOWED_DESCRIPTOR_MODEL = {
    ("kmeans", "hist"),
    ("kmeans", "hog"),
    ("kmeans", "lbp"),
    ("kmeans", "bovw"),
    ("dbscan", "hist"),
    ("dbscan", "hog"),
    ("dbscan", "lbp"),
    ("dbscan", "bovw"),
    ("hierarchical", "hog"),
    ("hierarchical", "hist"),
    ("hierarchical", "lbp"),
    ("hierarchical", "bovw"),
}


def find_output_dir():
    for path in _OUTPUT_CANDIDATES:
        if os.path.isdir(path):
            return path
    return _OUTPUT_CANDIDATES[0]


def fix_image_path(path):
    if isinstance(path, str) and os.path.exists(path):
        return path
    if not isinstance(path, str):
        return path

    norm = path.replace("\\", "/")
    marker = "images/data/test/"
    idx = norm.find(marker)
    if idx != -1:
        relative = norm[idx + len(marker) :]
        candidate = os.path.join(_DATA_PATH, relative)
        if os.path.exists(candidate):
            return candidate
    return path


@st.cache_data
def load_outputs(output_dir):
    clustering = {}
    metrics = {}

    if not os.path.isdir(output_dir):
        return clustering, metrics

    for filename in os.listdir(output_dir):
        if filename.startswith("save_clustering_") and filename.endswith(".xlsx"):
            body = filename[len("save_clustering_") : -len(".xlsx")]
            if "_" not in body:
                continue
            descriptor, model = body.rsplit("_", 1)
            if "combined" in descriptor.lower():
                continue
            if (model.lower(), descriptor.lower()) not in ALLOWED_DESCRIPTOR_MODEL:
                continue
            path = os.path.join(output_dir, filename)
            df = pd.read_excel(path)
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
            clustering[(model, descriptor)] = df

        if filename == "save_metric.xlsx":
            path = os.path.join(output_dir, filename)
            df = pd.read_excel(path)
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
            if "descriptor" in df.columns:
                df = df[~df["descriptor"].astype(str).str.lower().str.contains("combined")]
            metrics["kmeans"] = df

        if filename.startswith("save_metric_") and filename.endswith(".xlsx"):
            path = os.path.join(output_dir, filename)
            df = pd.read_excel(path)
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
            if "descriptor" in df.columns:
                df = df[~df["descriptor"].astype(str).str.lower().str.contains("combined")]

            key = filename[len("save_metric_") : -len(".xlsx")]
            if key != "all":
                metrics[key] = df

    return clustering, metrics


@st.cache_data
def make_3d_plot(df, selected_cluster):
    tmp = df.copy()
    clusters = sorted(tmp["cluster"].unique())

    color_map = {}
    for i, cid in enumerate(clusters):
        hue = i / max(1, len(clusters))
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
        color_map[str(cid)] = f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"

    tmp["cluster"] = tmp["cluster"].astype(str)
    fig = px.scatter_3d(
        tmp,
        x="x",
        y="y",
        z="z",
        color="cluster",
        color_discrete_map=color_map,
        title="3D clustering view",
    )

    focus = df[df["cluster"] == selected_cluster]
    fig.add_scatter3d(
        x=focus["x"],
        y=focus["y"],
        z=focus["z"],
        mode="markers",
        marker=dict(color="red", size=7),
        name=f"Cluster {selected_cluster}",
    )
    return fig


def dominant_class(df, cluster_id):
    subset = df[df["cluster"] == cluster_id]
    if subset.empty:
        return "(empty)"
    label = int(subset["label"].mode().iloc[0])
    return CLASS_NAMES[label]


def main():
    st.set_page_config(page_title="Unified Clustering Dashboard", layout="wide")
    st.title("Unified Clustering Dashboard")

    output_dir = find_output_dir()
    st.caption(f"Results folder: {output_dir}")

    clustering, metrics = load_outputs(output_dir)
    if len(clustering) == 0:
        st.error("No clustering result file found. Run pipeline_unified.py first.")
        return

    models = sorted({model for model, _ in clustering.keys()})
    selected_model = st.sidebar.selectbox("Model", models)

    descriptors = sorted([descriptor for model, descriptor in clustering.keys() if model == selected_model])
    selected_descriptor = st.sidebar.selectbox("Descriptor", descriptors)

    df = clustering[(selected_model, selected_descriptor)]
    required = {"x", "y", "z", "label", "cluster"}
    if not required.issubset(set(df.columns)):
        st.error("Selected file does not have required columns: x, y, z, label, cluster.")
        return

    non_noise_clusters = sorted([int(c) for c in df["cluster"].unique() if int(c) >= 0])
    if len(non_noise_clusters) == 0:
        st.warning("Only noise labels detected for this result.")
        return

    options = [f"{cid} - {dominant_class(df, cid)}" for cid in non_noise_clusters]
    selected = st.sidebar.selectbox("Cluster", options)
    selected_cluster = int(selected.split(" - ")[0])

    filtered = df[df["cluster"] == selected_cluster]

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader(f"{selected_model.upper()} + {selected_descriptor.upper()}")
        st.plotly_chart(make_3d_plot(df, selected_cluster), width="stretch")

    with col_right:
        st.metric("Points", len(df))
        st.metric("Clusters", len([c for c in df["cluster"].unique() if int(c) >= 0]))
        st.metric("Noise", int((df["cluster"] == -1).sum()))
        st.metric("Selected cluster size", len(filtered))

    st.subheader("Cluster composition")
    comp = (
        filtered["label"]
        .value_counts()
        .rename(index=lambda i: CLASS_NAMES[int(i)])
        .sort_values(ascending=False)
        .reset_index()
    )
    comp.columns = ["Class", "Count"]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(comp, width="stretch", hide_index=True)
    with c2:
        fig_comp = px.bar(comp, x="Class", y="Count", color="Class", title="Class distribution")
        fig_comp.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_comp, width="stretch")

    st.subheader("Metrics")
    metric_df = metrics.get(selected_model)
    if metric_df is None:
        st.info("No metric file found for this model.")
    else:
        row = metric_df
        if "descriptor" in metric_df.columns:
            row = metric_df[metric_df["descriptor"].astype(str).str.lower() == selected_descriptor.lower()]
        if row.empty:
            st.info("No metric row found for selected descriptor.")
        else:
            st.dataframe(row, width="stretch", hide_index=True)

    st.subheader("Sample images")
    if "image_path" not in filtered.columns:
        st.info("No image_path column in selected file.")
    else:
        preview = filtered.sample(n=min(15, len(filtered)), random_state=42)
        cols = st.columns(5)
        for idx, (_, item) in enumerate(preview.iterrows()):
            path = fix_image_path(item["image_path"])
            try:
                img = Image.open(path)
                label_name = CLASS_NAMES[int(item["label"])]
                cols[idx % 5].image(img, caption=label_name, width="stretch")
            except Exception:
                cols[idx % 5].write("Image not available")

    st.subheader("Global metrics comparison")
    metric_frames = []
    for model_name, df_metric in metrics.items():
        tmp = df_metric.copy()
        if "model" not in tmp.columns:
            tmp.insert(0, "model", model_name)
        metric_frames.append(tmp)

    if len(metric_frames) > 0:
        all_metrics = pd.concat(metric_frames, ignore_index=True)
        st.dataframe(all_metrics, width="stretch", hide_index=True)
        if {"descriptor", "ami", "model"}.issubset(set(all_metrics.columns)):
            pivot = all_metrics.pivot_table(index="descriptor", columns="model", values="ami", aggfunc="first")
            st.write("AMI pivot by descriptor/model")
            st.table(pivot.round(4))


if __name__ == "__main__":
    main()
