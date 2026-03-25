import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from PIL import Image

from .constant import CLASS_NAMES

_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def colorize_cluster(cluster_data, selected_cluster):
    df_plot = cluster_data.copy()
    df_plot['cluster'] = df_plot['cluster'].astype(str)  # couleurs catégorielles
    fig = px.scatter_3d(df_plot, x='x', y='y', z='z', color='cluster')
    filtered = cluster_data[cluster_data['cluster'] == selected_cluster]
    fig.add_scatter3d(
        x=filtered['x'], y=filtered['y'], z=filtered['z'],
        mode='markers', marker=dict(color='red', size=10),
        name=f'Cluster {selected_cluster}'
    )
    return fig


@st.cache_data
def plot_metric(df_metric):
    fig = px.bar(
        df_metric, x='descriptor', y='ami', color='descriptor',
        title="Score AMI par descripteur",
        labels={'ami': 'Adjusted Mutual Information', 'descriptor': 'Descripteur'}
    )
    st.plotly_chart(fig)


# ------------------------------------------------------------------
# Chargement des fichiers (chemins relatifs au script)
# ------------------------------------------------------------------
def _load(filename):
    path = os.path.join(_DIR, "output", filename)
    df = pd.read_excel(path)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns='Unnamed: 0', inplace=True)
    return df


df_bovw_kmeans = _load("save_clustering_bovw_sift_kmeans.xlsx")
try:
    df_bovw_dbscan = _load("save_clustering_bovw_sift_dbscan.xlsx")
    BOVW_DBSCAN_AVAILABLE = True
except FileNotFoundError:
    BOVW_DBSCAN_AVAILABLE = False
df_metric_bovw = _load("save_metric_bovw.xlsx")

try:
    df_combined_kmeans = _load("save_clustering_combined_kmeans.xlsx")
    df_combined_dbscan = _load("save_clustering_combined_dbscan.xlsx")
    df_metric_combined = _load("save_metric_combined.xlsx")
    COMBINED_AVAILABLE = True
except FileNotFoundError:
    COMBINED_AVAILABLE = False


# ------------------------------------------------------------------
# Interface
# ------------------------------------------------------------------
tab1, tab2 = st.tabs(["Analyse par descripteur", "Analyse global"])

with tab1:
    st.write('## Résultat de Clustering des données Snacks')
    st.sidebar.write("#### Veuillez sélectionner les clusters à analyser")

    # Sélection du descripteur
    descriptor_options = ["BOVW_KMEANS"]
    if BOVW_DBSCAN_AVAILABLE:
        descriptor_options += ["BOVW_DBSCAN"]
    else:
        st.sidebar.info("⚠️ Lancez `BovW()` dans `pipeline2.py` pour activer BOVW_DBSCAN.")
    if COMBINED_AVAILABLE:
        descriptor_options += ["COMBINED_KMEANS", "COMBINED_DBSCAN"]
    else:
        st.sidebar.info("⚠️ Lancez `combined_pipeline()` dans `pipeline2.py` pour activer les descripteurs COMBINED.")

    descriptor = st.sidebar.selectbox('Sélectionner un descripteur', descriptor_options)

    if descriptor == "BOVW_KMEANS":
        df = df_bovw_kmeans
    elif descriptor == "BOVW_DBSCAN":
        df = df_bovw_dbscan
    elif descriptor == "COMBINED_KMEANS":
        df = df_combined_kmeans
    elif descriptor == "COMBINED_DBSCAN":
        df = df_combined_dbscan

    # Sélection dynamique des clusters (exclut -1 = bruit DBSCAN)
    available_clusters = sorted([int(c) for c in df['cluster'].unique() if int(c) >= 0])
    n_noise = int((df['cluster'] == -1).sum())
    if n_noise > 0:
        st.sidebar.warning(f"⚠️ DBSCAN : {n_noise} points de bruit (label -1) non affichés")

    if len(available_clusters) == 0:
        st.warning("Aucun cluster exploitable à afficher pour ce descripteur.")
        st.stop()

    def dominant_class(cluster_id):
        subset = df[df['cluster'] == cluster_id]
        if len(subset) == 0:
            return "(vide)"
        return CLASS_NAMES[int(subset['label'].mode().iloc[0])]

    cluster_options = [f"Cluster {c}  —  {dominant_class(c)}" for c in available_clusters]
    selected_option = st.sidebar.selectbox('Sélectionner un Cluster', cluster_options)
    selected_cluster = int(selected_option.split()[1])

    filtered_data = df[df['cluster'] == selected_cluster]

    st.write(f"### Analyse du descripteur {descriptor}")
    st.write(f"#### Cluster {selected_cluster}  ·  {len(filtered_data)} images  ·  classe dominante : **{dominant_class(selected_cluster)}**")

    # --- Composition du cluster ---
    st.write("#### 📊 Composition du cluster")
    class_counts = (
        filtered_data['label']
        .value_counts()
        .rename(index=lambda i: CLASS_NAMES[int(i)])
        .sort_values(ascending=False)
        .reset_index()
    )
    class_counts.columns = ['Classe', "Nombre d'images"]
    class_counts['%'] = (class_counts["Nombre d'images"] / len(filtered_data) * 100).round(1).astype(str) + ' %'

    col_table, col_bar = st.columns([1, 2])
    with col_table:
        st.dataframe(class_counts, hide_index=True, use_container_width=True)
    with col_bar:
        fig_comp = px.bar(
            class_counts, x='Classe', y="Nombre d'images",
            color='Classe',
            title=f"Répartition des classes — Cluster {selected_cluster}",
        )
        fig_comp.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_comp, use_container_width=True)

    # --- Visualisation 3D ---
    st.write(f"#### Visualisation 3D du clustering — {descriptor}")
    fig = colorize_cluster(df, selected_cluster)
    st.plotly_chart(fig)

    # --- Images ---
    if 'image_path' in filtered_data.columns and len(filtered_data) > 0:
        st.write(f"#### Images du cluster {selected_cluster}  ({len(filtered_data)} images au total)")
        n_display = min(15, len(filtered_data))
        cols = st.columns(5)
        for i, (_, row) in enumerate(filtered_data.head(n_display).iterrows()):
            try:
                img = Image.open(row['image_path'])
                cols[i % 5].image(img, caption=f"✅ {CLASS_NAMES[int(row['label'])]}", use_column_width=True)
            except Exception as e:
                cols[i % 5].write(f"Erreur : {e}")
    else:
        st.write("Aucune image disponible pour ce cluster.")

with tab2:
    st.write('## Analyse Globale des descripteurs')

    all_dfs = [df_metric_bovw]
    if COMBINED_AVAILABLE:
        all_dfs.append(df_metric_combined)
    df_all = pd.concat(all_dfs, ignore_index=True)

    plot_metric(df_all)
    st.write('## Métriques détaillées')
    st.dataframe(df_all)
