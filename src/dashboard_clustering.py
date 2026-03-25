import streamlit as st
import pandas as pd
import numpy as np
import colorsys
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
import cv2
from PIL import Image

import os
from .constant import CLASS_NAMES, DATA_PATH

_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.abspath(DATA_PATH)


def fix_image_path(path):
    """
    Corrige un chemin d'image obsolète (ex : après renommage du dossier projet).
    Si le chemin stocké n'existe plus, extrait la partie relative après
    'images/data/test/' et reconstruit le chemin absolu depuis DATA_PATH.

    Parameters
    ----------
    path : str
        Chemin absolu stocké dans le fichier Excel (potentiellement périmé).

    Returns
    -------
    str
        Chemin corrigé si le fichier est trouvé, chemin original sinon.
    """
    if os.path.exists(path):
        return path
    # Normaliser les séparateurs pour la recherche du marqueur
    norm = path.replace("\\", "/")
    marker = "images/data/test/"
    idx = norm.find(marker)
    if idx != -1:
        relative = norm[idx + len(marker):]
        new_path = os.path.join(_DATA_PATH, relative)
        if os.path.exists(new_path):
            return new_path
    return path  # fallback : chemin original


def format_mtime(path):
    """
    Retourne la date de dernière modification d'un fichier au format lisible.
    """
    if not os.path.exists(path):
        return "fichier introuvable"
    ts = os.path.getmtime(path)
    return pd.to_datetime(ts, unit='s').strftime("%Y-%m-%d %H:%M:%S")


@st.cache_data
def colorize_cluster(cluster_data, selected_cluster):
    """
    Génère un graphique 3D interactif colorisé par cluster, en mettant
    en évidence le cluster sélectionné en rouge.

    Paramètres : 
    - cluster_data : pd.DataFrame
        DataFrame contenant les colonnes 'x', 'y', 'z' (coordonnées t-SNE 3D)
        et 'cluster' (identifiant de cluster entier).
    - selected_cluster : int
        Identifiant du cluster à mettre en évidence (affiché en rouge et
        en taille plus grande).

    Return :
        plotly.graph_objs.Figure
        Figure Plotly 3D scatter avec tous les clusters colorisés et le
        cluster sélectionné superposé en rouge.
    """
    df_plot = cluster_data.copy()
    # Tri numérique pour une attribution de couleur stable entre exécutions
    unique_clusters = sorted(df_plot['cluster'].unique())
    n_clusters = len(unique_clusters)

    # Génère une palette de couleurs bien distinctes (espace HSV)
    # puis mappe chaque cluster à une couleur unique.
    color_map = {}
    for i, cluster_id in enumerate(unique_clusters):
        hue = i / max(n_clusters, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        color_map[str(cluster_id)] = f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"

    df_plot['cluster'] = df_plot['cluster'].astype(str)  # couleurs catégorielles
    fig = px.scatter_3d(
        df_plot,
        x='x',
        y='y',
        z='z',
        color='cluster',
        color_discrete_map=color_map
    )
    filtered_data = cluster_data[cluster_data['cluster'] == selected_cluster]
    fig.add_scatter3d(x=filtered_data['x'], y=filtered_data['y'], z=filtered_data['z'],
                    mode='markers', marker=dict(color='red', size=10),
                    name=f'Cluster {selected_cluster}')
    return fig

@st.cache_data
def plot_metric(df_metric):
    """
    Affiche un graphique en barres comparant le score AMI (Adjusted Mutual
    Information) pour chaque descripteur, directement dans l'interface
    Streamlit.

    Paramètres :
    - df_metric : pd.DataFrame
        DataFrame contenant au minimum les colonnes :
        - 'descriptor' : nom du descripteur (ex. 'HISTOGRAM', 'HOG', ...)
        - 'ami'        : score AMI associé au clustering de ce descripteur.
        
        La figure est rendue directement via `st.plotly_chart`.
    """
    fig1 = px.bar(df_metric, x='descriptor', y='ami',
                  color='descriptor',
                  title="Score AMI par descripteur",
                  labels={'ami': 'Adjusted Mutual Information', 'descriptor': 'Descripteur'})
    st.plotly_chart(fig1)

        
# Chargement des données du clustering (chemins relatifs au script)
df_hist = pd.read_excel(os.path.join(_DIR, "output", "save_clustering_hist_kmeans.xlsx"))

# Résultats Histogram DBSCAN (générés par hist_dbscan_kmeans_pipeline())
try:
    df_hist_dbscan = pd.read_excel(os.path.join(_DIR, "output", "save_clustering_hist_dbscan.xlsx"))
    HIST_DBSCAN_AVAILABLE = True
except FileNotFoundError:
    HIST_DBSCAN_AVAILABLE = False

# Résultats BoVW (générés par BovW())
try:
    df_bovw_kmeans = pd.read_excel(os.path.join(_DIR, "output", "save_clustering_bovw_sift_kmeans.xlsx"))
    BOVW_KMEANS_AVAILABLE = True
except FileNotFoundError:
    BOVW_KMEANS_AVAILABLE = False

try:
    df_bovw_kmeans_pca = pd.read_excel(os.path.join(_DIR, "output", "save_clustering_bovw_sift_kmeans_pca.xlsx"))
    BOVW_KMEANS_PCA_AVAILABLE = True
except FileNotFoundError:
    BOVW_KMEANS_PCA_AVAILABLE = False

try:
    df_bovw_dbscan = pd.read_excel(os.path.join(_DIR, "output", "save_clustering_bovw_sift_dbscan.xlsx"))
    BOVW_DBSCAN_AVAILABLE = True
except FileNotFoundError:
    BOVW_DBSCAN_AVAILABLE = False

try:
    df_bovw_dbscan_pca = pd.read_excel(os.path.join(_DIR, "output", "save_clustering_bovw_sift_dbscan_pca.xlsx"))
    BOVW_DBSCAN_PCA_AVAILABLE = True
except FileNotFoundError:
    BOVW_DBSCAN_PCA_AVAILABLE = False

# Résultats HOG / COLOR (optionnels selon le pipeline exécuté)
try:
    df_hog = pd.read_excel(os.path.join(_DIR, "output", "save_clustering_hog_kmeans.xlsx"))
    HOG_AVAILABLE = True
except FileNotFoundError:
    HOG_AVAILABLE = False

try:
    df_hog_dbscan = pd.read_excel(os.path.join(_DIR, "output", "save_clustering_hog_dbscan.xlsx"))
    HOG_DBSCAN_AVAILABLE = True
except FileNotFoundError:
    HOG_DBSCAN_AVAILABLE = False

try:
    df_color = pd.read_excel(os.path.join(_DIR, "output", "save_clustering_color_kmeans.xlsx"))
    COLOR_AVAILABLE = True
except FileNotFoundError:
    COLOR_AVAILABLE = False

# Métriques : charger séparément le fichier global et le fichier hist
def _load_metric_file(filename):
    path = os.path.join(_DIR, "output", filename)
    try:
        df = pd.read_excel(path)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns="Unnamed: 0")
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df_metric_global = _load_metric_file("save_metric.xlsx")
df_metric_hist = _load_metric_file("save_metric_hist.xlsx")
df_metric_bovw = _load_metric_file("save_metric_bovw.xlsx")
df_metric_hog = _load_metric_file("save_metric_hog.xlsx")

# Chargement optionnel des résultats Hu Moments (générés par hu_moments_pipeline())
try:
    df_hu_kmeans = pd.read_excel(os.path.join(_DIR, "output", "save_clustering_hu_kmeans.xlsx"))
    df_hu_dbscan = pd.read_excel(os.path.join(_DIR, "output", "save_clustering_hu_dbscan.xlsx"))
    df_metric_hu = pd.read_excel(os.path.join(_DIR, "output", "save_metric_hu.xlsx"))
    if 'Unnamed: 0' in df_metric_hu.columns:
        df_metric_hu.drop(columns="Unnamed: 0", inplace=True)
    HU_AVAILABLE = True
except FileNotFoundError:
    HU_AVAILABLE = False

# Chargement des résultats GIST (générés par gist_pipeline())
try:
    df_gist_kmeans = pd.read_excel(os.path.join(_DIR, "output", "save_clustering_gist_kmeans.xlsx"))
    df_gist_dbscan = pd.read_excel(os.path.join(_DIR, "output", "save_clustering_gist_dbscan.xlsx"))
    df_metric_gist = pd.read_excel(os.path.join(_DIR, "output", "save_metric_gist.xlsx"))
    if 'Unnamed: 0' in df_metric_gist.columns:
        df_metric_gist.drop(columns="Unnamed: 0", inplace=True)
    GIST_AVAILABLE = True
except FileNotFoundError:
    GIST_AVAILABLE = False

# Chargement optionnel des résultats HOG+HSV (générés par hog_hsv_kmeans())
try:
    df_hog_hsv = pd.read_excel(os.path.join(_DIR, "output", "save_clustering_hog_hsv_kmeans.xlsx"))
    df_metric_hog_hsv = pd.read_excel(os.path.join(_DIR, "output", "save_metric_hog_hsv.xlsx"))
    if 'Unnamed: 0' in df_metric_hog_hsv.columns:
        df_metric_hog_hsv.drop(columns="Unnamed: 0", inplace=True)
    HOG_HSV_AVAILABLE = True
except FileNotFoundError:
    HOG_HSV_AVAILABLE = False

# Création de deux onglets
tab1, tab2 = st.tabs(["Analyse par descripteur", "Analyse global" ])

# Onglet numéro 1
with tab1:

    st.write('## Résultat de Clustering des données Snacks')
    st.sidebar.write("####  Veuillez sélectionner les clusters à analyser" )
    # Sélection des descripteurs
    descriptor_options = ["HIST_KMEANS"]
    if HIST_DBSCAN_AVAILABLE:
        descriptor_options += ["HIST_DBSCAN"]
    else:
        st.sidebar.info("Lancez `hist_dbscan_kmeans_pipeline()` dans `pipeline.py` pour activer HIST_DBSCAN.")
    if BOVW_KMEANS_AVAILABLE:
        descriptor_options += ["BOVW_KMEANS"]
    else:
        st.sidebar.info("Lancez `BovW()` dans `pipeline2.py` pour activer BOVW_KMEANS.")
    if BOVW_KMEANS_PCA_AVAILABLE:
        descriptor_options += ["BOVW_KMEANS_PCA"]
    else:
        st.sidebar.info("Lancez `BovW()` dans `pipeline2.py` pour activer BOVW_KMEANS_PCA.")
    if BOVW_DBSCAN_AVAILABLE:
        descriptor_options += ["BOVW_DBSCAN"]
    else:
        st.sidebar.info("Lancez `BovW()` dans `pipeline2.py` pour activer BOVW_DBSCAN.")
    if BOVW_DBSCAN_PCA_AVAILABLE:
        descriptor_options += ["BOVW_DBSCAN_PCA"]
    else:
        st.sidebar.info("Lancez `BovW()` dans `pipeline2.py` pour activer BOVW_DBSCAN_PCA.")
    if HOG_AVAILABLE:
        descriptor_options += ["HOG_KMEANS"]
    else:
        st.sidebar.info("Lancez `pipeline()` ou `hog_dbscan_kmeans_pipeline()` dans `pipeline.py` pour activer HOG_KMEANS.")
    if HOG_DBSCAN_AVAILABLE:
        descriptor_options += ["HOG_DBSCAN"]
    else:
        st.sidebar.info("Lancez `hog_dbscan_kmeans_pipeline()` dans `pipeline.py` pour activer HOG_DBSCAN.")
    if COLOR_AVAILABLE:
        descriptor_options += ["COLOR_HSV"]
    else:
        st.sidebar.info("Lancez `pipeline()` dans `pipeline.py` pour activer COLOR_HSV.")
    if HU_AVAILABLE:
        descriptor_options += ["HU_KMEANS", "HU_DBSCAN"]
    else:
        st.sidebar.info("Lancez `python pipeline.py` pour activer les descripteurs Hu Moments.")
    if GIST_AVAILABLE:
        descriptor_options += ["GIST_KMEANS", "GIST_DBSCAN"]
    else:
        st.sidebar.info("Lancez `python pipeline.py` pour activer les descripteurs GIST.")
    if HOG_HSV_AVAILABLE:
        descriptor_options += ["HOG+HSV"]
    else:
        st.sidebar.info("Lancez `hog_hsv_kmeans()` dans `pipeline.py` pour activer le descripteur HOG+HSV.")

    descriptor = st.sidebar.selectbox('Sélectionner un descripteur', descriptor_options)
    metric_source_file = ""
    if descriptor in ["HIST_KMEANS", "HISTOGRAM"]:
        df = df_hist
        metric_source_file = os.path.join(_DIR, "output", "save_clustering_hist_kmeans.xlsx")
    elif descriptor == "HIST_DBSCAN":
        df = df_hist_dbscan
        metric_source_file = os.path.join(_DIR, "output", "save_clustering_hist_dbscan.xlsx")
    elif descriptor == "BOVW_KMEANS":
        df = df_bovw_kmeans
        metric_source_file = os.path.join(_DIR, "output", "save_clustering_bovw_sift_kmeans.xlsx")
    elif descriptor == "BOVW_KMEANS_PCA":
        df = df_bovw_kmeans_pca
        metric_source_file = os.path.join(_DIR, "output", "save_clustering_bovw_sift_kmeans_pca.xlsx")
    elif descriptor == "BOVW_DBSCAN":
        df = df_bovw_dbscan
        metric_source_file = os.path.join(_DIR, "output", "save_clustering_bovw_sift_dbscan.xlsx")
    elif descriptor == "BOVW_DBSCAN_PCA":
        df = df_bovw_dbscan_pca
        metric_source_file = os.path.join(_DIR, "output", "save_clustering_bovw_sift_dbscan_pca.xlsx")
    elif descriptor in ["HOG", "HOG_KMEANS"]:
        df = df_hog
        metric_source_file = os.path.join(_DIR, "output", "save_clustering_hog_kmeans.xlsx")
    elif descriptor == "HOG_DBSCAN":
        df = df_hog_dbscan
        metric_source_file = os.path.join(_DIR, "output", "save_clustering_hog_dbscan.xlsx")
    elif descriptor == "COLOR_HSV":
        df = df_color
        metric_source_file = os.path.join(_DIR, "output", "save_clustering_color_kmeans.xlsx")
    elif descriptor == "HU_KMEANS":
        df = df_hu_kmeans
        metric_source_file = os.path.join(_DIR, "output", "save_clustering_hu_kmeans.xlsx")
    elif descriptor == "HU_DBSCAN":
        df = df_hu_dbscan
        metric_source_file = os.path.join(_DIR, "output", "save_clustering_hu_dbscan.xlsx")
    elif descriptor == "GIST_KMEANS":
        df = df_gist_kmeans
        metric_source_file = os.path.join(_DIR, "output", "save_clustering_gist_kmeans.xlsx")
    elif descriptor == "GIST_DBSCAN":
        df = df_gist_dbscan
        metric_source_file = os.path.join(_DIR, "output", "save_clustering_gist_dbscan.xlsx")
    elif descriptor == "HOG+HSV":
        df = df_hog_hsv
        metric_source_file = os.path.join(_DIR, "output", "save_clustering_hog_hsv_kmeans.xlsx")

    st.sidebar.caption(f"Dernière MAJ du fichier sélectionné : {format_mtime(metric_source_file)}")
    # Clusters réellement présents dans les données (exclut -1 = bruit DBSCAN)
    available_clusters = sorted([c for c in df['cluster'].unique() if c >= 0])
    n_noise_total = int((df['cluster'] == -1).sum())

    if n_noise_total > 0:
        st.sidebar.warning(f"⚠️ DBSCAN : {n_noise_total} points de bruit (label -1) non affichés")

    # Classe dominante par cluster
    def dominant_class(cluster_id):
        """
        Retourne le nom de la classe la plus représentée dans un cluster donné.

        Parameters
        ----------
        cluster_id : int
            Identifiant du cluster à analyser (doit correspondre à une valeur
            de la colonne 'cluster' du DataFrame courant `df`).

        Returns
        -------
        str
            Nom de la classe dominante (issu de CLASS_NAMES), ou '(vide)'
            si le cluster ne contient aucune image.
        """
        subset = df[df['cluster'] == cluster_id]
        if len(subset) == 0:
            return "(vide)"
        dominant_label = int(subset['label'].mode().iloc[0])
        return CLASS_NAMES[dominant_label]

    if len(available_clusters) == 0:
        st.warning(
            "Aucun cluster exploitable à afficher pour ce descripteur "
            "(uniquement du bruit DBSCAN ou fichier vide)."
        )
        st.stop()

    selected_cluster = st.sidebar.selectbox(
        'Sélectionner un Cluster',
        available_clusters,
        format_func=lambda c: f"Cluster {c}  —  {dominant_class(c)}"
    )

    # Filtrer les données en fonction du cluster sélectionné
    filtered_data = df[df['cluster'] == selected_cluster]

    st.write(f"###  Analyse du descripteur {descriptor}")
    st.write(f"#### Cluster {selected_cluster}  ·  {len(filtered_data)} images  ·  classe dominante : **{dominant_class(selected_cluster)}**")

    # --- Composition du cluster ---
    st.write("####  Composition du cluster")
    class_counts = (
        filtered_data['label']
        .value_counts()
        .rename(index=lambda i: CLASS_NAMES[int(i)])
        .sort_values(ascending=False)
        .reset_index()
    )
    class_counts.columns = ['Classe', 'Nombre d\'images']
    class_counts['%'] = (class_counts['Nombre d\'images'] / len(filtered_data) * 100).round(1).astype(str) + ' %'

    col_table, col_bar = st.columns([1, 2])
    with col_table:
        st.dataframe(class_counts, hide_index=True, use_container_width=True)
    with col_bar:
        fig_comp = px.bar(
            class_counts,
            x='Classe', y='Nombre d\'images',
            color='Classe',
            title=f"Répartition des classes — Cluster {selected_cluster}",
        )
        fig_comp.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_comp, use_container_width=True)

    st.write(f"####  Visualisation 3D du clustering avec descripteur {descriptor}")
    # Création d'un graph 3D des clusters
    fig = colorize_cluster(df, selected_cluster)
    st.plotly_chart(fig)

    # Affichage des images du cluster sélectionné
    if 'image_path' in filtered_data.columns and len(filtered_data) > 0:
        st.write(f"#### Images du cluster {selected_cluster}  ({len(filtered_data)} images au total)")
        n_display = min(15, len(filtered_data))
        shuffle_images = st.sidebar.checkbox("Mélanger les images affichées", value=True)
        if shuffle_images:
            display_data = filtered_data.sample(n=n_display)
        else:
            display_data = filtered_data.head(n_display)
        cols = st.columns(5)
        for i, (_, row) in enumerate(display_data.iterrows()):
            try:
                img = Image.open(fix_image_path(row['image_path']))
                cols[i % 5].image(img, caption=CLASS_NAMES[int(row['label'])], width='stretch')
            except Exception as e:
                cols[i % 5].write(f"Erreur : {e}")

# Onglet numéro 2
with tab2:
    st.write('## Analyse Global des descripteurs')

    # Combinaison de toutes les métriques disponibles
    all_metrics_dfs = []
    if not df_metric_global.empty:
        all_metrics_dfs.append(df_metric_global)
    if not df_metric_hist.empty:
        all_metrics_dfs.append(df_metric_hist)
    if not df_metric_bovw.empty:
        all_metrics_dfs.append(df_metric_bovw)
    if not df_metric_hog.empty:
        all_metrics_dfs.append(df_metric_hog)
    if HU_AVAILABLE:
        all_metrics_dfs.append(df_metric_hu)
    if GIST_AVAILABLE:
        all_metrics_dfs.append(df_metric_gist)
    if HOG_HSV_AVAILABLE:
        all_metrics_dfs.append(df_metric_hog_hsv)
    if len(all_metrics_dfs) == 0:
        st.info("Aucune métrique disponible. Exécutez d'abord un pipeline pour générer les fichiers dans output/.")
    else:
        df_all_metrics = pd.concat(all_metrics_dfs, ignore_index=True)
        # Unifier le nom historique HISTOGRAM -> HIST_KMEANS pour éviter les doublons
        if 'descriptor' in df_all_metrics.columns:
            df_all_metrics['descriptor'] = df_all_metrics['descriptor'].replace({'HISTOGRAM': 'HIST_KMEANS'})
        # En cas de doublons (même descripteur/modèle), garder la dernière occurrence
        # (le fichier save_metric_hist est ajouté après save_metric.xlsx)
        if {'descriptor', 'name_model'}.issubset(df_all_metrics.columns):
            df_all_metrics = df_all_metrics.drop_duplicates(
                subset=['descriptor', 'name_model'], keep='last'
            ).reset_index(drop=True)
        plot_metric(df_all_metrics)
        st.write('## Métriques détaillées')
        st.dataframe(df_all_metrics)
