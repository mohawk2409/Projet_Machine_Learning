import pandas as pd
from sklearn.manifold import TSNE
import numpy as np

def conversion_3d(X, n_components=3,perplexity=50,random_state=42, early_exaggeration=10,max_iter=3000):
    """
    Conversion des vecteurs de N dimensions vers une dimension précise (n_components) pour la visualisation
    Input : X (array-like) : données à convertir en 3D
            n_components (int) : nombre de dimensions cibles (par défaut : 3)
            perplexity (float) : valeur de perplexité pour t-SNE (par défaut : 50)
            random_state (int) : graine pour la génération de nombres aléatoires (par défaut : 42)
            early_exaggeration (float) : facteur d'exagération pour t-SNE (par défaut : 10)
            max_iter (int) : nombre d'itérations pour t-SNE (par défaut : 3000)
    Output : X_3d (array-like) : données converties en 3D
    """
    tsne = TSNE(n_components=n_components,
                random_state=random_state,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
                max_iter=max_iter
               )
    X = np.array(X)
    X_3d = tsne.fit_transform(X)
    return X_3d


def create_df_to_export(data_3d, l_true_label, l_cluster, l_path_img=None):
    """
    Création d'un DataFrame pour stocker les données et les labels
    Input : data_3d (array-like) : données converties en 3D
            l_true_label (list) : liste des labels vrais
            l_cluster (list) : liste des labels de cluster
            l_path_img (list) : liste des chemins des images
    Output : df (DataFrame) : DataFrame contenant les données et les labels
    """
    df = pd.DataFrame(data_3d, columns=['x', 'y', 'z'])
    df['label'] = l_true_label
    df['cluster'] = l_cluster
    if l_path_img is not None:
        df['image_path'] = l_path_img
    
    return df
