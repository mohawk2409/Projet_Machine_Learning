import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import transform
import itertools

try:
    from .constant import IMAGE_SIZE, DATA_PATH, CLASS_NAMES
except ImportError:
    from constant import IMAGE_SIZE, DATA_PATH, CLASS_NAMES


def load_images_from_directory(data_path=DATA_PATH, image_size=IMAGE_SIZE):
    """
    Charge les images depuis le répertoire organisé par classes (Snacks dataset).
    Input : data_path (str) : chemin vers le répertoire contenant les sous-dossiers de classes
            image_size (tuple) : taille cible pour le redimensionnement (largeur, hauteur)
    Output : images (list) : liste des images en niveaux de gris redimensionnées
             labels (list) : liste des labels entiers correspondants
             paths (list) : liste des chemins des images
             images_color (list) : liste des images en couleur BGR redimensionnées
    """
    images = []
    images_color = []
    labels = []
    paths = []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_path, class_name)
        if not os.path.isdir(class_dir):
            print(f"Attention : dossier introuvable pour la classe '{class_name}' : {class_dir}")
            continue

        for filename in os.listdir(class_dir):
            filepath = os.path.join(class_dir, filename)
            img = cv2.imread(filepath)
            if img is None:
                continue
            # Redimensionner l'image couleur
            color_resized = cv2.resize(img, image_size)
            # Convertir en niveaux de gris (pour HOG / SIFT)
            gray_resized = cv2.cvtColor(color_resized, cv2.COLOR_BGR2GRAY)
            images.append(gray_resized)
            images_color.append(color_resized)
            labels.append(label_idx)
            paths.append(filepath)

    return images, np.array(labels), paths, images_color


def load_train_test_split(data_path=DATA_PATH, image_size=IMAGE_SIZE,
                          test_size=0.2, random_state=42):
    """
    Charge toutes les images depuis data_path et les divise en ensembles
    d'entraînement (80%) et de test (20%) de façon stratifiée par classe.

    La stratification garantit que chaque classe est représentée
    proportionnellement dans les deux ensembles.

    Pourquoi train/test en clustering ?
      - TRAIN : construire le vocabulaire (BoVW), ajuster le scaler,
                entraîner le modèle KMeans (fit).
      - TEST  : prédire les clusters sur des données jamais vues (predict),
                calculer les métriques → mesure la généralisation du modèle.

    Output (8 valeurs) :
      images_train, labels_train, paths_train, images_color_train,
      images_test,  labels_test,  paths_test,  images_color_test
    """
    from sklearn.model_selection import train_test_split as sk_split

    images, labels, paths, images_color = load_images_from_directory(data_path, image_size)
    indices = np.arange(len(images))

    idx_train, idx_test = sk_split(
        indices, test_size=test_size, random_state=random_state, stratify=labels
    )

    images_train       = [images[i]        for i in idx_train]
    images_test        = [images[i]        for i in idx_test]
    labels_train       = labels[idx_train]
    labels_test        = labels[idx_test]
    paths_train        = [paths[i]         for i in idx_train]
    paths_test         = [paths[i]         for i in idx_test]

    print(f"  Split : {len(images_train)} images train  |  {len(images_test)} images test")
    return (images_train, labels_train, paths_train,
            images_test,  labels_test,  paths_test)


def compute_gray_histograms(images):
    """
    Calcule les histogrammes de niveau de gris pour les images.
    Input : images (list) : liste des images en niveaux de gris
    Output : descriptors (list) : liste des descripteurs d'histogrammes de niveau de gris
    """
    descriptors = []
    
    for image in images:

        img = np.array(image, dtype=np.uint8)

        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        descriptors.append(hist.flatten())
    descriptors=np.array(descriptors)
        
    return descriptors

def compute_hog_descriptors(images):
    """
    Calcule les descripteurs HOG pour les images en niveaux de gris.
    Input : images (array) : tableau numpy des images
    Output : descriptors (array) : tableau numpy des descripteurs HOG
    """
    descriptors = []
    for image in images:
        fd = hog(
            image,
            orientations=6,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
            transform_sqrt=True
        )
        descriptors.append(fd)
    descriptors = np.array(descriptors)
    return descriptors


def compute_hsv_histograms(images_color, bins_h=30, bins_s=12, bins_v=8):
    """
    Calcule des histogrammes couleur en espace HSV pour chaque image.
    C'est le descripteur le plus adapté pour distinguer des catégories
    d'aliments, car la couleur est leur attribut visuel le plus discriminant.

    L'espace HSV (Hue-Saturation-Value) est préféré à RGB/BGR car :
    - H (teinte) encode la couleur pure, indépendamment de la luminosité
    - S (saturation) distingue les couleurs vives des tons neutres
    - V (luminosité) capture l'exposition de l'image
    On concatène les trois histogrammes en un seul vecteur.

    Entrées : images_color (list) : liste d'images BGR (uint8)
            bins_h (int) : nombre de bins pour la teinte (défaut 18 → 20° par bin)
            bins_s (int) : nombre de bins pour la saturation (défaut 8)
            bins_v (int) : nombre de bins pour la valeur (défaut 8)
    Sortie : descriptors (np.array) : shape (N, bins_h + bins_s + bins_v)
    """
    descriptors = []
    for img_bgr in images_color:
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # Histogramme de la teinte (H) : plage [0, 180] pour OpenCV
        hist_h = cv2.calcHist([img_hsv], [0], None, [bins_h], [0, 180]).flatten()
        # Histogramme de la saturation (S) : plage [0, 256]
        hist_s = cv2.calcHist([img_hsv], [1], None, [bins_s], [0, 256]).flatten()
        # Histogramme de la valeur/luminosité (V) : plage [0, 256]
        hist_v = cv2.calcHist([img_hsv], [2], None, [bins_v], [0, 256]).flatten()
        # Concaténation et normalisation L1
        hist = np.concatenate([hist_h, hist_s, hist_v]).astype(float)
        if hist.sum() > 0:
            hist /= hist.sum()
        descriptors.append(hist)
    return np.array(descriptors)


def compute_hsv_2d_histogram(images_color, bins_h=20, bins_s=10, bins_v=8):
    """
    Calcule un histogramme 2D joint H×S + histogramme 1D V pour chaque image BGR.

    Pourquoi 2D ?
    Un histogramme marginal H seul met dans le même bin "rouge vif" et "rouge pâle".
    L'histogramme joint H×S capture la combinaison teinte+saturation : beaucoup plus
    discriminant pour des aliments (fraise = rouge+saturé, jambon = rouge+peu saturé).

    Sortie : shape (N, bins_h * bins_s + bins_v)
    """
    descriptors = []
    for img_bgr in images_color:
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # Histogramme 2D joint H×S : H ∈ [0,180], S ∈ [0,256] (OpenCV)
        hist_hs = cv2.calcHist(
            [img_hsv], [0, 1], None,
            [bins_h, bins_s], [0, 180, 0, 256]
        ).flatten().astype(float)
        # Histogramme 1D V (luminosité, moins discriminant)
        hist_v = cv2.calcHist(
            [img_hsv], [2], None, [bins_v], [0, 256]
        ).flatten().astype(float)
        hist = np.concatenate([hist_hs, hist_v])
        if hist.sum() > 0:
            hist /= hist.sum()  # normalisation L1
        descriptors.append(hist)
    return np.array(descriptors)


def compute_hu_moments(images):
    """
    Calcule les 7 moments de Hu pour chaque image en niveaux de gris.

    Les moments de Hu sont des invariants de forme : ils sont insensibles
    aux translations, rotations et changements d'échelle. Ils capturent
    la géométrie globale (silhouette) de l'objet dans l'image.

    Méthode (source : learnopencv.com/shape-matching-using-hu-moments-c-python/) :
      3. Log-transform   → H_i = -sign(h_i) * log10(|h_i|)
         → ramène les 7 valeurs dans une plage comparable
         (sans ça, h0 ~ 1e-3 et h6 ~ 1e-20, incomparables)

    Entrée  : images (list) : images en niveaux de gris (uint8)
    Sortie : descriptors (np.array) : shape (N, 7)
    """
    descriptors = []
    for image in images:
        # Moments de l'image
        mom = cv2.moments(image)
        # 7 moments de Hu (shape (7,1) → flatten en (7,))
        hu = cv2.HuMoments(mom).flatten()
        # Transformation logarithmique pour normaliser les ordres de grandeur
        # Ajoute 1e-10 pour éviter log(0) si un moment est exactement nul
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        descriptors.append(hu)
    return np.array(descriptors)


def compute_lbp_descriptor(images, radius=1, n_points=8):
    """
    Calcule le descripteur LBP (Local Binary Pattern) pour chaque image.

    LBP parcourt chaque pixel et compare son intensité à ses n_points voisins
    situés sur un cercle de rayon `radius`. Pour chaque voisin, on met 1 si
    le voisin est >= au pixel central, 0 sinon. On obtient un code binaire
    (ex : 01101100) qui encode la texture locale.

    Avec `method='uniform'` : seuls les patterns ayant au plus 1 transition
    0→1 (ou 1→0) sont conservés comme bins distincts. Les autres sont
    groupés dans un seul bin. Résultat : n_points + 2 = 10 bins.
    LBP est insensible à la luminosité uniforme (invariant monotone).

    Input  : images (list) : images en niveaux de gris
             radius (int) : rayon du cercle voisin (défaut 1)
             n_points (int) : nombre de voisins sur le cercle (défaut 8)
    Output : descriptors (np.array) : shape (N, n_points + 2)
    """
    from skimage.feature import local_binary_pattern
    n_bins = n_points + 2   # nombre de bins pour method='uniform'
    descriptors = []
    for image in images:
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(float)
        if hist.sum() > 0:
            hist /= hist.sum()
        descriptors.append(hist)
    return np.array(descriptors)
