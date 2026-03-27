import os

PATH_OUTPUT = os.path.join(os.path.dirname(__file__), "output")
MODEL_CLUSTERING = "kmeans"
IMAGE_SIZE = (64, 64)  # Taille de redimensionnement des images

# Chemin vers les données Snacks (relatif au dossier src/)
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "images", "data", "test")

CLASS_NAMES = ["apple", "banana", "cake", "candy", "carrot", "cookie",
               "doughnut", "grape", "hot dog", "ice cream", "juice",
               "muffin", "orange", "pineapple", "popcorn", "pretzel",
               "salad", "strawberry", "waffle", "watermelon"]