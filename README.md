# Projet Clustering Snacks (Version Unified)

Ce projet lance un pipeline de clustering non supervise puis un dashboard Streamlit.

Version recommandee a utiliser:
1. [src/pipeline_unified.py](src/pipeline_unified.py)
2. [src/dashboard_unified.py](src/dashboard_unified.py)

## 1. Lancement le plus simple (Docker, tous OS)

Prerequis:
1. Docker Desktop (Windows/macOS) ou Docker Engine (Linux)

Depuis la racine du projet:

```bash
docker build -t snacks-unified .
docker run --rm -p 8000:8000 -v "${PWD}/src/output:/app/src/output" snacks-unified
```

Sur Windows PowerShell, vous pouvez aussi utiliser:

```powershell
docker run --rm -p 8000:8000 -v "${PWD}\src\output:/app/src/output" snacks-unified
```

Puis ouvrir:
1. http://localhost:8000

Notes:
1. Au demarrage, le conteneur execute d'abord le pipeline puis lance le dashboard.
2. Le premier lancement peut prendre plusieurs minutes (extraction des descripteurs).
3. Les fichiers Excel sont ecrits dans [src/output](src/output) et persistés sur la machine hote via le volume `-v`.

## 2. Lancement local sans Docker

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.pipeline_unified
streamlit run src/dashboard_unified.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline_unified
streamlit run src/dashboard_unified.py
```

Puis ouvrir:
1. http://localhost:8501

## 3. Compatibilite macOS et CPU ARM

Oui, le projet peut fonctionner sur Mac et sur CPU ARM.

Resume:
1. macOS Intel: OK
2. macOS Apple Silicon (M1/M2/M3): OK
3. Linux ARM64: OK en general

Recommandations:
1. Utiliser Docker est la solution la plus portable entre machines.
2. Sur Apple Silicon, Docker utilisera une image ARM64 automatiquement.
3. En local, preferer Python 3.10 a 3.12 pour une compatibilite maximale des wheels OpenCV/scikit.

## 4. Sorties finales generees

Le pipeline unified produit les fichiers finaux dans [src/output](src/output):

1. Clustering par modele/descripteur (kmeans, dbscan, hierarchical)
2. Fichiers metriques consolides:
1. [src/output/save_metric_kmeans.xlsx](src/output/save_metric_kmeans.xlsx)
2. [src/output/save_metric_dbscan.xlsx](src/output/save_metric_dbscan.xlsx)
3. [src/output/save_metric_hierarchical.xlsx](src/output/save_metric_hierarchical.xlsx)
4. [src/output/save_metric_all.xlsx](src/output/save_metric_all.xlsx)

## 5. Depannage rapide

Probleme OpenCV dans Docker (ex: libxcb.so.1):
1. Le Dockerfile utilise deja une variante headless d'OpenCV pour eviter ce probleme.
2. Rebuild sans cache:

```bash
docker build --no-cache -t snacks-unified .
```

Si le port est deja pris:
1. Docker: changer le mapping, ex: -p 8501:8000
2. Local streamlit: streamlit run dashboard_unified.py --server.port 8502

Si vous voulez verifier uniquement la generation des fichiers (sans dashboard):

```powershell
docker run --rm -v "${PWD}\src\output:/app/src/output" snacks-unified python -m src.pipeline_unified
```

Puis verifier le contenu du dossier local:

```powershell
Get-ChildItem .\src\output
```

## 6. Fichiers a utiliser

Pour les consignes finales, utiliser uniquement:
1. [src/pipeline_unified.py](src/pipeline_unified.py)
2. [src/dashboard_unified.py](src/dashboard_unified.py)
3. [Dockerfile](Dockerfile)