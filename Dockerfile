FROM python:3.10-slim

WORKDIR /app

# Minimal dependencies required by pipeline_unified.py + dashboard_unified.py
RUN pip install --no-cache-dir \
	numpy \
	opencv-contrib-python-headless \
	pandas \
	'plotly>=5.18.0' \
	scikit-image \
	scikit-learn \
	'streamlit>=1.35.0' \
	matplotlib \
	openpyxl \
	pypdf

COPY src ./src
COPY images ./images

EXPOSE 8000

# 1) Generate final unified outputs
# 2) Start unified dashboard
CMD ["sh", "-c", "python -m src.pipeline_unified && streamlit run src/dashboard_unified.py --server.port=8000 --server.address=0.0.0.0"]