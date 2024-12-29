# Użyj oficjalnego obrazu Python jako bazowego
FROM python:3.10-slim

# Ustaw katalog roboczy
WORKDIR /apka

# Skopiuj wymagane pliki do kontenera
COPY . /apka

# Zainstaluj wymagane pakiety
RUN pip install --no-cache-dir -r requirements.txt

# Skopiuj modele do kontenera
COPY model/model.pkl /app/model/model.pkl
COPY notebooks/yolov5/runs/train/exp7/weights/last.pt /app/notebooks/yolov5/runs/train/exp7/weights/last.pt

# Uruchom aplikację Streamlit
CMD ["streamlit", "run", "app.py"]