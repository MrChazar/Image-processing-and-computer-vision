import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import torch

with open('../model/model.pkl', 'rb') as f:
    conv_model = pickle.load(f)

yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='../notebooks/yolov5/runs/train/exp2/weights/last.pt', force_reload=True)


def preprocess_image_conv_model(image):
    image = cv2.resize(image, (120, 60))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def draw_bounding_boxes(image, results):
    for *box, conf, cls in results.xyxy[0]:
        label = f'{results.names[int(cls)]} {conf:.2f}'
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image



st.title("Model wykrywania wad podstawy")

uploaded_file = st.file_uploader("Wybierz zdjęcie x-ray:", type=["jpg", "jpeg", "png"])
models_to_use = st.multiselect("Wybierz modele do użycia:", ["Konwolucyjny", "YOLOv5"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_with_boxes = image
    if "Konwolucyjny" in models_to_use:
        processed_image_conv = preprocess_image_conv_model(image)
        prediction_conv = conv_model.predict(processed_image_conv)
        st.write(f"Predykcja (Konwolucyjny): {'Skolioza' if prediction_conv[0][0] > 0.8 else 'Normalny'}")
        st.write(f"Pewność (Konwolucyjny): {prediction_conv[0][0]:.2f}")

    if "YOLOv5" in models_to_use:
        processed_image_yolo = image
        prediction_yolo = yolo_model(processed_image_yolo)
        image_with_boxes = draw_bounding_boxes(image.copy(), prediction_yolo)
        st.write(f"Predykcja (YOLOv5): {prediction_yolo}")


    st.image(image_with_boxes, caption='Przesłane zdjęcie z bounding boxami', width=300)
