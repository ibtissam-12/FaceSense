import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

CNN_PATH = "emotion_recognition/improved_aug_emotion_model.keras"
MOBILENET_PATH = "emotion_recognition/mobilenet_fixed.h5"
RESNET_PATH = "emotion_recognition/resnet50_emotion_model.keras"

emotion_classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

@st.cache_resource
def load_model(path):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        st.error(f"Erreur chargement modèle : {path}")
        st.code(str(e))
        return None

def predict_cnn(model, image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48))

    img = np.expand_dims(gray, axis=-1)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)
    idx = np.argmax(preds)

    return emotion_classes[idx], float(preds[0][idx])

def predict_rgb_model(model, image_bgr, size):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (size, size))

    img = image_rgb.astype("float32")
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)
    idx = np.argmax(preds)

    return emotion_classes[idx], float(preds[0][idx])

st.title("FaceSense Dashboard")
st.write("Reconnaissance des émotions faciales avec plusieurs modèles")

model_choice = st.sidebar.selectbox(
    "Choisir le modèle",
    ["CNN amélioré", "MobileNetV2", "ResNet50"]
)

mode = st.sidebar.radio(
    "Choisir le mode",
    ["Tester avec image", "Tester avec caméra"]
)

if model_choice == "CNN amélioré":
    model = load_model(CNN_PATH)
elif model_choice == "MobileNetV2":
    model = load_model(MOBILENET_PATH)
else:
    model = load_model(RESNET_PATH)

def predict_selected_model(image_bgr):
    if model is None:
        return "model_error", 0.0

    if model_choice == "CNN amélioré":
        return predict_cnn(model, image_bgr)

    if model_choice == "MobileNetV2":
        return predict_rgb_model(model, image_bgr, 160)

    if model_choice == "ResNet50":
        return predict_rgb_model(model, image_bgr, 224)

if mode == "Tester avec image":
    uploaded_file = st.file_uploader("Importer une image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        emotion, confidence = predict_selected_model(image_bgr)

        st.image(image, caption="Image importée", use_container_width=True)
        st.success(f"Émotion détectée : {emotion}")
        st.info(f"Confiance : {confidence:.2f}")

elif mode == "Tester avec caméra":
    camera_image = st.camera_input("Prendre une photo")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        emotion, confidence = predict_selected_model(image_bgr)

        st.image(image, caption="Image caméra", use_container_width=True)
        st.success(f"Émotion détectée : {emotion}")
        st.info(f"Confiance : {confidence:.2f}")