import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# =========================
# MODEL PATH
# =========================

MODEL_PATH = "emotion_recognition/improved_aug_emotion_model.keras"

# =========================
# EMOTION CLASSES
# =========================

emotion_classes = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model

model = load_model()

# =========================
# PREDICTION FUNCTION
# =========================

def predict_emotion(image_bgr):

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        (48, 48)
    )

    img = np.expand_dims(
        gray,
        axis=-1
    )

    img = np.expand_dims(
        img,
        axis=0
    )

    preds = model.predict(
        img,
        verbose=0
    )

    idx = np.argmax(preds)

    emotion = emotion_classes[idx]

    confidence = float(preds[0][idx])

    return emotion, confidence

# =========================
# DASHBOARD
# =========================

st.title("FaceSense Dashboard")

st.write(
    "Facial Emotion Recognition System"
)

# =========================
# SIDEBAR
# =========================

mode = st.sidebar.radio(
    "Choisir le mode",
    [
        "Tester avec image",
        "Tester avec caméra"
    ]
)

# =========================
# IMAGE MODE
# =========================

if mode == "Tester avec image":

    uploaded_file = st.file_uploader(
        "Importer une image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_np = np.array(image)

        image_bgr = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2BGR
        )

        emotion, confidence = predict_emotion(
            image_bgr
        )

        st.image(
            image,
            caption="Image importée",
            use_container_width=True
        )

        st.success(
            f"Émotion détectée : {emotion}"
        )

        st.info(
            f"Confiance : {confidence:.2f}"
        )

# =========================
# CAMERA MODE
# =========================

elif mode == "Tester avec caméra":

    camera_image = st.camera_input(
        "Prendre une photo"
    )

    if camera_image is not None:

        image = Image.open(
            camera_image
        ).convert("RGB")

        image_np = np.array(image)

        image_bgr = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2BGR
        )

        emotion, confidence = predict_emotion(
            image_bgr
        )

        st.image(
            image,
            caption="Image caméra",
            use_container_width=True
        )

        st.success(
            f"Émotion détectée : {emotion}"
        )

        st.info(
            f"Confiance : {confidence:.2f}")