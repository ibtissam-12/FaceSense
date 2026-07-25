import streamlit as st
import cv2
import os
import numpy as np
import tensorflow as tf
import av
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from PIL import Image

DATASET_PATH = "dataset"
EMOTION_MODEL_PATH = "emotion_recognition/improved_aug_emotion_model.keras"

emotion_classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="FaceSense", page_icon="🎭", layout="wide")


# =========================
# CHARGEMENT (inchangé dans sa logique)
# =========================
@st.cache_resource
def load_emotion_model():
    return tf.keras.models.load_model(EMOTION_MODEL_PATH)


def load_known_embeddings():
    known_embeddings = {}
    if not os.path.isdir(DATASET_PATH):
        os.makedirs(DATASET_PATH, exist_ok=True)
        return known_embeddings

    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)
        if os.path.isdir(person_path):
            images = os.listdir(person_path)
            if len(images) > 0:
                img_path = os.path.join(person_path, images[0])
                try:
                    embedding = DeepFace.represent(
                        img_path=img_path,
                        model_name="Facenet",
                        enforce_detection=False
                    )[0]["embedding"]
                    known_embeddings[person] = embedding
                except Exception as e:
                    st.sidebar.warning(f"Erreur chargement {person} : {e}")
    return known_embeddings


@st.cache_resource
def warmup_deepface():
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    DeepFace.extract_faces(img_path=dummy, detector_backend="opencv", enforce_detection=False)
    DeepFace.represent(img_path=dummy, model_name="Facenet", enforce_detection=False)
    return True


# =========================
# DISTANCE (inchangé)
# =========================
def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)


# =========================
# PREDICTION EMOTION (inchangé)
# =========================
def predict_emotion(emotion_model, face_img):
    try:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48))
        img = np.expand_dims(gray, axis=-1)
        img = np.expand_dims(img, axis=0)
        preds = emotion_model.predict(img, verbose=0)
        idx = np.argmax(preds)
        return emotion_classes[idx], preds[0][idx]
    except:
        return "unknown", 0.0


# =========================
# PROCESSOR (inchangé dans sa logique, hérite juste de last_boxes déjà en place)
# =========================
class FaceSenseProcessor(VideoProcessorBase):
    def __init__(self, emotion_model, known_embeddings):
        self.emotion_model = emotion_model
        self.known_embeddings = known_embeddings  # référence directe -> mutations reflétées en direct
        self.frame_count = 0
        self.last_boxes = []

    def draw_boxes(self, img, boxes):
        for (x, y, w, h, label, color) in boxes:
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return img

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % 2 != 0:
            img = self.draw_boxes(img, self.last_boxes)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        img = cv2.resize(img, (640, 480))

        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend="opencv",
            enforce_detection=False
        )

        current_boxes = []

        for face in faces:
            region = face["facial_area"]
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            face_img = img[y:y+h, x:x+w]

            name = "Unknown"
            best_dist = 999

            try:
                embedding = DeepFace.represent(
                    img_path=face_img,
                    model_name="Facenet",
                    enforce_detection=False
                )[0]["embedding"]

                for person, ref_embedding in self.known_embeddings.items():
                    dist = cosine_distance(embedding, ref_embedding)
                    if dist < 10 and dist < best_dist:
                        best_dist = dist
                        name = person
            except:
                name = "Unknown"

            emotion, emotion_conf = predict_emotion(self.emotion_model, face_img)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} | {emotion} ({emotion_conf:.2f})"

            current_boxes.append((x, y, w, h, label, color))

        self.last_boxes = current_boxes
        img = self.draw_boxes(img, current_boxes)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =========================
# ÉTAT PARTAGÉ (session_state, mutable en direct)
# =========================
if "known_embeddings" not in st.session_state:
    st.session_state.known_embeddings = load_known_embeddings()

with st.spinner("Chargement des modèles..."):
    warmup_deepface()
    emotion_model = load_emotion_model()


# =========================
# INTERFACE
# =========================
st.title("🎭 FaceSense")
st.caption("Reconnaissance faciale + détection d'émotion en temps réel")

tab_live, tab_dataset = st.tabs(["📷 Détection en direct", "👤 Gérer les personnes"])

with tab_live:
    col_video, col_info = st.columns([3, 1])

    with col_video:
        known_embeddings_ref = st.session_state.known_embeddings  # NOUVEAU : référence locale, capturée dans le thread principal

        webrtc_streamer(
            key="facesense",
            video_processor_factory=lambda: FaceSenseProcessor(
                emotion_model, known_embeddings_ref  # NOUVEAU : on passe la référence locale, pas st.session_state directement
            ),
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )

    with col_info:
        st.metric("Personnes enregistrées", len(st.session_state.known_embeddings))
        st.markdown("**Légende**")
        st.markdown("🟩 Visage reconnu")
        st.markdown("🟥 Visage inconnu")
        if st.session_state.known_embeddings:
            st.markdown("**Base actuelle**")
            for name in st.session_state.known_embeddings.keys():
                st.markdown(f"- {name}")
        else:
            st.info("Aucune personne enregistrée. Va dans l'onglet 'Gérer les personnes'.")

with tab_dataset:
    st.subheader("Ajouter une personne")

    col_form, col_preview = st.columns([2, 1])

    with col_form:
        new_name = st.text_input("Nom de la personne")
        new_photo = st.file_uploader("Photo du visage", type=["jpg", "jpeg", "png"], key="uploader")

        if st.button("Ajouter", type="primary"):
            if not new_name.strip():
                st.error("Merci de saisir un nom.")
            elif new_photo is None:
                st.error("Merci d'importer une photo.")
            else:
                person_dir = os.path.join(DATASET_PATH, new_name.strip())
                os.makedirs(person_dir, exist_ok=True)
                img_path = os.path.join(person_dir, "img1.jpg")

                image = Image.open(new_photo).convert("RGB")
                image.save(img_path)

                try:
                    embedding = DeepFace.represent(
                        img_path=img_path,
                        model_name="Facenet",
                        enforce_detection=False
                    )[0]["embedding"]

                    st.session_state.known_embeddings[new_name.strip()] = embedding
                    st.success(f"'{new_name.strip()}' ajouté(e) — visible immédiatement dans la détection en direct.")
                except Exception as e:
                    st.error(f"Erreur lors du calcul de l'embedding : {e}")

    with col_preview:
        if new_photo is not None:
            st.image(new_photo, caption="Aperçu", use_container_width=True)

    st.divider()
    st.subheader("Supprimer une personne")

    if st.session_state.known_embeddings:
        to_delete = st.selectbox("Choisir une personne", list(st.session_state.known_embeddings.keys()))
        if st.button("Supprimer", type="secondary"):
            del st.session_state.known_embeddings[to_delete]
            st.success(f"'{to_delete}' supprimé(e) de la base en mémoire.")
            st.rerun()
    else:
        st.info("Aucune personne à supprimer pour le moment.")