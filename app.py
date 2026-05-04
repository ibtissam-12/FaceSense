import streamlit as st
import numpy as np
import cv2
from deepface import DeepFace
import tempfile
import os

st.title("🎭 Face Recognition en temps réel")

# Charger automatiquement toutes les personnes du dataset
DATASET_PATH = "dataset"

def get_known_faces():
    known_faces = {}
    if not os.path.exists(DATASET_PATH):
        return known_faces
    for person_name in os.listdir(DATASET_PATH):
        person_folder = os.path.join(DATASET_PATH, person_name)
        if os.path.isdir(person_folder):
            images = [f for f in os.listdir(person_folder) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                known_faces[person_name] = os.path.join(person_folder, images[0])
    return known_faces

known_faces = get_known_faces()

# Afficher les personnes connues
if known_faces:
    st.sidebar.title("👥 Personnes connues")
    for name in known_faces:
        st.sidebar.write(f"✅ {name}")
else:
    st.warning("⚠️ Aucune personne dans le dataset! Crée des dossiers dans 'dataset/'")

st.markdown("---")

# Option 1: Upload image de référence manuelle
st.subheader("Option 1: Upload une image")
uploaded = st.file_uploader("Upload image de référence", type=["jpg", "jpeg", "png"])
manual_name = st.text_input("Nom de la personne", placeholder="ex: ibtissam")

ref_path_manual = None
if uploaded and manual_name:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        f.write(uploaded.read())
        ref_path_manual = f.name
    # Ajouter au dataset automatiquement
    person_folder = os.path.join(DATASET_PATH, manual_name)
    os.makedirs(person_folder, exist_ok=True)
    save_path = os.path.join(person_folder, "img1.jpg")
    with open(save_path, "wb") as f:
        uploaded.seek(0)
        f.write(uploaded.read())
    st.success(f"✅ {manual_name} ajouté au dataset!")
    known_faces = get_known_faces()

st.markdown("---")

# Webcam
st.subheader("📷 Reconnaissance faciale")
picture = st.camera_input("Prends une photo")

if picture:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        f.write(picture.getvalue())
        cam_path = f.name

    frame = cv2.imdecode(np.frombuffer(picture.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    recognized_name = "Unknown"
    recognized = False
    best_distance = 1.0

    # Comparer avec toutes les personnes du dataset
    if known_faces:
        with st.spinner("🔍 Analyse en cours..."):
            for name, ref_path in known_faces.items():
                try:
                    result = DeepFace.verify(
                        img1_path=ref_path,
                        img2_path=cam_path,
                        model_name="Facenet",
                        enforce_detection=False
                    )
                    if result["verified"] and result["distance"] < best_distance:
                        best_distance = result["distance"]
                        recognized_name = name
                        recognized = True
                except Exception:
                    continue

    # Dessiner rectangle
    try:
        faces = DeepFace.extract_faces(cam_path, enforce_detection=False)
        for face in faces:
            region = face["facial_area"]
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            color = (0, 255, 0) if recognized else (0, 0, 255)
            cv2.rectangle(rgb, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(rgb, (x, y-35), (x+w, y), color, -1)
            cv2.putText(rgb, recognized_name, (x+5, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    except Exception:
        pass

    # Afficher résultat
    if recognized:
        st.success(f"✅ Reconnu: **{recognized_name}** (distance: {round(best_distance, 3)})")
    else:
        st.error("❌ Visage inconnu!")

    st.image(rgb, caption="Résultat", width=700)

    os.unlink(cam_path)