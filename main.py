import cv2
import os
import numpy as np
import tensorflow as tf
from deepface import DeepFace

DATASET_PATH = "dataset"
EMOTION_MODEL_PATH = "emotion_recognition/improved_aug_emotion_model.keras"

emotion_classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH)

# =========================
# 1. CHARGER DATASET PERSONNES
# =========================
known_embeddings = {}

print("Chargement dataset...")

for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)

    if os.path.isdir(person_path):
        images = os.listdir(person_path)

        if len(images) > 0:
            img_path = os.path.join(person_path, images[0])

            embedding = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]

            known_embeddings[person] = embedding

print("Dataset chargé ✔")

# =========================
# 2. DISTANCE
# =========================
def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

# =========================
# 3. PREDICTION EMOTION
# =========================
def predict_emotion(face_img):
    try:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48))

        img = np.expand_dims(gray, axis=-1)
        img = np.expand_dims(img, axis=0)

        preds = emotion_model.predict(img, verbose=0)
        idx = np.argmax(preds)

        emotion = emotion_classes[idx]
        confidence = preds[0][idx]

        return emotion, confidence

    except:
        return "unknown", 0.0

# =========================
# 4. CAMERA
# =========================
cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 2 != 0:
        continue

    frame = cv2.resize(frame, (640, 480))

    faces = DeepFace.extract_faces(
        img_path=frame,
        detector_backend="opencv",
        enforce_detection=False
    )

    for face in faces:
        region = face["facial_area"]
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]

        face_img = frame[y:y+h, x:x+w]

        name = "Unknown"
        best_dist = 999

        try:
            embedding = DeepFace.represent(
                img_path=face_img,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]

            for person, ref_embedding in known_embeddings.items():
                dist = cosine_distance(embedding, ref_embedding)

                if dist < 10 and dist < best_dist:
                    best_dist = dist
                    name = person

        except:
            name = "Unknown"

        emotion, emotion_conf = predict_emotion(face_img)

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        label = f"{name} | {emotion} ({emotion_conf:.2f})"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("FaceSense - Face + Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()