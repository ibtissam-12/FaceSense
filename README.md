# FaceSense – Real-Time Face Recognition & Emotion Analysis

## Overview

FaceSense is an AI-powered application for real-time face recognition and facial emotion detection, combining identity recognition (FaceNet via DeepFace) and emotion classification (CNN trained on FER2013) within a single Streamlit web application.

## Features

- Real-time face detection (OpenCV via DeepFace)
- Real-time identity recognition (FaceNet embeddings, cosine/Euclidean distance)
- Real-time emotion classification (custom CNN)
- Live webcam stream via WebRTC (works in-browser, no local OpenCV window needed)
- Manage known people directly from the app (add/remove, no manual file editing)

## Technologies Used

- Python
- Streamlit + streamlit-webrtc
- TensorFlow / Keras
- OpenCV
- DeepFace (FaceNet)
- NumPy, Pillow

## Installation

```bash
pip install -r requirements.txt
```

## Run locally

```bash
streamlit run dashboard.py
```

The app opens at `http://localhost:8501`. Your browser will ask for camera permission.

## Managing known people

Known individuals are no longer added by manually editing the `dataset/` folder. Use the **"Gérer les personnes"** tab in the app:
1. Upload a photo and enter a name → click "Ajouter"
2. The person becomes recognizable immediately in the live detection tab, no restart needed
3. Remove a person from the same tab if needed

## Dataset (model training)

FER2013 Dataset — grayscale images classified into seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

## Models Evaluated (training/research phase)

Three architectures were compared for emotion classification: a CNN trained from scratch, MobileNetV2 (transfer learning), and ResNet50 (transfer learning). ResNet50 achieved the best accuracy (63.84%). See the full report (`rapport_projet_deep_learning.pdf`) for details.

**Note:** the deployed app (`dashboard.py`) currently runs the CNN-from-scratch model for emotion classification.

## Methodology

1. Data preprocessing
2. Data augmentation
3. Model training
4. Transfer Learning
5. Fine-tuning
6. Evaluation

## Deployment

Deployed on Streamlit Community Cloud: *[lien à ajouter après déploiement]*

## Project structure

- `dashboard.py` — main deployable app (Streamlit, WebRTC, real-time face + emotion recognition, dataset management via UI). **This is the app to run/deploy.**
- `main.py` — standalone local script (OpenCV window, `cv2.imshow`) using the same core logic. Useful for quick local testing without Streamlit, but requires manual editing of `dataset/` and does not work when deployed to the cloud (no local webcam access on a server).
## Authors

Ibtissam Gaamouche
Siham Al Yassoul

AI & Digital Transformation Engineering Students
