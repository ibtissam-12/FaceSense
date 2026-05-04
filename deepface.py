import cv2
from deepface import DeepFace

# 🎯 Ouvrir webcam
video_capture = cv2.VideoCapture(0)

# 🎯 Image de référence (ta base de données)
reference_image = "dataset/img3.jpg"
reference_name = "sara"

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    try:
        # 🔥 DeepFace compare frame avec image connue
        result = DeepFace.verify(
            img1_path=frame,               # image webcam
            img2_path=reference_image,     # image connue
            enforce_detection=False        # évite crash si aucun visage détecté
        )

        # 🎯 Si match
        if result["verified"]:
            name = reference_name
            color = (0, 255, 0)
        else:
            name = "Inconnu"
            color = (0, 0, 255)

    except:
        name = "Erreur"
        color = (0, 0, 255)

    # 🎯 Dessiner texte sur l’écran
    cv2.putText(frame, name, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2)

    # 🎯 Affichage vidéo
    cv2.imshow("FaceSense - DeepFace", frame)

    # Quitter avec q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()










