import cv2
import face_recognition

# 1. Charger image connue
known_image = face_recognition.load_image_file("dataset/img.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

known_face_encodings = [known_encoding]
known_face_names = ["nom"]

# 2. Webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # réduire image pour vitesse
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    # 3. détecter visages
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            name = known_face_names[0]

        face_names.append(name)

    # 4. afficher résultats
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()