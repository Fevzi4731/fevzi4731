from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2

# Modeli yükle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Etiketleri eşleştirmek istersen (örnek)
etiket_cevir = {
    "happy": "Mutlu",
    "sad": "Üzgün",
    "surprised": "Şaşkın",
    "angry": "Kızgın",
    "mutlu": "Mutlu",
    "uzgun": "uzgun",
    "saskin": "saskin",
    "kizgin": "Kizgin"
}

def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for face_landmarks in face_landmarks_list:
        # Nokta verilerini hazırla
        koordinatlar = []
        for landmark in face_landmarks:
            koordinatlar.append(round(landmark.x, 4))
            koordinatlar.append(round(landmark.y, 4))

        # Model ile tahmin et
        sonuc = model.predict([koordinatlar])[0]
        sonuc_yazi = etiket_cevir.get(sonuc.lower(), sonuc)

        # Görüntü üzerine yaz
        annotated_image = cv2.putText(annotated_image,
                                      sonuc_yazi,
                                      (60, 60),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      2,
                                      (0, 255, 0),
                                      4)
    return annotated_image


# MediaPipe setup
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Kamerayı başlat
cam = cv2.VideoCapture(0)
while cam.isOpened():
    basari, frame = cam.read()
    if basari:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = detector.detect(mp_image)
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)

        # Sonucu göster
        cv2.imshow("Yüz İfade Tahmini", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break

cam.release()
cv2.destroyAllWindows()
