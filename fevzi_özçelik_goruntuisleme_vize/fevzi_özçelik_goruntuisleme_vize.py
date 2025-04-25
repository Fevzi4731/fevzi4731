import cv2
import mediapipe as mp
import random
import math
import numpy as np
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Çöp görseli yükle
trash_img = cv2.imread("trash.png", cv2.IMREAD_UNCHANGED)
trash_size = 60
score = 0

# İlk pozisyon ve görünürlük durumu
trash_x, trash_y = random.randint(100, 500), random.randint(100, 400)
trash_visible = True
trash_hide_time = 0  # Gizlenme zamanı başlangıçta yok

# Kamera başlat
cam = cv2.VideoCapture(0)

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def overlay_png(background, overlay, x, y):
    h, w = overlay.shape[:2]
    if y + h > background.shape[0] or x + w > background.shape[1]:
        return background

    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            background[y:y+h, x:x+w, c] * (1 - alpha) + overlay[:, :, c] * alpha
        )
    return background

while True:
    success, frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    finger_x, finger_y = -100, -100
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            finger_x = int(handLms.landmark[8].x * w)
            finger_y = int(handLms.landmark[8].y * h)

            cv2.circle(frame, (finger_x, finger_y), 10, (0, 255, 0), -1)

    # Çöp görselini göster
    if trash_visible:
        resized_trash = cv2.resize(trash_img, (trash_size, trash_size))
        frame = overlay_png(frame, resized_trash, trash_x, trash_y)

        # Dokunma kontrolü
        if distance((finger_x, finger_y), (trash_x + trash_size//2, trash_y + trash_size//2)) < trash_size//2:
            trash_visible = False
            trash_hide_time = time.time()  # Şu anı not al
            score += 1

    else:
        # 1 saniye sonra yeni çöp gelsin
        if time.time() - trash_hide_time > 1.0:
            trash_x, trash_y = random.randint(50, 600), random.randint(50, 400)
            trash_visible = True

    # Skor yazısı
    cv2.putText(frame, f"Score: {score}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Göster
    cv2.imshow("Trash Collector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
