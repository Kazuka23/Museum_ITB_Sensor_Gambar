import math
import os
import random
import time
import cv2
import mediapipe as mp
import numpy as np

# 1. Konfigurasi Path Folder Otomatis
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_WARNA = os.path.join(BASE_DIR, "assets", "warna.png")
PATH_SKETSA = os.path.join(BASE_DIR, "assets", "sketsa.png")

WIDTH, HEIGHT = 1280, 720
BRUSH_RADIUS = 130  # Ukuran lingkaran diperbesar sedikit agar lebih cepat mewarnai
COMPLETE_THRESHOLD = 0.90  # 90% bagian berwarna terkena kuas = Selesai!

# Margin zona tengah kamera (0.22 berarti cukup gerakkan tangan di 56% area tengah kamera
# untuk menjangkau 100% ujung kiri-kanan dan atas-bawah layar)
CAM_MARGIN_X = 0.22
CAM_MARGIN_Y = 0.22

# 2. Memuat Gambar dari Folder assets
img_color_raw = cv2.imread(PATH_WARNA)
img_sketch_raw = cv2.imread(PATH_SKETSA)

if img_color_raw is None:
    print(f"[ERROR] Gambar tidak ditemukan di: {PATH_WARNA}")
    exit()

img_color = cv2.resize(img_color_raw, (WIDTH, HEIGHT))

if img_sketch_raw is not None:
    img_sketch = cv2.resize(img_sketch_raw, (WIDTH, HEIGHT))
else:
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_sketch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# --- DETEKSI PINTAR: Cari bagian gambar yang memang punya warna (bukan background putih polos) ---
diff = cv2.absdiff(img_color, img_sketch)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, target_area_mask = cv2.threshold(diff_gray, 10, 255, cv2.THRESH_BINARY)
total_target_pixels = np.count_nonzero(target_area_mask)

# Fallback jika seluruh gambar berbeda
if total_target_pixels < 1000:
    target_area_mask[:] = 255
    total_target_pixels = WIDTH * HEIGHT

# 3. Inisialisasi MediaPipe Hands & Drawing Skeleton
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,  # Lebih mudah mendeteksi tangan di kondisi agak gelap/miring
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

reveal_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)


def reset_canvas():
    global reveal_mask, smooth_x, smooth_y, target_x, target_y
    reveal_mask[:] = 0
    rand_x = random.randint(BRUSH_RADIUS + 50, WIDTH - BRUSH_RADIUS - 50)
    rand_y = random.randint(BRUSH_RADIUS + 50, HEIGHT - BRUSH_RADIUS - 50)
    smooth_x, smooth_y = rand_x, rand_y
    target_x, target_y = rand_x, rand_y


smooth_x, smooth_y = 0, 0
target_x, target_y = 0, 0
reset_canvas()

is_animating = False
anim_start_time = 0
ANIM_DURATION = 1.6

WINDOW_NAME = "Museum ITB Interactive"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

# CATATAN: Fungsi Mouse Callback sudah dihapus sepenuhnya!

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi Tangan & Gambar Skeleton
    results = hands.process(rgb_frame)
    hand_detected = False

    if results.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            # Ambil titik tengah telapak tangan (Landmark 9)
            palm = hand_landmarks.landmark[9]

            # Pemetaan dengan Margin: Tangan tidak perlu ke pinggir kamera untuk mencapai ujung layar!
            norm_x = (palm.x - CAM_MARGIN_X) / (1.0 - 2 * CAM_MARGIN_X)
            norm_y = (palm.y - CAM_MARGIN_Y) / (1.0 - 2 * CAM_MARGIN_Y)

            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))

            target_x = int(norm_x * WIDTH)
            target_y = int(norm_y * HEIGHT)

    # Pergerakan lingkaran yang mulus (Smoothing)
    smooth_x = int(smooth_x + (target_x - smooth_x) * 0.25)
    smooth_y = int(smooth_y + (target_y - smooth_y) * 0.25)
    smooth_x = max(0, min(WIDTH, smooth_x))
    smooth_y = max(0, min(HEIGHT, smooth_y))

    # --- MODE 1: SEDANG MEWARNAI ---
    if not is_animating:
        if hand_detected:
            cv2.circle(reveal_mask, (smooth_x, smooth_y), BRUSH_RADIUS, 255, -1)

        canvas = img_sketch.copy()
        canvas[reveal_mask == 255] = img_color[reveal_mask == 255]

        # Lingkaran kuas putih transparan ala Museum ITB
        overlay = canvas.copy()
        cv2.circle(
            overlay, (smooth_x, smooth_y), BRUSH_RADIUS, (255, 255, 255), -1
        )
        cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)
        cv2.circle(
            canvas, (smooth_x, smooth_y), BRUSH_RADIUS, (255, 255, 255), 4
        )
        cv2.circle(
            canvas, (smooth_x, smooth_y), BRUSH_RADIUS - 8, (255, 255, 255), 1
        )

        # Hitung persentase HANYA pada objek yang berwarna (bukan background kosong)
        colored_target = np.count_nonzero(
            cv2.bitwise_and(reveal_mask, target_area_mask)
        )
        raw_ratio = colored_target / total_target_pixels
        display_percent = min(100, int((raw_ratio / COMPLETE_THRESHOLD) * 100))

        percent_text = f"Progress: {display_percent}%"
        cv2.putText(
            canvas,
            percent_text,
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            4,
        )
        cv2.putText(
            canvas,
            percent_text,
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        if raw_ratio >= COMPLETE_THRESHOLD:
            is_animating = True
            anim_start_time = time.time()

    # --- MODE 2: ANIMASI BOUNCE SAAT SELESAI LALU RESET ---
    else:
        elapsed = time.time() - anim_start_time
        if elapsed < ANIM_DURATION:
            progress = elapsed / ANIM_DURATION
            scale = 1.0 + 0.12 * math.sin(progress * math.pi * 3) * (
                1.0 - progress
            )

            new_w, new_h = int(WIDTH * scale), int(HEIGHT * scale)
            resized = cv2.resize(img_color, (new_w, new_h))

            canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            if scale >= 1.0:
                start_x = (new_w - WIDTH) // 2
                start_y = (new_h - HEIGHT) // 2
                canvas = resized[
                    start_y : start_y + HEIGHT, start_x : start_x + WIDTH
                ]
            else:
                start_x = (WIDTH - new_w) // 2
                start_y = (HEIGHT - new_h) // 2
                canvas[
                    start_y : start_y + new_h, start_x : start_x + new_w
                ] = resized

            cv2.putText(
                canvas,
                "SELESAI!",
                (WIDTH // 2 - 140, HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 0),
                8,
            )
            cv2.putText(
                canvas,
                "SELESAI!",
                (WIDTH // 2 - 140, HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (100, 255, 100),
                4,
            )
        else:
            is_animating = False
            reset_canvas()
            canvas = img_sketch.copy()

    # --- PREVIEW WEBCAM & KOTAK BATAS ZONA GERAK (POJOK KANAN BAWAH) ---
    cam_w, cam_h = 280, 180
    cam_small = cv2.resize(frame, (cam_w, cam_h))

    # Gambar kotak kuning tipis di dalam preview kamera sebagai panduan "Zona Aktif"
    zx1, zy1 = int(cam_w * CAM_MARGIN_X), int(cam_h * CAM_MARGIN_Y)
    zx2, zy2 = int(cam_w * (1 - CAM_MARGIN_X)), int(cam_h * (1 - CAM_MARGIN_Y))
    cv2.rectangle(cam_small, (zx1, zy1), (zx2, zy2), (0, 255, 255), 1)

    border_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    status_text = "Tangan Terdeteksi" if hand_detected else "Mencari Tangan..."

    y_offset, x_offset = HEIGHT - cam_h - 20, WIDTH - cam_w - 20
    canvas[y_offset : HEIGHT - 20, x_offset : WIDTH - 20] = cam_small
    cv2.rectangle(
        canvas, (x_offset, y_offset), (WIDTH - 20, HEIGHT - 20), border_color, 3
    )

    cv2.rectangle(
        canvas, (x_offset, y_offset - 30), (WIDTH - 20, y_offset), border_color, -1
    )
    cv2.putText(
        canvas,
        status_text,
        (x_offset + 10, y_offset - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.imshow(WINDOW_NAME, canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break
    elif key == ord("r"):
        reset_canvas()

cap.release()
cv2.destroyAllWindows()