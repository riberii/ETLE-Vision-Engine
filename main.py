"""
ETLE Vision Engine - Simulasi Sistem Tilang Elektronik
=========================================================
Fitur:
  1. Toggle ON/OFF sistem deteksi pakai SPASI (real-time, tanpa restart)
  2. Logging otomatis tiap pelanggaran ke CSV (waktu asli komputer, plat
     simulasi, jenis kendaraan, warna) + foto bukti ke folder terpisah
  3. Plat nomor simulasi bergaya plat Indonesia, di-generate sekali per
     kendaraan (pakai object tracking, bukan per-frame) dan dijamin unik
     selama sesi berjalan.

Catatan jujur soal keterbatasan:
  - Deteksi MEREK mobil (Toyota, Honda, dst) TIDAK bisa dilakukan dengan
    YOLOv8n biasa, karena model ini cuma dilatih di dataset COCO yang
    cuma tau kategori umum (mobil/motor/bus/truk), bukan logo merek.
    Untuk itu butuh model klasifikasi terpisah yang dilatih khusus di
    dataset logo/merek mobil. Function detect_brand() di bawah adalah
    placeholder yang bisa lo ganti nanti kalau mau nambahin model kedua.
  - Warna kendaraan dideteksi pakai heuristik warna rata-rata (HSV),
    bukan model ML - cukup akurat untuk demo, tapi bisa meleset kalau
    pencahayaan aneh atau mobilnya banyak pantulan/glare.
"""

import cv2
import os
import csv
import random
import string
from datetime import datetime
from ultralytics import YOLO

# =========================================================
# KONFIGURASI
# =========================================================
MODEL_PATH = "yolov8n.pt"
CAMERA_INDEX = 2
CAMERA_BACKEND = cv2.CAP_MSMF

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
VIOLATION_LINE_Y = 400

VEHICLE_CLASSES = [2, 3, 5, 7]  # 2=car, 3=motorcycle, 5=bus, 7=truck
CLASS_NAMES_ID = {2: "Mobil", 3: "Motor", 5: "Bus", 7: "Truk"}

CSV_FILE = "data_pelanggaran_etle.csv"
SNAPSHOT_DIR = "bukti_pelanggaran"

# Kode wilayah plat nomor Indonesia (contoh umum, gak lengkap semua tapi
# representatif buat simulasi)
AREA_CODES = [
    "A", "B", "D", "E", "F", "G", "H", "K", "L", "M", "N", "P", "R", "S",
    "T", "W", "Z", "AA", "AB", "AD", "AE", "AG", "BA", "BB", "BD", "BE",
    "BG", "BH", "BK", "BL", "BM", "BN", "DA", "DB", "DD", "DE", "DG",
    "DH", "DK", "DL", "DM", "DN", "DR", "DS", "DT", "EA", "EB",
]

# =========================================================
# STATE GLOBAL
# =========================================================
used_plates = set()        # semua plat yang udah pernah dipakai (anti-dupe)
track_plates = {}          # track_id -> plat nomor (persist selama mobil kelihatan)
track_brands = {}          # track_id -> merek (placeholder)
violated_ids = set()       # track_id yang sudah tercatat melanggar (biar gak dobel log)


# =========================================================
# FUNGSI BANTUAN
# =========================================================
def generate_unique_plate():
    """Generate plat nomor gaya Indonesia, dijamin belum pernah dipakai."""
    while True:
        area = random.choice(AREA_CODES)
        numbers = "".join(random.choices(string.digits, k=random.choice([3, 4])))
        suffix = "".join(random.choices(string.ascii_uppercase, k=random.choice([2, 3])))
        plate = f"{area} {numbers} {suffix}"
        if plate not in used_plates:
            used_plates.add(plate)
            return plate


def detect_color_name(crop):
    """Estimasi nama warna dominan dari crop bounding box pakai HSV mean."""
    if crop is None or crop.size == 0:
        return "Tidak Diketahui"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.mean(hsv)[:3]

    if v < 50:
        return "Hitam"
    if s < 40 and v > 200:
        return "Putih"
    if s < 60:
        return "Silver/Abu-abu"
    if h < 10 or h >= 170:
        return "Merah"
    if h < 25:
        return "Oranye"
    if h < 35:
        return "Kuning"
    if h < 85:
        return "Hijau"
    if h < 130:
        return "Biru"
    if h < 150:
        return "Ungu"
    return "Merah"


def detect_brand(crop):
    """
    PLACEHOLDER. YOLOv8n (COCO) gak bisa deteksi merek mobil.
    Kalau nanti lo punya model klasifikasi logo/merek terpisah,
    panggil model itu di sini dan return hasilnya.
    """
    return "Tidak Terdeteksi"


def init_csv():
    """Bikin file CSV + header kalau belum ada."""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Waktu (Real Time)",
                "Plat Nomor (Simulasi)",
                "Jenis Kendaraan",
                "Merek (Placeholder)",
                "Warna Kendaraan",
                "File Bukti Foto",
            ])


def log_violation(plate, vehicle_type, brand, color_name, snapshot_path):
    """Tulis satu baris pelanggaran ke CSV, pakai waktu asli komputer."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, plate, vehicle_type, brand, color_name, snapshot_path])
    return timestamp


def save_snapshot(frame, plate):
    """Simpan foto bukti pelanggaran, nama file pakai plat + timestamp."""
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    safe_plate = plate.replace(" ", "")
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{safe_plate}_{ts_file}.jpg"
    path = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(path, frame)
    return path


def draw_status_bar(frame, etle_active, width):
    """Tampilkan status ETLE ON/OFF + instruksi tombol di pojok kiri atas."""
    if etle_active:
        text = "ETLE: AKTIF"
        color = (0, 255, 0)
    else:
        text = "ETLE: NONAKTIF (PAUSED)"
        color = (0, 0, 255)

    cv2.rectangle(frame, (0, 0), (width, 60), (30, 30, 30), -1)
    cv2.putText(frame, text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, "[SPASI] Toggle ON/OFF   |   [Q] Keluar",
                (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


# =========================================================
# SETUP AWAL
# =========================================================
init_csv()

model = YOLO(MODEL_PATH)

# Gunakan ini agar OpenCV otomatis menyesuaikan tanpa dibatasi MSMF
CAMERA_INDEX = 2
cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

etle_active = True

print("ETLE Vision Engine Aktif. Tekan SPASI buat toggle, Q buat keluar.")

# =========================================================
# MAIN LOOP
# =========================================================
while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Menunggu kamera hardware siap...")
        key = cv2.waitKey(500) & 0xFF
        if key == ord("q"):
            break
        continue

    height, width, _ = frame.shape

    # Garis batas pelanggaran selalu digambar biar operator tetap
    # bisa lihat posisinya walau sistem lagi di-pause
    cv2.line(frame, (0, VIOLATION_LINE_Y), (width, VIOLATION_LINE_Y), (0, 0, 255), 3)
    cv2.putText(frame, "VIOLATION LINE", (10, VIOLATION_LINE_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if etle_active:
        results = model.track(
            frame, persist=True, classes=VEHICLE_CLASSES, conf=0.4, verbose=False
        )
        boxes = results[0].boxes

        if boxes is not None and boxes.id is not None:
            for box in boxes:
                track_id = int(box.id[0])
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                vehicle_type = CLASS_NAMES_ID.get(cls_id, "Kendaraan")

                # Plat nomor: generate SEKALI per track_id, lalu dipakai terus
                if track_id not in track_plates:
                    track_plates[track_id] = generate_unique_plate()
                plate = track_plates[track_id]

                # Crop buat deteksi warna & (nanti) merek
                crop_x1, crop_y1 = max(0, x1), max(0, y1)
                crop_x2, crop_y2 = min(width, x2), min(height, y2)
                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                color_name = detect_color_name(crop)

                if track_id not in track_brands:
                    track_brands[track_id] = detect_brand(crop)
                brand = track_brands[track_id]

                center_x = int((x1 + x2) / 2)
                center_y = y2
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

                # Teks plat nomor simulasi nempel di atas box
                cv2.putText(frame, plate, (x1, max(20, y1 - 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                is_violation = center_y > VIOLATION_LINE_Y

                if is_violation:
                    box_color = (0, 0, 255)
                    cv2.putText(frame, "VIOLATION!", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Cuma log SEKALI per kendaraan, bukan tiap frame dia
                    # masih di bawah garis
                    if track_id not in violated_ids:
                        violated_ids.add(track_id)
                        snapshot_path = save_snapshot(frame, plate)
                        ts = log_violation(plate, vehicle_type, brand, color_name, snapshot_path)
                        print(f"[PELANGGARAN] {ts} | {plate} | {vehicle_type} | "
                              f"Warna: {color_name} | Bukti: {snapshot_path}")
                else:
                    box_color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2 if not is_violation else 3)

    draw_status_bar(frame, etle_active, width)
    cv2.imshow("ETLE App - AI Vision Engine", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        etle_active = not etle_active
        status_text = "AKTIF" if etle_active else "NONAKTIF"
        print(f"[TOGGLE] Sistem ETLE sekarang: {status_text}")

cap.release()
cv2.destroyAllWindows()
