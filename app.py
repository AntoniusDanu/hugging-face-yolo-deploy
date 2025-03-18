from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO
import shutil
import os

# Inisialisasi FastAPI
app = FastAPI()

# Load model YOLO & PaddleOCR sekali saat startup
MODEL_PATH = "./best.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model {MODEL_PATH} tidak ditemukan!")

yolo_model = YOLO(MODEL_PATH)
ocr = PaddleOCR(lang='en')

# 📌 Fungsi untuk membaca plat nomor
def detect_plate(image_path):
    # Deteksi objek dengan YOLO
    results = yolo_model(image_path)
    
    # Pastikan hasil deteksi tidak kosong
    if not results or len(results[0].boxes) == 0:
        return {"error": "Tidak ada plat nomor terdeteksi"}

    # Ambil bounding box pertama
    boxes = results[0].boxes.xyxy.numpy()
    x1, y1, x2, y2 = map(int, boxes[0])

    # Baca gambar
    image = cv2.imread(image_path)
    
    # Pastikan koordinat tidak keluar dari ukuran gambar
    h, w, _ = image.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    
    # Potong area plat nomor
    plate_image = image[y1:y2, x1:x2]
    
    # Cek apakah pemotongan berhasil
    if plate_image.size == 0:
        return {"error": "Gagal memotong plat nomor"}

    # Simpan gambar hasil crop
    plate_path = "plate.jpg"
    cv2.imwrite(plate_path, plate_image)
    
    # Jalankan OCR
    ocr_results = ocr.ocr(plate_path, cls=True)
    text = ocr_results[0][0][1][0] if ocr_results else "Tidak terbaca"
    
    return {"plate_number": text, "bounding_box": boxes[0].tolist()}

# ✅ Endpoint utama untuk mengecek API
@app.get("/")
def home():
    return {"message": "🚀 ALPR API is running!"}

# ✅ Endpoint untuk upload gambar
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = "temp.jpg"
    
    # Simpan file sementara
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Proses gambar dengan YOLO + OCR
    result = detect_plate(file_path)

    return result

# ✅ Jalankan aplikasi
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
