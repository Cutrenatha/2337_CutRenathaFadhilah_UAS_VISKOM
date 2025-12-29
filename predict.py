from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Folder default untuk prediksi
PREDICT_FOLDER = "predict"


class FruitPredictor:
    def __init__(self, model_path=None):
        """Initialize predictor dengan trained model"""

        # Auto-detect model jika tidak diisi
        if model_path is None:
            model_path = self.find_latest_model()

        if not os.path.exists(model_path):
            print(f"[ERROR] Model tidak ditemukan: {model_path}")
            print("Jalankan train.py terlebih dahulu.")
            sys.exit(1)

        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        print("Model loaded.\n")

    def find_latest_model(self):
        """Mencari model terbaru di runs/classify"""
        runs_path = Path("runs/classify")

        if not runs_path.exists():
            return "runs/classify/fruit_freshness/weights/best.pt"

        model_folders = sorted(
            [f for f in runs_path.iterdir()
             if f.is_dir() and f.name.startswith("fruit_freshness")]
        )

        if not model_folders:
            return "runs/classify/fruit_freshness/weights/best.pt"

        latest_folder = model_folders[-1]
        return str(latest_folder / "weights" / "best.pt")

    def predict_single_image(self, image_path, show_result=True):
        """Prediksi satu gambar"""

        if not os.path.exists(image_path):
            print(f"Image tidak ditemukan: {image_path}")
            return None

        results = self.model(image_path, verbose=False)

        probs = results[0].probs
        pred_idx = probs.top1
        pred_label = results[0].names[pred_idx]
        confidence = probs.top1conf.item()

        # Output terminal sederhana
        print(f"{Path(image_path).name} -> {pred_label} ({confidence*100:.1f}%)")

        if show_result:
            self.display_result(image_path, pred_label, confidence)

        return {
            "label": pred_label,
            "confidence": confidence,
            "is_fresh": "fresh" in pred_label.lower()
        }

    def display_result(self, image_path, pred_label, confidence):
        """Menampilkan gambar dengan hasil prediksi (rapi & tidak ketutup)"""

        img = cv2.imread(image_path)
        if img is None:
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(
            f"{pred_label.upper()} - {confidence*100:.1f}%",
            fontsize=14,
            fontweight="bold",
            pad=12
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def predict_batch(self, folder_path):
        """Prediksi seluruh gambar dalam folder"""

        folder = Path(folder_path)
        if not folder.exists():
            print(f"Folder '{folder_path}' tidak ditemukan.")
            return

        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        images = [f for f in folder.iterdir()
                  if f.suffix.lower() in image_extensions]

        if not images:
            print("Tidak ada gambar untuk diprediksi.")
            return

        print(f"Memproses {len(images)} gambar...\n")

        fresh_count = 0
        for img_path in images:
            result = self.predict_single_image(str(img_path), show_result=True)
            if result and result["is_fresh"]:
                fresh_count += 1

        rotten_count = len(images) - fresh_count

        print("\nRingkasan Prediksi")
        print("-" * 30)
        print(f"Total  : {len(images)}")
        print(f"Fresh  : {fresh_count}")
        print(f"Rotten : {rotten_count}")
        print("-" * 30)


def main():
    predictor = FruitPredictor()

    if not os.path.exists(PREDICT_FOLDER):
        print(f"Folder '{PREDICT_FOLDER}' belum ada.")
        print("Buat folder 'predict' dan isi dengan gambar buah.")
        return

    predictor.predict_batch(PREDICT_FOLDER)


if __name__ == "__main__":
    main()
