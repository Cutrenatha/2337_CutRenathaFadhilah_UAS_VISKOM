from ultralytics import YOLO
import os
from pathlib import Path
import torch

def check_dataset():
    """Check dataset structure"""
    dataset_path = Path("dataset")
    
    if not dataset_path.exists():
        print("\n[ERROR] Folder 'dataset' tidak ditemukan!")
        print("Download dataset dari Kaggle terlebih dahulu.\n")
        return False
    
    train_path = dataset_path / "train"
    test_path = dataset_path / "test"
    
    if not train_path.exists():
        print("\n[ERROR] Folder 'dataset/train' tidak ditemukan!\n")
        return False
    
    # Count images
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    
    total_train = 0
    print("\nTrain Set:")
    for class_dir in sorted(train_path.iterdir()):
        if class_dir.is_dir():
            count = len([f for f in class_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            total_train += count
            print(f"  {class_dir.name:20s} : {count:5d} images")
    print(f"  {'Total':20s} : {total_train:5d} images")
    
    total_test = 0
    if test_path.exists():
        print("\nTest Set:")
        for class_dir in sorted(test_path.iterdir()):
            if class_dir.is_dir():
                count = len([f for f in class_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                total_test += count
                print(f"  {class_dir.name:20s} : {count:5d} images")
        print(f"  {'Total':20s} : {total_test:5d} images")
    
    print("="*70 + "\n")
    return True


def check_device():
    """Check available device"""
    print("="*70)
    print("DEVICE CONFIGURATION")
    print("="*70)
    
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\nDevice      : GPU ({gpu_name})")
        print(f"CUDA        : {torch.version.cuda}")
        print(f"Performance : Fast training (~1-3 hours)")
    else:
        device = 'cpu'
        print(f"\nDevice      : CPU")
        print(f"Performance : Slower training (~3-8 hours)")
        print(f"\nTip: Use Google Colab for free GPU access")
    
    print("="*70 + "\n")
    return device


def train_fruit_classifier():
    """Train YOLOv8 classification model"""
    
    device = check_device()
    
    print("="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    
    # Configure based on device
    if device == 'cpu':
        model_name = "yolov8n-cls.pt"
        config = {
            'data': "dataset",
            'epochs': 30,
            'patience': 10,
            'imgsz': 128,
            'batch': 4,
            'workers': 2,
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 15,
            'translate': 0.1,
            'scale': 0.5,
            'fliplr': 0.5,
            'mixup': 0.0,
            'val': True,
            'save': True,
            'save_period': 5,
            'device': 'cpu',
            'pretrained': True,
            'verbose': False,  # Disable verbose untuk output lebih bersih
            'seed': 42,
            'project': 'runs/classify',
            'name': 'fruit_freshness',
            'exist_ok': False,
        }
        print(f"\nModel       : YOLOv8-Nano (CPU optimized)")
        print(f"Epochs      : 30")
        print(f"Image Size  : 128x128")
        print(f"Batch Size  : 4")
        print(f"Device      : CPU")
    else:
        model_name = "yolov8m-cls.pt"
        config = {
            'data': "dataset",
            'epochs': 100,
            'patience': 20,
            'imgsz': 224,
            'batch': 32,
            'workers': 8,
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 15,
            'translate': 0.1,
            'scale': 0.5,
            'fliplr': 0.5,
            'mixup': 0.15,
            'val': True,
            'save': True,
            'save_period': 10,
            'device': 0,
            'pretrained': True,
            'verbose': False,  # Disable verbose untuk output lebih bersih
            'seed': 42,
            'project': 'runs/classify',
            'name': 'fruit_freshness',
            'exist_ok': False,
        }
        print(f"\nModel       : YOLOv8-Medium (GPU optimized)")
        print(f"Epochs      : 100")
        print(f"Image Size  : 224x224")
        print(f"Batch Size  : 32")
        print(f"Device      : GPU")
    
    print("="*70 + "\n")
    
    # Load model
    print("Loading model...")
    model = YOLO(model_name)
    print("Model loaded.\n")
    
    # Start training
    print("="*70)
    print("STARTING TRAINING")
    print("="*70)
    print()
    
    results = model.train(**config)
    
    # Training complete
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"\nModel saved to : {results.save_dir}/weights/best.pt")
    print(f"Results saved  : {results.save_dir}")
    print("="*70 + "\n")
    
    return results


def validate_model(model_path="runs/classify/fruit_freshness/weights/best.pt"):
    """Validate model performance"""
    
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model not found: {model_path}\n")
        return None
    
    print("="*70)
    print("VALIDATION")
    print("="*70)
    print()
    
    model = YOLO(model_path)
    metrics = model.val(data="dataset", split='test', batch=32, imgsz=224, verbose=False)
    
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    print(f"\nTop-1 Accuracy : {metrics.top1*100:.2f}%")
    print(f"Top-5 Accuracy : {metrics.top5*100:.2f}%")
    print("="*70 + "\n")
    
    return metrics


if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("FRUIT FRESHNESS DETECTION - TRAINING SCRIPT")
    print("="*70 + "\n")
    
    # Check dataset
    if not check_dataset():
        exit(1)
    
    # Confirm training
    response = input("Start training? (y/n): ").lower().strip()
    if response != 'y':
        print("\nTraining cancelled.\n")
        exit(0)
    
    print()
    
    # Train model
    results = train_fruit_classifier()
    
    # Ask for validation
    validate = input("\nRun validation? (y/n): ").lower().strip()
    if validate == 'y':
        print()
        validate_model()
    
    print("Done. Use predict.py or app.py for inference.\n")