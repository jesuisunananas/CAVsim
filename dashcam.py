from ultralytics import YOLO
import torch

#device = 'mps' if torch.backends.mps.is_available() else 'cpu'
#print(f"Using device: {device}")

# Auto-detect GPU (CUDA for NVIDIA)
if torch.cuda.is_available():
    device = 0  # Use first GPU (or 'cuda' or specific GPU id)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = 'cpu'
    print("⚠️  No GPU detected, using CPU")

model = YOLO("yolov8n.pt")

results = model.train(
    data='data.yaml',
    epochs=150,
    
    # SPEED OPTIMIZATIONS
    imgsz=640,
    batch=64,
    device=device,
    workers=8,
    cache='ram',
    
    project='10k/runs/train',
    name='bdd10k_yolov8n_gpu',
    exist_ok=True,
    
    # REDUCE AUGMENTATIONS (major speedup)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,                  # Reduce from 1.0 -> 0.5 (mosaic is slow)
    mixup=0.15,
    copy_paste=0.1,              # Disable copy-paste augmentation
    
    # OPTIMIZER SETTINGS
    optimizer='SGD',            # Adam is often faster than SGD
    lr0=0.01,                   # Lower learning rate for Adam
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,             # Reduce warmup
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    
    # VALIDATION & SAVING (reduce overhead)
    val=True,
    save=True,
    save_period=20,              # Save less frequently (20 instead of 10)
    patience=50,                 # Reduce patience for early stopping
    plots=False,                 # Disable plots during training (save at end)
    
    # OTHER SPEEDUPS
    amp=True,                    # Automatic Mixed Precision (faster on supported hardware)
    fraction=1.0,                # Use full dataset
    
    verbose=True,
    single_cls=False,
    rect=False,                  # Don't use rectangular training (slightly faster)
    cos_lr=True,                # Disable cosine LR (simpler, faster)
    close_mosaic=10,             # Disable mosaic in last 10 epochs

    #conf=0.25,
    #iou=0.45,
)

print("\n" + "="*60)
print("Training completed!")
print("="*60)
print(f"Best model saved at: {results.save_dir}/weights/best.pt")
print(f"Last model saved at: {results.save_dir}/weights/last.pt")
print(f"Results saved in: {results.save_dir}")