from ultralytics import YOLO
import torch
from pathlib import Path

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

model_path = '10k/runs/train/bdd10k_yolov8n_fast/weights/best.pt'
model = YOLO(model_path)

print(f"Loaded model: {model_path}")

data_yaml = Path("data.yaml").resolve()
print(f"Using data config: {data_yaml}")

print("\n" + "="*60)
print("Running Validation with Plot Generation")
print("="*60 + "\n")

metrics = model.val(
    data=str(data_yaml),
    split='val',
    batch=16,
    device=device,
    
    plots=True,
    save_json=True,
    save_conf=True,
    
    conf=0.25,
    iou=0.45,
    max_det=300,
    
    verbose=True,
    cache=False,
    
    project='10k/runs/val',
    name='validation_plots',
    exist_ok=True
)

print("\n" + "="*60)
print("VALIDATION RESULTS")
print("="*60)
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"mAP75:    {metrics.box.map75:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")

print("\n" + "="*60)
print("Per-Class Results")
print("="*60)

class_names = ['car', 'truck', 'bus', 'person', 'bike', 'motor', 'rider', 'traffic light', 'traffic sign', 'train']

if hasattr(metrics.box, 'ap_class_index'):
    for i, class_idx in enumerate(metrics.box.ap_class_index):
        if i < len(metrics.box.ap):
            print(f"{class_names[class_idx]:15s} - mAP50: {metrics.box.ap[i]:.4f}")

print("\n" + "="*60)
print("PLOTS SAVED")
print("="*60)
print(f"Results directory: {metrics.save_dir}")
print(f"\nGenerated plots:")
print(f"  - confusion_matrix.png")
print(f"  - F1_curve.png")
print(f"  - P_curve.png")
print(f"  - R_curve.png")
print(f"  - PR_curve.png")
print(f"\nOpen the directory to view plots:")
print(f"  open {metrics.save_dir}")