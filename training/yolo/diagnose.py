from pathlib import Path
import yaml

def diagnose_dataset(yaml_file='data.yaml'):
    """Diagnose dataset issues that cause empty validation results"""
    
    print("="*60)
    print("DATASET DIAGNOSIS")
    print("="*60)
    
    # Load yaml
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    
    base_path = Path(config['path'])
    train_path = base_path / config.get('train', '')
    val_path = base_path / config.get('val', '')
    
    print(f"\n📁 Base path: {base_path}")
    print(f"📁 Train path: {train_path}")
    print(f"📁 Val path: {val_path}")
    
    # Check directories exist
    print("\n" + "="*60)
    print("DIRECTORY CHECK")
    print("="*60)
    
    for name, path in [('Train images', train_path), 
                       ('Val images', val_path),
                       ('Train labels', Path(str(train_path).replace('images', 'labels'))),
                       ('Val labels', Path(str(val_path).replace('images', 'labels')))]:
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
        
        if exists and 'images' in str(path):
            img_count = len(list(path.glob('*.jpg'))) + len(list(path.glob('*.png')))
            print(f"   → {img_count} images")
        elif exists and 'labels' in str(path):
            lbl_count = len(list(path.glob('*.txt')))
            print(f"   → {lbl_count} label files")
    
    # Check validation labels
    val_labels_path = Path(str(val_path).replace('images', 'labels'))
    
    if val_labels_path.exists():
        print("\n" + "="*60)
        print("VALIDATION LABELS CHECK")
        print("="*60)
        
        label_files = list(val_labels_path.glob('*.txt'))
        
        if not label_files:
            print("❌ No label files found in validation set!")
            return
        
        empty_count = 0
        total_boxes = 0
        sample_labels = []
        
        for lbl_file in label_files[:10]:  # Check first 10
            with open(lbl_file, 'r') as f:
                lines = f.readlines()
                if not lines:
                    empty_count += 1
                else:
                    total_boxes += len(lines)
                    if len(sample_labels) < 3:
                        sample_labels.append((lbl_file.name, lines[0].strip()))
        
        print(f"Total label files: {len(label_files)}")
        print(f"Empty label files: {empty_count}")
        print(f"Average boxes per image: {total_boxes / len(label_files[:10]):.1f}")
        
        if sample_labels:
            print("\nSample labels (first line of first 3 files):")
            for name, line in sample_labels:
                print(f"  {name}: {line}")
        
        if empty_count == len(label_files[:10]):
            print("\n❌ ALL LABELS ARE EMPTY!")
            print("This is why validation shows 'nan' - there are no ground truth boxes.")
    
    # Check class distribution
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION (first 100 labels)")
    print("="*60)
    
    class_counts = {}
    checked = 0
    
    for lbl_file in list(val_labels_path.glob('*.txt'))[:100]:
        with open(lbl_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
        checked += 1
    
    if class_counts:
        class_names = config.get('names', {})
        for class_id in sorted(class_counts.keys()):
            class_name = class_names.get(class_id, f"Class {class_id}")
            count = class_counts[class_id]
            print(f"  {class_id}: {class_name:15s} - {count} instances")
    else:
        print("❌ No class instances found!")
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    
    if not val_path.exists():
        print("❌ ISSUE: Validation image directory doesn't exist")
        print("   FIX: Check data.yaml path or create validation split")
    elif not val_labels_path.exists():
        print("❌ ISSUE: Validation labels directory doesn't exist")
        print("   FIX: Run label extraction script for validation data")
    elif not label_files:
        print("❌ ISSUE: No label files in validation directory")
        print("   FIX: Extract labels for validation images")
    elif empty_count == len(label_files[:10]):
        print("❌ ISSUE: All label files are empty")
        print("   FIX: Re-run label extraction - current labels have no bounding boxes")
    elif not class_counts:
        print("❌ ISSUE: No valid bounding boxes found")
        print("   FIX: Check label format and re-extract if needed")
    else:
        print("✅ Dataset looks OK - issue might be with training")
        print("   Check: Did training show any mAP values > 0?")
        print("   Check: Are you using the correct model weights?")

if __name__ == "__main__":
    diagnose_dataset()