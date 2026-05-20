import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# BDD100K to YOLO class mapping
CLASS_MAPPING = {
    'car': 0,
    'truck': 1,
    'bus': 2,
    'person': 3,
    'bike': 4,
    'motor': 5,
    'rider': 6,
    'traffic light': 7,
    'traffic sign': 8,
    'train': 9
}

def convert_bbox_to_yolo(bbox, img_width=1280, img_height=720):
    """Convert BDD100K bbox to YOLO format"""
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return x_center, y_center, width, height

def extract_labels_from_individual_jsons(json_dir, images_dir, output_labels_dir, output_images_dir=None):
    """
    Extract labels from individual JSON files and optionally filter images
    Only processes images that have corresponding JSON files
    """
    json_path = Path(json_dir)
    images_path = Path(images_dir)
    
    # Get all JSON files
    json_files = list(json_path.glob('*.json'))
    print(f"Found {len(json_files)} JSON files in {json_dir}")
    
    # Get stems of JSON files (images we want to keep)
    json_stems = {f.stem for f in json_files}
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(images_path.glob(f'*{ext}'))
    
    print(f"Found {len(image_files)} images in {images_dir}")
    
    # Filter images that have corresponding JSON
    matched_images = [img for img in image_files if img.stem in json_stems]
    print(f"Matched {len(matched_images)} images with JSON files")
    
    # Create output directories
    os.makedirs(output_labels_dir, exist_ok=True)
    if output_images_dir:
        os.makedirs(output_images_dir, exist_ok=True)
    
    # Process each matched image
    processed = 0
    skipped_no_labels = 0
    
    for img_path in tqdm(matched_images, desc="Processing labels"):
        image_stem = img_path.stem
        json_file = json_path / f"{image_stem}.json"
        
        # Read JSON
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract labels
        labels_data = []
        if 'frames' in data and len(data['frames']) > 0:
            # Format with 'frames' array
            frame = data['frames'][0]
            if 'objects' in frame:
                for obj in frame['objects']:
                    category = obj.get('category', '')
                    
                    if category not in CLASS_MAPPING:
                        continue
                    
                    if 'box2d' not in obj:
                        continue
                    
                    bbox = obj['box2d']
                    class_id = CLASS_MAPPING[category]
                    
                    x_center, y_center, width, height = convert_bbox_to_yolo(bbox)
                    labels_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        elif 'labels' in data:
            # Alternative format with direct 'labels' array
            for label in data['labels']:
                category = label.get('category', '')
                
                if category not in CLASS_MAPPING:
                    continue
                
                if 'box2d' not in label:
                    continue
                
                bbox = label['box2d']
                class_id = CLASS_MAPPING[category]
                
                x_center, y_center, width, height = convert_bbox_to_yolo(bbox)
                labels_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Write label file
        label_file = Path(output_labels_dir) / f"{image_stem}.txt"
        with open(label_file, 'w') as f:
            if labels_data:
                f.write('\n'.join(labels_data))
            else:
                skipped_no_labels += 1
        
        # Copy image to output directory if specified
        if output_images_dir:
            output_img = Path(output_images_dir) / img_path.name
            if not output_img.exists():
                shutil.copy2(img_path, output_img)
        
        processed += 1
    
    print(f"\n✅ Processed {processed} images with JSON files")
    print(f"⚠️  {skipped_no_labels} images had no valid labels (empty label files created)")
    print(f"📁 Labels saved to: {output_labels_dir}")
    if output_images_dir:
        print(f"📁 Images copied to: {output_images_dir}")
    
    return processed

def process_all_splits(copy_images=False):
    """
    Process train, val, and test splits
    
    Args:
        copy_images: If True, copy only matched images to a new directory
    """
    
    base_10k = Path("~/Developer/CAVsim/10k").expanduser()
    base_100k = Path("~/Downloads/100k").expanduser()
    
    # Map of splits
    splits = {
        'train': {
            'images': base_10k / 'images' / 'train',
            'labels_out': base_10k / ('labels_filtered' if copy_images else 'labels') / 'train',
            'images_out': base_10k / 'images_filtered' / 'train' if copy_images else None,
            'json_dir': base_100k / 'train'
        },
        'val': {
            'images': base_10k / 'images' / 'val',
            'labels_out': base_10k / ('labels_filtered' if copy_images else 'labels') / 'val',
            'images_out': base_10k / 'images_filtered' / 'val' if copy_images else None,
            'json_dir': base_100k / 'val'
        },
        'test': {
            'images': base_10k / 'images' / 'test',
            'labels_out': base_10k / ('labels_filtered' if copy_images else 'labels') / 'test',
            'images_out': base_10k / 'images_filtered' / 'test' if copy_images else None,
            'json_dir': base_100k / 'test'
        }
    }
    
    total_extracted = 0
    
    for split_name, paths in splits.items():
        print(f"\n{'='*60}")
        print(f"Processing: {split_name.upper()}")
        print(f"{'='*60}\n")
        
        if not paths['images'].exists():
            print(f"⚠️  Images directory not found: {paths['images']}")
            continue
        
        if not paths['json_dir'].exists():
            print(f"⚠️  JSON directory not found: {paths['json_dir']}")
            continue
        
        count = extract_labels_from_individual_jsons(
            json_dir=str(paths['json_dir']),
            images_dir=str(paths['images']),
            output_labels_dir=str(paths['labels_out']),
            output_images_dir=str(paths['images_out']) if paths['images_out'] else None
        )
        
        total_extracted += count
    
    print(f"\n{'='*60}")
    print(f"🎉 DONE! Total images processed: {total_extracted}")
    print(f"{'='*60}")
    
    if copy_images:
        print(f"\n📝 Don't forget to update your data.yaml:")
        print("\nChange:")
        print("  train: images/train")
        print("  val: images/val")
        print("\nTo:")
        print("  train: images_filtered/train")
        print("  val: images_filtered/val")
        print("\nAnd the script will automatically use labels_filtered/")

def create_classes_file(output_file):
    """Create classes.txt file"""
    with open(output_file, 'w') as f:
        for name, idx in sorted(CLASS_MAPPING.items(), key=lambda x: x[1]):
            f.write(f"{name}\n")
    print(f"📝 Created classes file: {output_file}")

if __name__ == "__main__":
    # Check if JSON directory exists
    json_dir = Path("~/Downloads/100k/train").expanduser()
    
    if not json_dir.exists():
        print("⚠️  BDD100K JSON directory not found!")
        print(f"\nExpected directory: {json_dir}")
        print("\nPlease ensure your 100k folder structure looks like:")
        print("  ~/Downloads/100k/")
        print("  ├── train/")
        print("  │   ├── 0a0a0b1a-7c39d841.json")
        print("  │   ├── 0a0a0b1a-27d9fc44.json")
        print("  │   └── ...")
        print("  └── val/")
        print("      └── ...")
    else:
        # Ask user if they want to copy filtered images
        print("Do you want to copy only matched images to a new 'images_filtered' directory?")
        print("This will ensure you only train on images that have labels.")
        print("(y/n): ", end='')
        
        response = input().strip().lower()
        copy_images = response in ['y', 'yes']
        
        # Process all splits
        process_all_splits(copy_images=copy_images)
        
        # Create classes file
        output_dir = Path("~/Developer/CAVsim/10k").expanduser()
        create_classes_file(output_dir / 'classes.txt')
        
        if copy_images:
            print("\n" + "="*60)
            print("IMPORTANT: Update your data.yaml")
            print("="*60)
            print("Change:")
            print("  train: images/train")
            print("  val: images/val")
            print("\nTo:")
            print("  train: images_filtered/train")
            print("  val: images_filtered/val")
            print("\nNote: labels_filtered/ will be used automatically")