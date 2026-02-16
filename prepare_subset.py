"""
Dataset Subset Preparation
Creates a subset of 100 images per class from the source dataset.
"""
import os
import shutil
import random
from pathlib import Path

def prepare_subset(source_dir, target_dir, num_images=100, seed=42):
    """
    Create a subset of the dataset with limited images per class.
    
    Args:
        source_dir: Path to source dataset (must have training_set/cat and training_set/dog)
        target_dir: Path to create subset dataset
        num_images: Number of images per class to sample
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Define paths
    source_train = os.path.join(source_dir, 'training_set')
    source_test = os.path.join(source_dir, 'test_set')
    
    target_train = os.path.join(target_dir, 'train')
    target_val = os.path.join(target_dir, 'validation')
    
    # Create target directories
    for class_name in ['cat', 'dog']:
        os.makedirs(os.path.join(target_train, class_name), exist_ok=True)
        os.makedirs(os.path.join(target_val, class_name), exist_ok=True)
    
    print(f"Preparing dataset subset: {num_images} images per class")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print("="*60)
    
    # Process each class
    for class_name in ['cat', 'dog']:
        source_class_dir = os.path.join(source_train, class_name)
        
        # Get all images
        all_images = [f for f in os.listdir(source_class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\n{class_name.upper()}: Found {len(all_images)} images")
        
        # Randomly sample
        if len(all_images) < num_images:
            print(f"  Warning: Only {len(all_images)} images available, using all")
            selected = all_images
        else:
            selected = random.sample(all_images, num_images)
        
        # Split 80/20 for train/validation
        split_idx = int(len(selected) * 0.8)
        train_images = selected[:split_idx]
        val_images = selected[split_idx:]
        
        print(f"  Train: {len(train_images)} images")
        print(f"  Validation: {len(val_images)} images")
        
        # Copy training images
        for img in train_images:
            src = os.path.join(source_class_dir, img)
            dst = os.path.join(target_train, class_name, img)
            shutil.copy2(src, dst)
        
        # Copy validation images
        for img in val_images:
            src = os.path.join(source_class_dir, img)
            dst = os.path.join(target_val, class_name, img)
            shutil.copy2(src, dst)
    
    print("\n" + "="*60)
    print("Dataset subset prepared successfully!")
    print(f"Location: {target_dir}")
    
    return target_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset subset')
    parser.add_argument('--source', type=str, required=True, help='Source dataset directory')
    parser.add_argument('--target', type=str, default='data_subset', help='Target directory')
    parser.add_argument('--num_images', type=int, default=100, help='Images per class')
    
    args = parser.parse_args()
    
    prepare_subset(args.source, args.target, args.num_images)
