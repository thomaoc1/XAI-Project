import os
import shutil
import random

def is_valid_imagefolder(root_dir, classes):
    return all(
        os.path.isdir(os.path.join(root_dir, cls)) and
        any(fname.endswith('.jpg') for fname in os.listdir(os.path.join(root_dir, cls)))
        for cls in classes
    )

def reorganize_for_imagefolder(root_dir):
    classes = ['cat', 'dog']

    if is_valid_imagefolder(root_dir, classes):
        print(f"{root_dir} already organized in ImageFolder format.")
        return True  # already organized

    print(f"Reorganizing {root_dir} into ImageFolder format...")
    for cls in classes:
        class_dir = os.path.join(root_dir, cls)
        os.makedirs(class_dir, exist_ok=True)

    for filename in os.listdir(root_dir):
        if filename.endswith('.jpg'):
            label = 'cat' if filename.startswith('cat') else 'dog'
            src = os.path.join(root_dir, filename)
            dst = os.path.join(root_dir, label, filename)
            shutil.move(src, dst)
    return False  # reorganization done

def create_train_val_split(data_dir, val_dir, n_val_per_class=2500, seed=42):
    random.seed(seed)
    classes = ['cat', 'dog']

    if is_valid_imagefolder(val_dir, classes):
        print(f"{val_dir} already contains validation data in ImageFolder format.")
        return True  # validation already exists

    print(f"Creating validation split in {val_dir}...")
    os.makedirs(val_dir, exist_ok=True)

    for cls in classes:
        src_cls_dir = os.path.join(data_dir, cls)
        val_cls_dir = os.path.join(val_dir, cls)
        os.makedirs(val_cls_dir, exist_ok=True)

        images = [f for f in os.listdir(src_cls_dir) if f.endswith('.jpg')]
        images.sort()
        random.shuffle(images)

        if len(images) <= n_val_per_class:
            print(f"Not enough images in {cls} to create validation set. Skipping...")
            continue

        val_images = images[:n_val_per_class]

        for img in val_images:
            src_path = os.path.join(src_cls_dir, img)
            dst_path = os.path.join(val_cls_dir, img)
            shutil.move(src_path, dst_path)

    print(f"Validation set created: {n_val_per_class} images per class (if available).")
    return False  # split done

def main():
    data_dir = 'dataset/dogs-vs-cats/train'
    val_dir = 'dataset/dogs-vs-cats/validation'
    classes = ['cat', 'dog']

    already_train_ok = reorganize_for_imagefolder(data_dir)
    already_val_ok = is_valid_imagefolder(val_dir, classes)

    if already_train_ok and already_val_ok:
        print("Dataset is already properly formatted. No action needed.")
    else:
        create_train_val_split(data_dir, val_dir, n_val_per_class=2500, seed=42)

    if os.path.exists('dataset/deepfake-dataset'):
        os.rename('dataset/deepfake-dataset', 'dataset/deepfake')

if __name__ == '__main__':
    main()
