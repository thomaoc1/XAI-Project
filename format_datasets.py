import os
import shutil
import random

def reorganize_for_imagefolder(root_dir):
    classes = ['cat', 'dog']
    for cls in classes:
        class_dir = os.path.join(root_dir, cls)
        os.makedirs(class_dir, exist_ok=True)

    for filename in os.listdir(root_dir):
        if filename.endswith('.jpg'):
            label = 'cat' if filename.startswith('cat') else 'dog'
            src = os.path.join(root_dir, filename)
            dst = os.path.join(root_dir, label, filename)
            shutil.move(src, dst)

def create_train_val_split(data_dir, val_dir, n_val_per_class=2500, seed=42):
    random.seed(seed)
    classes = ['cat', 'dog']

    os.makedirs(val_dir, exist_ok=True)

    for cls in classes:
        src_cls_dir = os.path.join(data_dir, cls)
        val_cls_dir = os.path.join(val_dir, cls)
        os.makedirs(val_cls_dir, exist_ok=True)

        images = [f for f in os.listdir(src_cls_dir) if f.endswith('.jpg')]
        images.sort()  # ensure reproducibility before shuffle
        random.shuffle(images)

        val_images = images[:n_val_per_class]

        for img in val_images:
            src_path = os.path.join(src_cls_dir, img)
            dst_path = os.path.join(val_cls_dir, img)
            shutil.move(src_path, dst_path)

    print(f"Validation set created: {n_val_per_class} images per class.")
    print("Remaining images stay in train set.")

def main():
    data_dir = 'dataset/dogs-vs-cats/train'
    val_dir = 'dataset/dogs-vs-cats/validation'

    reorganize_for_imagefolder(data_dir)
    create_train_val_split(data_dir, val_dir, n_val_per_class=2500, seed=42)

    if os.path.exists('dataset/deepfake-dataset'):
        os.rename('dataset/deepfake-dataset', 'dataset/deepfake')

if __name__ == '__main__':
    main()
