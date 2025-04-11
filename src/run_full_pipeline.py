import argparse

import src.classification.train as train_classifier
import src.datagen.generate_dataset as generate_data
import src.anomalydetection.train as train_vae
import src.anomalydetection.eval_vae as eval_vae

from src.config import DatasetConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Full pipeline")
    parser.add_argument('dataset', type=str, choices=['deepfake', 'dogs-vs-cats'])
    parser.add_argument('--target_class', type=str, default=None, choices=['fake', 'real', 'cat', 'dog'])
    parser.add_argument('--classifier_batch_size', type=int, default=64)
    parser.add_argument('--vae_batch_size', type=int, default=64)
    parser.add_argument('--classifier_epochs', type=int, default=5)
    parser.add_argument('--vae_epochs', type=int, default=50)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = DatasetConfig(args.dataset, target_class=args.target_class)

    print("==== 1. Training Classifier ====")
    train_classifier.main(
        cfg=config,
        num_epochs=args.classifier_epochs,
        batch_size=args.classifier_batch_size,
    )

    print("==== 2. Generating Clean Heatmaps ====")
    generate_data.main(
        cfg=config,
        batch_size=64,
        target_class_name=args.target_class
    )

    print("==== 3. Training VAE ====")
    train_vae.main(
        cfg=config,
        n_epochs=args.vae_epochs,
        batch_size=args.vae_batch_size,
    )

    print("==== 4. Eval VAE ====")
    eval_vae.main(
        cfg=config,
    )