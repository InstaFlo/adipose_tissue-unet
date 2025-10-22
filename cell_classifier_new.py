#!/usr/bin/env python3
import os
import math
import time
import argparse
from pathlib import Path

# Quiet some TF logs (optional)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

# Use only tensorflow.keras (avoid mixing with standalone keras)
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense


def get_parser():
    p = argparse.ArgumentParser('Classify adipocyte tiles')
    p.add_argument('--out-dir', type=str, default='output',
                   help='Directory to write CSV outputs.')
    p.add_argument('--weight_dir', type=str, required=True,
                   help='Path to the trained model .h5 (model or weights).')
    p.add_argument('--image-path', type=str, required=True,
                   help='Directory with tiles in subfolders (e.g., blurry/ empty/ tissue/).')
    p.add_argument('--n_cpu', type=int, default=0,
                   help='(Ignored by Keras 3 predict) Kept for CLI compatibility.')
    return p


def build_inception_head(input_shape=(299, 299, 3), num_classes=3):
    """
    Fallback architecture matching a typical InceptionV3 classifier head.
    Only used if the provided .h5 is weights-only.
    """
    base = InceptionV3(include_top=False, weights=None, input_shape=input_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base.input, outputs=out)


def try_load_model_or_weights(h5_path: Path):
    """
    First attempt: load a full serialized Keras model with compile=False (skip legacy optimizer configs).
    Fallback: rebuild a plausible InceptionV3 head and load weights.
    """
    h5_path = str(h5_path)
    try:
        print('Trying to load full model (compile=False)...')
        m = load_model(h5_path, compile=False)
        print('Loaded full model from h5.')
        return m
    except Exception as e1:
        print(f'Full model load failed: {e1}')
        print('Attempting weights-only load with reconstructed InceptionV3 head...')
        m = build_inception_head()
        m.load_weights(h5_path)  # will error if shapes don’t match
        print('Loaded weights into reconstructed model.')
        return m


def main():
    args = get_parser().parse_args()

    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accept paths with spaces; expand ~
    image_root = Path(os.path.expanduser(args.image_path))
    if not image_root.exists():
        raise FileNotFoundError(f'Image path not found: {image_root}')

    # Keras generator over subfolders; we don’t use labels (class_mode=None)
    img_size = (299, 299)
    batch_size = 1  # keep 1 unless you know your memory headroom

    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_gen = datagen.flow_from_directory(
        directory=str(image_root),
        target_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        class_mode=None  # unlabeled inference
    )

    nbr_test_samples = test_gen.samples
    if nbr_test_samples == 0:
        raise RuntimeError(
            f'No images found under {image_root}/*/*. '
            f'Ensure your tiles are in subfolders (e.g., blurry/, empty/, tissue/).'
        )

    print(f'Number of total patches to classify: {nbr_test_samples}')
    print('Loading model and weights from training process ...')
    model = try_load_model_or_weights(Path(args.weight_dir))

    # Predict
    print('Begin to predict for testing data ...')
    steps = math.ceil(nbr_test_samples / batch_size)

    # Keras 3 removed use_multiprocessing/workers for predict; keep it simple
    preds = model.predict(test_gen, steps=steps, verbose=1)

    # Trim to exact N if the last batch was padded
    preds = preds[:nbr_test_samples]

    # Sanity: predictions should be (N, 3)
    if preds.ndim != 2 or preds.shape[1] != 3:
        raise ValueError(
            f'Expected predictions with shape (N, 3) for classes '
            f'[empty, not_adipocyte, adipocyte], got {preds.shape}'
        )

    # OPTIONAL: If you want to reorder columns to a specific header,
    # you can compute a mapping here. For now we assume model outputs
    # are already in [empty, not_adipocyte, adipocyte] order.
    header = ['image', 'empty', 'not_adipocyte', 'adipocyte']

    # Write outputs
    print('Begin to write cell probabilities ..')
    # Save the seed used (for reproducibility of shuffling/augments if any)
    with open(out_dir / 'seed.csv', 'w') as fseed:
        fseed.write(f"{np.random.randint(0, 1_000_001)}\n")

    out_csv = out_dir / 'out.inception.adipocytes.csv'
    with open(out_csv, 'w') as f:
        f.write(','.join(header) + '\n')
        # test_gen.filenames lists relative paths like "tissue/slide_r0_c0.png"
        for i, relpath in enumerate(test_gen.filenames[:nbr_test_samples]):
            probs = preds[i]
            row = ','.join([Path(relpath).name] + [f'{p:.6f}' for p in probs])
            f.write(row + '\n')

    print(f'Done. Wrote: {out_csv}')


if __name__ == '__main__':
    main()
