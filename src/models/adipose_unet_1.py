"""
Optimized U-Net Training for Sybr Gold + Eosin Adipose Segmentation
TensorFlow 2.13 / Python 3.10

Optimized for REGION segmentation (not thin boundaries)
Dataset: 469 train / 136 val images (40% negative samples)
Target: ~80% Dice coefficient

Key Changes:
- Region-optimized loss function (prioritizes Dice over BCE)
- Moderate augmentation for 469 images
- Proper checkpoint management (best model, not last)
- Backward compatibility with old softmax model
- Fixed learning rate bug
- Cleaned duplicate augmentation code
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import json
import math

import numpy as np
np.random.seed(865)

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout,
    Concatenate, Lambda, Reshape, Add,
)
from tensorflow.keras.optimizers.experimental import AdamW 
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
)
from tensorflow.keras.mixed_precision import set_global_policy

import cv2
import tifffile as tiff

# Local utils
sys.path.append('.')
from src.utils.runtime import funcname
from src.utils.model import dice_coef
from src.utils.data import augment_pair_moderate  # Use moderate augmentation


# ---- Loss Functions (Optimized for Region Segmentation) ------------------

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Focal loss for handling hard examples"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    eps = tf.constant(1e-6, tf.float32)
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
    
    ce = -y_true*tf.math.log(y_pred) - (1-y_true)*tf.math.log(1-y_pred)
    p_t = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
    focal_weight = tf.pow(1.0 - p_t, gamma)
    alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
    
    return tf.reduce_mean(alpha_t * focal_weight * ce)


def dice_loss(y_true, y_pred):
    """Dice loss - measures region overlap directly"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    smooth = tf.constant(1.0, tf.float32)
    
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    
    return 1.0 - (2.0*inter + smooth) / (denom + smooth)


def combined_loss_region(y_true, y_pred):
    """
    Loss optimized for REGION segmentation (not thin boundaries).
    
    Rationale:
    - Dice (1.2x): Primary metric for region overlap
    - BCE (0.3x): Reduced emphasis (boundaries less critical for blobs)
    - Focal (0.3x): Minimal (larger regions = fewer hard negatives)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    focal = focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0)
    
    return 0.3*bce + 1.2*dice + 0.3*focal

def combined_loss_stable(y_true, y_pred):
    """More stable loss for initial training"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    focal = focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0)
    
    # More balanced weights for stability
    return 0.5*bce + 1.0*dice + 0.5*focal
# ---- Efficient Data Pipeline ----------------------------------------------

class TileDataset:
    """Efficient data loading with caching and prefetching"""
    
    def __init__(self, images_dir: Path, masks_dir: Path, batch_size: int, 
                 augment: bool = True, cache_size: int = 100):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.batch_size = batch_size
        self.augment = augment
        self.cache_size = cache_size
        
        # Load file lists
        self.image_files = sorted(self.images_dir.glob("*.jpg"))
        self.mask_files = {p.stem: p for p in self.masks_dir.glob("*.tif")}
        
        # Filter to paired files only
        self.pairs = []
        for img_path in self.image_files:
            if img_path.stem in self.mask_files:
                self.pairs.append((img_path, self.mask_files[img_path.stem]))
        
        print(f"Found {len(self.pairs)} paired tiles in {images_dir.name}")
        
        # Cache for frequently accessed tiles
        self.cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        
        # Calculate normalization stats
        self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        """Compute mean and std on a sample of images"""
        sample_size = min(50, len(self.pairs))
        sample_pixels = []
        
        for img_path, _ in self.pairs[:sample_size]:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            sample_pixels.append(img.flatten())
        
        all_pixels = np.concatenate(sample_pixels)
        self.mean = np.mean(all_pixels)
        self.std = np.std(all_pixels)
        print(f"Dataset stats - Mean: {self.mean:.2f}, Std: {self.std:.2f}")
    
    def load_pair(self, img_path: Path, mask_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load and cache a single image-mask pair"""
        cache_key = img_path.stem
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Load image as grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        
        # Load mask
        mask = tiff.imread(str(mask_path)).astype(np.float32)
        if mask.ndim == 3:
            mask = mask.squeeze()
        
        # Cache if space available
        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = (img.copy(), mask.copy())
        
        return img, mask
    
    def __len__(self):
        return len(self.pairs)
    
    def generator(self):
        """Python generator for tf.data.Dataset"""
        rng = np.random.RandomState()
        indices = np.arange(len(self.pairs))
        
        while True:
            rng.shuffle(indices)
            
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                images = []
                masks = []
                
                for idx in batch_indices:
                    img_path, mask_path = self.pairs[idx]
                    img, mask = self.load_pair(img_path, mask_path)
                    
                    # Augmentation using moderate pipeline from data.py
                    if self.augment:
                        img, mask = augment_pair_moderate(img, mask, rng)
                    
                    # Normalize using percentile clipping
                    p1, p99 = np.percentile(img, (1, 99))
                    scale = max(p99 - p1, 1e-3)
                    img = np.clip((img - p1) / scale, 0, 1).astype(np.float32)
                    
                    images.append(img)
                    masks.append(mask)
                
                # Pad last batch if needed
                while len(images) < self.batch_size:
                    images.append(images[-1])
                    masks.append(masks[-1])
                
                yield (
                    np.array(images, dtype=np.float32),
                    np.array(masks, dtype=np.float32)
                )
    
    def create_dataset(self):
        """Create tf.data.Dataset with prefetching"""
        output_signature = (
            tf.TensorSpec(shape=(self.batch_size, 1024, 1024), dtype=tf.float32),
            tf.TensorSpec(shape=(self.batch_size, 1024, 1024), dtype=tf.float32)
        )
        
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=output_signature
        )
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


# ---- Model with Backward Compatibility ------------------------------------

class AdiposeUNet:
    def __init__(self, checkpoint_name: str, freeze_encoder: bool = True):
        self.checkpoint_name = checkpoint_name
        self.freeze_encoder = freeze_encoder
        self.net: Model | None = None
        self.checkpoint_dir = Path(f"checkpoints/{checkpoint_name}_1024_finetune")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def build_model(self, init_nb: int = 44, dropout_rate: float = 0.3):
        """Build U-Net with sigmoid output for binary segmentation"""
        K.set_image_data_format('channels_last')
        
        inputs = Input(shape=(1024, 1024), dtype='float32')
        x = Reshape((1024, 1024, 1))(inputs)
        
        # Encoder
        down1 = Conv2D(init_nb, 3, activation='relu', padding='same', name='down1_conv1')(x)
        down1 = Conv2D(init_nb, 3, activation='relu', padding='same', name='down1_conv2')(down1)
        down1pool = MaxPooling2D((2, 2), strides=(2, 2), name='down1_pool')(down1)
        
        down2 = Conv2D(init_nb * 2, 3, activation='relu', padding='same', name='down2_conv1')(down1pool)
        down2 = Conv2D(init_nb * 2, 3, activation='relu', padding='same', name='down2_conv2')(down2)
        down2pool = MaxPooling2D((2, 2), strides=(2, 2), name='down2_pool')(down2)
        
        down3 = Conv2D(init_nb * 4, 3, activation='relu', padding='same', name='down3_conv1')(down2pool)
        down3 = Conv2D(init_nb * 4, 3, activation='relu', padding='same', name='down3_conv2')(down3)
        down3pool = MaxPooling2D((2, 2), strides=(2, 2), name='down3_pool')(down3)
        
        # Bottleneck with dilated convolutions
        dilate1 = Conv2D(init_nb * 8, 3, activation='relu', padding='same', dilation_rate=1, name='dilate1')(down3pool)
        dilate1 = Dropout(dropout_rate, name='dropout_dilate1')(dilate1)
        dilate2 = Conv2D(init_nb * 8, 3, activation='relu', padding='same', dilation_rate=2, name='dilate2')(dilate1)
        dilate3 = Conv2D(init_nb * 8, 3, activation='relu', padding='same', dilation_rate=4, name='dilate3')(dilate2)
        dilate4 = Conv2D(init_nb * 8, 3, activation='relu', padding='same', dilation_rate=8, name='dilate4')(dilate3)
        dilate5 = Conv2D(init_nb * 8, 3, activation='relu', padding='same', dilation_rate=16, name='dilate5')(dilate4)
        dilate6 = Conv2D(init_nb * 8, 3, activation='relu', padding='same', dilation_rate=32, name='dilate6')(dilate5)
        dilate_all_added = Add(name='dilate_add')([dilate1, dilate2, dilate3, dilate4, dilate5, dilate6])
        
        # Decoder
        up3 = UpSampling2D((2, 2), name='up3_upsample')(dilate_all_added)
        up3 = Conv2D(init_nb * 4, 3, activation='relu', padding='same', name='up3_conv1')(up3)
        up3 = Concatenate(axis=-1, name='up3_concat')([down3, up3])
        up3 = Conv2D(init_nb * 4, 3, activation='relu', padding='same', name='up3_conv2')(up3)
        up3 = Conv2D(init_nb * 4, 3, activation='relu', padding='same', name='up3_conv3')(up3)
        up3 = Dropout(dropout_rate, name='dropout_up3')(up3)
        
        up2 = UpSampling2D((2, 2), name='up2_upsample')(up3)
        up2 = Conv2D(init_nb * 2, 3, activation='relu', padding='same', name='up2_conv1')(up2)
        up2 = Concatenate(axis=-1, name='up2_concat')([down2, up2])
        up2 = Conv2D(init_nb * 2, 3, activation='relu', padding='same', name='up2_conv2')(up2)
        up2 = Conv2D(init_nb * 2, 3, activation='relu', padding='same', name='up2_conv3')(up2)
        up2 = Dropout(dropout_rate, name='dropout_up2')(up2)
        
        up1 = UpSampling2D((2, 2), name='up1_upsample')(up2)
        up1 = Conv2D(init_nb, 3, activation='relu', padding='same', name='up1_conv1')(up1)
        up1 = Concatenate(axis=-1, name='up1_concat')([down1, up1])
        up1 = Conv2D(init_nb, 3, activation='relu', padding='same', name='up1_conv2')(up1)
        up1 = Conv2D(init_nb, 3, activation='relu', padding='same', name='up1_conv3')(up1)
        up1 = Dropout(dropout_rate, name='dropout_up1')(up1)
        
        # Sigmoid output for binary segmentation
        x = Conv2D(1, 1, activation='sigmoid', dtype='float32', name='out_sigmoid')(up1)
        x = Lambda(lambda z: K.squeeze(z, axis=-1), name='squeeze')(x)
        
        self.net = Model(inputs=inputs, outputs=x)
        
        # Freeze encoder if requested
        if self.freeze_encoder:
            self.freeze_encoder_layers()
        
        return self.net
    
    def freeze_encoder_layers(self):
        """Freeze encoder for transfer learning"""
        frozen_layers = [
            'down1_conv1', 'down1_conv2', 'down1_pool',
            'down2_conv1', 'down2_conv2', 'down2_pool',
            'down3_conv1', 'down3_conv2', 'down3_pool',
        ]
        
        for layer in self.net.layers:
            if layer.name in frozen_layers:
                layer.trainable = False
        
        print(f"Frozen {len(frozen_layers)} encoder layers for transfer learning")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning"""
        for layer in self.net.layers:
            layer.trainable = True
        print("Unfrozen all layers for fine-tuning")
    
    def compile_model(self, lr: float = 1e-4, weight_decay: float = 1e-5):
        """Compile with optimizer and loss (FIXED: now uses lr parameter)"""
        optimizer = AdamW(
            learning_rate=lr,  # FIXED: was hardcoded to 1e-4
            weight_decay=weight_decay,
            epsilon=1e-7,
            clipnorm=1.0
        )
        self.net.compile(
            optimizer=optimizer,
            loss=combined_loss_stable,  # Region-optimized loss
            metrics=[dice_coef, 'binary_accuracy']
        )
    
    def load_pretrained_weights(self, h5_path: str):
        """
        Load weights from old model (with softmax output) into new model (sigmoid).
        Skips mismatched final layer automatically.
        """
        import h5py
        from tensorflow.python.keras.saving import hdf5_format
        
        print(f"\nLoading pretrained weights from: {h5_path}")
        print("Note: Final layer mismatch (softmax→sigmoid) will be skipped automatically")
        
        with h5py.File(h5_path, 'r') as f:
            group = f['model_weights'] if 'model_weights' in f else f
            
            # Load by name, skip mismatches (the final layer)
            try:
                hdf5_format.load_weights_from_hdf5_group_by_name(
                    group, 
                    self.net.layers,
                    skip_mismatch=True
                )
                print("✓ Loaded pretrained weights (skipped final layer mismatch)")
            except Exception as e:
                print(f"Warning: Weight loading encountered issues: {e}")
                print("Attempting partial load...")
                hdf5_format.load_weights_from_hdf5_group_by_name(group, self.net.layers)
                print("✓ Partial weight loading complete")
    
    def save_weights_modern(self, suffix: str = "finetuned"):
        """Save weights in modern TF2 format"""
        weights_path = self.checkpoint_dir / f"weights_{suffix}.weights.h5"
        self.net.save_weights(str(weights_path))
        print(f"✓ Saved weights to {weights_path}")


# ---- Training with Proper Checkpoint Management ---------------------------

def train_model(
    data_root: Path,
    pretrained_weights: str,
    batch_size: int = 2,
    epochs_phase1: int = 50,
    epochs_phase2: int = 100,
    use_mixed_precision: bool = False,  # Default off for stability
):
    """
    Two-phase training with proper checkpoint management:
    Phase 1: Frozen encoder, train decoder
    Phase 2: Unfreeze all, fine-tune (loads BEST from phase 1)
    """
    
    # Mixed precision (off by default for stability)
    if use_mixed_precision:
        set_global_policy('mixed_float16')
        print("✓ Mixed precision enabled (float16)")
    else:
        print("✓ Using float32 precision")
    
    # Setup data
    data_root = Path(data_root)
    train_images = data_root / "dataset" / "train" / "images"
    train_masks = data_root / "dataset" / "train" / "masks"
    val_images = data_root / "dataset" / "val" / "images"
    val_masks = data_root / "dataset" / "val" / "masks"
    
    train_dataset = TileDataset(train_images, train_masks, batch_size, augment=True)
    val_dataset = TileDataset(val_images, val_masks, batch_size, augment=False)
    
    train_ds = train_dataset.create_dataset()
    val_ds = val_dataset.create_dataset()
    
    steps_per_epoch = max(1, math.ceil(len(train_dataset)/batch_size))
    validation_steps = max(1, math.ceil(len(val_dataset)/batch_size))
    
    print(f"\n{'='*60}")
    print(f"Dataset Configuration:")
    print(f"{'='*60}")
    print(f"  Training:   {len(train_dataset)} tiles ({steps_per_epoch} steps/epoch)")
    print(f"  Validation: {len(val_dataset)} tiles ({validation_steps} steps/epoch)")
    print(f"  Augmentation: Moderate (optimized for 469 images)")
    print(f"{'='*60}\n")
    
    # Build model
    model = AdiposeUNet("adipose_sybreosin", freeze_encoder=True)
    model.build_model(init_nb=44, dropout_rate=0.3)
    model.compile_model(lr=1e-4, weight_decay=1e-5)
    
    # Load pretrained weights (backward compatible)
    if pretrained_weights and Path(pretrained_weights).exists():
        model.load_pretrained_weights(pretrained_weights)
    else:
        print("WARNING: No pretrained weights found, training from scratch")
    
    model.net.summary()
    
    # ==== PHASE 1: Frozen Encoder ====
    print(f"\n{'='*60}")
    print(f"PHASE 1: Training decoder only ({epochs_phase1} epochs)")
    print(f"Learning rate: 1e-4")
    print(f"{'='*60}\n")
    
    callbacks_phase1 = [
        ModelCheckpoint(
            filepath=str(model.checkpoint_dir / "phase1_best.weights.h5"),
            monitor='val_dice_coef',
            mode='max',  # Higher dice is better
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_dice_coef',
            mode ='max',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_dice_coef',
            mode='max',
            patience=25,
            restore_best_weights=False,  # We'll load manually
            verbose=1
        ),
        CSVLogger(str(model.checkpoint_dir / "phase1_training.log")),
    ]
    
    model.net.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs_phase1,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks_phase1,
        verbose=1
    )
    
    # Save final phase 1 weights for reference
    model.save_weights_modern("phase1_final")
    
    # ==== PHASE 2: Full Fine-tuning ====
    print(f"\n{'='*60}")
    print(f"PHASE 2: Fine-tuning all layers ({epochs_phase2} epochs)")
    print(f"Learning rate: 1e-5 (10x lower)")
    print(f"{'='*60}\n")
    
    # CRITICAL: Load best Phase 1 model (not last epoch)
    best_phase1_path = model.checkpoint_dir / "phase1_best.weights.h5"
    if best_phase1_path.exists():
        print(f"Loading BEST Phase 1 model from: {best_phase1_path}")
        model.net.load_weights(str(best_phase1_path))
        print("✓ Loaded best Phase 1 weights\n")
    else:
        print("WARNING: Best Phase 1 weights not found, using last epoch weights\n")
    
    # Unfreeze and recompile with lower learning rate
    model.unfreeze_encoder()
    model.compile_model(lr=1e-5, weight_decay=1e-5)  # FIXED: 10x lower LR
    
    callbacks_phase2 = [
        ModelCheckpoint(
            filepath=str(model.checkpoint_dir / "phase2_best.weights.h5"),
            monitor='val_dice_coef',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_dice_coef',
            mode='max',
            factor=0.5,
            patience=15,
            min_lr=1e-8,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_dice_coef',
            mode='max',
            patience=30,
            restore_best_weights=False,  # We saved best already
            verbose=1
        ),
        CSVLogger(str(model.checkpoint_dir / "phase2_training.log"))
    ]
    
    model.net.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs_phase2,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks_phase2,
        verbose=1
    )
    
    # Save final weights
    model.save_weights_modern("phase2_final")
    
    # Load and save best overall model
    best_phase2_path = model.checkpoint_dir / "phase2_best.weights.h5"
    if best_phase2_path.exists():
        model.net.load_weights(str(best_phase2_path))
        model.save_weights_modern("best_overall")
    
    print(f"\n{'='*60}")
    print("✓ Training Complete!")
    print(f"{'='*60}")
    print(f"Checkpoint directory: {model.checkpoint_dir}")
    print(f"\nBest models saved:")
    print(f"  - phase1_best.weights.h5  (best Phase 1 model)")
    print(f"  - phase2_best.weights.h5  (best Phase 2 model)")
    print(f"  - weights_best_overall.weights.h5  (final best)")
    print(f"{'='*60}\n")
    
    return model


# ---- CLI ------------------------------------------------------------------

def main():
    # GPU memory growth
    for g in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    
    tf.random.set_seed(865)
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(
        description="Train U-Net for adipose region segmentation (Sybr Gold + Eosin)"
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='~/Data_for_ML/Meat_Luci_Tulane',
        help='Root directory containing dataset/ folder'
    )
    parser.add_argument(
        '--pretrained-weights',
        type=str,
        default='checkpoints/unet_1024_dilation/weights_loss_val.weights.h5',
        help='Path to pretrained weights (old softmax model compatible)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Batch size (2-4 recommended for 1024x1024 tiles)'
    )
    parser.add_argument(
        '--epochs-phase1',
        type=int,
        default=50,
        help='Epochs for phase 1 (frozen encoder)'
    )
    parser.add_argument(
        '--epochs-phase2',
        type=int,
        default=100,
        help='Epochs for phase 2 (full fine-tuning)'
    )
    parser.add_argument(
        '--mixed-precision',
        action='store_true',
        help='Enable mixed precision training (faster but may be unstable)'
    )
    
    args = parser.parse_args()
    
    # Expand paths
    data_root = Path(args.data_root).expanduser()
    
    # Validate data exists
    if not (data_root / "dataset" / "train").exists():
        raise FileNotFoundError(
            f"Training data not found at {data_root}/dataset/train. "
            "Run the dataset builder first!"
        )
    
    # Train model
    model = train_model(
        data_root=data_root,
        pretrained_weights=args.pretrained_weights,
        batch_size=args.batch_size,
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2,
        use_mixed_precision=args.mixed_precision
    )
    
    print("\n✓ Training complete! Use the best model for inference:")
    print(f"  Best model: checkpoints/adipose_sybreosin_1024_finetune/weights_best_overall.weights.h5")


if __name__ == "__main__":
    main()