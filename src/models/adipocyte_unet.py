"""
U-Net (TF2/Keras) compatible with Python 3.10 / TensorFlow 2.13
- Removes legacy keras imports and merge API
- Uses tensorflow.keras throughout
- Replaces fit_generator -> fit
- Uses Concatenate / Add layers in place of merge
- Replaces K.set_image_dim_ordering -> K.set_image_data_format
"""

import os
import sys
import argparse
import logging
import random
from os import path, makedirs

import numpy as np
np.random.seed(865)

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout,
    Concatenate, Conv2DTranspose, Lambda, Reshape, BatchNormalization,
    Add, Multiply, Maximum, Average,
)
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
)
from tensorflow.keras.losses import binary_crossentropy

import tifffile as tiff

# Local utils
sys.path.append('.')
from src.utils.runtime import funcname, gpu_selection  # noqa: E402
from src.utils.model import (  # noqa: E402
    dice_coef, dice_coef_loss, KerasHistoryPlotCallback, KerasSimpleLoggerCallback,
    jaccard_coef, jaccard_coef_int, weighted_bce_dice_loss, weighted_dice_loss,
    weighted_bce_loss, weighted_dice_coeff
)
from src.utils.data import random_transforms  # noqa: E402
from src.utils.isbi_utils import isbi_get_data_montage  # noqa: E402
from src.utils.clr_callback import CyclicLR  # noqa: E402


# ---- Losses -----------------------------------------------------------------

def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2.0 * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1.0 - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


# ---- Model ------------------------------------------------------------------

class UNet:
    def __init__(self, checkpoint_name: str):
        self.config = {
            'data_path': 'data',
            'input_shape': (1024, 1024),
            'output_shape': (1024, 1024),
            'transform_train': True,
            'batch_size': 2,
            'nb_epoch': 400,
        }

        self.checkpoint_name = checkpoint_name
        self.net: Model | None = None
        self.imgs_trn = None
        self.msks_trn = None
        self.imgs_val = None
        self.msks_val = None
        self.imgs_trn2 = None
        self.msks_trn2 = None
        self.imgs_val2 = None
        self.msks_val2 = None
        self.imgs_trn3 = None
        self.msks_trn3 = None
        self.imgs_val3 = None
        self.msks_val3 = None
        self.imgs_trn4 = None
        self.msks_trn4 = None
        self.imgs_val4 = None
        self.msks_val4 = None

    @property
    def checkpoint_path(self) -> str:
        return f"checkpoints/{self.checkpoint_name}_{self.config['input_shape'][0]}_dilation"

    # ---- Data I/O ------------------------------------------------------------
    def load_data(self):
        self.imgs_trn = np.load('montage_imgs/gtex_montage_img2_trn.npy')
        self.msks_trn = np.load('montage_imgs/gtex_montage_msk2_trn.npy')

        self.imgs_val = np.load('montage_imgs/gtex_montage_img2_val.npy')
        self.msks_val = np.load('montage_imgs/gtex_montage_msk2_val.npy')

        self.imgs_trn2 = np.load('montage_imgs/julius_montage_img2_trn.npy')
        self.msks_trn2 = np.load('montage_imgs/julius_montage_msk2_trn.npy')

        self.imgs_val2 = np.load('montage_imgs/julius_montage_img2_val.npy')
        self.msks_val2 = np.load('montage_imgs/julius_montage_msk2_val.npy')

        self.imgs_trn3 = np.load('montage_imgs/NDOG_montage_img2_trn.npy')
        self.msks_trn3 = np.load('montage_imgs/NDOG_montage_msk2_trn.npy')

        self.imgs_val3 = np.load('montage_imgs/NDOG_montage_img2_val.npy')
        self.msks_val3 = np.load('montage_imgs/NDOG_montage_msk2_val.npy')

        self.imgs_trn4 = np.load('montage_imgs/exeter_montage_img2_trn.npy')
        self.msks_trn4 = np.load('montage_imgs/exeter_montage_msk2_trn.npy')

        self.imgs_val4 = np.load('montage_imgs/exeter_montage_img2_val.npy')
        self.msks_val4 = np.load('montage_imgs/exeter_montage_msk2_val.npy')

    # ---- Build / Compile -----------------------------------------------------
    def compile(self, init_nb: int = 44, lr: float = 1e-4, loss=bce_dice_loss):
        # TF2 data format
        K.set_image_data_format('channels_last')

        # Input: start as HW then expand channel dim to (H, W, 1)
        inputs = Input(shape=self.config['input_shape'], dtype='float32')
        x = Reshape(self.config['input_shape'] + (1,))(inputs)

        # Down path
        down1 = Conv2D(init_nb, 3, activation='relu', padding='same')(x)
        down1 = Conv2D(init_nb, 3, activation='relu', padding='same')(down1)
        down1pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

        down2 = Conv2D(init_nb * 2, 3, activation='relu', padding='same')(down1pool)
        down2 = Conv2D(init_nb * 2, 3, activation='relu', padding='same')(down2)
        down2pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

        down3 = Conv2D(init_nb * 4, 3, activation='relu', padding='same')(down2pool)
        down3 = Conv2D(init_nb * 4, 3, activation='relu', padding='same')(down3)
        down3pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)

        # Bottleneck with stacked dilated convolutions
        dilate1 = Conv2D(init_nb * 8, 3, activation='relu', padding='same', dilation_rate=1)(down3pool)
        dilate2 = Conv2D(init_nb * 8, 3, activation='relu', padding='same', dilation_rate=2)(dilate1)
        dilate3 = Conv2D(init_nb * 8, 3, activation='relu', padding='same', dilation_rate=4)(dilate2)
        dilate4 = Conv2D(init_nb * 8, 3, activation='relu', padding='same', dilation_rate=8)(dilate3)
        dilate5 = Conv2D(init_nb * 8, 3, activation='relu', padding='same', dilation_rate=16)(dilate4)
        dilate6 = Conv2D(init_nb * 8, 3, activation='relu', padding='same', dilation_rate=32)(dilate5)
        dilate_all_added = Add()([dilate1, dilate2, dilate3, dilate4, dilate5, dilate6])

        # Up path
        up3 = UpSampling2D((2, 2))(dilate_all_added)
        up3 = Conv2D(init_nb * 4, 3, activation='relu', padding='same')(up3)
        up3 = Concatenate(axis=-1)([down3, up3])
        up3 = Conv2D(init_nb * 4, 3, activation='relu', padding='same')(up3)
        up3 = Conv2D(init_nb * 4, 3, activation='relu', padding='same')(up3)

        up2 = UpSampling2D((2, 2))(up3)
        up2 = Conv2D(init_nb * 2, 3, activation='relu', padding='same')(up2)
        up2 = Concatenate(axis=-1)([down2, up2])
        up2 = Conv2D(init_nb * 2, 3, activation='relu', padding='same')(up2)
        up2 = Conv2D(init_nb * 2, 3, activation='relu', padding='same')(up2)

        up1 = UpSampling2D((2, 2))(up2)
        up1 = Conv2D(init_nb, 3, activation='relu', padding='same')(up1)
        up1 = Concatenate(axis=-1)([down1, up1])
        up1 = Conv2D(init_nb, 3, activation='relu', padding='same')(up1)
        up1 = Conv2D(init_nb, 3, activation='relu', padding='same')(up1)

        x = Conv2D(2, 1, activation='softmax')(up1)
        x = Lambda(lambda z: z[:, :, :, 1], output_shape=self.config['output_shape'])(x)
        self.net = Model(inputs=inputs, outputs=x)

        # Optimizer & compile
        optimizer = RMSprop(learning_rate=lr)
        self.net.compile(optimizer=optimizer, loss=loss, metrics=[dice_coef])

    # ---- Training ------------------------------------------------------------
    def train(self):
        logger = logging.getLogger(funcname())

        gen_trn = self.batch_gen_trn(
            imgs=self.imgs_trn, imgs2=self.imgs_trn2, imgs3=self.imgs_trn3, imgs4=self.imgs_trn4,
            msks=self.msks_trn, msks2=self.msks_trn2, msks3=self.msks_trn3, msks4=self.msks_trn4,
            batch_size=self.config['batch_size'], transform=self.config['transform_train'], val=False,
        )
        gen_val = self.batch_gen_trn(
            imgs=self.imgs_val, imgs2=self.imgs_val2, imgs3=self.imgs_val3, imgs4=self.imgs_val4,
            msks=self.msks_val, msks2=self.msks_val2, msks3=self.msks_val3, msks4=self.msks_val4,
            batch_size=self.config['batch_size'], transform=self.config['transform_train'], val=True,
        )

        csv_logger = CSVLogger('training.log')
        clr_triangular = CyclicLR(mode='triangular')
        clr_triangular._reset(new_base_lr=1e-5, new_max_lr=5e-4)

        callbacks = [
            clr_triangular,
            EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=300, verbose=1, mode='min'),
            ModelCheckpoint(self.checkpoint_path + '/weights_loss_val.weights', monitor='val_loss', save_best_only=True, verbose=1),
            ModelCheckpoint(self.checkpoint_path + '/weights_loss_trn.weights', monitor='loss', save_best_only=True, verbose=1),
            csv_logger,
        ]

        logger.info(f"Training for {self.config['nb_epoch']} epochs.")
        # TF2: use fit with Python generator
        self.net.fit(
            gen_trn,
            steps_per_epoch=100,
            epochs=self.config['nb_epoch'],
            validation_data=gen_val,
            validation_steps=20,
            verbose=1,
            callbacks=callbacks,
        )

    # ---- Batch generator -----------------------------------------------------
    def batch_gen_trn(
        self, imgs, imgs2, imgs3, imgs4, msks, msks2, msks3, msks4, batch_size, transform=True, rng=np.random, val=False
    ):
        H, W = imgs.shape
        H2, W2 = imgs2.shape
        H3, W3 = imgs3.shape
        H4, W4 = imgs4.shape
        wdw_H, wdw_W = self.config['input_shape']

        _mean, _std = np.mean(imgs), np.std(imgs)
        _mean2, _std2 = np.mean(imgs2), np.std(imgs2)
        _mean3, _std3 = np.mean(imgs3), np.std(imgs3)
        _mean4, _std4 = np.mean(imgs4), np.std(imgs4)

        normalize = lambda x: (x - _mean) / (_std + 1e-10)
        normalize2 = lambda x: (x - _mean2) / (_std2 + 1e-10)
        normalize3 = lambda x: (x - _mean3) / (_std3 + 1e-10)
        normalize4 = lambda x: (x - _mean4) / (_std4 + 1e-10)

        while True:
            img_batch = np.zeros((batch_size,) + self.config['input_shape'], dtype=imgs.dtype)
            msk_batch = np.zeros((batch_size,) + self.config['output_shape'], dtype=msks.dtype)

            for batch_idx in range(batch_size):
                rand_var = random.random()
                if rand_var < 0.25:
                    y0, x0 = rng.randint(0, H - wdw_H), rng.randint(0, W - wdw_W)
                    y1, x1 = y0 + wdw_H, x0 + wdw_W
                    img_batch[batch_idx] = imgs[y0:y1, x0:x1]
                    msk_batch[batch_idx] = msks[y0:y1, x0:x1]
                if 0.25 <= rand_var < 0.50:
                    if val:
                        y0, x0 = rng.randint(0, H2 - wdw_H), 0
                    else:
                        y0, x0 = rng.randint(0, H2 - wdw_H), rng.randint(0, W2 - wdw_W)
                    y1, x1 = y0 + wdw_H, x0 + wdw_W
                    img_batch[batch_idx] = imgs2[y0:y1, x0:x1]
                    msk_batch[batch_idx] = msks2[y0:y1, x0:x1]
                if 0.50 <= rand_var <= 0.75:
                    y0, x0 = rng.randint(0, H3 - wdw_H), rng.randint(0, W3 - wdw_W)
                    y1, x1 = y0 + wdw_H, x0 + wdw_W
                    img_batch[batch_idx] = imgs3[y0:y1, x0:x1]
                    msk_batch[batch_idx] = msks3[y0:y1, x0:x1]
                if rand_var > 0.75:
                    y0, x0 = rng.randint(0, H4 - wdw_H), rng.randint(0, W4 - wdw_W)
                    y1, x1 = y0 + wdw_H, x0 + wdw_W
                    img_batch[batch_idx] = imgs4[y0:y1, x0:x1]
                    msk_batch[batch_idx] = msks4[y0:y1, x0:x1]
                # Per-dataset normalization
                if rand_var < 0.25:
                    img_batch = normalize(img_batch)
                elif 0.25 <= rand_var < 0.50:
                    img_batch = normalize2(img_batch)
                elif 0.50 <= rand_var < 0.75:
                    img_batch = normalize3(img_batch)
                else:
                    img_batch = normalize4(img_batch)
            yield img_batch, msk_batch

    # ---- Inference -----------------------------------------------------------
    def predict(self, imgs):
        imgs = (imgs - np.mean(imgs)) / (np.std(imgs) + 1e-10)
        return self.net.predict(imgs).round()


# ---- CLI --------------------------------------------------------------------

def main():
    # GPU memory growth (TF2)
    for g in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    tf.random.set_seed(865)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(funcname())

    prs = argparse.ArgumentParser()
    prs.add_argument('--name', help='name used for checkpoints', default='unet', type=str)

    subprs = prs.add_subparsers(title='actions', description='Choose from:')

    subprs_trn = subprs.add_parser('train', help='Train the model.')
    subprs_trn.set_defaults(which='train')
    subprs_trn.add_argument('-w', '--weights', help='path to keras weights')

    subprs_sbt = subprs.add_parser('predict', help='Run inference on a TIFF stack/image.')
    subprs_sbt.set_defaults(which='predict')
    subprs_sbt.add_argument('-w', '--weights', help='path to weights', required=True)
    subprs_sbt.add_argument('-t', '--tiff', help='path to image', required=True)

    args = vars(prs.parse_args())
    assert args['which'] in ['train', 'predict']

    model = UNet(args['name'])

    if not path.exists(model.checkpoint_path):
        makedirs(model.checkpoint_path)

    def load_weights():
        if args.get('weights'):
            logger.info('Loading weights from %s.' % args['weights'])
            # Make sure model is built before loading
            if model.net is None:
                model.compile()
            model.net.load_weights(args['weights'])

    if args['which'] == 'train':
        model.compile()
        load_weights()
        model.net.summary()
        model.load_data()
        model.train()

    elif args['which'] == 'predict':
        out_path = f"{model.checkpoint_path}/test-volume-masks.tif"
        model.config['input_shape'] = (1024, 1024)
        model.config['output_shape'] = (1024, 1024)
        model.compile()
        load_weights()
        model.net.summary()
        imgs_sbt = tiff.imread(args['tiff'])
        msks_sbt = model.predict(imgs_sbt)
        logger.info('Writing predicted masks to %s' % out_path)
        tiff.imsave(out_path, msks_sbt.astype(np.uint8))


if __name__ == "__main__":
    main()
