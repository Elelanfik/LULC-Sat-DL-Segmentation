import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, 
    UpSampling2D, concatenate, Dropout, ZeroPadding2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K

import numpy as np
import cv2
import pandas as pd
import math
import matplotlib.pyplot as plt
import albumentations as A

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

# Constants and Configuration
INPUT_SHAPE = (256, 256, 3)
NUM_CLASSES = 8
BATCH_SIZE = 16
INITIAL_LR = 1e-4
DROPOUT_RATE = 0.3

# Class definitions for land cover types
CLASS_COLORS = {
    'Forest': [0, 255, 0],
    'Agricultural Land': [165, 42, 42],
    'Road': [70, 130, 180],
    'Grassland': [50, 205, 50],
    'Water Bodies': [0, 0, 255],
    'Shrubland': [173, 255, 47],
    'Built-up': [255, 105, 45],
    'Unlabelled': [0, 0, 0],
}

# Data Encoding/Decoding Utilities
def encode_categorical(mask, num_classes=NUM_CLASSES):
    """Convert RGB mask to one-hot encoded categorical mask"""
    cat_encoded = np.zeros((mask.shape[0], mask.shape[1], num_classes), dtype='float32')
    for i, value in enumerate(CLASS_COLORS.values()):
        condition = np.all(mask == value, axis=-1)
        cat_encoded[condition, i] = 1.0
    return cat_encoded

def decode_prediction(pred, num_classes=NUM_CLASSES):
    """Convert model predictions back to RGB mask"""
    argmax_idx = np.array([np.argmax(x) for x in pred.reshape((-1, num_classes))])
    argmax_idx = argmax_idx.reshape(pred.shape[:-1])
    
    decoded_mask = np.zeros(shape=(*pred.shape[:-1], 3), dtype='uint8')
    
    for i, value in enumerate(CLASS_COLORS.values()):
        decoded_mask[argmax_idx==i] = value
    return decoded_mask

# Custom Data Generator
class LandCoverDataGenerator(Sequence):
    """
    Custom data generator for land cover segmentation that supports:
    - Loading and preprocessing images/masks
    - On-the-fly augmentation
    - Batch processing
    """
    
    def __init__(self, img_paths, mask_paths, batch_size, augment=False, shuffle=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.img_paths))
        self.on_epoch_end()
        
    def __len__(self):
        return math.ceil(len(self.img_paths) / self.batch_size)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_img_paths = [self.img_paths[i] for i in batch_indexes]
        batch_mask_paths = [self.mask_paths[i] for i in batch_indexes]
        return self._load_batch(batch_img_paths, batch_mask_paths)
    
    def _load_batch(self, img_paths, mask_paths):
        batch_img = []
        batch_mask = []
        
        for img_path, mask_path in zip(img_paths, mask_paths):
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
            mask = encode_categorical(mask)
            
            if self.augment:
                augmented = self._augment(image, mask)
                batch_img.extend(augmented[0])
                batch_mask.extend(augmented[1])
            else:
                batch_img.append(image.astype(np.float32) / 255.0)
                batch_mask.append(mask.astype(np.float32))
                
        return np.array(batch_img), np.array(batch_mask)
    
    def _augment(self, image, mask):
        """Apply Albumentations augmentations"""
        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.2),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ])
        augmented = transform(image=image, mask=mask)
        return (
            [image/255.0, augmented['image']/255.0],
            [mask, augmented['mask']]
        )

# Custom Metrics and Losses
def dice_coef(y_true, y_pred, smooth=1):
    """Dice coefficient metric"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss function"""
    return 1 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """Combined Dice and Crossentropy loss"""
    return dice_loss(y_true, y_pred) + tf.keras.losses.categorical_crossentropy(y_true, y_pred)

class MeanIoUWrapper(tf.keras.metrics.Metric):
    """Wrapper for MeanIoU to work with categorical outputs"""
    def __init__(self, num_classes, name='iou_metric', **kwargs):
        super().__init__(name=name, **kwargs)
        self.iou_metric = MeanIoU(num_classes=num_classes)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        self.iou_metric.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        return self.iou_metric.result()
    
    def reset_states(self):
        self.iou_metric.reset_states()

# Model Architecture
def decoder_block(x, skip, filters):
    """UNet-style decoder block with skip connections"""
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
    skip = ZeroPadding2D(((0, x.shape[1]-skip.shape[1]), (0, x.shape[2]-skip.shape[2])))(skip)
    x = concatenate([x, skip])
    
    x = Conv2D(filters, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(filters, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    
    return x

def build_xception_unet(input_shape, num_classes, dropout_rate):
    """Build Xception-UNet hybrid model"""
    inputs = Input(input_shape)
    
    # Xception backbone (encoder)
    xception = Xception(include_top=False, weights="imagenet", input_tensor=inputs)
    
    # Skip connections from Xception
    s1 = xception.get_layer("block1_conv2_act").output
    s2 = xception.get_layer("block3_sepconv2_bn").output
    s3 = xception.get_layer("block4_sepconv2_bn").output 
    s4 = xception.get_layer("block13_sepconv2_act").output
    x = xception.get_layer("block14_sepconv2_act").output
    
    # Decoder path
    x = decoder_block(x, s4, 512)
    x = decoder_block(x, s3, 256)
    x = decoder_block(x, s2, 128)
    x = decoder_block(x, s1, 64)
    
    # Final upsampling and output
    x = UpSampling2D(size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Conv2D(num_classes, 1, activation="softmax")(x)
    
    return Model(inputs, outputs, name="Xception_U-Net")

# Training Pipeline
def train_model():
    # Load data splits
    train_df = pd.read_csv('/path/to/train.csv')
    val_df = pd.read_csv('/path/to/val.csv')
    test_df = pd.read_csv('/path/to/test.csv')
    
    # Create data generators
    train_gen = LandCoverDataGenerator(
        train_df['image_path'].tolist(),
        train_df['mask_path'].tolist(),
        batch_size=BATCH_SIZE,
        augment=True,
        shuffle=True
    )
    
    val_gen = LandCoverDataGenerator(
        val_df['image_path'].tolist(),
        val_df['mask_path'].tolist(),
        batch_size=BATCH_SIZE,
        augment=False,
        shuffle=False
    )
    
    # Build and compile model
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_xception_unet(INPUT_SHAPE, NUM_CLASSES, DROPOUT_RATE)
        optimizer = Adam(learning_rate=INITIAL_LR)
        model.compile(
            optimizer=optimizer,
            loss=combined_loss,
            metrics=[
                dice_coef,
                MeanIoUWrapper(num_classes=NUM_CLASSES)
            ]
        )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_dice_coef', patience=20, restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, mode='min', verbose=1),
        ModelCheckpoint(
            filepath='best_model.weights.h5',
            save_weights_only=True,
            monitor='val_dice_coef',
            mode='max',
            save_best_only=True
        )
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=100,
        callbacks=callbacks
    )
    
    return model, history

# Visualization Utilities
def plot_training_history(history):
    """Plot training metrics"""
    metrics = ['loss', 'dice_coef', 'iou_metric']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Plot training and validation metrics
        plt.plot(history.history[metric], label=f'Training {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        
        plt.title(f'Training vs Validation {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        
        plt.savefig(f'training_{metric}.png')
        plt.close()

if __name__ == "__main__":
    model, history = train_model()
    plot_training_history(history)