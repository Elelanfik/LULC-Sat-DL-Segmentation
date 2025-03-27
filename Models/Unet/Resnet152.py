"""


A deep learning model combining ResNet152 encoder with UNet decoder for semantic segmentation 
of land cover types from satellite/aerial imagery. Implements advanced training techniques 
including multi-GPU support, custom loss functions, and comprehensive evaluation metrics.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Input, backend as K
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, 
    UpSampling2D, concatenate, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.utils import Sequence
from tensorflow.keras.regularizers import l2

import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import pandas as pd
import math

# Configure GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error setting GPU memory growth: {e}")

# Constants
INPUT_SHAPE = (256, 256, 3)
NUM_CLASSES = 8
BATCH_SIZE = 16
INITIAL_LR = 1e-4
DROPOUT_RATE = 0.3
L2_REG = 0.1

# Land cover class definitions
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

class LandCoverDataGenerator(Sequence):
    """
    Data generator for land cover segmentation that supports:
    - Loading and preprocessing of images/masks
    - On-the-fly augmentation
    - Batch processing
    
    Args:
        img_paths (list): List of image file paths
        mask_paths (list): List of mask file paths
        batch_size (int): Number of samples per batch
        augment (bool): Whether to apply data augmentation
        shuffle (bool): Whether to shuffle data after each epoch
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
            mask = self._encode_mask(mask)
            
            if self.augment:
                augmented = self._augment(image, mask)
                batch_img.extend(augmented[0])
                batch_mask.extend(augmented[1])
            else:
                batch_img.append(image.astype(np.float32) / 255.0)
                batch_mask.append(mask.astype(np.float32))
                
        return np.array(batch_img), np.array(batch_mask)
    
    def _encode_mask(self, mask):
        """Convert RGB mask to one-hot encoded categorical mask"""
        encoded = np.zeros((mask.shape[0], mask.shape[1], NUM_CLASSES), dtype='float32')
        for i, color in enumerate(CLASS_COLORS.values()):
            encoded[np.all(mask == color, axis=-1), i] = 1.0
        return encoded
    
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

def decoder_block(x, skip, filters, dropout_rate=DROPOUT_RATE):
    """
    UNet decoder block with skip connection
    
    Args:
        x: Input tensor from previous layer
        skip: Skip connection tensor from encoder
        filters: Number of filters for convolutional layers
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Output tensor after processing
    """
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
    x = concatenate([x, skip])
    
    x = Conv2D(filters, 3, padding='same', kernel_regularizer=l2(L2_REG))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    
    x = Conv2D(filters, 3, padding='same', kernel_regularizer=l2(L2_REG))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    
    return x

def build_resnet152_unet(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
    """
    Build ResNet152-UNet hybrid model
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    inputs = Input(input_shape)
    
    # ResNet152 backbone (encoder)
    resnet = ResNet152(include_top=False, weights="imagenet", input_tensor=inputs)
    
    # Skip connections from ResNet
    s1 = resnet.get_layer("conv1_relu").output
    s2 = resnet.get_layer("conv2_block3_out").output
    s3 = resnet.get_layer("conv3_block8_out").output
    s4 = resnet.get_layer("conv4_block36_out").output
    x = resnet.get_layer("conv5_block3_out").output
    
    # Decoder path
    x = decoder_block(x, s4, 512)
    x = decoder_block(x, s3, 256)
    x = decoder_block(x, s2, 128)
    x = decoder_block(x, s1, 64)
    
    # Final output
    x = UpSampling2D(size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)
    outputs = Conv2D(num_classes, 1, activation="softmax")(x)
    
    return Model(inputs, outputs, name="ResNet152_U-Net")

# Custom Metrics
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

def dice_coef(y_true, y_pred, smooth=1):
    """Dice similarity coefficient metric"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def jaccard_coef(y_true, y_pred, smooth=1):
    """Jaccard similarity coefficient (IoU) metric"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def combined_loss(y_true, y_pred):
    """Combined Dice and Crossentropy loss"""
    dice_loss = 1 - dice_coef(y_true, y_pred)
    crossentropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return dice_loss + crossentropy

def train_model():
    """Main training pipeline"""
    # Load dataset
    train_df = pd.read_csv('/path/to/train.csv')
    val_df = pd.read_csv('/path/to/val.csv')
    
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
        model = build_resnet152_unet()
        optimizer = Adam(learning_rate=INITIAL_LR)
        model.compile(
            optimizer=optimizer,
            loss=combined_loss,
            metrics=[
                dice_coef,
                jaccard_coef,
                MeanIoUWrapper(num_classes=NUM_CLASSES)
            ]
        )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_dice_coef',
            patience=20,
            restore_best_weights=True,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=10,
            mode='min',
            verbose=1
        ),
        ModelCheckpoint(
            filepath='best_model.weights.h5',
            save_weights_only=True,
            monitor='val_dice_coef',
            mode='max',
            save_best_only=True,
            verbose=1
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

def plot_training_history(history):
    """Visualize training metrics"""
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