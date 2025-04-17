# Standard Library Imports
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# TensorFlow / Keras Imports
import tensorflow as tf
from tensorflow.keras import layers, models, Input, backend as K
from tensorflow.keras.applications import Xception, MobileNetV2
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, UpSampling2D,
    concatenate, Dropout, ZeroPadding2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.utils import Sequence
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import he_normal

# Additional Libraries
import albumentations as A
from IPython.display import clear_output


clear_output()
# GPU MEMORY GROWTH
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth set for GPUs.")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")
# ENCODE AND DECODE METHOD       
        
mask_rgb_values = {
    'Forest': [0, 255, 0],
    'Agricultural Land': [165, 42, 42],
    'Road': [70, 130, 180],
    'Grassland': [50, 205, 50],
    'Water Bodies': [0, 0, 255],
    'Shrubland': [173, 255, 47],
    'Built-up': [255, 105, 45],
    'Unlabelled': [0, 0, 0],
}

def encode_categorical(mask, num_classes=8):
    cat_encoded = np.zeros((mask.shape[0], mask.shape[1], num_classes), dtype='float32')
    for i, value in enumerate(mask_rgb_values.values()):
        condition = np.all(mask == value, axis=-1)
        cat_encoded[condition, i] = 1.0
    return cat_encoded


def decode_prediction(pred, num_classes=8):
  argmax_idx = np.array([np.argmax(x) for x in pred.reshape((-1, num_classes))])
  argmax_idx = argmax_idx.reshape(pred.shape[:-1])

  decoded_mask=np.zeros(shape=(*pred.shape[:-1], 3), dtype='uint8')

  for i, value in enumerate(mask_rgb_values.values()):
    decoded_mask[argmax_idx==i] = value
  return decoded_mask   
  


# Load CSVs
train_df = pd.read_csv('/home/rdadmin/Tsiyon/notebook/SPLITS/train.csv')
val_df = pd.read_csv('/home/rdadmin/Tsiyon/notebook/SPLITS/val.csv')
test_df = pd.read_csv('/home/rdadmin/Tsiyon/notebook/SPLITS/test.csv')

# Extract paths from the CSVs
train_img_paths = train_df['image_path'].tolist()
train_mask_paths = train_df['mask_path'].tolist()

val_img_paths = val_df['image_path'].tolist()
val_mask_paths = val_df['mask_path'].tolist()

test_img_paths = test_df['image_path'].tolist()
test_mask_paths = test_df['mask_path'].tolist()
  
  
  
class ImageDataGenerator(Sequence):
    def __init__(self,
                 img_paths,
                 mask_paths,
                 batch_size,
                 augment,
                 num_classes=8,
                 shuffle=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.augment = augment

        self.indexes = np.arange(len(self.img_paths))
        self.on_epoch_end()
        
    def __len__(self):
        return math.ceil(len(self.img_paths) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # Get batch indexes
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_img_paths = [self.img_paths[i] for i in batch_indexes]
        batch_mask_paths = [self.mask_paths[i] for i in batch_indexes]

        # Load and preprocess batch
        X, y = self.__load_batch(batch_img_paths, batch_mask_paths)
        return X, y
    
    def __load_batch(self, batch_img_paths, batch_mask_paths):
        batch_img = []
        batch_mask = []

        for img_path, mask_path in zip(batch_img_paths, batch_mask_paths):
            # Load image and mask
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)

            # Encode mask as categorical
            mask = encode_categorical(mask, num_classes=self.num_classes)
            
            # Perform augmentation if enabled
            if self.augment:
                augmented_images, augmented_masks = self.data_augmentation(image, mask)
                batch_img.extend(augmented_images)
                batch_mask.extend(augmented_masks)
            else:
                batch_img.append(image.astype(np.float32) / 255.0)
                batch_mask.append(mask.astype(np.float32))

        return np.array(batch_img, dtype='float32'), np.array(batch_mask, dtype='float32')   

    def data_augmentation(self, image, mask):
        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.2),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ])

        augmented = transform(image=image, mask=mask)
        augmented_image = augmented['image']
        augmented_mask = augmented['mask']

        original_image = image.astype(np.float32) / 255.0
        original_mask = mask.astype(np.float32) 
        augmented_image = augmented_image.astype(np.float32) / 255.0
        augmented_mask = augmented_mask.astype(np.float32)

        return [original_image, augmented_image], [original_mask, augmented_mask]   
        
#DATA PREPATION
        
class_colors = {    
    'Forest': [0, 255, 0],
    'Agricultural Land': [165, 42, 42],
    'Road': [70, 130, 180],
    'Grassland': [50, 205, 50],
    'Water Bodies': [0, 0, 255],
    'Shrubland': [173, 255, 47],
    'Built-up': [255, 105, 45],
    'Unlabelled': [0, 0, 0],
}

# METRICS
def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def combined_dice_crossentropy_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    crossentropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return dice + crossentropy


class MeanIoUWrapper(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='iou_metric', **kwargs):
        super(MeanIoUWrapper, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.iou_metric = MeanIoU(num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        self.iou_metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.iou_metric.result()

    def reset_states(self):
        self.iou_metric.reset_states()

def iou_metric(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-6)  
    
def dice_coef_metric(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    
def jacard_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection + 1.0)
input_shape = (256, 256, 3)
n_classes = 8


def decoder_block(x, y, filters, dropout_rate=0.3):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)

    # Adjust the number of filters in y to match x
    y = layers.Conv2D(filters, (1, 1), padding='same')(y)

    # Apply ZeroPadding to the feature map y to match the shape of x
    if x.shape[1] != y.shape[1] or x.shape[2] != y.shape[2]:
        y = layers.ZeroPadding2D(((0, x.shape[1] - y.shape[1]), (0, x.shape[2] - y.shape[2])))(y)

    x = layers.Add()([x, y])
    x = layers.Conv2D(filters, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = layers.Conv2D(filters, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    return x

def xception_linknet(input_shape, classes, dropout):
    inputs = Input(input_shape)
    xception = Xception(include_top=False, weights="imagenet", input_tensor=inputs)

    # Extract specific layers for skip connections
    s1 = xception.get_layer("block1_conv2_act").output  # 128x128x64
    s2 = xception.get_layer("block3_sepconv2_bn").output  # 64x64x128
    s3 = xception.get_layer("block4_sepconv2_bn").output  # 32x32x256
    s4 = xception.get_layer("block13_sepconv2_act").output  # 16x16x728
    x = xception.get_layer("block14_sepconv2_act").output  # 8x8x2048

    # Decoder
    x = decoder_block(x, s4, 1024)
    x = decoder_block(x, s3, 512)
    x = decoder_block(x, s2, 256)
    x = decoder_block(x, s1, 128)
    x = UpSampling2D(size=(2, 2))(x)
    x = Dropout(dropout)(x)
    outputs = Conv2D(classes, 1, activation="softmax")(x)

    model = Model(inputs, outputs, name="Xception_LinkNet")
    return model


def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def combined_dice_crossentropy_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    crossentropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return dice + crossentropy

class MeanIoUWrapper(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='iou_metric', **kwargs):
        super(MeanIoUWrapper, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.iou_metric = MeanIoU(num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        self.iou_metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.iou_metric.result()

    def reset_states(self):
        self.iou_metric.reset_states()

def iou_metric(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-6)  
    
def dice_coef_metric(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    
def jacard_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection + 1.0)
    # Define the Dice loss function
def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
#MODEL TRAINING
train_generator = ImageDataGenerator(
    img_paths=train_img_paths,
    mask_paths=train_mask_paths,
    batch_size=16,
    augment=True,  # Enable augmentation for training
    num_classes=8,
    shuffle=True  # Shuffle training data
)

val_generator = ImageDataGenerator(
    img_paths=val_img_paths,
    mask_paths=val_mask_paths,
    batch_size=16,
    augment=True,  # Disable augmentation for validation
    num_classes=8,
    shuffle=True  # No shuffling for validation
)

test_generator = ImageDataGenerator(
    img_paths=test_img_paths,
    mask_paths=test_mask_paths,
    batch_size=8,
    augment=False,  # No augmentation for testing
    num_classes=8,
    shuffle=False  # No shuffling for testing
)


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # Reinitialize the model
    model = xception_linknet(input_shape, classes=n_classes, dropout=0.3)
    adamw_optimizer = Adam(learning_rate=1e-4)
    model.compile(
        optimizer=adamw_optimizer, 
        loss=combined_dice_crossentropy_loss, 
        metrics=[dice_coef_metric, jacard_coef, MeanIoUWrapper(num_classes=n_classes)]
    )
    
    # Load the latest checkpoint weights
    latest_checkpoint_path = "/home/rdadmin/Tsiyon/NEWONE/MODELSAVED/LINKNETXCEP_latest_latest.weights.h5"
    try:
        model.load_weights(latest_checkpoint_path)
        print(f"Loaded weights from {latest_checkpoint_path}")
    except Exception as e:
        print(f"Error loading weights: {e}")

    # Callbacks
    earlystop = EarlyStopping(
        monitor='val_dice_coef_metric', 
        patience=20, 
        restore_best_weights=True, 
        mode='max'
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.3, 
        patience=10, 
        mode='min', 
        verbose=1
    )
    
    best_checkpoint = ModelCheckpoint(
        filepath="/home/rdadmin/Tsiyon/NEWONE/MODELSAVED/LINKNETXCEP_best.weights.h5",
        save_weights_only=True,
        monitor='val_dice_coef_metric',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    last_checkpoint = ModelCheckpoint(
        filepath="/home/rdadmin/Tsiyon/NEWONE/MODELSAVED/LINKNETXCEP_latest.weights.h5",
        save_weights_only=True,
        verbose=1
    )

    callbacks = [earlystop, reduce_lr, best_checkpoint, last_checkpoint]

# Resume training
history = model.fit(
    train_generator,
    validation_data=val_generator,
    initial_epoch=history.epoch[-1] if 'history' in locals() else 0,
    epochs=100,
    callbacks=callbacks
)
def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    smoothed_values = np.convolve(values, weights, 'valid')
    return smoothed_values

# Extract data from history
loss = history.history['loss']
val_loss = history.history['val_loss']
dice_coef = history.history['dice_coef_metric']
val_dice_coef = history.history['val_dice_coef_metric']
iou = history.history['jacard_coef']
val_iou = history.history['val_jacard_coef']
iou1 = history.history['iou_metric']
val_iou1 = history.history['val_iou_metric']

window_size = 5

smoothed_loss = moving_average(loss, window_size)
smoothed_val_loss = moving_average(val_loss, window_size)
smoothed_dice_coef = moving_average(dice_coef, window_size)
smoothed_val_dice_coef = moving_average(val_dice_coef, window_size)
smoothed_iou = moving_average(iou, window_size)
smoothed_val_iou = moving_average(val_iou, window_size)
smoothed_iou1 = moving_average(iou1, window_size)
smoothed_val_iou1 = moving_average(val_iou1, window_size)

smoothed_epochs = range(1, len(smoothed_loss) + 1)

# Set x-axis ticks to step by 1
x_ticks = np.arange(10, len(smoothed_loss) + 1, step=10)

# Plot Training and Validation Loss
plt.plot(smoothed_epochs, smoothed_loss, 'y', label='Training loss')
plt.plot(smoothed_epochs, smoothed_val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(x_ticks)
plt.legend()
plt.savefig('/home/rdadmin/Tsiyon/NEWONE/RESULT/LINKNETXCEPLOSS.png')
plt.close()

# Plot Training and Validation Dice Coefficient
plt.plot(smoothed_epochs, smoothed_dice_coef, 'c', label="Training")
plt.plot(smoothed_epochs, smoothed_val_dice_coef, 'r', label="Validation")
plt.title("Training Vs Validation Dice-Coef")
plt.xlabel("Epochs")
plt.ylabel("Dice-Coef")
plt.xticks(x_ticks)
plt.legend()
plt.savefig('/home/rdadmin/Tsiyon/NEWONE/RESULT/LINKNETXCEPdice.png')
plt.close()

# Plot Training and Validation IOU (Jaccard Coefficient)
plt.plot(smoothed_epochs, smoothed_iou, 'c', label="Training")
plt.plot(smoothed_epochs, smoothed_val_iou, 'r', label="Validation")
plt.title("Training Vs Validation jac Metric")
plt.xlabel("Epochs")
plt.ylabel("IOU")
plt.xticks(x_ticks)
plt.legend()
plt.savefig('/home/rdadmin/Tsiyon/NEWONE/RESULT/LINKNETXCEPJAC.png')
plt.close()

# Plot Training and Validation IOU Metric
plt.plot(smoothed_epochs, smoothed_iou1, 'c', label="Training")
plt.plot(smoothed_epochs, smoothed_val_iou1, 'r', label="Validation")
plt.title("Training Vs Validation IOU Metric")
plt.xlabel("Epochs")
plt.ylabel("IOU")
plt.xticks(x_ticks)
plt.legend()
plt.savefig('/home/rdadmin/Tsiyon/NEWONE/RESULT/LINKNETXCEPIOU.png')
plt.close()

