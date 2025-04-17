# IMPORT PACKAGE
import os
import cv2
import math
import random
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import clear_output
from random import randint
from glob import glob
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import Sequence
import albumentations as A
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, ReLU, Add, Softmax, Dropout, LeakyReLU, UpSampling2D, concatenate, ZeroPadding2D, Activation, Input, LeakyReLU, DepthwiseConv2D
from tensorflow.keras import backend as K
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
from skmultilearn.model_selection import iterative_train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.mixed_precision import Policy, set_global_policy
from tensorflow.keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, Lambda, Reshape, Dropout, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, concatenate, UpSampling2D, multiply, add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.mixed_precision import Policy, set_global_policy
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
  
  
# CUSTOM DATA GENERATOR  
  
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
 

# MODEL DEFINITION
# Convolution Block with Batch Normalization and Dropout

def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1, padding='same', use_bias=False):
    x = Conv2D(
        num_filters, 
        kernel_size=kernel_size,
        dilation_rate=dilation_rate, 
        padding=padding, 
        use_bias=use_bias, 
        #kernel_initializer='he_normal',
        #kernel_regularizer=l2(0.1)  # Added L2 regularization
    )(block_input)
    #x = BatchNormalization()(x)  # Added Batch Normalization
    x = Dropout(0.3)(x)
    return Activation('relu')(x)

def DilatedSpatialPyramidPooling(input):
    dims = input.shape
    x = Conv2D(256, (1, 1), padding='same', use_bias=False)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    atrous_block1 = Conv2D(256, (3, 3), padding='same', dilation_rate=(1, 1), use_bias=False)(input)
    atrous_block1 = BatchNormalization()(atrous_block1)
    atrous_block1 = ReLU()(atrous_block1)

    atrous_block6 = Conv2D(256, (3, 3), padding='same', dilation_rate=(6, 6), use_bias=False)(input)
    atrous_block6 = BatchNormalization()(atrous_block6)
    atrous_block6 = ReLU()(atrous_block6)

    atrous_block12 = Conv2D(256, (3, 3), padding='same', dilation_rate=(12, 12), use_bias=False)(input)
    atrous_block12 = BatchNormalization()(atrous_block12)
    atrous_block12 = ReLU()(atrous_block12)

    atrous_block18 = Conv2D(256, (3, 3), padding='same', dilation_rate=(18, 18), use_bias=False)(input)
    atrous_block18 = BatchNormalization()(atrous_block18)
    atrous_block18 = ReLU()(atrous_block18)

    pooling = AveragePooling2D(pool_size=(dims[-3], dims[-2]))(input)
    pooling = Conv2D(256, (1, 1), padding='same', use_bias=False)(pooling)
    pooling = BatchNormalization()(pooling)
    pooling = ReLU()(pooling)
    pooling = UpSampling2D(size=(dims[-3] // pooling.shape[1], dims[-2] // pooling.shape[2]), interpolation='bilinear')(pooling)

    concat = concatenate([x, atrous_block1, atrous_block6, atrous_block12, atrous_block18, pooling])
    output = Conv2D(256, (1, 1), padding='same', use_bias=False)(concat)
    output = BatchNormalization()(output)
    output = ReLU()(output)

    return output

def DeeplabV3Plus(image_size, num_classes):
    input = Input(shape=(image_size, image_size, 3))
    mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input)

    x = mobilenet.get_layer('block_13_expand_relu').output  # Changed to an appropriate layer in MobileNetV2
    x = DilatedSpatialPyramidPooling(x)

    input_a = UpSampling2D(size=(image_size // x.shape[1], image_size // x.shape[2]), interpolation='bilinear')(x)
    input_b = mobilenet.get_layer('block_4_expand_relu').output  # Changed to an appropriate layer in MobileNetV2
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    # Upsample 'input_b' to match 'input_a' size
    input_b = UpSampling2D(size=(input_a.shape[1] // input_b.shape[1], input_a.shape[2] // input_b.shape[2]), interpolation='bilinear')(input_b)
    x = concatenate([input_a, input_b], axis=-1)

    x = convolution_block(x)
    x = convolution_block(x)
    x = UpSampling2D(size=(image_size // x.shape[1], image_size // x.shape[2]), interpolation='bilinear')(x)
    output = Conv2D(num_classes, kernel_size=(1, 1), padding='same', activation='softmax')(x)
    return Model(inputs=input, outputs=output, name='DeepLabV3')

#MODEL TRAINING
train_generator = ImageDataGenerator(
    img_paths=train_img_paths,
    mask_paths=train_mask_paths,
    batch_size=8,
    augment=True,  # Enable augmentation for training
    num_classes=8,
    shuffle=True  # Shuffle training data
)

val_generator = ImageDataGenerator(
    img_paths=val_img_paths,
    mask_paths=val_mask_paths,
    batch_size=8,
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



for image, mask in train_generator:
    print(image.shape)
    print(mask.shape)
    break 
    
img, label = test_generator.__getitem__(4)
img.shape, label.shape

#METRICS
# Example usagevvv
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

    
# MODEL TRAINING

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # Reinitialize the model
    model = DeeplabV3Plus(image_size=256, num_classes=8)
    adamw_optimizer = Adam(learning_rate=1e-4, weight_decay=1e-5)
    model.compile(
        optimizer=adamw_optimizer, 
        loss=combined_dice_crossentropy_loss, 
        metrics=[dice_coef_metric, jacard_coef, MeanIoUWrapper(num_classes=n_classes)]
    )
    
    # Load the latest checkpoint weights
    latest_checkpoint_path = "/home/rdadmin/Tsiyon/NEWONE/MODELSAVED/DEPMOBIV_latest_latest.weights.h5"
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
        factor=0.2, 
        patience=10, 
        mode='min', 
        verbose=1
    )
    
    best_checkpoint = ModelCheckpoint(
        filepath="/home/rdadmin/Tsiyon/NEWONE/MODELSAVED/1DEPMOBIV_best.weights.h5",
        save_weights_only=True,
        monitor='val_dice_coef_metric',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    last_checkpoint = ModelCheckpoint(
        filepath="/home/rdadmin/Tsiyon/NEWONE/MODELSAVED/DEPMOBIV_latest.weights.h5",
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
plt.savefig('/home/rdadmin/Tsiyon/NEWONE/RESULT/DEPMOBIVLOSS.png')
plt.show()
plt.close()

# Plot Training and Validation Dice Coefficient
plt.plot(smoothed_epochs, smoothed_dice_coef, 'c', label="Training")
plt.plot(smoothed_epochs, smoothed_val_dice_coef, 'r', label="Validation")
plt.title("Training Vs Validation Dice-Coef")
plt.xlabel("Epochs")
plt.ylabel("Dice-Coef")
plt.xticks(x_ticks)
plt.legend()
plt.savefig('/home/rdadmin/Tsiyon/NEWONE/RESULT/DEPMOBIVdice.png')
plt.show()
plt.close()

# Plot Training and Validation IOU (Jaccard Coefficient)
plt.plot(smoothed_epochs, smoothed_iou, 'c', label="Training")
plt.plot(smoothed_epochs, smoothed_val_iou, 'r', label="Validation")
plt.title("Training Vs Validation jac Metric")
plt.xlabel("Epochs")
plt.ylabel("IOU")
plt.xticks(x_ticks)
plt.legend()
plt.savefig('/home/rdadmin/Tsiyon/NEWONE/RESULT/DEPMOBIVJAC.png')
plt.show()
plt.close()

# Plot Training and Validation IOU Metric
plt.plot(smoothed_epochs, smoothed_iou1, 'c', label="Training")
plt.plot(smoothed_epochs, smoothed_val_iou1, 'r', label="Validation")
plt.title("Training Vs Validation IOU Metric")
plt.xlabel("Epochs")
plt.ylabel("IOU")
plt.xticks(x_ticks)
plt.legend()
plt.savefig('/home/rdadmin/Tsiyon/RESULT_FIGUREDEPMOBIVIOU.png')
plt.show()
plt.close()



           
