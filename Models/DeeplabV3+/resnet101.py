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
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, ReLU, Add, Softmax, Dropout, LeakyReLU, UpSampling2D, concatenate, ZeroPadding2D, Activation, Input, LeakyReLU
from tensorflow.keras import backend as K
from tensorflow.keras.applications import ResNet101  
from tensorflow.keras.regularizers import l2
from skmultilearn.model_selection import iterative_train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.initializers import he_normal
#from tensorflow_addons.optimizers import AdamW
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

class_names = list(class_colors.keys())
class_rgb_values = list(class_colors.values())

img_paths = glob('/home/rdadmin/Tsiyon/DATA/SELEN/*IMAGE*/*.jpg')

mask_paths = [path.replace('.jpg', '.png').replace('IMAGE', 'MASK') for path in img_paths]

class_indicators = []

for mask_path in mask_paths:
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    mask = tf.convert_to_tensor(mask, dtype=tf.int32)

    mask_classes = tf.zeros(len(class_rgb_values), dtype=tf.int32)

    flattened_mask = tf.reshape(mask, [-1, 3])

    flattened_mask_1d = tf.reduce_sum(flattened_mask * [1, 256, 65536], axis=1)  # Convert RGB tuple to a single value
    
    unique_colors = tf.unique(flattened_mask_1d)[0]

    mask_classes = tf.Variable(mask_classes)  # Make it mutable

    for color in unique_colors.numpy():  # Convert to NumPy array for easy comparison
        color_list = [(color % 256), ((color // 256) % 256), (color // 65536)]  # Recover the RGB values from the combined value
        if color_list in class_rgb_values:
            class_idx = class_rgb_values.index(color_list)
            # Use tensor_scatter_nd_update to modify the class
            mask_classes = tf.tensor_scatter_nd_update(mask_classes, [[class_idx]], [1])
    
    # Convert mask_classes to NumPy and append to the list
    class_indicators.append(mask_classes.numpy())

class_indicators = np.array(class_indicators)  


X = np.array(img_paths).reshape(-1, 1)  # Convert to 2D array (n_samples, 1)
y = np.array(class_indicators)  # Make sure class_indicators is a 2D array if it's multilabel

# Check if y is a 2D array; if not, you can reshape it
if len(y.shape) == 1:
    y = y.reshape(-1, 1)  # Convert y to a 2D array (n_samples, 1)

# First split (train_val, test)
X_train_val, y_train_val, X_test, y_test = iterative_train_test_split(X, y, test_size=0.1)

# Second split (train, val) using the train_val data
X_train, y_train, X_val, y_val = iterative_train_test_split(X_train_val, y_train_val, test_size=0.1111)

train_img_paths = [path[0] for path in X_train.tolist()]  # Flatten 2D array to 1D
val_img_paths = [path[0] for path in X_val.tolist()]      # Flatten 2D array to 1D
test_img_paths = [path[0] for path in X_test.tolist()]    # Flatten 2D array to 1D

train_mask_paths = [path.replace('.jpg', '.png').replace('IMAGE', 'MASK') for path in train_img_paths]
val_mask_paths = [path.replace('.jpg', '.png').replace('IMAGE', 'MASK') for path in val_img_paths]
test_mask_paths = [path.replace('.jpg', '.png').replace('IMAGE', 'MASK') for path in test_img_paths]


def get_class_distribution(y):
    return np.sum(y, axis=0)

train_class_distribution = get_class_distribution(y_train)
val_class_distribution = get_class_distribution(y_val)
test_class_distribution = get_class_distribution(y_test)

print("Train class distribution:", dict(zip(class_names, train_class_distribution)))
print("Validation class distribution:", dict(zip(class_names, val_class_distribution)))
print("Test class distribution:", dict(zip(class_names, test_class_distribution)))

print("Number of training images:", len(X_train))
print("Number of validation images:", len(X_val))
print("Number of test images:", len(X_test))

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
    x = Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, 
               padding=padding, use_bias=use_bias, kernel_initializer=he_normal())(block_input)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    return Activation('relu')(x)

# Dilated Spatial Pyramid Pooling with adjusted dilation rates
def DilatedSpatialPyramidPooling(dspp_input):
    dims = list(dspp_input.shape)
    x = AveragePooling2D(pool_size=(dims[1], dims[2]))(dspp_input)
    x = convolution_block(x, num_filters=256, kernel_size=1, use_bias=True)
    out_pool = UpSampling2D(size=(dims[1] // x.shape[1], dims[2] // x.shape[2]), interpolation='bilinear')(x)

    out_1 = convolution_block(dspp_input, num_filters=256, kernel_size=1, dilation_rate=1)
    out_3 = convolution_block(dspp_input, num_filters=256, kernel_size=3, dilation_rate=3)  # Added smaller dilation rate
    out_9 = convolution_block(dspp_input, num_filters=256, kernel_size=3, dilation_rate=9)
    out_15 = convolution_block(dspp_input, num_filters=256, kernel_size=3, dilation_rate=15)
    out_21 = convolution_block(dspp_input, num_filters=256, kernel_size=3, dilation_rate=21)  # Added larger dilation rate

    # Apply Dropout layers to out_3, out_9, and out_15
    out_3 = Dropout(0.2)(out_3)
    out_9 = Dropout(0.2)(out_9)
    out_15 = Dropout(0.2)(out_15)

    x = concatenate([out_pool, out_1, out_3, out_9, out_15, out_21], axis=-1)  # Combine multi-scale features
    output = convolution_block(x, num_filters=256, kernel_size=1)
    return output

# Full Model without Morphological Post-Processing Layer
def DeeplabV3Plus(image_size, num_classes):
    input = Input(shape=(image_size, image_size, 3))
    resnet101 = ResNet101(weights="imagenet", include_top=False, input_tensor=input)

    x = resnet101.get_layer("conv4_block23_out").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]), interpolation="bilinear"
    )(x)
    input_b = resnet101.get_layer("conv2_block3_out").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    input_b = UpSampling2D(
        size=(input_a.shape[1] // input_b.shape[1], input_a.shape[2] // input_b.shape[2]),
        interpolation="bilinear",
    )(input_b)
    x = concatenate([input_a, input_b], axis=-1)

    x = convolution_block(x, num_filters=256)
    x = convolution_block(x, num_filters=256)
    x = convolution_block(x, num_filters=128)

    x = UpSampling2D(size=(image_size // x.shape[1], image_size // x.shape[2]), interpolation="bilinear")(x)
    output = Conv2D(num_classes, kernel_size=(1, 1), padding="same", activation="softmax")(x)

    return Model(inputs=input, outputs=output, name="DeepLabV3")


#MODEL TRAINING
train_generator = ImageDataGenerator(
    img_paths=train_img_paths,
    mask_paths=train_mask_paths,
    batch_size=8,
    augment=True,  # Augmentation is enabled for training
    num_classes=8,
    shuffle=True  # Shuffling enabled for training
)

val_generator = ImageDataGenerator(
    img_paths=val_img_paths,
    mask_paths=val_mask_paths,
    batch_size=8,
    augment=False,  # No augmentation for validation
    num_classes=8,
    shuffle=False  # No shuffling for validation (ensure deterministic behavior)
)

test_generator = ImageDataGenerator(
    img_paths=test_img_paths,
    mask_paths=test_mask_paths,
    batch_size=8,
    augment=False,  # No augmentation for testing
    num_classes=8,
    shuffle=False  # No shuffling for testing (ensure deterministic behavior)
)
def compute_class_weights(class_distribution):
    total_samples = np.sum(class_distribution)
    class_weights = total_samples / (len(class_distribution) * (class_distribution + 1e-6))  
    return class_weights

train_class_distribution = get_class_distribution(y_train)
val_class_distribution = get_class_distribution(y_val)
test_class_distribution = get_class_distribution(y_test)

train_class_weights = compute_class_weights(train_class_distribution)

print("Class weights for training set:", dict(zip(class_names, train_class_weights)))

val_class_weights = compute_class_weights(val_class_distribution)
test_class_weights = compute_class_weights(test_class_distribution)

print("Class weights for validation set:", dict(zip(class_names, val_class_weights)))
print("Class weights for test set:", dict(zip(class_names, test_class_weights)))

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
    latest_checkpoint_path = "/home/rdadmin/Tsiyon/NEWONE/MODELSAVED/DEEPV3RES101_latest.weights.h5"
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
        filepath="/home/rdadmin/Tsiyon/NEWONE/MODELSAVED/DEEPV3RES101_best.weights.h5",
        save_weights_only=True,
        monitor='val_dice_coef_metric',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    last_checkpoint = ModelCheckpoint(
        filepath="/home/rdadmin/Tsiyon/NEWONE/MODELSAVED/DEEPV3RES101_latest.weights.h5",
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
x_ticks = np.arange(1, len(smoothed_loss) + 1, step=1)

# Plot Training and Validation Loss
plt.plot(smoothed_epochs, smoothed_loss, 'y', label='Training loss')
plt.plot(smoothed_epochs, smoothed_val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(x_ticks)
plt.legend()
plt.savefig('/home/rdadmin/Tsiyon/NEWONE/RESULT/DEPRES101LOSS.png')
plt.close()

# Plot Training and Validation Dice Coefficient
plt.plot(smoothed_epochs, smoothed_dice_coef, 'c', label="Training")
plt.plot(smoothed_epochs, smoothed_val_dice_coef, 'r', label="Validation")
plt.title("Training Vs Validation Dice-Coef")
plt.xlabel("Epochs")
plt.ylabel("Dice-Coef")
plt.xticks(x_ticks)
plt.legend()
plt.savefig('/home/rdadmin/Tsiyon/NEWONE/RESULT/DEPRES101dice.png')
plt.close()

# Plot Training and Validation IOU (Jaccard Coefficient)
plt.plot(smoothed_epochs, smoothed_iou, 'c', label="Training")
plt.plot(smoothed_epochs, smoothed_val_iou, 'r', label="Validation")
plt.title("Training Vs Validation jac Metric")
plt.xlabel("Epochs")
plt.ylabel("IOU")
plt.xticks(x_ticks)
plt.legend()
plt.savefig('/home/rdadmin/Tsiyon/NEWONE/RESULT/DEPRES101JAC.png')
plt.close()

# Plot Training and Validation IOU Metric
plt.plot(smoothed_epochs, smoothed_iou1, 'c', label="Training")
plt.plot(smoothed_epochs, smoothed_val_iou1, 'r', label="Validation")
plt.title("Training Vs Validation IOU Metric")
plt.xlabel("Epochs")
plt.ylabel("IOU")
plt.xticks(x_ticks)
plt.legend()
plt.savefig('/home/rdadmin/Tsiyon/NEWONE/RESULT/DEPRES101IOU.png')
plt.close()



           
