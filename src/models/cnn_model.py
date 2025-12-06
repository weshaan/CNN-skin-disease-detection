# ===========================================================
# SKIN CANCER DETECTION USING DENSENET121 (TRANSFER LEARNING)
# ===========================================================

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------
# PARAMETERS
# ---------------------------
IMG_SIZE = 224
BATCH = 32
EPOCHS = 10   # change as needed
TRAIN_DIR = "/content/data/train"    # change path if needed
VAL_DIR   = "/content/data/val"

# ---------------------------
# DATA PIPELINE
# ---------------------------
train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1/255.0)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical"
)

num_classes = len(train_data.class_indices)

# ---------------------------
# BUILD DENSENET MODEL
# ---------------------------
base = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze layers
for layer in base.layers:
    layer.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------------------
# TRAIN MODEL
# ---------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ---------------------------
# OPTIONAL FINE-TUNING (MUCH BETTER ACCURACY)
# ---------------------------
# Unfreeze last DenseNet blocks
for layer in base.layers[-60:]:  
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# ---------------------------
# SAVE MODEL
# ---------------------------
model.save("densenet_skin_cancer.h5")
print("Model saved as densenet_skin_cancer.h5")
