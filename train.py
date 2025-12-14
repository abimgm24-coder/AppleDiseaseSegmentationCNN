import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset/train')
VAL_DIR = os.path.join(BASE_DIR, 'dataset/val')
SAVE_MODEL = os.path.join(BASE_DIR, 'saved_models/apple_disease_cnn.h5')

# Image settings
IMG_SIZE = (128,128)
BATCH_SIZE = 8

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Create model
model = create_model(input_shape=(128,128,3), num_classes=4)

# Train model
model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Save model
os.makedirs(os.path.dirname(SAVE_MODEL), exist_ok=True)
model.save(SAVE_MODEL)
print(f"Model saved at {SAVE_MODEL}")
