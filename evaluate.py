import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'saved_models/apple_disease_cnn.h5')
TEST_DIR = os.path.join(BASE_DIR, 'dataset/val')  # using val as test for demo

# Load model
model = load_model(MODEL_PATH)

# Class names
classes = ['black_rot', 'healthy', 'rust', 'scab']

# Evaluate one example image per class
for class_name in classes:
    class_folder = os.path.join(TEST_DIR, class_name)
    imgs = os.listdir(class_folder)
    if len(imgs) == 0:
        continue
    img_path = os.path.join(class_folder, imgs[0])
    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    pred_class = classes[np.argmax(pred)]
    print(f"Actual: {class_name}, Predicted: {pred_class}")
