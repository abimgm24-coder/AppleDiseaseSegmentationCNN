from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join('saved_models', 'apple_disease_cnn.h5')
model = load_model(MODEL_PATH)

# Classes
classes = ['black_rot', 'healthy', 'rust', 'scab']

@app.route("/", methods=['GET','POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            os.makedirs('uploads', exist_ok=True)
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)
            img = image.load_img(filepath, target_size=(128,128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            pred = model.predict(img_array)
            prediction = classes[np.argmax(pred)]
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
