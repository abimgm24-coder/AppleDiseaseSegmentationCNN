# Apple Disease Detection using CNN

This project detects diseases in apple leaves using Convolutional Neural Networks.

## Folder Structure

- classification: contains model, training, and evaluation scripts
- segmentation: (optional) leaf segmentation scripts
- templates: HTML files for Flask web app
- dataset: train and validation images
- saved_models: trained model files

## How to Run

1. Install dependencies: `pip install -r requirements.txt`
2. Train model: `python classification/train.py`
3. Run Flask app: `python app.py`
4. Open browser: `http://127.0.0.1:5000`
