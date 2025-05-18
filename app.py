from flask import Flask, render_template, redirect, url_for, request
import torch as tc
import torch.nn as nn 
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import io
import os
import base64

from SimpleFNN import SimpleFNN

app = Flask(__name__)

MODEL = 'mnist.pth'

INPUT_SIZE = 28*28
HIDDEN_SIZE = 128
NUM_CLASSES = 10

device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
print(f"Using Device : {device}")

try:
    model = SimpleFNN()
    model.load_state_dict(tc.load(MODEL, map_location=device))
    model.eval()
    
except FileNotFoundError:
    print(f"[ERROR] : Model file '{MODEL}' not found.")
    print("[INFO] : Please check the model path.")
    model = None

except Exception as e:
    print(f"[ERROR] : Error while loading the model : {e}")
    print("[INFO] : Please check the model file.")
    model = None
    
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('L')

        transform = transforms.Compose([
            transforms.Resize((28 , 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307), (0.3081))
        ])

        img_tensor = transform(img)

        # img_tensor = img_tensor.view(1, 1, 28, 28)
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor
    except Exception as e:
        print(f"[ERROR] : Error while preprocessing the image : {e}")
        return None
    
def predict_digit(image_tensor, model):
    if image_tensor is None or model is None:
        return -1, 0.0
    
    try:
        model.eval()
        with tc.no_grad():
            outputs = model(image_tensor)
            print(f"[DEBUG] Raw model outputs (logits): {outputs}")
            
            probabilities = F.softmax(outputs, dim=1)
            print(f"[DEBUG] Probabilities for each class: {probabilities.cpu().numpy().tolist()}")
            
            _, predicted = tc.max(outputs.data, 1)
            
            confidence = probabilities[0][predicted.item()].item()
            
        # Return the predicted digit, its confidence, and the full probability distribution for debugging
        return predicted.item(), confidence, probabilities.cpu().numpy().flatten().tolist()
    
    except Exception as e:
        print(f"[ERROR] : Error while predicting the digit : {e}")
        return -1, 0.0, []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return redirect(url_for('index'))
    
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file =request.files['file']
    
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        image_bytes = file.read()
        mime_type = file.mimetype
        
        img_base64 = base64.b64encode(image_bytes).decode('utf-8')
        image_data_url = f"data:{mime_type};base64,{img_base64}"
        
        processed_image_tensor = preprocess_image(image_bytes)
        
        if processed_image_tensor is not None:
            predicted_digit, confidence, all_probabilities = predict_digit(processed_image_tensor, model) # Modified to get all_probabilities
            
            if predicted_digit != -1:
                return render_template('index.html', predicted_digit=predicted_digit, confidence=confidence, image_data_url=image_data_url, all_probabilities=all_probabilities)
            else:
                return render_template('index.html', error="Error during prediction.", image_data_url=image_data_url, all_probabilities=all_probabilities if 'all_probabilities' in locals() else [])
        else:
            return render_template('index.html', error="Error in image preprocessing.", image_data_url=image_data_url)
        
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
    