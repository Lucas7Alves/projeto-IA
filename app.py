from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D
from joblib import load
import os

app = Flask(__name__)

# Carregar o modelo Random Forest
MODEL_PATH = 'random_forest_pneumonia.pkl'
rf_model = load(MODEL_PATH)

def get_feature_extractor():
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

feature_extractor = get_feature_extractor()

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = preprocess_input(image) 
    image = np.expand_dims(image, axis=0)  
    
    
    features = feature_extractor.predict(image)
    return features

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/care", methods=["GET"])
def care():
    return render_template("care.html")

@app.route("/diagnostic", methods=["GET"])
def diagnostic():
    return render_template("diagnostic.html")

@app.route("/login", methods=["GET"])
def login():
    return render_template("login.html")

@app.route("/signup", methods=["GET"])
def signup():
    return render_template("signup.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo foi enviado"}), 400
    
    file = request.files["file"]
    
    try:
        # Pré-processamento
        image = Image.open(file).convert("RGB")
        features = preprocess_image(image)
        
        # Verificação crítica: shape das features
        print("Shape das features:", features.shape)  # Deve ser (1, 1024) para DenseNet121
        
        # Predição
        prediction = rf_model.predict(features)[0]  # Índice da classe (0, 1, ou 2)
        confidence = np.max(rf_model.predict_proba(features))  # Probabilidade máxima
        
        # Mapeamento das classes
        CLASSES = ['bacterial', 'normal', 'viral']
        label = CLASSES[prediction]
        
        return jsonify({
            "prediction": label,
            "confidence": float(confidence)  # Garante que é um float serializável
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)