import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model, scaler, and features
model = joblib.load('house_price_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')
model_features = joblib.load('model_features.pkl')

# Load the dataset to get unique locations for the form
df = pd.read_csv('Bengaluru_House_Data.csv')
locations = sorted(df['location'].dropna().unique())

@app.route('/')
def home():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    form_values = [x for x in request.form.values()]
    location = form_values[0]
    other_features = form_values[1:]
    
    # Prepare the input features
    input_features = [0] * len(model_features)
    for i, feature in enumerate(model_features):
        if feature == f'location_{location}':
            input_features[i] = 1
        elif feature in ['total_sqft', 'bath', 'balcony', 'bhk']:
            input_features[i] = float(other_features.pop(0))
    
    # Scale the numerical features
    input_features_scaled = scaler.transform([input_features[:4]])[0]
    input_features[:4] = input_features_scaled
    
    # Make prediction
    prediction = model.predict([input_features])

    output = int(round(prediction[0], 2))
    return render_template('index.html', prediction_text=f'Predicted House Price: ₹{output*100000}', locations=locations)

if __name__ == "__main__":
    app.run(debug=True)
