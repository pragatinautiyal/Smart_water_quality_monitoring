from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)
model = joblib.load("water_quality_model.pkl")
features = ['pH', 'TDS', 'Chlorine', 'EC', 'TOC', 'THM', 'Sulfate', 'Turbidity']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_future_quality():
    data = request.json
    prev = np.array(data['previous'])
    curr = np.array(data['current'])

    rate_of_change = (curr - prev)
    predictions = []
    unsafe_detected = False

    for step in range(1, 6):
        future = curr + step * rate_of_change
        future_df = pd.DataFrame([future], columns=features)
        pred = model.predict(future_df)[0]
        status = "Safe" if pred == 1 else "Unsafe"
        predictions.append({
            "step": step,
            "input": future.round(2).tolist(),
            "prediction": status
        })
        if status == "Unsafe":
            unsafe_detected = True

    result = {
        "future_predictions": predictions,
        "final_alert": "Unsafe water predicted!" if unsafe_detected else "Water remains safe."
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
