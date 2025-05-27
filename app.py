from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import sqlite3
from datetime import datetime
import uuid  # For unique request IDs

app = Flask(__name__)

# Load trained model
model = joblib.load("model.joblib")

# Features used during training
feature_columns = [
    'Ammonia-Total (as N)',
    'Conductivity @25°C',
    'pH',
    'Total Hardness (as CaCO3)',
    'HighCond_LowHard'
]

# Class labels
label_mapping = {
    0: 'Excellent',
    1: 'Fair',
    2: 'Good',
    3: 'Moderate',
    4: 'Poor',
    5: 'Unsuitable'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_future_quality():
    data = request.json
    prev = np.array(data['previous'])
    curr = np.array(data['current'])

    rate_of_change = curr - prev
    unsafe_detected = False
    predictions = []

    # Generate unique request ID for this batch
    request_id = str(uuid.uuid4())

    # Open DB connection
    conn = sqlite3.connect('water_quality_monitor.db')
    cursor = conn.cursor()

    for step in range(1, 6):
        future = curr + step * rate_of_change
        future_df = pd.DataFrame([{
            'Ammonia-Total (as N)': future[0],
            'Conductivity @25°C': future[1],
            'pH': future[2],
            'Total Hardness (as CaCO3)': future[3],
        }])

        # Compute derived feature
        future_df['HighCond_LowHard'] = (
            (future_df['Conductivity @25°C'] > 0) &
            (future_df['Total Hardness (as CaCO3)'] < 300)
        ).astype(int)

        future_df = future_df[feature_columns]

        prediction = model.predict(future_df)[0]
        label = label_mapping.get(prediction, str(prediction))

        predictions.append({
            "step": step,
            "input": future.round(2).tolist(),
            "prediction": label
        })

        if prediction >= 4:
            unsafe_detected = True

        # Insert each prediction step into DB, final_alert stored only on last step
        cursor.execute('''
            INSERT INTO water_quality (
                request_id,
                timestamp,
                ammonia_total,
                conductivity,
                pH,
                total_hardness,
                highcond_lowhard,
                prediction,
                final_alert
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            request_id,
            datetime.now().isoformat(),
            float(future[0]), float(future[1]),
            float(future[2]), float(future[3]),
            int(future_df['HighCond_LowHard'][0]),
            label,
            "⚠️ Unsafe water predicted!" if unsafe_detected and step == 5 else None
        ))

    conn.commit()
    conn.close()

    result = {
        "future_predictions": predictions,
        "final_alert": "⚠️ Unsafe water predicted!" if unsafe_detected else "✅ Water remains safe."
    }

    return jsonify(result)

@app.route('/history', methods=['GET'])
def get_history():
    conn = sqlite3.connect('water_quality_monitor.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT request_id, timestamp, ammonia_total, conductivity, pH, total_hardness, highcond_lowhard, prediction, final_alert
        FROM water_quality
        ORDER BY timestamp DESC
        LIMIT 20
    ''')
    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            "request_id": row[0],
            "timestamp": row[1],
            "ammonia_total": row[2],
            "conductivity": row[3],
            "pH": row[4],
            "total_hardness": row[5],
            "highcond_lowhard": row[6],
            "prediction": row[7],
            "final_alert": row[8]
        })

    return jsonify({"history": history})

if __name__ == '__main__':
    app.run(debug=True)
