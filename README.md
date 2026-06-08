# Smart Water Quality Monitoring System

Smart Water Quality Monitoring System is an AI-powered web application designed to analyze and predict future water quality conditions using historical water parameter trends.

The system combines machine learning, predictive analytics, database storage, and web development to forecast water safety and classify water quality based on chemical parameters.

The project is split into two parts:

* **backend** – Flask + Random Forest model
* **frontend** – HTML, CSS, JavaScript interface

---

## Features

* Water quality prediction using machine learning
* Future water quality forecasting for **5 prediction steps**
* Multi-class water quality classification:

  * Excellent
  * Good
  * Fair
  * Moderate
  * Poor
  * Unsuitable
* Historical trend-based future prediction
* Unsafe water detection alerts
* Prediction history tracking using SQLite
* Automatic derived feature generation
* Real-time prediction through an interactive web interface

---

## Tech Stack

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Python
* Flask

### Machine Learning

* Random Forest
* Scikit-learn

### Data Processing

* NumPy
* Pandas

### Database

* SQLite

### Model Storage

* Joblib

---

## Project Structure

```text
Smart_water_quality_monitoring/
├── backend/
│   ├── app.py
│   ├── water_quality_monitor.db
│
├── datasets/
│   ├── Filtered_water_parameters.csv
│   ├── WQI_Results_on_Dataset.csv
│
├── frontend/
│   ├── index.html
│
├── model/
│   ├── data_cleaning_01.ipynb
│   ├── data_cleaning_02.ipynb
│   ├── data_cleaning_03.ipynb
│   ├── train_model.ipynb
│   ├── model.zip
│
└── README.md
```

---

## Local Setup

### 1. Clone and open the project

```bash
git clone https://github.com/pragatinautiyal/Smart_water_quality_monitoring.git
cd Smart_water_quality_monitoring
```

### 2. Install dependencies

Install required Python libraries:

```bash
pip install flask numpy pandas scikit-learn joblib
```

### 3. Run the application

Navigate to the backend folder and start the Flask server:

```bash
cd backend
python app.py
```

Application runs on:

```text
http://127.0.0.1:5000
```

---

## How It Works

### 1. User Input

Users enter:

* Previous water parameter values
* Current water parameter values

Input features include:

* Ammonia-Total (as N)
* Conductivity @25°C
* pH
* Total Hardness (as CaCO3)

### 2. Trend Calculation

The system calculates the **rate of change** between previous and current readings.

### 3. Future Prediction

Based on the calculated trend, the system predicts water quality for the next **5 future steps**.

### 4. Feature Engineering

A derived feature (`HighCond_LowHard`) is automatically computed to improve prediction performance.

### 5. Machine Learning Classification

The trained **Random Forest model** classifies future water quality into:

* Excellent
* Good
* Fair
* Moderate
* Poor
* Unsuitable

### 6. Output Results

The system returns:

* Future water quality predictions
* Water safety status
* Unsafe water alerts
* Prediction history

---

## Dataset Information

The project uses processed water quality datasets stored in CSV format.

Dataset includes:

* **8500+ water quality records**
* Water Quality Index (WQI) labels
* Water chemical parameters
* Feature-engineered attributes
* Multi-class water quality categories

---

## Available Routes

### Home Page

```http
GET /
```

Loads the water quality monitoring interface.

---

### Predict Future Water Quality

```http
POST /predict
```

#### Request Body

```json
{
  "previous": [45, 120, 7.2, 350],
  "current": [50, 135, 7.5, 320]
}
```

#### Available Input Features

* Ammonia-Total (as N)
* Conductivity @25°C
* pH
* Total Hardness (as CaCO3)

---

### View Prediction History

```http
GET /history
```

Returns the latest prediction records stored in SQLite.

---

## Database Storage

All predictions are stored in a **SQLite database** with:

* Request ID
* Timestamp
* Water parameter values
* Prediction result
* Safety alert

This enables historical monitoring and analysis of water quality trends.

---

## Model Pipeline

The machine learning workflow includes:

1. Data Cleaning
2. Feature Engineering
3. Model Training using Random Forest
4. Model Serialization using Joblib
5. Flask-based Deployment

---

## Future Improvements

* Real-time sensor integration
* Interactive analytics dashboard
* Cloud database integration
* Graph-based water trend visualization
* SMS/Email alert system
* Additional water quality parameters
* Better model monitoring and evaluation

---

