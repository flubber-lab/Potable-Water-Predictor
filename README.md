### Water Potability Predictor App

Welcome to the Water Potability Predictor App! This application is designed to predict whether a given water sample is potable (safe for drinking) based on various water quality parameters. The app is built using Streamlit and utilizes a Random Forest Classifier model trained on the water_potability.csv dataset.

## Table of Contents

Overview
Features
Installation
Usage
Model Training

Water potability is a critical factor in ensuring public health. This app allows users to input water quality parameters such as pH, hardness, solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes, and turbidity. The app then predicts whether the water is safe for drinking based on the trained Random Forest model.

## Features

**User-friendly Interface:** Built with Streamlit, the app provides an intuitive and easy-to-use interface.
**Real-time Prediction:** Users can input water quality parameters and get instant predictions on water potability.
**Model Insights:** The app uses a Random Forest Classifier, which is known for its accuracy and robustness in classification tasks.

### Installation

To run this app locally, follow these steps:

## Clone the Repository:
```
git clone https://github.com/flubber-lab/Potable-Water-Predictor.git
cd water-potability-prediction```

Set Up a Virtual Environment (optional but recommended):

```python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate` ```

## Install Dependencies:

`pip install -r requirements.txt`

## Run the App:

`streamlit run app.py`

## Access the App:

Open your web browser and go to http://localhost:8501 to use the app.

**Usage**

**Input Parameters:** Enter the values for the various water quality parameters in the input fields provided.
Predict: Click the "Predict" button to get the prediction on whether the water is potable.
Result: The app will display whether the water is safe for drinking or not.
Model Training

The Random Forest Classifier was trained using the water_potability.csv dataset. The dataset contains various water quality parameters and a target variable indicating whether the water is potable (1) or not (0).
```
### Steps for Model Training:

**Data Preprocessing:** Handle missing values, normalize data, and split the dataset into training and testing sets.
**Model Training:** Train a Random Forest Classifier on the training data.
**Model Evaluation:** Evaluate the model's performance using metrics such as accuracy, precision, recall, and F1-score.
**Model Saving:** Save the trained model using joblib or pickle for later use in the Streamlit app.

### Example Code for Model Training:

```
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv('water_potability.csv')

# Handle missing values
data.fillna(data.mean(), inplace=True)

# Split the data into features and target
X = data.drop('Potability', axis=1)
y = data['Potability']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Save the model
pickle.dump(model, 'water_potability_model.pkl')```
