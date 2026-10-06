# AI Industrial Maintenance Copilot

An AI-based predictive maintenance system that predicts the Remaining Useful Life (RUL) of industrial machinery from sensor data and provides a maintenance condition assessment.

## Objective

The objective of this project is to estimate how many operational cycles a machine has remaining before failure and use the prediction to support maintenance decisions.

## Dataset

The project uses the NASA C-MAPSS FD004 turbofan engine dataset.

The dataset contains:

- Engine identifiers
- Operating cycles
- Operating settings
- Multiple sensor measurements
- Remaining Useful Life (RUL)

## Approach

The project follows this pipeline:

Dataset
→ Data Analysis
→ Sensor Selection
→ Feature Engineering
→ Data Cleaning
→ Train/Test Split
→ Feature Scaling
→ Model Training
→ Model Evaluation
→ Model Saving
→ Streamlit Deployment

## Feature Engineering

Five important sensors were selected based on statistical analysis and correlation:

- sensor_14
- sensor_13
- sensor_15
- sensor_11
- sensor_4

For each selected sensor, additional features were created:

- Rolling mean
- Rolling standard deviation
- Trend

This resulted in 20 model features.

## Models

Three regression approaches were evaluated:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

A tuned Random Forest Regressor was selected as the final model.

## Final Model Performance

Tuned Random Forest:

- MAE: 45.05 cycles
- RMSE: 61.82 cycles
- R²: 0.5142

## Important Features

The most influential features in the final model were:

| Feature | Importance |
|---|---:|
| sensor_13 | 0.247005 |
| sensor_11 | 0.126967 |
| sensor_4 | 0.105417 |
| sensor_15 | 0.091204 |
| sensor_14 | 0.082514 |

## Maintenance Assessment

The predicted RUL is converted into a maintenance condition:

| Predicted RUL | Condition |
|---:|---|
| ≤ 30 cycles | CRITICAL |
| 31–75 cycles | WARNING |
| 76–150 cycles | MONITOR |
| > 150 cycles | NORMAL |

The application also provides a maintenance recommendation based on the predicted condition.

## Deployment

The trained model is integrated into a Streamlit web application.

The application provides:

- Sensor input interface
- RUL prediction
- Machine condition assessment
- Maintenance recommendation
- Feature importance visualization
- Model interpretation

## Saved Model Components

The following files are used by the application:

- `tuned_random_forest.pkl` — trained Random Forest model
- `scaler.pkl` — feature scaler
- `feature_names.pkl` — model feature order

## Example Prediction

For one test sample:

- Actual RUL: 220 cycles
- Predicted RUL: 175.48 cycles

The application classified the machine condition as:

**NORMAL**

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt 