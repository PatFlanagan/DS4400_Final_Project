Powerlifting Performance Prediction and Anomaly Detection

Overview

This project applies machine learning techniques to the OpenPowerlifting dataset to:
- Predict an athlete’s final competition total (TotalKg) using early attempt data and athlete characteristics
- Identify anomalous performances using prediction residuals

The goal is to support:
- Coaching decisions (predicting performance mid meet)
- Data quality analysis (detecting unusual or erroneous entries)


Project Structure

FinalProject/
  src/
    DataLoader.py
    linear_regression.py
    (other models)
  data/
    (dataset information stored here, actual dataset is downloaded from DataLoader.py)
  README.txt


IMPORTANT: Data Setup

This project uses the OpenPowerlifting dataset, which is very large (~700MB) and is NOT included in this repository.

You MUST run the data loader before running any models.

Run the following command:

    python src/DataLoader.py

This script will:
- Download the dataset from OpenPowerlifting
- Extract the zip file
- Move the CSV file into the data/ directory

After running, you should have:

data/
  openpowerlifting-<previously-updated-date>.csv


Running Models

After loading the dataset, you can run individual models.

Example (Linear Regression):

    python src/linear_regression.py

This will:
- Load the dataset
- Train a linear regression model
- Output:
  - MSE and R^2 metrics
  - Feature coefficients
  - Top anomalous predictions


Features Used (Baseline Model)

Input features:
- Squat1Kg
- Bench1Kg
- Deadlift1Kg
- BodyweightKg
- Age

Target variable:
- TotalKg


Methods

Implemented:
- Linear Regression (sklearn)
- Linear Regression (closed-form solution)

Planned / Optional:
- Ridge Regression
- Random Forest
- Gradient Boosting


Evaluation Metrics

- Mean Squared Error (MSE)
- R^2 Score


Anomaly Detection

Anomalies are identified using residuals:

    residual = actual - predicted

Large residuals may indicate:
- Unusual performances
- Potential data errors
- Missing contextual features


Notes

- The dataset is large, so only required columns are loaded for efficiency
- All features are standardized before training
- Sklearn and closed form implementations should produce nearly identical results
