# 🚲 Bike Rental Demand Prediction

## Project Overview

The Bike Rental Demand Prediction project uses Machine Learning
to estimate the number of bikes that may be rented in one hour.

The prediction is based on different factors such as:

- Weather conditions
- Temperature
- Humidity
- Wind speed
- Season
- Date
- Time
- Holiday information

## Business Objective

The main objective is to predict bike rental demand so that
bike-sharing companies can improve bike availability, reduce
customer waiting time, and plan their operations effectively.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle
- Matplotlib
- Seaborn

## Machine Learning Algorithms

The following regression algorithms were evaluated:

1. Multiple Linear Regression
2. Support Vector Regression
3. Random Forest Regression
4. Gradient Boosting Regression

## Model Evaluation Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

## Project Features

- Interactive Streamlit dashboard
- Date and time selection
- Weather condition selection
- Temperature and feels-like temperature inputs
- Humidity and wind speed inputs
- Automatic feature engineering
- Bike rental demand prediction
- Prediction result and demand summary

## Project Structure

```text
Bike-Rental-Demand-Prediction/
│
├── app.py
├── bike_rental_model.pkl
├── requirements.txt
└── README.md