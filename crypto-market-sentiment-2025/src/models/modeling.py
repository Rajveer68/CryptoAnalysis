from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import os

# Load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess the data
def preprocess_data(data):
    # Handle missing values
    data.fillna(method='ffill', inplace=True)
    
    # Convert date column to datetime
    data['date'] = pd.to_datetime(data['date'])
    
    # Feature engineering: Extracting features from date
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    
    # Drop unnecessary columns
    data.drop(columns=['date'], inplace=True)
    
    return data

# Train a machine learning model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2

# Main function to execute the modeling process
def main():
    # Determine the correct file path
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(root_dir, 'Dataset', 'crypto_sentiment_prediction_dataset.csv')
    
    # Load and preprocess data
    data = load_data(file_path)
    processed_data = preprocess_data(data)
    
    # Define features and target variable
    X = processed_data.drop(columns=['current_price_usd'])
    y = processed_data['current_price_usd']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    mse, r2 = evaluate_model(model, X_test, y_test)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

if __name__ == "__main__":
    main()