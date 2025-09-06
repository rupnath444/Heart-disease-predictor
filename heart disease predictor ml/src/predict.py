import pandas as pd
import numpy as np
import pickle
from pathlib import Path

class HeartDiseasePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_models()
    
    def load_models(self):
        """Load the trained model, scaler, and feature names"""
        try:
            model_path = Path(__file__).parent.parent / "models"
            self.model = pickle.load(open(model_path / "heartmodel.pkl", 'rb'))
            self.scaler = pickle.load(open(model_path / "scaler.pkl", 'rb'))
            self.feature_names = pickle.load(open(model_path / "feature_names.pkl", 'rb'))
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def preprocess_input(self, user_input):
        """
        Preprocess user input to match training data format
        user_input: dictionary with user's health parameters
        """
        # Create DataFrame from user input
        df = pd.DataFrame([user_input])
        
        # Define categorical columns (same as training)
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        
        # Create dummy variables
        df = pd.get_dummies(df, columns=categorical_cols)
        
        # Ensure all feature columns are present (add missing ones with 0)
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match training data
        df = df[self.feature_names]
        
        # Scale continuous features
        continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        df[continuous_cols] = self.scaler.transform(df[continuous_cols])
        
        return df
    
    def predict(self, user_input):
        """
        Make prediction for user input
        Returns: (prediction, probability)
        """
        try:
            # Preprocess input
            processed_input = self.preprocess_input(user_input)
            
            # Make prediction
            prediction = self.model.predict(processed_input)[0]
            probability = self.model.predict_proba(processed_input)[0]
            
            return {
                'prediction': int(prediction),
                'probability_no_disease': round(probability[0] * 100, 2),
                'probability_disease': round(probability[1] * 100, 2),
                'risk_level': self.get_risk_level(probability[1])
            }
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def get_risk_level(self, probability):
        """Determine risk level based on probability"""
        if probability >= 0.7:
            return "High Risk"
        elif probability >= 0.4:
            return "Moderate Risk"
        else:
            return "Low Risk"

# Test the predictor
if __name__ == "__main__":
    predictor = HeartDiseasePredictor()
    
    # Example user input
    test_input = {
        'age': 63,
        'sex': 1,
        'cp': 3,
        'trestbps': 145,
        'chol': 233,
        'fbs': 1,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 2.3,
        'slope': 0,
        'ca': 0,
        'thal': 1
    }
    
    result = predictor.predict(test_input)
    if result:
        print(f"Prediction: {'Heart Disease' if result['prediction'] == 1 else 'No Heart Disease'}")
        print(f"Probability of Heart Disease: {result['probability_disease']}%")
        print(f"Risk Level: {result['risk_level']}")