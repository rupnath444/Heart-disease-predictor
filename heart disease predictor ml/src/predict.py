import pandas as pd
import pickle
from pathlib import Path

class HeartDiseasePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_models()

    def load_models(self):
        try:
            model_dir = Path(__file__).parent.parent / "models"
            self.model = pickle.load(open(model_dir / "heartmodel.pkl", "rb"))
            self.scaler = pickle.load(open(model_dir / "scaler.pkl", "rb"))
            self.feature_names = pickle.load(open(model_dir / "feature_names.pkl", "rb"))
            print("Models loaded successfully.")
        except Exception as e:
            print("Error loading models:", e)

    def preprocess_input(self, user_input):
        df = pd.DataFrame([user_input])
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        df = pd.get_dummies(df, columns=categorical_cols)
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]
        numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        return df

    def predict(self, user_input):
        try:
            processed = self.preprocess_input(user_input)
            prediction = self.model.predict(processed)[0]
            prob = self.model.predict_proba(processed)[0]
            risk_level = self.get_risk_level(prob[1])
            return {
                'prediction': int(prediction),
                'probability_no_disease': round(prob[0]*100, 2),
                'probability_disease': round(prob[1]*100, 2),
                'risk_level': risk_level
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def get_risk_level(self, probability):
        if probability >= 0.7:
            return "High Risk"
        elif probability >= 0.4:
            return "Moderate Risk"
        else:
            return "Low Risk"

if __name__ == "__main__":
    predictor = HeartDiseasePredictor()
    sample_input = {
        'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
        'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
        'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
    }
    result = predictor.predict(sample_input)
    if result:
        print(f"Prediction: {'Heart Disease' if result['prediction'] == 1 else 'No Heart Disease'}")
        print(f"Probability of Disease: {result['probability_disease']}%")
        print(f"Risk Level: {result['risk_level']}")
