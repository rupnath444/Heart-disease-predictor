import pandas as pd
import pickle
import os
from pathlib import Path

class HeartDiseasePredictor:

    def __init__(self):
        self.best_model = None
        self.all_models = None
        self.scaler = None
        self.feature_names = None
        self.model_info = None
        self.load_models()

    def load_models(self):
        """Load saved models and preprocessing tools"""
        try:
            if os.path.exists("models"):
                model_dir = Path("models")
            else:
                model_dir = Path(__file__).parent.parent / "models"

            # Load best model
            self.best_model = pickle.load(open(model_dir / "heartmodel.pkl", "rb"))
            # Load scaler for data preprocessing
            self.scaler = pickle.load(open(model_dir / "scaler.pkl", "rb"))
            # Load feature names
            self.feature_names = pickle.load(open(model_dir / "feature_names.pkl", "rb"))

            # Try loading all models and info if available
            try:
                self.all_models = pickle.load(open(model_dir / "all_models.pkl", "rb"))
                self.model_info = pickle.load(open(model_dir / "model_info.pkl", "rb"))
                print(f"✅ All models loaded successfully!")
                print(f" Best model: {self.model_info['best_model_name']}")
                print(f" Available models: {len(self.all_models)}")
            except:
                print("⚠️ Only best model loaded (all_models.pkl not found)")
                self.all_models = None
                self.model_info = None

        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
        return True

    def preprocess_input(self, user_input):
        """Convert user input dict to model input format"""
        try:
            df = pd.DataFrame([user_input])
            categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
            df = pd.get_dummies(df, columns=categorical_cols)

            # Add missing columns with 0
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0

            # Reorder columns to match training
            df = df[self.feature_names]

            # Scale numeric columns
            numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            cols_to_scale = [col for col in numeric_cols if col in df.columns]
            if cols_to_scale:
                df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])

            return df
        except Exception as e:
            print(f"❌ Error preprocessing input: {e}")
            return None

    def predict(self, user_input, model_name=None):
        """Make prediction with specified model or best model"""
        try:
            processed = self.preprocess_input(user_input)
            if processed is None:
                return None

            if model_name is None:
                model = self.best_model
                model_display_name = self.model_info['best_model_name'] if self.model_info else "Best Model"
            else:
                if self.all_models is None:
                    print("⚠️ Only best model available. Using best model instead.")
                    model = self.best_model
                    model_display_name = "Best Model"
                elif model_name in self.all_models:
                    model = self.all_models[model_name]['model']
                    model_display_name = model_name.replace('_', ' ').title()
                else:
                    print(f"❌ Model '{model_name}' not found. Available models:")
                    for name in self.all_models.keys():
                        print(f" - {name}")
                    return None

            prediction = model.predict(processed)[0]

            try:
                prob = model.predict_proba(processed)[0]
                prob_no_disease = round(prob[0] * 100, 2)
                prob_disease = round(prob[1] * 100, 2)
                has_prob = True
            except:
                prob_no_disease = None
                prob_disease = None
                has_prob = False

            if has_prob:
                risk_level = self.get_risk_level(prob[1])
            else:
                risk_level = "High Risk" if prediction == 1 else "Low Risk"

            result = {
                'model_used': model_display_name,
                'prediction': int(prediction),
                'prediction_text': 'Heart Disease Risk' if prediction == 1 else 'Low Heart Disease Risk',
                'risk_level': risk_level,
                'has_probability': has_prob
            }

            if has_prob:
                result['probability_no_disease'] = prob_no_disease
                result['probability_disease'] = prob_disease

            return result

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None

    def predict_all_models(self, user_input):
        """Get predictions from all models for comparison"""
        if self.all_models is None:
            print("⚠️ Only best model available")
            return self.predict(user_input)

        results = {}
        print("🔄 Getting predictions from all models...")
        for model_name in self.all_models.keys():
            result = self.predict(user_input, model_name)
            if result:
                results[model_name] = result
        return results

    def get_risk_level(self, probability):
        """Return risk category based on probability"""
        if probability >= 0.7:
            return "High Risk"
        elif probability >= 0.4:
            return "Moderate Risk"
        else:
            return "Low Risk"

    def show_model_info(self):
        """Print info about models"""
        print("\n" + "="*50)
        print("AVAILABLE MODELS")
        print("="*50)

        if self.model_info:
            print(f"Best Model: {self.model_info['best_model_name']}")
            print(f"Training Date: {self.model_info['training_date']}")
            print(f"Dataset Size: {self.model_info['dataset_size']} patients")
            print(f"Features Used: {self.model_info['feature_count']}")

        if self.all_models:
            print(f"\nAll Available Models ({len(self.all_models)}):")
            for i, (key, data) in enumerate(self.all_models.items(), 1):
                print(f"{i}. {key.replace('_',' ').title()}")
                metrics = data['metrics']
                print(f" Accuracy: {metrics['accuracy']*100:.1f}%")
                print(f" Precision: {metrics['precision']*100:.1f}%")
                print(f" Recall: {metrics['recall']*100:.1f}%")
        else:
            print("Only best model loaded")

def main():
    print("Heart Disease Prediction System")
    print("="*40)

    predictor = HeartDiseasePredictor()
    predictor.show_model_info()

    sample_input = {
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

    print("\n" + "="*50)
    print("SAMPLE PREDICTION")
    print("="*50)

    print("Using best model:")
    result = predictor.predict(sample_input)
    if result:
        print(f"Model: {result['model_used']}")
        print(f"Prediction: {result['prediction_text']}")
        print(f"Risk Level: {result['risk_level']}")
        if result['has_probability']:
            print(f"Probability of Disease: {result['probability_disease']}%")
            print(f"Probability of No Disease: {result['probability_no_disease']}%")

    if predictor.all_models:
        print("\n" + "="*50)
        print("ALL MODELS COMPARISON")
        print("="*50)
        all_results = predictor.predict_all_models(sample_input)
        for model_name, res in all_results.items():
            print(f"\n{model_name.replace('_',' ').title()}:")
            print(f" Prediction: {res['prediction_text']}")
            print(f" Risk Level: {res['risk_level']}")
            if res['has_probability']:
                print(f" Disease Probability: {res['probability_disease']}%")

if __name__ == "__main__":
    main()
