from flask import Flask, render_template, request, jsonify
import sys
from pathlib import Path
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add project path
sys.path.append(str(Path(__file__).parent))
from predict import HeartDiseasePredictor

# Flask app
app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Initialize predictor
predictor = HeartDiseasePredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = {
            'age': float(request.form['age']),
            'sex': int(request.form['sex']),
            'cp': int(request.form['cp']),
            'trestbps': float(request.form['trestbps']),
            'chol': float(request.form['chol']),
            'fbs': int(request.form['fbs']),
            'restecg': int(request.form['restecg']),
            'thalach': float(request.form['thalach']),
            'exang': int(request.form['exang']),
            'oldpeak': float(request.form['oldpeak']),
            'slope': int(request.form['slope']),
            'ca': int(request.form['ca']),
            'thal': int(request.form['thal'])
        }

        result = predictor.predict(user_input)
        if result:
            return jsonify({
                'success': True,
                'prediction': result['prediction'],
                'probability': result['probability_disease'],
                'risk_level': result['risk_level'],
                'message': 'Prediction successful'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Error making prediction'
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

# --------------------------
if __name__ == "__main__":
    # Run Flask only on localhost
    app.run(host='127.0.0.1', port=5000, debug=True)
