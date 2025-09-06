# ❤️ Heart Disease Risk Predictor

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive heart disease risk prediction system featuring both a **Flask web interface** and a **command-line tool**, powered by machine learning models trained on clinical heart disease datasets.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Web Application](#-web-application)
- [Command Line Tool](#-command-line-tool)
- [API Reference](#-api-reference)
- [Data Schema](#-data-schema)
- [Model Details](#-model-details)
- [Training Pipeline](#-training-pipeline)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Disclaimers](#-disclaimers)


---

## 🎯 Overview

This project provides a comprehensive heart disease risk assessment tool that combines:

- **Web Interface**: User-friendly browser-based form for clinical data input
- **REST API**: JSON endpoint for programmatic access and integration
- **CLI Tool**: Interactive terminal-based assessment with visual feedback
- **ML Pipeline**: Robust preprocessing and prediction using scikit-learn

The system uses standardized features and one-hot encoded categorical variables to provide accurate risk predictions with probability scores and interpretable risk levels.

---

## ✨ Features

- 🌐 **Interactive Web UI**: Clean, responsive interface for data collection and result visualization
- 🔗 **RESTful API**: Simple JSON API endpoint for seamless integration
- 💻 **Command Line Interface**: Interactive terminal tool with guided prompts
- 🧠 **Machine Learning**: Multiple model evaluation with optimized preprocessing pipeline
- 📊 **Risk Assessment**: Categorized risk levels (Low/Moderate/High) with probability scores
- 🔄 **Consistent Predictions**: Reusable predictor class ensures reproducible results
- 📈 **Visual Feedback**: Risk level indicators and probability displays

---

## 📂 Project Structure

```
project-root/
├── src/
│   ├── app.py              # Flask web server and API endpoints
│   ├── predict.py          # Core prediction class and inference logic
│   ├── train_model.py      # Model training and evaluation pipeline
│   ├── heart.csv           # Primary training dataset
│   └── Cleavelandheart.csv # Extended reference dataset
├── models/
│   ├── heartmodel.pkl      # Trained classifier (auto-generated)
│   ├── scaler.pkl          # Feature scaler (auto-generated)
│   └── feature_names.pkl   # Feature schema (auto-generated)
├── templates/
│   └── index.html          # Web interface template
├── static/
│   ├── script.js           # Frontend JavaScript logic
│   └── style.css           # UI styling and layout
├── test_prediction.py      # Interactive CLI prediction tool
├── requirements.txt        # Python dependencies
└── README.md              # This documentation
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd heart-disease-predictor
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   
   # Linux/macOS
   source .venv/bin/activate
   
   # Windows
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (first time setup)
   ```bash
   python src/train_model.py
   ```

---

## 🌐 Web Application

### Starting the Server

```bash
python src/app.py
```

The application will be available at `http://127.0.0.1:5000`

### Using the Web Interface

1. Navigate to the application URL in your browser
2. Fill out the medical assessment form with patient data
3. Click "Predict Risk" to get results
4. View the diagnosis, risk level, and probability score

---

## 💻 Command Line Tool

Launch the interactive CLI assessment:

```bash
python test_prediction.py
```

### Features

- **Guided Input**: Plain-language medical explanations for each parameter
- **Visual Feedback**: ASCII risk level bar display
- **Same Accuracy**: Uses identical model artifacts as the web application
- **User Friendly**: Clear prompts and validation for medical professionals

---

## 🔌 API Reference

### Prediction Endpoint

**POST** `/predict`

Accepts form data or JSON and returns risk prediction results.

#### Request Parameters

All parameters are required:

| Parameter | Type | Description | Valid Range |
|-----------|------|-------------|-------------|
| `age` | float | Patient age in years | 1-120 |
| `sex` | int | Gender (0=female, 1=male) | 0, 1 |
| `cp` | int | Chest pain type | 0-3 |
| `trestbps` | float | Resting blood pressure (mmHg) | 80-200 |
| `chol` | float | Serum cholesterol (mg/dl) | 100-600 |
| `fbs` | int | Fasting blood sugar >120 mg/dl | 0, 1 |
| `restecg` | int | Resting ECG results | 0-2 |
| `thalach` | float | Maximum heart rate achieved | 60-220 |
| `exang` | int | Exercise induced angina | 0, 1 |
| `oldpeak` | float | ST depression induced by exercise | 0.0-10.0 |
| `slope` | int | Slope of peak exercise ST segment | 0-2 |
| `ca` | int | Number of major vessels | 0-3 |
| `thal` | int | Thalassemia type | 1-3 |

#### Response Format

```json
{
  "success": true,
  "prediction": 1,
  "probability": 0.85,
  "risk_level": "High Risk",
  "message": "Prediction completed successfully"
}
```

#### Example Request

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "age=57&sex=1&cp=2&trestbps=130&chol=236&fbs=0&restecg=1&thalach=165&exang=0&oldpeak=1.0&slope=2&ca=0&thal=2"
```

---

## 📊 Data Schema

### Input Features

| Feature | Description | Type | Values |
|---------|-------------|------|---------|
| **age** | Age in years | Continuous | Numeric |
| **sex** | Gender | Categorical | 0=Female, 1=Male |
| **cp** | Chest pain type | Categorical | 0=Typical angina, 1=Atypical angina, 2=Non-anginal pain, 3=Asymptomatic |
| **trestbps** | Resting blood pressure | Continuous | mmHg |
| **chol** | Serum cholesterol | Continuous | mg/dl |
| **fbs** | Fasting blood sugar > 120 mg/dl | Binary | 0=False, 1=True |
| **restecg** | Resting ECG results | Categorical | 0=Normal, 1=ST-T wave abnormality, 2=LV hypertrophy |
| **thalach** | Maximum heart rate achieved | Continuous | BPM |
| **exang** | Exercise induced angina | Binary | 0=No, 1=Yes |
| **oldpeak** | ST depression induced by exercise | Continuous | Numeric |
| **slope** | Slope of peak exercise ST segment | Categorical | 0=Upsloping, 1=Flat, 2=Downsloping |
| **ca** | Number of major vessels colored by fluoroscopy | Ordinal | 0-3 |
| **thal** | Thalassemia | Categorical | 1=Normal, 2=Fixed defect, 3=Reversible defect |

### Risk Level Thresholds

- **High Risk**: Probability ≥ 0.7 (70%)
- **Moderate Risk**: Probability ≥ 0.4 (40%)
- **Low Risk**: Probability < 0.4 (40%)

---

## 🧠 Model Details

### Preprocessing Pipeline

1. **Feature Scaling**: StandardScaler applied to continuous variables
2. **Categorical Encoding**: One-hot encoding for categorical features
3. **Feature Alignment**: Consistent column ordering using saved feature schema

### Model Selection

The training pipeline evaluates multiple algorithms:

- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**
- **Support Vector Machine (RBF kernel)**
- **Decision Tree**
- **Random Forest**

The best performing model is automatically selected and saved.

### Model Artifacts

- `heartmodel.pkl`: Trained classifier
- `scaler.pkl`: Fitted StandardScaler
- `feature_names.pkl`: Column order for consistent preprocessing

---

## 🏋️ Training Pipeline

### Running Training

```bash
python src/train_model.py
```

### Training Process

1. **Data Loading**: Reads from `src/heart.csv`
2. **Preprocessing**: Feature scaling and encoding
3. **Model Evaluation**: Cross-validation across multiple algorithms
4. **Model Selection**: Best performer based on accuracy/F1-score
5. **Artifact Generation**: Saves model, scaler, and feature schema
6. **Performance Metrics**: Displays training results and model comparison

### Custom Training

To retrain with different parameters or datasets:

1. Modify `src/train_model.py`
2. Update dataset path or preprocessing steps
3. Run training script
4. New artifacts will be generated automatically

---

## 🚀 Deployment

### Local Development

```bash
# Development server
python src/app.py
```

### Production Deployment

For production deployment, consider:

1. **WSGI Server**: Use Gunicorn or uWSGI
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:8000 src.app:app
   ```

2. **Environment Variables**: Set Flask environment
   ```bash
   export FLASK_ENV=production
   ```

3. **Reverse Proxy**: Configure nginx for static file serving

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "src/app.py"]
```

---

## 🛠️ Troubleshooting

### Common Issues

**Model artifacts not found**
```bash
# Solution: Run training script
python src/train_model.py
```

**Template/static file errors**
- Ensure `app.py` is in `src/` directory
- Verify `templates/` and `static/` are in project root

**Dependency conflicts**
```bash
# Create fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate  # or fresh_env\Scripts\activate on Windows
pip install -r requirements.txt
```

**Port already in use**
- Change port in `app.py`: `app.run(port=5001)`
- Or kill existing process: `lsof -ti:5000 | xargs kill`

### Debug Mode

Enable Flask debug mode for development:

```python
# In src/app.py
app.run(debug=True, host='0.0.0.0', port=5000)
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make your changes
5. Run tests: `python -m pytest`
6. Commit changes: `git commit -m 'Add amazing feature'`
7. Push to branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

### Contribution Guidelines

- **Code Quality**: Follow PEP 8 style guidelines
- **Testing**: Add tests for new features
- **Documentation**: Update README and docstrings
- **Validation**: Ensure input validation and error handling
- **Performance**: Consider efficiency for large-scale usage

### Areas for Improvement

- Enhanced UI/UX design
- Additional model algorithms
- Comprehensive test coverage
- Input validation strengthening
- API rate limiting
- Logging and monitoring

---

## ⚠️ Disclaimers

**IMPORTANT MEDICAL DISCLAIMER**

- This software is for **educational and demonstration purposes only**
- **NOT a certified medical device or diagnostic tool**
- **NOT intended for clinical diagnosis or treatment decisions**
- All clinical applications must be supervised by qualified healthcare professionals
- Users assume full responsibility for any clinical interpretations
- Consult healthcare providers for medical advice and diagnosis

**Technical Limitations**

- Model trained on limited historical datasets
- Performance may vary across different populations
- Regular model updates and validation recommended
- No guarantee of prediction accuracy

---

## 📈 Roadmap

### Version 2.0 Features
- [ ] Docker containerization with docker-compose
- [ ] Separate requirements files (production vs development)
- [ ] Enhanced input validation and sanitization
- [ ] User authentication and session management
- [ ] Prediction history and analytics dashboard

### Version 3.0 Vision
- [ ] Real-time model retraining capabilities
- [ ] Multi-model ensemble predictions
- [ ] Advanced visualization and reporting
- [ ] Integration with EMR systems
- [ ] Mobile application development

---

## 🙏 Acknowledgments

- **Dataset Sources**: Heart disease datasets from UCI ML Repository
- **Medical Community**: Healthcare professionals who validate clinical approaches
- **Open Source**: scikit-learn, Flask, and Python ecosystem contributors
- **Research**: Academic papers on cardiovascular risk assessment

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/heart-disease-predictor/issues)
- **Documentation**: This README and code comments
- **Community**: Discussions and feature requests welcome

---

*Built with ❤️ for healthcare innovation and educational purposes*
