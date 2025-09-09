# Enhanced Heart Disease Model Training - Multiple Models (Beginner Level)
# This code trains 5 different models and picks the best one

# Import the essential libraries
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load the data
print("Loading heart disease data...")
from pathlib import Path
data = pd.read_csv(Path(__file__).parent / "heart.csv")
print("Data loaded successfully!")
print(f"We have {len(data)} patients in our dataset")

# Step 2: Look at the data
print("\nLet's see what our data looks like:")
print(data.head())

# Step 3: Prepare the data
print("\nPreparing the data...")
data_copy = data.copy()

# Convert categorical variables to dummy variables
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
final_data = pd.get_dummies(data_copy, columns=categorical_columns)
print(f"After converting categories to numbers: {final_data.shape[1]} columns")

# Step 4: Scale the numerical columns
print("\nScaling numerical data...")
scaler = StandardScaler()
numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Fit scaler on all numerical columns at once
final_data[numerical_columns] = scaler.fit_transform(final_data[numerical_columns])

# Step 5: Separate input and output
print("\nSeparating patient info from heart disease diagnosis...")
X = final_data.drop('target', axis=1)
y = final_data['target']
print(f"We have {X.shape[1]} pieces of information about each patient")

# Step 6: Split data for training and testing
print("\nSplitting data for training and testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training on {len(X_train)} patients")
print(f"Testing on {len(X_test)} patients")

# Step 7: Create and test different models
print("\n" + "="*60)
print("TESTING DIFFERENT MODELS")
print("="*60)

# Store all model results
model_results = []

# Function to evaluate a model (helper function)
def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Train and evaluate a model, return results"""
    print(f"\nTesting {model_name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate different scores
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    # Print results
    print(f"  Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  Precision: {precision:.3f} (How many predicted diseases were actually diseases)")
    print(f"  Recall:    {recall:.3f} (How many actual diseases we caught)")
    print(f"  F1-Score:  {f1:.3f} (Overall balance of precision and recall)")
    
    return {
        'name': model_name,
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'overall_score': (accuracy + precision + recall + f1) / 4  # Simple average
    }

# Model 1: K-Nearest Neighbors (KNN)
print("\n1. K-NEAREST NEIGHBORS (KNN)")
print("   This model looks at similar patients to make predictions")

# Find best number of neighbors
best_knn_score = 0
best_k = 1
print("   Finding best number of neighbors to consider...")

for k in range(1, 11):
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    temp_score = accuracy_score(y_test, knn_temp.predict(X_test))
    
    if temp_score > best_knn_score:
        best_knn_score = temp_score
        best_k = k

print(f"   Best number of neighbors: {best_k}")

# Create final KNN model
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_result = evaluate_model(knn_model, "K-Nearest Neighbors", X_train, X_test, y_train, y_test)
model_results.append(knn_result)

# Model 2: Logistic Regression
print("\n2. LOGISTIC REGRESSION")
print("   This model uses mathematical equations to find patterns")

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_result = evaluate_model(lr_model, "Logistic Regression", X_train, X_test, y_train, y_test)
model_results.append(lr_result)

# Model 3: Support Vector Machine (SVM)
print("\n3. SUPPORT VECTOR MACHINE (SVM)")
print("   This model draws boundaries to separate healthy from disease patients")

svm_model = SVC(kernel='rbf', C=100, gamma=0.01, random_state=42, probability=True)
svm_result = evaluate_model(svm_model, "Support Vector Machine", X_train, X_test, y_train, y_test)
model_results.append(svm_result)

# Model 4: Decision Tree
print("\n4. DECISION TREE")
print("   This model asks yes/no questions like: 'Is age > 50? Is chest pain type 2?'")

dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_result = evaluate_model(dt_model, "Decision Tree", X_train, X_test, y_train, y_test)
model_results.append(dt_result)

# Model 5: Random Forest
print("\n5. RANDOM FOREST")
print("   This model combines many decision trees for better predictions")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_result = evaluate_model(rf_model, "Random Forest", X_train, X_test, y_train, y_test)
model_results.append(rf_result)

# Step 8: Compare all models and pick the best one
print("\n" + "="*80)
print("MODEL COMPARISON RESULTS")
print("="*80)

# Sort models by overall score
model_results.sort(key=lambda x: x['overall_score'], reverse=True)

print(f"{'Rank':<5} {'Model Name':<25} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1-Score':<9}")
print("-" * 80)

for i, result in enumerate(model_results, 1):
    print(f"{i:<5} {result['name']:<25} {result['accuracy']:.3f}      {result['precision']:.3f}       {result['recall']:.3f}    {result['f1_score']:.3f}")

# Get the best model
best_model_result = model_results[0]
best_model = best_model_result['model']
best_model_name = best_model_result['name']

print(f"\n🏆 WINNER: {best_model_name}")
print(f"   Overall Score: {best_model_result['overall_score']:.3f}")
print(f"   This model is correct {best_model_result['accuracy']*100:.1f}% of the time!")

# Step 9: Save the best model and alternatives
print(f"\nSaving models...")

# Create models folder
models_folder = "models"
if not os.path.exists(models_folder):
    os.makedirs(models_folder)

# Save the BEST model (this is what predict.py will use by default)
with open('models/heartmodel.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save the scaler (needed for preprocessing)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save column names (needed for preprocessing)
column_names = list(X.columns)
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(column_names, f)

# Save ALL models for comparison (optional - you can use different models later)
all_models = {}
for result in model_results:
    model_key = result['name'].lower().replace(' ', '_').replace('-', '_')
    all_models[model_key] = {
        'model': result['model'],
        'metrics': {
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1_score': result['f1_score']
        }
    }

with open('models/all_models.pkl', 'wb') as f:
    pickle.dump(all_models, f)

# Save model info for predict.py
model_info = {
    'best_model_name': best_model_name,
    'best_model_key': best_model_name.lower().replace(' ', '_').replace('-', '_'),
    'all_model_names': [result['name'] for result in model_results],
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_size': len(data),
    'feature_count': len(column_names)
}

with open('models/model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print("All models saved successfully!")

# Step 10: Final Summary
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Dataset size: {len(data)} patients")
print(f"Features used: {len(column_names)}")
print(f"Best Model: {best_model_name}")
print(f"Best Accuracy: {best_model_result['accuracy']*100:.1f}%")
print(f"Models trained: {len(model_results)}")

print(f"\nFiles saved:")
print(f"✅ models/heartmodel.pkl (Best model: {best_model_name})")
print(f"✅ models/scaler.pkl (Data preprocessing)")
print(f"✅ models/feature_names.pkl (Column names)")
print(f"✅ models/all_models.pkl (All 5 models)")
print(f"✅ models/model_info.pkl (Training information)")

print("\nYour heart disease prediction system is ready!")

# Step 11: Simple test
print(f"\n🧪 Testing the best model ({best_model_name}):")
sample_patient = X_test.iloc[0:1]
prediction = best_model.predict(sample_patient)[0]

try:
    probability = best_model.predict_proba(sample_patient)[0]
    confidence = max(probability) * 100
    
    if prediction == 1:
        print("✅ Prediction: Heart Disease Risk")
    else:
        print("✅ Prediction: Low Heart Disease Risk")
    print(f"✅ Confidence: {confidence:.1f}%")
    
except:
    # Some models don't support predict_proba
    if prediction == 1:
        print("✅ Prediction: Heart Disease Risk")
    else:
        print("✅ Prediction: Low Heart Disease Risk")

print(f"\nReady to use! Run predict.py to make predictions on new patients.")
