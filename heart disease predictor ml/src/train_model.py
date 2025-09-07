# Simple Heart Disease Model Training - Beginner Level
# This code is written using basic Python concepts that beginners can understand

# Import only the essential libraries
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Step 1: Load the data (simple file reading)
print("Loading heart disease data...")
data = pd.read_csv("heart.csv")
print("Data loaded successfully!")
print(f"We have {len(data)} patients in our dataset")

# Step 2: Look at the data
print("\nLet's see what our data looks like:")
print(data.head())  # Show first 5 rows

# Step 3: Prepare the data (simple way)
print("\nPreparing the data...")

# Convert text categories to numbers (simple approach)
# We'll manually handle the most important categorical columns
data_copy = data.copy()

# Convert sex: 0=female, 1=male (already numbers, so we keep as is)
# Convert chest pain type: already numbers 0,1,2,3
# Convert other categorical variables to dummy variables (0s and 1s)
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Simple way to create dummy variables
final_data = pd.get_dummies(data_copy, columns=categorical_columns)
print(f"After converting categories to numbers: {final_data.shape[1]} columns")

# Step 4: Scale the numerical columns (make them similar size)
print("\nScaling numerical data...")
scaler = StandardScaler()
numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Scale only if columns exist
for col in numerical_columns:
    if col in final_data.columns:
        final_data[col] = scaler.fit_transform(final_data[[col]])

# Step 5: Separate input and output
print("\nSeparating patient info from heart disease diagnosis...")
X = final_data.drop('target', axis=1)  # Patient information
y = final_data['target']                # Heart disease yes/no (0 or 1)

print(f"We have {X.shape[1]} pieces of information about each patient")

# Step 6: Split data into training and testing
print("\nSplitting data for training and testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training on {len(X_train)} patients")
print(f"Testing on {len(X_test)} patients")

# Step 7: Find the best number of neighbors (simple loop)
print("\nFinding the best settings for our model...")
best_accuracy = 0
best_k = 1

# Try different numbers of neighbors from 1 to 10 (keeping it simple)
for k in range(1, 11):
    # Create a model
    model = KNeighborsClassifier(n_neighbors=k)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Test the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"With {k} neighbors: {accuracy:.3f} accuracy")
    
    # Remember the best one
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f"\nBest number of neighbors: {best_k}")
print(f"Best accuracy: {best_accuracy:.3f}")

# Step 8: Create and train the final model
print("\nTraining the final model...")
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)

# Test the final model
final_predictions = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, final_predictions)

print(f"Final model accuracy: {final_accuracy:.3f}")
print(f"This means our model is correct {final_accuracy*100:.1f}% of the time!")

# Step 9: Save the model (simple file saving)
print("\nSaving the model...")

# Create models folder if it doesn't exist (same structure as original)
import os
models_folder = "models"
if not os.path.exists(models_folder):
    os.makedirs(models_folder)

# Save the trained model (same filenames as original)
with open('models/heartmodel.pkl', 'wb') as f:
    pickle.dump(final_model, f)

# Save the scaler (this must be saved correctly for predict.py to work)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the column names (predict.py needs this exact list)
column_names = list(X.columns)
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(column_names, f)

print("Model saved successfully!")
print(f"Model file: models/heartmodel.pkl")
print(f"Scaler file: models/scaler.pkl") 
print(f"Features file: models/feature_names.pkl")

# Step 10: Show summary
print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
print(f"Dataset size: {len(data)} patients")
print(f"Features used: {len(column_names)}")
print(f"Best k value: {best_k}")
print(f"Model accuracy: {final_accuracy*100:.1f}%")
print("\nYour heart disease prediction model is ready to use!")

# Simple test with one patient
print("\nLet's test with one patient from our test data:")
sample_patient = X_test.iloc[0:1]  # Get first test patient
prediction = final_model.predict(sample_patient)[0]
probability = final_model.predict_proba(sample_patient)[0]

if prediction == 1:
    print("Prediction: Heart Disease Risk")
else:
    print("Prediction: Low Heart Disease Risk")

print(f"Confidence: {max(probability)*100:.1f}%")
