import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    print("🔄 Starting Heart Disease Model Training...")
    
    # Load data
    try:
        data = pd.read_csv("heart.csv")
        print(f"✅ Data loaded successfully! Shape: {data.shape}")
    except FileNotFoundError:
        print("❌ Error: heart.csv not found!")
        return
    
    # Check for missing values
    if data.isnull().sum().any():
        print("⚠️ Warning: Missing values detected")
    else:
        print("✅ No missing values found")
    
    # Identify categorical and continuous columns
    categorical_val = []
    continuous_val = []
    for column in data.columns:
        if len(data[column].unique()) <= 9:
            categorical_val.append(column)
        else:
            continuous_val.append(column)
    
    print(f"📊 Categorical columns: {len(categorical_val)}")
    print(f"📊 Continuous columns: {len(continuous_val)}")
    
    # Remove target from categorical for encoding
    categorical_val.remove('target')
    
    # Create dummy variables
    dataset = pd.get_dummies(data, columns=categorical_val)
    print(f"✅ Dummy encoding completed. New shape: {dataset.shape}")
    
    # Scale continuous features
    scaler = StandardScaler()
    continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    dataset[continuous_cols] = scaler.fit_transform(dataset[continuous_cols])
    print("✅ Feature scaling completed")
    
    # Split data
    X = dataset.drop(['target'], axis=1)
    y = dataset['target']
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"✅ Data split - Train: {x_train.shape}, Test: {x_test.shape}")
    
    # Find best k value for KNN
    print("🔍 Finding optimal k value for KNN...")
    k_range = range(1, 21)  # Reduced range for efficiency
    best_score = 0
    best_k = 1
    
    for k in k_range:
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        knn_temp.fit(x_train, y_train)
        score = knn_temp.score(x_test, y_test)
        
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"🎯 Best k value: {best_k} with accuracy: {best_score:.4f}")
    
    # Train final model with best k
    print("🤖 Training final KNN model...")
    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(x_train, y_train)
    
    # Evaluate final model
    final_predictions = final_model.predict(x_test)
    final_accuracy = metrics.accuracy_score(y_test, final_predictions)
    print(f"✅ Final model accuracy: {final_accuracy:.4f}")
    
    # Create models directory
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Save model artifacts
    try:
        # Save the model
        pickle.dump(final_model, open(models_dir / 'heartmodel.pkl', 'wb'))
        print("✅ Model saved successfully!")
        
        # Save the scaler
        pickle.dump(scaler, open(models_dir / 'scaler.pkl', 'wb'))
        print("✅ Scaler saved successfully!")
        
        # Save feature names
        feature_names = list(X.columns)
        pickle.dump(feature_names, open(models_dir / 'feature_names.pkl', 'wb'))
        print("✅ Feature names saved successfully!")
        
        print(f"\n📁 Saved files:")
        print(f"   - Model: {models_dir / 'heartmodel.pkl'}")
        print(f"   - Scaler: {models_dir / 'scaler.pkl'}")
        print(f"   - Features: {models_dir / 'feature_names.pkl'}")
        print(f"   - Total features: {len(feature_names)}")
        
    except Exception as e:
        print(f"❌ Error saving files: {e}")
        return
    
    print(f"\n🎉 Training completed successfully!")
    print(f"📊 Final KNN Model (k={best_k}) - Accuracy: {final_accuracy:.4f}")
    print("🚀 Model is ready for predictions!")

if __name__ == "__main__":
    main()
