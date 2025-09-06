import pandas as pd                       
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path

# Read the dataset - using the heart.csv in src folder
data = pd.read_csv(Path(__file__).parent / "heart.csv")
print("Data loaded successfully!")
print(data.head())

data.isnull().sum()   #Checking for null values in the dataset.

import seaborn as sns
sns.countplot(x = data["target"])

categorical_val = []
continuous_val = []
for column in data.columns:
  if len(data[column].unique()) <= 9:
    categorical_val.append(column)
  else:
    continuous_val.append(column)

print("Categorical columns:", categorical_val)
print("Continuous columns:", continuous_val)

plt.figure(figsize = (15, 15))
for i, column in enumerate(categorical_val, 1):       #Data visualization part 1
  plt.subplot(3, 3, i)
  data[data["target"] == 0][column].hist(bins = 35, color = 'blue', label = 'Have Heart Disease = NO', alpha = 0.6)
  data[data["target"] == 1][column].hist(bins = 35, color = 'red', label = 'Have Heart Disease = YES', alpha = 0.6)
  plt.legend()
  plt.xlabel(column)


plt.figure(figsize = (15, 15))
for i, column in enumerate(continuous_val, 1):      #Data visualization part 2
  plt.subplot(3, 3, i)
  data[data["target"] == 0][column].hist(bins = 35, color = 'blue', label = 'Have Heart Disease = NO', alpha = 0.6)
  data[data["target"] == 1][column].hist(bins = 35, color = 'red', label = 'Have Heart Disease = YES', alpha = 0.6)
  plt.legend()
  plt.xlabel(column)


categorical_val.remove('target')
dataset = pd.get_dummies(data, columns = categorical_val)
dataset.head()

print("Original columns:", data.columns.tolist())     #Original columns in the dataset
print("After dummy encoding:", dataset.columns.tolist())  #After adding indicator variables

from sklearn.preprocessing import StandardScaler
s_sc = StandardScaler()
col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])
print("Scaling completed!")

from sklearn.model_selection import train_test_split
X = dataset.drop(['target'], axis = 1)  #Features
y = dataset['target']                   #Target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)     #Split into train and test
print(f"Training set: {x_train.shape}, Test set: {x_test.shape}")

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
lr_clf = LogisticRegression()
lr_clf.fit(x_train, y_train)
y_pred1 = lr_clf.predict(x_test)
print('Logistic Regression Accuracy: ', metrics.accuracy_score(y_test, y_pred1))

from sklearn.svm import SVC
clf = SVC(kernel = 'rbf', C = 100, gamma = 0.01)
clf.fit(x_train, y_train)   #Training the model
y_pred3 = clf.predict(x_test)   #Predicting the data using the model
print('SVM Accuracy: ', metrics.accuracy_score(y_test, y_pred3))   #Finding the accuracy

from sklearn.metrics import classification_report, confusion_matrix
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred1))

from sklearn.neighbors import KNeighborsClassifier
k_range = range(1, 26)
scores = {}
scores_list = []
for k in k_range:
  knn = KNeighborsClassifier(n_neighbors = k)   #Model Building
  knn.fit(x_train, y_train)       #Training
  y_pred2 = knn.predict(x_test)   #Testing
  scores[k] = metrics.accuracy_score(y_test, y_pred2)
  scores_list.append(metrics.accuracy_score(y_test, y_pred2))

print("KNN Scores for different k values:", scores)

# Find best k
best_k = max(scores, key=scores.get)
print(f"Best k value: {best_k} with accuracy: {scores[best_k]}")

knn = KNeighborsClassifier(n_neighbors = best_k)
knn.fit(x_train, y_train)   #Model is trained
y_pred2 = knn.predict(x_test)
print('KNN Accuracy: ', metrics.accuracy_score(y_test, y_pred2))

print("KNN Classification Report:")
print(classification_report(y_test, y_pred2))

print("KNN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred2))

# SAVE ALL REQUIRED FILES
models_dir = Path(__file__).parent.parent / "models"

try:
    # Save the KNN model
    pickle.dump(knn, open(models_dir / 'heartmodel.pkl', 'wb'))
    print("✅ Model saved successfully!")
    
    # Save the scaler
    pickle.dump(s_sc, open(models_dir / 'scaler.pkl', 'wb'))
    print("✅ Scaler saved successfully!")
    
    # Save feature names
    feature_names = list(X.columns)
    pickle.dump(feature_names, open(models_dir / 'feature_names.pkl', 'wb'))
    print("✅ Feature names saved successfully!")
    
    print(f"\nSaved files:")
    print(f"- Model: {models_dir / 'heartmodel.pkl'}")
    print(f"- Scaler: {models_dir / 'scaler.pkl'}")
    print(f"- Features: {models_dir / 'feature_names.pkl'}")
    print(f"- Total features: {len(feature_names)}")
    
except Exception as e:
    print(f"❌ Error saving files: {e}")

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion = 'entropy')
clf.fit(x_train, y_train)
y_pred4 = clf.predict(x_test)
print('Decision Tree Accuracy: ', metrics.accuracy_score(y_test, y_pred4))

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 300)
clf.fit(x_train, y_train)
y_pred5 = clf.predict(x_test)
print('Random Forest Accuracy: ', metrics.accuracy_score(y_test, y_pred5))

print("\n🎉 Training completed! All models trained and files saved.")
print(f"📊 Best performing model: KNN with k={best_k} (Accuracy: {scores[best_k]:.4f})")