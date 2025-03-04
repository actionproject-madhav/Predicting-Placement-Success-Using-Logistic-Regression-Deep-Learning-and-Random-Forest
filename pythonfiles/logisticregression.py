import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score)
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv('placementdata.csv')

# Convert binary columns
binary_columns = ['ExtracurricularActivities', 'PlacementTraining', 'PlacementStatus']
for col in binary_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'Placed': 1, 'NotPlaced': 0})

# Select important features
top_features = [
    'CGPA',
    'Internships',
    'Projects',
    'AptitudeTestScore',
    'SSC_Marks'
]

# Prepare data
X = df[top_features]
y = df['PlacementStatus']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numerical_cols = ['CGPA', 'Internships', 'Projects', 'AptitudeTestScore', 'SSC_Marks']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols)
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Logistic Regression Model
model = LogisticRegression(random_state=42)
model.fit(X_train_processed, y_train)

# Evaluate
y_pred = model.predict(X_test_processed)
y_pred_prob = model.predict_proba(X_test_processed)[:, 1]

print("\nLogistic Regression Performance")
print("=" * 60)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_prob):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Placed', 'Placed'],
                yticklabels=['Not Placed', 'Placed'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


plot_confusion_matrix(y_test, y_pred, "Logistic Regression Confusion Matrix")


# Prediction function
def predict_placement(student_data):
    student_df = pd.DataFrame([student_data])
    student_processed = preprocessor.transform(student_df[top_features])

    prediction_prob = model.predict_proba(student_processed)[0][1]
    prediction = 'Placed' if prediction_prob > 0.5 else 'Not Placed'

    return {
        'prediction': prediction,
        'confidence': f"{prediction_prob:.1%}",
        'key_factors': student_data
    }


# Test prediction
sample_student = {
    'CGPA': 7.8,
    'Internships': 1,
    'Projects': 2,
    'AptitudeTestScore': 75,
    'SSC_Marks': 85
}

print("\nSample Logistic Regression Prediction:")
print(predict_placement(sample_student))