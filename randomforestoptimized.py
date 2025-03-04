import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score)
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('placementdata.csv')

# Convert binary columns
binary_columns = ['ExtracurricularActivities', 'PlacementTraining', 'PlacementStatus']
for col in binary_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'Placed': 1, 'NotPlaced': 0})

# Select only important features (adjust based on your feature importance results)
top_features = [
    'CGPA',  # Typically most important feature
    'Internships',  # 2nd most important
    'Projects',  # 3rd important
    'AptitudeTestScore',  # 4th important
    'SSC_Marks'  # 5th important
]

# Prepare data with selected features
X = df[top_features]
y = df['PlacementStatus']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing setup
numerical_cols = [col for col in top_features if col in [
    'CGPA', 'Internships', 'Projects',
    'Workshops/Certifications', 'AptitudeTestScore',
    'SoftSkillsRating', 'SSC_Marks', 'HSC_Marks'
]]

binary_cols = [col for col in top_features if col in [
    'ExtracurricularActivities', 'PlacementTraining'
]]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('pass', 'passthrough', binary_cols)
    ])

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Initialize and train optimized Random Forest
rf_optimized = RandomForestClassifier(
    n_estimators=150,
    max_depth=7,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42
)
rf_optimized.fit(X_train_processed, y_train)

# Make predictions
y_pred = rf_optimized.predict(X_test_processed)
y_proba = rf_optimized.predict_proba(X_test_processed)[:, 1]

# Evaluate performance
print("Optimized Random Forest Performance (Top Features Only)")
print("=" * 60)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_proba):.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))


# Confusion Matrix plot
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


plot_confusion_matrix(y_test, y_pred, "Optimized RF Confusion Matrix")

# Feature Importance visualization
importances = rf_optimized.feature_importances_
feature_names = numerical_cols + binary_cols

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names, palette='viridis')
plt.title('Optimized Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()


# Example prediction function using important features
def predict_placement(student_data):
    # Create DataFrame from input
    student_df = pd.DataFrame([student_data])

    # Select and preprocess features
    student_processed = preprocessor.transform(student_df[top_features])

    # Make prediction
    prediction = rf_optimized.predict(student_processed)
    probability = rf_optimized.predict_proba(student_processed)[0][1]

    return {
        'prediction': 'Placed' if prediction[0] == 1 else 'Not Placed',
        'probability': f"{probability:.1%}",
        'decisive_factors': dict(zip(top_features, student_df[top_features].values[0]))
    }


# Test sample prediction
sample_student = {
    'CGPA': 8.2,
    'Internships': 2,
    'Projects': 3,
    'AptitudeTestScore': 88,
    'SSC_Marks': 92
}

print("\nSample Prediction:")
print(predict_placement(sample_student))