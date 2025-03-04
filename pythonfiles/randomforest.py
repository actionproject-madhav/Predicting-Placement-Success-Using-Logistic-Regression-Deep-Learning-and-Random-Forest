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

# Load and preprocess data (same as before)
df = pd.read_csv('placementdata.csv')

# Convert binary columns
binary_columns = ['ExtracurricularActivities', 'PlacementTraining', 'PlacementStatus']
for col in binary_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'Placed': 1, 'NotPlaced': 0})

# Prepare data
X = df.drop(['StudentID', 'PlacementStatus'], axis=1)
y = df['PlacementStatus']

# Split data (same 80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing (same transformers)
numerical_cols = ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications',
                  'AptitudeTestScore', 'SoftSkillsRating', 'SSC_Marks', 'HSC_Marks']
binary_cols = ['ExtracurricularActivities', 'PlacementTraining']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('pass', 'passthrough', binary_cols)
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Initialize and train Random Forest
rf = RandomForestClassifier(n_estimators=100,
                           max_depth=5,
                           class_weight='balanced',
                           random_state=42)
rf.fit(X_train_processed, y_train)

# Make predictions
y_pred = rf.predict(X_test_processed)
y_proba = rf.predict_proba(X_test_processed)[:, 1]

# Evaluate performance
print("Random Forest Performance:")
print("=" * 60)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_proba):.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix plot
# Get feature importances and sort them
feature_importances = pd.DataFrame({
    'Feature': numerical_cols + binary_cols,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

# Select top N features (e.g., top 5)
top_features = feature_importances.head(5)['Feature'].tolist()
print("Top 5 Features:\n", top_features)