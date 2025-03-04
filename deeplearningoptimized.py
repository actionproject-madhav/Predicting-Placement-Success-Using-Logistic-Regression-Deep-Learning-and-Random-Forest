import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv('placementdata.csv')

# Convert binary columns
binary_columns = ['ExtracurricularActivities', 'PlacementTraining', 'PlacementStatus']
for col in binary_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'Placed': 1, 'NotPlaced': 0})

# Select only important features (same as RF selection)
top_features = [
    'CGPA',
    'Internships',
    'Projects',
    'AptitudeTestScore',
    'SSC_Marks'
]

# Prepare data with selected features
X = df[top_features]
y = df['PlacementStatus']

# Split data (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing setup
numerical_cols = ['CGPA', 'Internships', 'Projects', 'AptitudeTestScore', 'SSC_Marks']
binary_cols = []  # No binary features in top features

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols)
    ])

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Neural Network Architecture
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_processed.shape[1],)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train_processed, y_train,
                    validation_data=(X_test_processed, y_test),
                    epochs=20,
                    batch_size=32,
                    verbose=1)

# Evaluate on test set
y_pred_prob = model.predict(X_test_processed)
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate metrics
print("\nNeural Network Performance (Top Features Only)")
print("=" * 60)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_prob):.4f}\n")
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


plot_confusion_matrix(y_test, y_pred, "NN Confusion Matrix (Top Features)")

# Training history plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# Prediction function
def predict_placement(student_data):
    # Create DataFrame from input
    student_df = pd.DataFrame([student_data])

    # Preprocess features
    student_processed = preprocessor.transform(student_df[top_features])

    # Make prediction
    prediction_prob = model.predict(student_processed)[0][0]
    prediction = 'Placed' if prediction_prob > 0.5 else 'Not Placed'

    return {
        'prediction': prediction,
        'confidence': f"{prediction_prob:.1%}",
        'key_factors': {
            'CGPA': student_data['CGPA'],
            'Internships': student_data['Internships'],
            'Projects': student_data['Projects'],
            'AptitudeTestScore': student_data['AptitudeTestScore'],
            'SSC_Marks': student_data['SSC_Marks']
        }
    }


# Test prediction
sample_student = {
    'CGPA': 7.8,
    'Internships': 1,
    'Projects': 2,
    'AptitudeTestScore': 75,
    'SSC_Marks': 85
}

print("\nSample Neural Network Prediction:")
print(predict_placement(sample_student))