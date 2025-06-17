import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load data
# Load raw CSV
df = pd.read_csv('Training/coords.csv', header=None)  # Disable header since it's not properly saved

# Create expected column names
pose_columns = [f'pose_{i}_{axis}' for i in range(33) for axis in ['x', 'y', 'z', 'v']]
face_columns = [f'face_{i}_{axis}' for i in range(468) for axis in ['x', 'y', 'z', 'v']]
expected_columns = ['class'] + pose_columns + face_columns

# Set column names
df.columns = expected_columns


# The first column contains labels
X = df[expected_columns[1:]]
y = df['class']



# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Ridge Classifier': RidgeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

best_score = 0
best_model = None
best_model_name = ""

# Train and evaluate models
for name, model in models.items():
    pipeline = make_pipeline(StandardScaler(), model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.2f}")
    
    
    if acc > best_score:
        best_score = acc
        best_model = pipeline
        best_model_name = name

# Save the best model
model_path = "Model/model.pkl"
os.makedirs("Model", exist_ok=True)

with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

print(f"\n✅ Best Model: {best_model_name} with Accuracy: {best_score:.2f}")
print(f"Model saved to {model_path}")
