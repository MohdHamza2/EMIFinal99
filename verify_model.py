import joblib
import os

# Paths to your saved models
classification_model_path = "models/best_classification_model.joblib"
preprocessor_path = "models/preprocessor.joblib"

# Check if files exist
print("Checking model files...\n")

if os.path.exists(classification_model_path) and os.path.exists(preprocessor_path):
    print("✅ Model and preprocessor files found!")
else:
    print("❌ Model or preprocessor file missing!")

# Try loading the model
try:
    model = joblib.load(classification_model_path)
    preprocessor = joblib.load(preprocessor_path)
    print("\n✅ Model loaded successfully!")
    print(f"Model type: {type(model)}")
except Exception as e:
    print(f"\n⚠️ Error loading model: {e}")
