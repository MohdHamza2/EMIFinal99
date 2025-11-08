import joblib

try:
    preprocessor = joblib.load("models/preprocessor.joblib")
    print("✅ Preprocessor loaded successfully!")
    print("Type:", type(preprocessor))
except Exception as e:
    print("⚠️ Error loading preprocessor:", e)
