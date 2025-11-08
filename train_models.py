import os
import sys
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.data_loader import DataLoader
from features.data_preprocessing import DataPreprocessor
from models.model_trainer import ModelTrainer

def main():
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Initialize components
    print("Loading data...")
    data_loader = DataLoader()
    df = data_loader.load_data()
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessor(target_column='emi_eligible')  # Adjust target column name if different
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_data(df)
    
    # Save preprocessor
    joblib.dump(preprocessor.preprocessor, models_dir / "preprocessor.joblib")
    
    # Initialize model trainer
    print("Initializing model training...")
    trainer = ModelTrainer(experiment_name="emi_prediction")
    
    # Train classification models
    print("\nTraining classification models...")
    clf_result = trainer.train_classification_models(X_train, y_train, X_val, y_val)
    
    # Save best classification model
    joblib.dump(clf_result['model'], models_dir / "best_classification_model.joblib")
    print(f"\nSaved best classification model: {clf_result['model_name']}")
    
    # Train regression models (for maximum EMI prediction)
    print("\nTraining regression models...")
    # For regression, we'll use the same features but a different target
    # Assuming 'max_emi' is your regression target column
    # You'll need to adjust this based on your actual target column name
    reg_preprocessor = DataPreprocessor(target_column='max_emi')
    _, _, _, y_train_reg, y_val_reg, _ = reg_preprocessor.preprocess_data(df)
    
    reg_result = trainer.train_regression_models(X_train, y_train_reg, X_val, y_val_reg)
    
    # Save best regression model
    joblib.dump(reg_result['model'], models_dir / "best_regression_model.joblib")
    print(f"\nSaved best regression model: {reg_result['model_name']}")
    
    print("\nModel training completed successfully!")

if __name__ == "__main__":
    main()
