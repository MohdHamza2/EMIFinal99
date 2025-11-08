import os
import sys
import joblib
import pandas as pd
import mlflow
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import mlflow.sklearn

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from src.data.data_loader import DataLoader
from src.features.data_preprocessing import DataPreprocessor
from src.models.model_trainer import ModelTrainer
from src.utils.data_utils import DataProcessor
from config.config import (
    ColumnNames, 
    ModelConfig, 
    Paths,
    MLflowConfig,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    # Create models directory
    models_dir = Path(Paths.MODELS_DIR)
    models_dir.mkdir(exist_ok=True, parents=True)
    
    # Create data directory
    data_dir = Path(Paths.DATA_DIR)
    data_dir.mkdir(exist_ok=True, parents=True)

def train_models():
    """Train and save the models."""
    try:
        # Setup MLflow
        mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
        mlflow.set_experiment(MLflowConfig.EXPERIMENT_NAME)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            logger.info("Starting model training...")
            
            # Initialize data loader
            logger.info("Loading data...")
            data_loader = DataLoader()
            df = data_loader.load_data()
            
            # Initialize data processor
            data_processor = DataProcessor()
            
            # Preprocess data for classification
            logger.info("Preprocessing data for classification...")
            X_train_clf, X_val_clf, X_test_clf, y_train_clf, y_val_clf, y_test_clf = preprocess_data(
                df, 
                target_column=ColumnNames.TARGET_CLASSIFICATION.value
            )
            
            # Save preprocessor for classification
            #preprocessor_path = Path(Paths.MODELS_DIR) / ModelConfig.PREPROCESSOR
            #joblib.dump(preprocessor.preprocessor, preprocessor_path)
            #logger.info(f"✅ Saved fitted preprocessor to {preprocessor_path}")
            
            # Log preprocessing parameters
            mlflow.log_param("target_column", ColumnNames.TARGET_CLASSIFICATION.value)
            mlflow.log_param("categorical_features", CATEGORICAL_FEATURES)
            mlflow.log_param("numerical_features", NUMERICAL_FEATURES)
            
            # Train classification models
            logger.info("Training classification models...")
            trainer = ModelTrainer()
            clf_result = trainer.train_classification_models(
                X_train_clf, y_train_clf, X_val_clf, y_val_clf
            )
            
            # Save the best classification model
            clf_model_path = Path(Paths.MODELS_DIR) / ModelConfig.CLASSIFICATION_MODEL
            joblib.dump(clf_result['model'], clf_model_path)
            logger.info(f"Saved classification model to {clf_model_path}")
            
            # Log classification metrics
            for metric_name, value in clf_result['metrics'].items():
                mlflow.log_metric(f"clf_{metric_name}", value)
            
            # ✅ FIX: End the MLflow run before starting regression
            mlflow.end_run()

            # Preprocess data for regression (if different target)
            if ColumnNames.TARGET_REGRESSION.value in df.columns:
                logger.info("Preprocessing data for regression...")
                X_train_reg, X_val_reg, X_test_reg, y_train_reg, y_val_reg, y_test_reg = preprocess_data(
                    df,
                    target_column=ColumnNames.TARGET_REGRESSION.value
                )
                
                # Train regression models
                logger.info("Training regression models...")
                reg_result = trainer.train_regression_models(
                    X_train_reg, y_train_reg, X_val_reg, y_val_reg
                )
                
                # Save the best regression model
                reg_model_path = Path(Paths.MODELS_DIR) / ModelConfig.REGRESSION_MODEL
                joblib.dump(reg_result['model'], reg_model_path)
                logger.info(f"Saved regression model to {reg_model_path}")
                
                # Log regression metrics
                for metric_name, value in reg_result['metrics'].items():
                    mlflow.log_metric(f"reg_{metric_name}", value)
            
            logger.info("Model training completed successfully!")
            
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        raise

def preprocess_data(df: pd.DataFrame, target_column: str):
    """Preprocess the data for model training.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        
    Returns:
        Tuple containing X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(target_column=target_column)
    
    # Preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_data(df)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    """Main function to run the training pipeline."""
    try:
        # Setup directories
        setup_directories()
        
        # Train models
        train_models()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
