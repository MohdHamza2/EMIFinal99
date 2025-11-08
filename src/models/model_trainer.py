import os
import mlflow
import mlflow.sklearn
import logging
from typing import Dict, Any, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

class ModelTrainer:
    def __init__(self, experiment_name: str = "emi_prediction"):
        """Initialize the ModelTrainer.
        
        Args:
            experiment_name (str): Name of the MLflow experiment. Defaults to "emi_prediction".
        """
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(__name__)
        self._setup_mlflow()
        
    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking."""
        mlflow.set_experiment(self.experiment_name)
        mlflow.sklearn.autolog()
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate classification metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray, optional): Predicted probabilities. Defaults to None.
            
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            
        return metrics
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred)
        }
    
    def train_classification_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                                  X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train multiple classification models and return the best one.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels
            
        Returns:
            Dict[str, Any]: Dictionary containing the best model and its metrics
        """
        self.logger.info("Training classification models...")
        
        models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'xgboost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            #'svc': SVC(probability=True, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }
        
        best_model = None
        best_metrics = {}
        best_model_name = ""
        
        for name, model in models.items():
            with mlflow.start_run(run_name=f"classification_{name}", nested=True):
                self.logger.info(f"Training {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = self._calculate_classification_metrics(y_val, y_pred, y_pred_proba)
                
                # Log metrics
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"val_{metric_name}", value)
                
                # Log model
                mlflow.sklearn.log_model(model, name)

                # --- Step 2: Save classification model locally ---
                #from pathlib import Path
                #import joblib
                #from config.paths import Paths  # make sure config.paths is imported at top

                #model_path = Path(Paths.MODELS_DIR) / f"{name}_classifier.joblib"
                #joblib.dump(model, model_path)
                #self.logger.info(f"✅ Saved classification model locally at {model_path}")
                # -------------------------------------------------
  
                # Update best model
                if not best_model or metrics['f1_score'] > best_metrics.get('f1_score', 0):
                    best_model = model
                    best_metrics = metrics
                    best_model_name = name
        
        self.logger.info(f"Best classification model: {best_model_name} with F1-score: {best_metrics['f1_score']:.4f}")
        
        return {
            'model': best_model,
            'metrics': best_metrics,
            'model_name': best_model_name
        }
    
    def train_regression_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train multiple regression models and return the best one.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation target
            
        Returns:
            Dict[str, Any]: Dictionary containing the best model and its metrics
        """
        self.logger.info("Training regression models...")
        
        models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(random_state=42),
            'xgboost': XGBRegressor(n_estimators=30,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0),
            #'svr': SVR(),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=50,
                max_depth=4,
                random_state=42)
        }
        
        best_model = None
        best_metrics = {}
        best_model_name = ""
        
        for name, model in models.items():
            with mlflow.start_run(run_name=f"regression_{name}"):
                self.logger.info(f"Training {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val)
                
                # Calculate metrics
                metrics = self._calculate_regression_metrics(y_val, y_pred)
                
                # Log metrics
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"val_{metric_name}", value)

                # Log model
                mlflow.sklearn.log_model(model, name)

                # --- Step 2: Save model locally ---
                #from pathlib import Path
                #import joblib
                #from config.paths import Paths
                #model_path = Path(Paths.MODELS_DIR) / f"{name}_regressor.joblib"
                #joblib.dump(model, model_path)
                #self.logger.info(f"✅ Saved regression model locally at {model_path}")
                # ----------------------------------
                
                
                # Update best model (using negative RMSE as we want to minimize it)
                if not best_model or metrics['rmse'] < best_metrics.get('rmse', float('inf')):
                    best_model = model
                    best_metrics = metrics
                    best_model_name = name
        
        self.logger.info(f"Best regression model: {best_model_name} with RMSE: {best_metrics['rmse']:.4f}")
        
        return {
            'model': best_model,
            'metrics': best_metrics,
            'model_name': best_model_name
        }

if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    from sklearn.datasets import make_classification, make_regression
    
    # Create sample data
    X_clf, y_clf = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, random_state=42)
    
    # Split data
    X_train_clf, X_val_clf, y_train_clf, y_val_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    
    X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train classification models
    print("Training classification models...")
    clf_result = trainer.train_classification_models(X_train_clf, y_train_clf, X_val_clf, y_val_clf)
    print(f"Best classification model: {clf_result['model_name']}")
    print(f"Metrics: {clf_result['metrics']}")
    
    # Train regression models
    print("\nTraining regression models...")
    reg_result = trainer.train_regression_models(X_train_reg, y_train_reg, X_val_reg, y_val_reg)
    print(f"Best regression model: {reg_result['model_name']}")
    print(f"Metrics: {reg_result['metrics']}")
