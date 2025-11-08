import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import logging

class DataPreprocessor:
    def __init__(self, target_column: str = 'emi_eligibility', test_size: float = 0.2, 
                 val_size: float = 0.1, random_state: int = 42):
        """Initialize the DataPreprocessor.
        
        Args:
            target_column (str): Name of the target column. Defaults to 'emi_eligible'.
            test_size (float): Proportion of data to use for testing. Defaults to 0.2.
            val_size (float): Proportion of training data to use for validation. Defaults to 0.1.
            random_state (int): Random seed for reproducibility. Defaults to 42.
        """
        self.target_column = target_column
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        self.preprocessor = None
        self.numerical_features = None
        self.categorical_features = None
        
    def _identify_features(self, df: pd.DataFrame) -> None:
        """Identify numerical and categorical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        # Identify numerical and categorical features
        self.numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Remove target variable if present
        if self.target_column in self.numerical_features:
            self.numerical_features.remove(self.target_column)
        if self.target_column in self.categorical_features:
            self.categorical_features.remove(self.target_column)
            
        self.logger.info(f"Identified numerical features: {self.numerical_features}")
        self.logger.info(f"Identified categorical features: {self.categorical_features}")
    
    def _create_preprocessor(self) -> ColumnTransformer:
        """Create the preprocessing pipeline.
        
        Returns:
            ColumnTransformer: Configured preprocessor
        """
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
            
        return preprocessor
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """Preprocess the data and split into train, validation, and test sets.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple containing:
                - X_train (pd.DataFrame): Training features
                - X_val (pd.DataFrame): Validation features
                - X_test (pd.DataFrame): Test features
                - y_train (pd.Series): Training target
                - y_val (pd.Series): Validation target
                - y_test (pd.Series): Test target
        """
        self.logger.info("Starting data preprocessing...")
        
        # Make a copy to avoid modifying the original dataframe
        df = df.copy()
        
        # Identify features
        self._identify_features(df)

        # Try converting numeric-like categorical columns back to numeric
        for col in self.categorical_features:
            # Try to convert to numeric if possible
            df[col] = pd.to_numeric(df[col], errors='ignore')

        # Remove very high-cardinality categorical features
        self.categorical_features = [
            col for col in self.categorical_features if df[col].nunique() < 100
        ]
  
        # Ensure categorical columns are all strings
        for col in self.categorical_features:
            df[col] = df[col].astype(str)

        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
    
                # --- Detect task type ---
        # Regression targets are numeric with many unique values
        y_unique = df[self.target_column].nunique()
        y_is_numeric = pd.api.types.is_numeric_dtype(df[self.target_column])
        is_classification = not y_is_numeric or y_unique < 20  # adjust threshold if needed

        if is_classification:
            self.logger.info("Detected task type: CLASSIFICATION")
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

            # Save label encoder for inference
            import joblib
            joblib.dump(label_encoder, "models/label_encoder.joblib")

            stratify_option_train = y
            stratify_option_val = True
        else:
            self.logger.info("Detected task type: REGRESSION")
            y = pd.to_numeric(y, errors='coerce')
            stratify_option_train = None
            stratify_option_val = False

        # --- Split datasets ---
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_option_train if is_classification else None
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=self.val_size / (1 - self.test_size),
            random_state=self.random_state,
            stratify=y_train_val if is_classification else None
        )

        
        # Create and fit preprocessor
        self.preprocessor = self._create_preprocessor()
        
        # Fit on training data
        self.preprocessor.fit(X_train)
        
        # Transform all datasets
        X_train_processed = self.preprocessor.transform(X_train)
        X_val_processed = self.preprocessor.transform(X_val)
        X_test_processed = self.preprocessor.transform(X_test)
        
        self.logger.info(f"Data split sizes - Train: {len(X_train_processed)}, "
                        f"Val: {len(X_val_processed)}, Test: {len(X_test_processed)}")
        
        import joblib
        joblib.dump(self.preprocessor, "models/preprocessor.joblib")

        return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test

if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    from data.data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    df = loader.load_data()
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_data(df)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")
