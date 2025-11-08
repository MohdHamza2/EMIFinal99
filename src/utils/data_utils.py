import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union, List
from pathlib import Path
import joblib
import logging
from datetime import datetime

from config.config import (
    ColumnNames, 
    CATEGORICAL_FEATURES, 
    NUMERICAL_FEATURES,
    FeatureEngineeringParams,
    ModelConfig
)

logger = logging.getLogger(__name__)

class DataProcessor:
    """Class for processing and transforming data for the EMI prediction system."""
    
    def __init__(self, preprocessor_path: Optional[str] = None):
        """Initialize the DataProcessor.
        
        Args:
            preprocessor_path: Path to a saved preprocessor. If None, a new one will be created.
        """
        self.preprocessor = None
        self.feature_names = None
        
        if preprocessor_path and Path(preprocessor_path).exists():
            self.load_preprocessor(preprocessor_path)
    
    def load_preprocessor(self, filepath: str) -> None:
        """Load a preprocessor from a file.
        
        Args:
            filepath: Path to the saved preprocessor.
        """
        try:
            self.preprocessor = joblib.load(filepath)
            logger.info(f"Loaded preprocessor from {filepath}")
            
            # Extract feature names from the preprocessor
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                self.feature_names = list(self.preprocessor.get_feature_names_out())
            else:
                # For older versions of scikit-learn
                self.feature_names = None
                
        except Exception as e:
            logger.error(f"Error loading preprocessor: {str(e)}")
            raise
    
    def save_preprocessor(self, filepath: str) -> None:
        """Save the preprocessor to a file.
        
        Args:
            filepath: Path where the preprocessor should be saved.
        """
        if self.preprocessor is None:
            raise ValueError("No preprocessor has been fitted.")
            
        try:
            joblib.dump(self.preprocessor, filepath)
            logger.info(f"Saved preprocessor to {filepath}")
        except Exception as e:
            logger.error(f"Error saving preprocessor: {str(e)}")
            raise
    
    def preprocess_data(
        self, 
        df: pd.DataFrame, 
        is_training: bool = False,
        target_column: str = ColumnNames.TARGET_CLASSIFICATION.value
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Preprocess the input data.
        
        Args:
            df: Input DataFrame containing the data to preprocess.
            is_training: Whether this is training data (fit the preprocessor).
            target_column: Name of the target column.
            
        Returns:
            Tuple containing:
                - Preprocessed features (DataFrame)
                - Target values (Series) if target_column is in df, else None
        """
        # Make a copy to avoid modifying the original DataFrame
        df = df.copy()
        
        # Extract target if present
        target = None
        if target_column in df.columns:
            target = df.pop(target_column)
        
        # Convert column names to string to avoid issues with numeric column names
        df.columns = df.columns.astype(str)
        
        # Ensure all expected columns are present
        self._validate_columns(df)
        
        # Apply feature engineering
        df = self._feature_engineering(df)
        
        # Apply preprocessing
        if is_training or self.preprocessor is None:
            self._fit_preprocessor(df)
        
        # Transform the data
        processed_data = self._transform_data(df)
        
        return processed_data, target
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that all required columns are present in the DataFrame."""
        required_columns = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
        missing_columns = required_columns - set(df.columns)
        
        if missing_columns:
            logger.warning(f"Missing columns in input data: {missing_columns}")
    
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to create new features."""
        df = df.copy()
        
        # 1. Calculate debt-to-income ratio
        if all(col in df.columns for col in [ColumnNames.EXISTING_EMI.value, ColumnNames.MONTHLY_INCOME.value]):
            df[ColumnNames.DEBT_TO_INCOME] = (
                (df[ColumnNames.EXISTING_EMI.value] / 
                 df[ColumnNames.MONTHLY_INCOME.value].replace(0, np.nan)) * 100
            ).fillna(0)
        
        # 2. Calculate expense-to-income ratio (simplified)
        # In a real scenario, you might have actual expense data
        if ColumnNames.MONTHLY_INCOME.value in df.columns:
            # This is a simplified version - adjust based on your actual data
            df[ColumnNames.EXPENSE_TO_INCOME] = (
                (df.get('monthly_expenses', df[ColumnNames.MONTHLY_INCOME.value] * 0.6) / 
                 df[ColumnNames.MONTHLY_INCOME.value].replace(0, np.nan)) * 100
            ).fillna(0)
        
        # 3. Calculate affordability ratio
        if all(col in df.columns for col in [ColumnNames.MONTHLY_INCOME.value, 'monthly_expenses']):
            df[ColumnNames.AFFORDABILITY_RATIO] = (
                (df[ColumnNames.MONTHLY_INCOME.value] - df['monthly_expenses']) / 
                df[ColumnNames.MONTHLY_INCOME.value].replace(0, np.nan) * 100
            ).fillna(0)
        
        # 4. Create credit score bins
        if ColumnNames.CREDIT_SCORE.value in df.columns:
            bins = [300, 579, 669, 739, 799, 900]
            labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Exceptional']
            df['credit_score_category'] = pd.cut(
                df[ColumnNames.CREDIT_SCORE.value], 
                bins=bins, 
                labels=labels,
                include_lowest=True
            )
        
        # 5. Create age groups
        if ColumnNames.AGE.value in df.columns:
            bins = [18, 25, 35, 45, 55, 65, 100]
            labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
            df['age_group'] = pd.cut(
                df[ColumnNames.AGE.value], 
                bins=bins, 
                labels=labels,
                include_lowest=True
            )
        
        return df
    
    def _fit_preprocessor(self, df: pd.DataFrame) -> None:
        """Fit the preprocessor on the training data."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.pipeline import Pipeline
        
        # Identify feature types
        numeric_features = [col for col in NUMERICAL_FEATURES if col in df.columns]
        categorical_features = [col for col in CATEGORICAL_FEATURES if col in df.columns]
        
        # Create transformers
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # Drop columns that are not explicitly specified
        )
        
        # Fit the preprocessor
        self.preprocessor.fit(df)
        
        # Store feature names
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            self.feature_names = list(self.preprocessor.get_feature_names_out())
    
    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using the fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted. Call _fit_preprocessor first.")
        
        # Transform the data
        transformed_data = self.preprocessor.transform(df)
        
        # Convert to DataFrame with feature names if available
        if self.feature_names is not None:
            return pd.DataFrame(transformed_data, columns=self.feature_names, index=df.index)
        else:
            return pd.DataFrame(transformed_data, index=df.index)

def calculate_emi(principal: float, rate: float, years: int) -> float:
    """Calculate EMI amount.
    
    Args:
        principal: Loan amount
        rate: Annual interest rate (percentage)
        years: Loan tenure in years
        
    Returns:
        EMI amount
    """
    if principal <= 0 or years <= 0:
        return 0.0
        
    rate_monthly = rate / 12 / 100
    months = years * 12
    
    if rate_monthly == 0:  # Handle 0% interest rate
        return principal / months
        
    emi = principal * rate_monthly * (1 + rate_monthly)**months / ((1 + rate_monthly)**months - 1)
    return emi

def generate_amortization_schedule(principal: float, rate: float, years: int) -> pd.DataFrame:
    """Generate a loan amortization schedule.
    
    Args:
        principal: Loan amount
        rate: Annual interest rate (percentage)
        years: Loan tenure in years
        
    Returns:
        DataFrame containing the amortization schedule
    """
    rate_monthly = rate / 12 / 100
    months = years * 12
    emi = calculate_emi(principal, rate, years)
    
    schedule = []
    balance = principal
    
    for month in range(1, int(months) + 1):
        interest = balance * rate_monthly
        principal_paid = emi - interest
        balance -= principal_paid
        
        if balance < 0:
            balance = 0
        
        schedule.append({
            'Month': month,
            'EMI': emi,
            'Principal': principal_paid,
            'Interest': interest,
            'Remaining Balance': max(0, balance)
        })
    
    return pd.DataFrame(schedule)

def load_data(filepath: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        sample_size: Number of rows to sample (for testing)
        
    Returns:
        Loaded DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        
        if sample_size and sample_size < len(df):
            df = df.sample(min(sample_size, len(df)), random_state=42)
            
        return df
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {str(e)}")
        raise
