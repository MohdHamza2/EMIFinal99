from enum import Enum
from typing import List, Dict, Any

class ColumnNames(Enum):
    """Enum for column names in the dataset."""
    # Target columns
    TARGET_CLASSIFICATION = "emi_eligibility"  # Binary: 1 for eligible, 0 for not eligible
    TARGET_REGRESSION = "max_monthly_emi"  # Maximum affordable EMI amount
    
    # Feature columns
    AGE = "age"
    MONTHLY_INCOME = "monthly_income"
    EMPLOYMENT_TYPE = "employment_type"
    EXISTING_EMI = "existing_emi"
    CREDIT_SCORE = "credit_score"
    LOAN_AMOUNT = "loan_amount"
    LOAN_TENURE = "loan_tenure"  # in years
    INTEREST_RATE = "interest_rate"  # annual percentage
    RESIDENCE_TYPE = "residence_type"
    
    # Derived features (will be created during preprocessing)
    DEBT_TO_INCOME = "debt_to_income_ratio"
    EXPENSE_TO_INCOME = "expense_to_income_ratio"
    AFFORDABILITY_RATIO = "affordability_ratio"

# List of categorical features for one-hot encoding
CATEGORICAL_FEATURES = [
    ColumnNames.EMPLOYMENT_TYPE.value,
    ColumnNames.RESIDENCE_TYPE.value
]

# List of numerical features for scaling
NUMERICAL_FEATURES = [
    ColumnNames.AGE.value,
    ColumnNames.MONTHLY_INCOME.value,
    ColumnNames.EXISTING_EMI.value,
    ColumnNames.CREDIT_SCORE.value,
    ColumnNames.LOAN_AMOUNT.value,
    ColumnNames.LOAN_TENURE.value,
    ColumnNames.INTEREST_RATE.value
]

# Employment type options
EMPLOYMENT_TYPES = [
    "Salaried",
    "Self-Employed",
    "Business",
    "Other"
]

# Residence type options
RESIDENCE_TYPES = [
    "Rented",
    "Owned",
    "Company Provided",
    "Other"
]

# Feature engineering parameters
class FeatureEngineeringParams:
    """Parameters for feature engineering."""
    # Thresholds for debt-to-income ratio (as percentage)
    DTI_THRESHOLD = 40  # Common threshold for loan approval
    
    # Threshold for expense-to-income ratio (as percentage)
    ETI_THRESHOLD = 60  # Common threshold for affordability
    
    # Minimum credit score for eligibility
    MIN_CREDIT_SCORE = 650
    
    # Minimum employment duration in months
    MIN_EMPLOYMENT_DURATION = 6

# Model configuration
class ModelConfig:
    """Model configuration parameters."""
    # Random seed for reproducibility
    RANDOM_STATE = 42
    
    # Test and validation split ratios
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    
    # Model file names
    CLASSIFICATION_MODEL = "best_classification_model.joblib"
    REGRESSION_MODEL = "best_regression_model.joblib"
    PREPROCESSOR = "preprocessor.joblib"

# Path configuration
class Paths:
    """File and directory paths."""
    # Directory paths
    MODELS_DIR = "models"
    DATA_DIR = "data"
    
    # File paths
    @staticmethod
    def get_data_file(filename: str) -> str:
        """Get the full path to a data file."""
        return str(Path(Paths.DATA_DIR) / filename)
    
    @staticmethod
    def get_model_file(filename: str) -> str:
        """Get the full path to a model file."""
        return str(Path(Paths.MODELS_DIR) / filename)

# MLflow configuration
class MLflowConfig:
    """MLflow configuration parameters."""
    TRACKING_URI = "http://localhost:5000"
    EXPERIMENT_NAME = "emi_prediction"

# Streamlit app configuration
class AppConfig:
    """Streamlit application configuration."""
    PAGE_TITLE = "EMI Prediction System"
    PAGE_ICON = "💰"
    LAYOUT = "wide"
    
    # Colors
    PRIMARY_COLOR = "#4CAF50"
    SECONDARY_COLOR = "#2196F3"
    
    # Plotly template
    PLOT_TEMPLATE = "plotly_white"

# Loan calculation parameters
class LoanParams:
    """Loan calculation parameters."""
    # Minimum and maximum loan amounts (in INR)
    MIN_LOAN_AMOUNT = 10000
    MAX_LOAN_AMOUNT = 5000000
    
    # Minimum and maximum loan tenures (in years)
    MIN_LOAN_TENURE = 1
    MAX_LOAN_TENURE = 30
    
    # Minimum and maximum interest rates (annual percentage)
    MIN_INTEREST_RATE = 1.0
    MAX_INTEREST_RATE = 20.0
    
    # Minimum and maximum credit scores
    MIN_CREDIT_SCORE = 300
    MAX_CREDIT_SCORE = 900
