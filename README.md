# EMI Prediction System

A comprehensive machine learning system for predicting EMI (Equated Monthly Installment) eligibility and calculating maximum affordable EMI amounts for loan applicants.

## Features

- **EMI Eligibility Prediction**: Binary classification to determine if a customer is eligible for a loan
- **Maximum EMI Prediction**: Regression model to predict the maximum affordable EMI
- **Interactive Dashboard**: User-friendly interface for making predictions and exploring data
- **Model Monitoring**: Integration with MLflow for tracking experiments and model performance
- **Data Exploration**: Interactive visualizations for exploring the dataset

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- MLflow (for model tracking)
- Streamlit (for the web interface)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd EMIFinal
   ```

2. **Create a virtual environment (recommended)**:
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

1. Place your dataset file in the project root directory with the name `emi_prediction_dataset.csv`
2. Ensure your dataset includes the following columns (adjust in `config/config.py` if different):
   - `age`: Customer's age
   - `monthly_income`: Monthly income in INR
   - `employment_type`: Type of employment (e.g., Salaried, Self-Employed)
   - `existing_emi`: Existing EMI payments (if any)
   - `credit_score`: Credit score (300-900)
   - `loan_amount`: Requested loan amount
   - `loan_tenure`: Loan tenure in years
   - `interest_rate`: Annual interest rate
   - `residence_type`: Type of residence
   - `emi_eligible`: Target for classification (1 for eligible, 0 for not eligible)
   - `max_emi`: Target for regression (maximum affordable EMI)

## Training the Models

1. **Start the MLflow tracking server** (in a separate terminal):
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns -h 0.0.0.0 -p 5000
   ```

2. **Train the models**:
   ```bash
   python train_and_save_models.py
   ```

   This will:
   - Preprocess the data
   - Train both classification and regression models
   - Save the best models to the `models/` directory
   - Log all experiments to MLflow

## Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the application** in your browser at `http://localhost:8501`

## Application Features

### Home
- Overview of the application
- Quick start guide
- System requirements

### EMI Predictor
- **Single Prediction**: Input customer details to get instant predictions
- **Batch Prediction**: Upload a CSV file for batch processing
- **Amortization Schedule**: View detailed payment schedule for loans

### Data Explorer
- Interactive visualizations of the dataset
- Filter and explore data by various criteria
- Statistical summaries and insights

### Model Monitoring
- Track model performance over time
- Compare different model versions
- View metrics and parameters in MLflow UI

### About
- Detailed documentation
- Setup instructions
- Contact information

## Project Structure

```
EMIFinal/
├── config/                  # Configuration files
│   └── config.py            # Global configuration parameters
├── data/                    # Data directory
│   └── emi_prediction_dataset.csv  # Input dataset
├── models/                  # Trained models and preprocessors
│   ├── best_classification_model.joblib
│   ├── best_regression_model.joblib
│   └── preprocessor.joblib
├── src/                     # Source code
│   ├── data/                # Data loading and processing
│   │   └── data_loader.py
│   ├── features/            # Feature engineering
│   │   └── data_preprocessing.py
│   ├── models/              # Model training and evaluation
│   │   └── model_trainer.py
│   ├── utils/               # Utility functions
│   │   └── data_utils.py
│   └── visualization/       # Visualization utilities
├── .gitignore
├── app.py                   # Main Streamlit application
├── requirements.txt          # Python dependencies
├── train_and_save_models.py  # Script to train and save models
└── README.md                # This file
```

## Deployment

### Streamlit Cloud

1. Push your code to a GitHub repository
2. Sign up for a free account at [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" and connect your GitHub repository
4. Select the branch and set the main file to `app.py`
5. Click "Deploy!"

### Heroku

1. Install the Heroku CLI and login:
   ```bash
   heroku login
   ```

2. Create a new Heroku app:
   ```bash
   heroku create emi-prediction-app
   ```

3. Set up the MLflow tracking URI:
   ```bash
   heroku config:set MLFLOW_TRACKING_URI=your-mlflow-uri
   ```

4. Deploy the app:
   ```bash
   git push heroku main
   ```

## Troubleshooting

- **MLflow Connection Issues**: Ensure the MLflow server is running and accessible at the specified URI
- **Model Loading Errors**: Verify that the model files exist in the `models/` directory
- **Dependency Issues**: Make sure all required packages are installed with the correct versions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Model tracking with [MLflow](https://mlflow.org/)
- Visualization with [Plotly](https://plotly.com/) and [Matplotlib](https://matplotlib.org/)
