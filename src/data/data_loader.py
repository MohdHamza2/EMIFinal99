import pandas as pd
import os
from pathlib import Path
import logging

class DataLoader:
    def __init__(self, data_path: str = None):
        """Initialize the DataLoader with the path to the dataset.
        
        Args:
            data_path (str, optional): Path to the dataset file. Defaults to None.
        """
        self.data_path = data_path or os.path.join(Path(__file__).parents[2], 'emi_prediction_dataset.csv')
        self.logger = logging.getLogger(__name__)
        
    def load_data(self) -> pd.DataFrame:
        """Load the dataset from the specified path.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            self.logger.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)
            self.logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df
        except FileNotFoundError:
            self.logger.error(f"File not found at {self.data_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    loader = DataLoader()
    df = loader.load_data()
    print(f"Loaded {len(df)} records")
