import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from loguru import logger
import typer

# Define the project root path reliably (assuming your project structure is standard)
PROJECT_ROOT = Path(__file__).resolve().parents[2] 

# --- Define Paths ---
# Path to the raw CSV file
RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'german_credit_data.csv'
# Path to save the processed (cleaned) data directory
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

app = typer.Typer() # Keep the typer app for command line execution


# --- 1. Load Raw Data ---
def load_raw_data() -> pd.DataFrame:
    """Loads the German Credit Data from the data/raw folder."""
    if not RAW_DATA_PATH.exists():
        logger.error(f"Raw data file not found at: {RAW_DATA_PATH}")
        raise FileNotFoundError(
            "Please ensure you've moved the downloaded CSV file there."
        )

    # Note: index_col=0 is used because the German Credit Data often has an extraneous index column.
    df = pd.read_csv(RAW_DATA_PATH, index_col=0)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} features from raw data.")
    return df

# --- 2. Process Data ---
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans, preprocesses, and prepares the data for model training.
    """
    logger.info("Starting data processing...")

    # Clean Target Variable (Assuming 'kredit' is the target)
    df = df.rename(columns={'kredit': 'target'})
    df['target'] = df['target'].astype(int)

    # German Credit Data uses 1=Good, 2=Bad. We must flip it to 0=Good, 1=Bad (standard ML practice)

    # Handle Categorical Features (One-Hot Encoding)
    categorical_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Handle Missing Data
    df = df.fillna(-999) 
    
    logger.success(f"Data processing complete. Final feature count: {len(df.columns) - 1}")
    return df

# --- 3. Save Train/Test Split ---
def save_train_test_split(df: pd.DataFrame):
    """Splits the processed data into train and test sets and saves them."""
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Perform the split (stratify ensures balanced target classes in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create the data/processed directory if it doesn't exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the split data frames using the compact 'pickle' format (.pkl)
    X_train.to_pickle(PROCESSED_DATA_DIR / 'X_train.pkl')
    X_test.to_pickle(PROCESSED_DATA_DIR / 'X_test.pkl')
    y_train.to_pickle(PROCESSED_DATA_DIR / 'y_train.pkl')
    y_test.to_pickle(PROCESSED_DATA_DIR / 'y_test.pkl')
    
    logger.success(f"Train/Test splits saved to: {PROCESSED_DATA_DIR}")


@app.command()
def main():
    """Runs the full data processing pipeline."""
    try:
        raw_df = load_raw_data()
        processed_df = process_data(raw_df)
        save_train_test_split(processed_df)
    except FileNotFoundError as e:
        logger.error(f"Execution failed: {e}")


if __name__ == "__main__":
    # The command is executed via the Typer app interface
    app()