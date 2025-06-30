import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def load_and_preprocess_data(file_path: str, sequence_length: int, test_size: float = 0.2, validation_size: float = 0.2):
    """
    Loads and preprocesses the synthetic tick data with moving average features.

    Args:
        file_path (str): The path to the CSV file.
        sequence_length (int): The length of the input sequences.
        test_size (float): The proportion of the dataset to allocate to the test set.
        validation_size (float): The proportion of the training set to allocate to the validation set.

    Returns:
        A tuple containing the training, validation, and test sets.
    """
    # Load the data
    df = pd.read_csv(file_path)

    # Calculate moving averages
    df['ma_5'] = df['price'].rolling(window=5).mean()
    df['ma_20'] = df['price'].rolling(window=20).mean()

    # Handle NaN values by back-filling
    df.bfill(inplace=True)

    # Select features
    features = ['price', 'ma_5', 'ma_20']
    data = df[features].values
    target = df['price'].values

    # Create sequences
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(target[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    # Split training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, shuffle=False)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

if __name__ == '__main__':
    # Example usage
    file_path = '../data/synthetic_ticks_custom.csv'
    sequence_length = 10
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_preprocess_data(file_path, sequence_length)

    print("Training set shape:", X_train.shape, y_train.shape)
    print("Validation set shape:", X_val.shape, y_val.shape)
    print("Test set shape:", X_test.shape, y_test.shape)