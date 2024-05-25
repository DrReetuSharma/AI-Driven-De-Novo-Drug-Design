import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    """
    Load dataset from a CSV file.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """
    Preprocess the dataset:
    - Handle missing values
    - Encode categorical variables
    - Standardize numerical features
    """
    # Handle missing values
    data = data.dropna()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['Activity', 'Target', 'Selectivity', 'Pharmacokinetics']
    
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    
    # Standardize numerical features (if any numerical features are present)
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    
    if numerical_columns.any():
        scaler = StandardScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    return data, label_encoders

def save_preprocessed_data(data, output_file_path):
    """
    Save the preprocessed dataset to a CSV file.
    """
    data.to_csv(output_file_path, index=False)

def main():
    # Load the dataset
    input_file_path = 'data/dataset.csv'
    data = load_data(input_file_path)
    
    # Preprocess the data
    preprocessed_data, label_encoders = preprocess_data(data)
    
    # Save the preprocessed data
    output_file_path = 'data/preprocessed_dataset.csv'
    save_preprocessed_data(preprocessed_data, output_file_path)
    
    print(f"Preprocessed data saved to {output_file_path}")

if __name__ == '__main__':
    main()
