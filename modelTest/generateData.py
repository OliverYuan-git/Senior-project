import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the CSV file as a tf.data.Dataset
def load_csv_dataset(file_path, batch_size=32):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=batch_size,
        label_name='quality',  # Use 'quality' as the label column
        na_value="?",
        num_epochs=1,
        ignore_errors=True
    )
    return dataset

# Fit a linear regression model to capture the relationship between features and quality
def fit_regression_model(df):
    features = df.drop(columns=['quality']).values
    labels = df['quality'].values
    model = LinearRegression()
    model.fit(features, labels)
    return model

# Generate new data that maintains the relationship between quality and features
def generate_similar_data(df, model, num_samples):
    features = df.drop(columns=['quality']).values
    generated_features = []
    
    for _ in range(num_samples):
        random_sample = features[np.random.randint(features.shape[0])]
        noise = np.random.normal(0, 0.01, size=random_sample.shape)
        new_sample = random_sample + noise
        predicted_quality = model.predict([new_sample])[0]
        new_sample = np.append(new_sample, predicted_quality)
        generated_features.append(new_sample)
    
    columns = list(df.columns)
    generated_df = pd.DataFrame(generated_features, columns=columns)
    return generated_df

# Main function to load, augment, and save dataset
def main():
    input_file_path = "winequality-red-corrected.csv"  # Path to your input CSV file
    output_file_path = "winequality-red-augmented.csv"  # Path to save the augmented CSV file
    
    # Step 1: Load the dataset into a pandas DataFrame
    df = pd.read_csv(input_file_path)
    
    # Step 2: Fit a regression model to capture the relationship between features and quality
    model = fit_regression_model(df)
    
    # Step 3: Generate new data that maintains the relationship between quality and features
    num_samples = len(df)  # Generate the same number of samples as the original dataset
    generated_df = generate_similar_data(df, model, num_samples)
    
    # Step 4: Adjust quality column to be an integer
    generated_df['quality'] = generated_df['quality'].round().astype(int)
    
    # Step 5: Save the augmented dataset to a new CSV file
    generated_df.to_csv(output_file_path, index=False)
    print(f"Augmented dataset saved to {output_file_path}")

if __name__ == "__main__":
    main()
