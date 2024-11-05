import tensorflow as tf
import pandas as pd
import numpy as np

file_path = 'winequality-red-corrected.csv'
data = pd.read_csv(file_path)
def normalize(data):
    for column in data.columns[:-1]:
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
    return data
normalized_data = normalize(data.copy())
features = normalized_data.iloc[:, :-1].values
labels = normalized_data['quality'].values

# Function to add Gaussian noise
def add_noise(features, noise_factor=0.01):
    noise = np.random.randn(*features.shape) * noise_factor
    return features + noise

# TensorFlow dataset creation
def create_dataset(features, labels, noise_factor=0.01):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    
    # Add noise to features for augmentation
    dataset = dataset.map(lambda x, y: (add_noise(x, noise_factor), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset

noise_factor = 0.02  # adjust
augmented_dataset = create_dataset(features, labels, noise_factor)

augmented_features = []
augmented_labels = []

for batch_features, batch_labels in augmented_dataset:
    augmented_features.append(batch_features.numpy())
    augmented_labels.append(batch_labels.numpy())

augmented_features = np.vstack(augmented_features)  
augmented_labels = np.hstack(augmented_labels)     
augmented_df = pd.DataFrame(augmented_features, columns=data.columns[:-1]) 
augmented_df['quality'] = augmented_labels 
output_file_path = 'myRedData.csv'
augmented_df.to_csv(output_file_path, index=False)

print(f"Augmented dataset saved to: {output_file_path}")
