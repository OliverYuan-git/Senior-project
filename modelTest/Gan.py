import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, optimizers
import matplotlib.pyplot as plt

# Step 1: Load the dataset and exclude the 'quality' column
data_path = 'winequality-white-corrected.csv'
data = pd.read_csv(data_path)
features = data.drop(columns=['quality'])  # Drop the 'quality' column

# Step 2: Preprocess the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(features)  # Only scale feature columns

# Step 3: Define GAN components
# Define the generator model
def build_generator(latent_dim, output_dim):
    model = tf.keras.Sequential([
        layers.Dense(128, activation="relu", input_dim=latent_dim),
        layers.Dense(256, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(output_dim, activation="tanh")  # Output dimension matches input data
    ])
    return model

# Define the discriminator model
def build_discriminator(input_dim):
    model = tf.keras.Sequential([
        layers.Dense(512, activation="relu", input_dim=input_dim),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")  # Binary classification output
    ])
    return model

# Parameters
latent_dim = 100
data_dim = data_scaled.shape[1]  # Number of features in the dataset

# Build the models
generator = build_generator(latent_dim, data_dim)
discriminator = build_discriminator(data_dim)

# Initialize fresh optimizers linked to their model variables
generator_optimizer = optimizers.Adam(0.0002, 0.5)
discriminator_optimizer = optimizers.Adam(0.0002, 0.5)

# Binary cross-entropy loss function
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# Training function for one step in the GAN
@tf.function
def train_step(real_samples):
    batch_size = tf.shape(real_samples)[0]

    # Train the discriminator
    noise = tf.random.normal([batch_size, latent_dim])
    fake_samples = generator(noise, training=True)
    
    real_labels = tf.ones((batch_size, 1))
    fake_labels = tf.zeros((batch_size, 1))
    with tf.GradientTape() as tape:
        real_loss = bce(real_labels, discriminator(real_samples, training=True))
        fake_loss = bce(fake_labels, discriminator(fake_samples, training=True))
        d_loss = (real_loss + fake_loss) / 2
    gradients_of_discriminator = tape.gradient(d_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # Train the generator
    misleading_labels = tf.ones((batch_size, 1))
    with tf.GradientTape() as tape:
        fake_samples = generator(noise, training=True)
        g_loss = bce(misleading_labels, discriminator(fake_samples, training=True))
    gradients_of_generator = tape.gradient(g_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    
    return d_loss, g_loss

# Step 4: Training the GAN
def train_gan(epochs=1000, batch_size=64):
    for epoch in range(epochs):
        idx = np.random.randint(0, data_scaled.shape[0], batch_size)
        real_samples = data_scaled[idx]
        real_samples = tf.convert_to_tensor(real_samples, dtype=tf.float32)

        # Run one step of training
        d_loss, g_loss = train_step(real_samples)

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch} / {epochs} - D Loss: {d_loss.numpy()}, G Loss: {g_loss.numpy()}")

# Train the GAN
train_gan(epochs=5000, batch_size=64)

# Step 5: Generate synthetic data
def generate_synthetic_data(generator, latent_dim, num_samples):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    generated_samples = generator.predict(noise)
    return scaler.inverse_transform(generated_samples)  # Revert scaling

# Generate synthetic data for the features
synthetic_data = generate_synthetic_data(generator, latent_dim, 4898)

# Save synthetic data to CSV in the current working directory
synthetic_data_df = pd.DataFrame(synthetic_data, columns=features.columns)  # Use feature column names only
synthetic_data_df.to_csv('synthetic_winequality_features6.csv', index=False)

print("Synthetic data saved to 'synthetic_winequality_features2.csv'")
