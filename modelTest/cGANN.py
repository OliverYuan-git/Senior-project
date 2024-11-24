import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data_path = 'winequality-white-corrected.csv'
data = pd.read_csv(data_path)

# Step 2: Preprocess the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

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

# Compile the discriminator
discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                      loss="binary_crossentropy", metrics=["accuracy"])

# Combine the generator and discriminator into the GAN model
discriminator.trainable = False
gan_input = layers.Input(shape=(latent_dim,))
generated_data = generator(gan_input)
gan_output = discriminator(generated_data)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss="binary_crossentropy")

# Step 4: Training the GAN
def train_gan(gan, generator, discriminator, data, latent_dim, epochs=1000, batch_size=64):
    for epoch in range(epochs):
        # Train the discriminator
        # Real samples
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_samples = data[idx]
        real_labels = np.ones((batch_size, 1))

        # Fake samples
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_samples = generator.predict(noise)
        fake_labels = np.zeros((batch_size, 1))

        # Train the discriminator on real and fake samples
        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        misleading_labels = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, misleading_labels)

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch} / {epochs} - D Loss: {d_loss[0]}, D Acc: {d_loss[1] * 100}%, G Loss: {g_loss}")

# Train the GAN
train_gan(gan, generator, discriminator, data_scaled, latent_dim, epochs=2000, batch_size=64)

# Step 5: Generate synthetic data
def generate_synthetic_data(generator, latent_dim, num_samples):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    generated_samples = generator.predict(noise)
    return scaler.inverse_transform(generated_samples)  # Revert scaling

# Generate synthetic data
synthetic_data = generate_synthetic_data(generator, latent_dim, 1000)

# Save synthetic data to CSV in the current working directory
synthetic_data_df = pd.DataFrame(synthetic_data, columns=data.columns)  # Ensure column names match
synthetic_data_df.to_csv('synthetic_winequality_data.csv', index=False)

print("Synthetic data saved to 'synthetic_winequality_data.csv'")
