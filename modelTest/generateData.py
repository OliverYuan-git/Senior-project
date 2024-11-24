import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load dataset
data = pd.read_csv('winequality-red-corrected.csv')
X = data.drop(columns=['quality'])
y = data['quality']
quality_counts = y.value_counts()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_scaled, y, epochs=50, verbose=0)
generated_X = np.random.normal(loc=X.mean(), scale=X.std(), size=(1600, X.shape[1]))
generated_X_scaled = scaler.transform(generated_X) 

generated_y = model.predict(generated_X_scaled).flatten()
noise = np.random.normal(0, 0.5, generated_y.shape)
generated_y = np.clip(np.round(generated_y + noise), 1, 10).astype(int)

generated_data = pd.DataFrame(generated_X, columns=X.columns)
generated_data['quality'] = generated_y

for quality_level, count in quality_counts.items():
    current_count = (generated_data['quality'] == quality_level).sum()
    if current_count < count:
    
        low_quality_indices = generated_data[generated_data['quality'] < quality_level].index
        needed_count = count - current_count
    
        if len(low_quality_indices) > 0:
            if len(low_quality_indices) < needed_count:
 
                indices_to_adjust = np.random.choice(low_quality_indices, size=needed_count, replace=True)
            else:

                indices_to_adjust = np.random.choice(low_quality_indices, size=needed_count, replace=False)
            
            generated_data.loc[indices_to_adjust, 'quality'] = quality_level


generated_data.to_csv('red-linear.csv', index=False)

