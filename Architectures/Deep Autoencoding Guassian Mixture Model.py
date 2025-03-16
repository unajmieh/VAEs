import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

# Load your IoT dataset into a pandas DataFrame
# Assuming your dataset is stored in a variable called 'iot_data'
# Replace 'target_column' with the name of the target variable column in your dataset
# Replace the column names with the actual feature names in your dataset
# X = iot_data.drop(['target_column'], axis=1)  # Features
# y = iot_data['target_column']  # Target variable

# Standardize the input data
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

# Define the input layer
input_layer = Input(shape=(X_train_1.shape[1],))

# Define the deep autoencoder architecture
encoded = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_layer)
encoded = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(encoded)
encoded = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01))(encoded)
encoded = Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01))(encoded)

decoded = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01))(encoded)
decoded = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(decoded)
decoded = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(decoded)
decoded = Dense(X_train_1.shape[1], activation='linear')(decoded)

# Define the autoencoder model
autoencoder = Model(input_layer, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
autoencoder.fit(X_train_1, X_train_1, epochs=50, batch_size=32)

# Use the encoder layer to transform the input data into a lower-dimensional representation
encoder = Model(input_layer, encoded)
X_encoded_4 = encoder.predict(X_test_1)

# Fit a Gaussian Mixture Model to the encoded data
gmm = GaussianMixture(n_components=3)  # Set the number of components as needed
gmm.fit(X_encoded_4)

# Predict the GMM component assignments for the encoded data
component_assignments = gmm.predict(X_encoded_4)

# Print the component assignments
print("Component Assignments:", component_assignments)