import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

# Load your IoT dataset into a pandas DataFrame
# Assuming your dataset is stored in a variable called 'iot_data'
# Replace 'target_column' with the name of the target variable column in your dataset
# Replace the column names with the actual feature names in your dataset
# X = iot_data.drop(['target_column'], axis=1)  # Features
# y = iot_data['target_column']  # Target variable

# Select the top 5 features based on the ANOVA F-test
# selector = SelectKBest(score_func=f_classif, k=5)
# X_new = selector.fit_transform(X, y)

# Define the input layer
input_layer = Input(shape=(X_train.shape[1],))

# Define the stacked encoder LSTM layers with dropout and L2 regularization
encoded = LSTM(16, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(input_layer)
encoded = Dropout(0.2)(encoded)
encoded = LSTM(8, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(encoded)
encoded = Dropout(0.2)(encoded)
encoded = LSTM(4, activation='relu', kernel_regularizer=regularizers.l2(0.01))(encoded)

# Define the stacked decoder LSTM layers with dropout and L2 regularization
decoded = RepeatVector(X_train.shape[1])(encoded)
decoded = LSTM(8, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(decoded)
decoded = Dropout(0.2)(decoded)
decoded = LSTM(16, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(decoded)
decoded = Dropout(0.2)(decoded)
decoded = LSTM(X_train.shape[1], activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(decoded)

# Define the autoencoder model
autoencoder = Model(input_layer, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
autoencoder.fit(X_train_1, X_train_1, epochs=50, batch_size=32)

# Use the encoder layer to transform the input data into a lower-dimensional representation
encoder = Model(input_layer, encoded)
X_encoded_3 = encoder.predict(X_test_1)

# Print the shape of the encoded data
print("Encoded data shape:", X_encoded_3.shape)

#------------------ Model 2 ------------ 
########################################

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

# Load your IoT dataset into a pandas DataFrame
# Assuming your dataset is stored in a variable called 'iot_data'
# Replace 'target_column' with the name of the target variable column in your dataset
# Replace the column names with the actual feature names in your dataset
X = iot_data.drop(['target_column'], axis=1)  # Features
y = iot_data['target_column']  # Target variable

# Select the top 5 features based on the ANOVA F-test
selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X, y)

# Define the input layer
input_layer = Input(shape=(X_new.shape[1],))

# Define the encoder LSTM layers with dropout and L2 regularization
encoded = LSTM(16, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(input_layer)
encoded = Dropout(0.2)(encoded)
encoded = LSTM(8, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(encoded)
encoded = Dropout(0.2)(encoded)
encoded = LSTM(4, activation='relu', kernel_regularizer=regularizers.l2(0.01))(encoded)

# Repeat the encoded representation to the original sequence length
decoded = RepeatVector(X_new.shape[1])(encoded)

# Define the decoder LSTM layers with dropout and L2 regularization
decoded = LSTM(8, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(decoded)
decoded = Dropout(0.2)(decoded)
decoded = LSTM(16, activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(decoded)
decoded = Dropout(0.2)(decoded)
decoded = LSTM(X_new.shape[1], activation='relu', kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(decoded)

# Define the autoencoder model
autoencoder = Model(input_layer, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
autoencoder.fit(X_new, X_new, epochs=50, batch_size=32)

# Use the encoder layer to transform the input data into a lower-dimensional representation
encoder = Model(input_layer, encoded)
X_encoded = encoder.predict(X_new)

# Print the shape of the encoded data
print("Encoded data shape:", X_encoded.shape)