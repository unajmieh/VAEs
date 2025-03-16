from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

# Define the input layer
input_layer = Input(shape=(input_dim,))

# Define the encoder architecture with sparsity constraint
encoded = Dense(64, activation='relu', activity_regularizer=regularizers.l1(0.01))(input_layer)
encoded = Dense(32, activation='relu', activity_regularizer=regularizers.l1(0.01))(encoded)
encoded = Dense(16, activation='relu', activity_regularizer=regularizers.l1(0.01))(encoded)
encoded = Dense(8, activation='relu', activity_regularizer=regularizers.l1(0.01))(encoded)

# Define the decoder architecture
decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(32, activation='relu')(decoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)

# Define the autoencoder model
autoencoder = Model(input_layer, decoded)

# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')