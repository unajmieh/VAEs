import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

def contractive_autoencoder(encoding_dim, lam=1e-4):
    # Define the input layer
    inputs = Input(shape=(X_train.shape[1],))
    
    # Encoder
    encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l2(lam))(inputs)
    
    # Decoder
    decoded = Dense(X_train.shape[1], activation='sigmoid')(encoded)
    
    # Define the contractive loss function
    def contractive_loss(y_true, y_pred):
        mse = K.mean(K.square(y_true - y_pred), axis=1)
        
        # Compute the Jacobian of the encoder output with respect to the input
        encoder = Model(inputs, encoded)
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            encoded_output = encoder(inputs)
        jacobian = tape.batch_jacobian(encoded_output, inputs)
        jacobian_norm = K.sqrt(K.sum(K.square(jacobian), axis=(1, 2)))
        
        # Compute the contractive loss
        return mse + lam * jacobian_norm
    
    # Create the autoencoder model
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=Adam(), loss=contractive_loss)
    
    return autoencoder

# Instantiate the contractive autoencoder model
encoding_dim = 32  # Dimension of the encoded representation
ca_model = contractive_autoencoder(encoding_dim)

# Example synthetic data (as placeholder for X_train)
X_train = np.random.rand(1000, 20)  # 1000 samples, 20 features

# Train the contractive autoencoder
ca_model.fit(X_train, X_train, epochs=50, batch_size=32)

print("Training complete!")
