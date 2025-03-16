

# Define the input layer
input_layer = Input(shape=(X_train_1.shape[1],))

# Define the encoder layers with L2 regularization
encoded_layer1 = Dense(16, activation='relu', activity_regularizer=regularizers.l2(0.01))(input_layer)
encoded_layer2 = Dense(8, activation='relu', activity_regularizer=regularizers.l2(0.01))(encoded_layer1)
encoded_layer3 = Dense(4, activation='relu', activity_regularizer=regularizers.l2(0.01))(encoded_layer2)
encoded_layer4 = Dense(2, activation='relu', activity_regularizer=regularizers.l2(0.01))(encoded_layer3)
encoded_layer5 = Dense(1, activation='relu', activity_regularizer=regularizers.l2(0.01))(encoded_layer4)

# Define the decoder layers with L2 regularization
decoded_layer1 = Dense(2, activation='relu', activity_regularizer=regularizers.l2(0.01))(encoded_layer5)
decoded_layer2 = Dense(4, activation='relu', activity_regularizer=regularizers.l2(0.01))(decoded_layer1)
decoded_layer3 = Dense(8, activation='relu', activity_regularizer=regularizers.l2(0.01))(decoded_layer2)
decoded_layer4 = Dense(16, activation='relu', activity_regularizer=regularizers.l2(0.01))(decoded_layer3)
decoded_layer5 = Dense(X_train_1.shape[1], activation='sigmoid', activity_regularizer=regularizers.l2(0.01))(decoded_layer4)

# Define the autoencoder model
autoencoder = Model(input_layer, decoded_layer5)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
autoencoder.fit(X_train_1, X_train_1, epochs=5, batch_size=32)

# Use the encoder layers to transform the input data into a lower-dimensional representation
encoder = Model(input_layer, encoded_layer5)

X_encoded_1 = encoder.predict(X_test_1)

# Print the shape of the encoded data
print("Encoded data shape:", X_encoded_1.shape)