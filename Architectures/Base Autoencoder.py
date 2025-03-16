import numpy as np 
from keras.models import Model  
from tensorflow.keras.models import Model
from keras.layers import Input, Dense, Dropout  
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from keras.callbacks import EarlyStopping
from config import AUTOENCODER_CONFIGS 

#---- This is the model 1 --------------------
##############################################
def create_autoencoder_base(input_shape, hidden_units): 
    input_layer = Input(shape=(input_shape,))
    encoded_layer = Dense(hidden_units, activation='relu')(input_layer)
    decoded_layer = Dense(hidden_units, activation='sigmoid')(encoded_layer)
    autoencoder = Model(input_layer, decoded_layer)
    autoencoder.compile(loss="adam", loss='mean_squared_error')
    return autoencoder

def train_autoencoder(autoencoder, X_train, batch_size, epochs): 
    early_stop = EarlyStopping(monitor='loss', patience=2)
    autoencoder.fit(X_train,X_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stop])


def encode_data(autoencoder, X_test): 
    encoder = Model(autoencoder.input, autoencoder.layers[1].output)
    return encoder.predict(X_test)

#----------This is the model 2---depth in the layers---------
#############################################################
def create_autoencoder_V1(input_shape):
    # this is the encode layers -- here there are more depth
    input_layer =  Input(shape=(input_shape,))
    encoded_layer = Dropout(0.2)(encoded_layer)
    encoded_layer = Dense(32, activation="relu")(encoded_layer)
    encoded_layer = Dropout(0.2)(encoded_layer)
    encoded_layer= Dense(3, activation='relu')(encoded_layer)

    # Define the decoder layers -- here there are more depths as well
    decoded_layer = Dense(32, activation='relu')(encoded_layer)
    decoded_layer=Dropout(0.2)(decoded_layer)
    decoded_layer=Dense(54, activation='relu')(decoded_layer)
    decoded_layer=Dense(input_shape, activation='sigmoid')(decoded_layer)   

    autoencoder_v1 = Model(input_layer, decoded_layer)
    return autoencoder_v1

def compile_autoencoder_v1(autoencoder): 
    autoencoder.compile(optimizer="adam", loss='mean_squared_error')

def train_autoencoder(autoencoder, epoch,) -> None:
    history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=BATCH_SIZE,)

# ----- This is the model 3 --  different optimizer and loss function --------
##############################################################################
def create_autoencoder_v2(input_shape, condition_shape):
    input_layer = Input(shape=(input_shape,))
    condition_layer = Input(shape=(condition_shape,))
    encoded_layer = Dropout(0.2)(input_layer)
    encoded_layer = Dense(64, activation='relu')(encoded_layer)
    encoded_layer = Dropout(0.2)(encoded_layer)
    encoded_layer = Dense(32, activation='relu')(encoded_layer)
    concatenated_layer = Concatenate()([encoded_layer, condition_layer])
    latent_layer = Dense(16, activation='relu')(concatenated_layer)
    decoded_layer = Dense(32, activation='relu')(latent_layer)
    decoded_layer = Dropout(0.2)(decoded_layer)
    decoded_layer = Dense(64, activation='relu')(decoded_layer)
    output_layer = Dense(input_shape, activation='sigmoid')(decoded_layer)
    autoencoder = Model(inputs=[input_layer, condition_layer], outputs=output_layer)
    return autoencoder

def compile_conditional_autoencoder(autoencoder):
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

def train_autoencoder_v2(autoencoder, X_train, conditions, epochs, batch_size):
    history = autoencoder.fit([X_train, conditions], X_train, epochs=epochs, batch_size=batch_size)
    return history

# Example usage:
input_shape = 784  # For MNIST data (28x28)
condition_shape = 10  # 10 classes for MNIST
autoencoder_v2 = create_autoencoder_v2(input_shape, condition_shape)
autoencoder_v2 = compile_conditional_autoencoder(autoencoder_v2)
epochs = 20
batch_size = 128
# Assuming X_train and conditions are already prepared
history = train_autoencoder_v2(autoencoder_v2, X_train, conditions, epochs, batch_size)

# ----- This is the model 4 - for categorical data type -------------
#####################################################################
def create_autoencoder_v3(input_shape, condition_shape):
    # Main Input Layer
    input_layer = Input(shape=(input_shape,))
    
    # Condition Input Layer (Categorical or one-hot encoded labels)
    condition_layer = Input(shape=(condition_shape,))
    
    # Encoder
    encoded_layer = Dropout(0.2)(input_layer)
    encoded_layer = Dense(64, activation='relu')(encoded_layer)
    encoded_layer = Dropout(0.2)(encoded_layer)
    encoded_layer = Dense(32, activation='relu')(encoded_layer)
    
    # Concatenation Layer
    concatenated_layer = Concatenate()([encoded_layer, condition_layer])
    
    # Further encoding to Latent Space
    latent_layer = Dense(16, activation='relu')(concatenated_layer)
    
    # Decoder
    decoded_layer = Dense(32, activation='relu')(latent_layer)
    decoded_layer = Dropout(0.2)(decoded_layer)
    decoded_layer = Dense(64, activation='relu')(decoded_layer)
    
    # Output Layer
    output_layer = Dense(input_shape, activation='sigmoid')(decoded_layer)
    
    # Create the autoencoder model
    autoencoder_v3 = Model(inputs=[input_layer, condition_layer], outputs=output_layer)
    
    return autoencoder_v3

def compile_autoencoder_v3(autoencoder):
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

def train_autoencoder_v3(autoencoder, X_train, conditions, epochs, batch_size):
    history = autoencoder.fit([X_train, conditions], X_train, epochs=epochs, batch_size=batch_size)
    return history

# Example usage:
input_shape = 100  # Example input shape
condition_shape = 5  # Example for one-hot encoded categorical labels (5 classes)
autoencoder_v3 = create_autoencoder_v3(input_shape, condition_shape)
autoencoder_v3 = compile_autoencoder_v3(autoencoder_v3)

# Example data
X_train = np.random.rand(1000, input_shape)  # 1000 samples
conditions = np.random.randint(0, 2, (1000, condition_shape))  # One-hot encoded conditions

# Train the model
epochs = 50
batch_size = 32
history = train_autoencoder_v2(autoencoder_v3, X_train, conditions, epochs, batch_size)
