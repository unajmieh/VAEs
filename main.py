# main.py  
import numpy as np  
from architectures.autoencoder_v1 import create_autoencoder, train_autoencoder, encode_data  
from architectures.autoencoder_v2 import create_autoencoder as create_autoencoder_v2, train_autoencoder as train_autoencoder_v2, encode_data as encode_data_v2  
from utils.data_loader import load_data  
from config import AUTOENCODER_CONFIGS  

# Load data  
X_train, X_test = load_data()  

# Using the first autoencoder  
config_v1 = AUTOENCODER_CONFIGS["version_1"]  
autoencoder_v1 = create_autoencoder(input_shape=X_train.shape[1], hidden_units=config_v1["hidden_units"])  
train_autoencoder(autoencoder_v1, X_train, config_v1["batch_size"], config_v1["epochs"])  
X_encoded_v1 = encode_data(autoencoder_v1, X_test)  

print("Encoded data from Autoencoder V1:\n", X_encoded_v1)  

# Using the second autoencoder  
config_v2 = AUTOENCODER_CONFIGS["version_2"]  
autoencoder_v2 = create_autoencoder(input_shape=X_train.shape[1], hidden_units=config_v2["hidden_units"])  
train_autoencoder(autoencoder_v2, X_train, config_v2["batch_size"], config_v2["epochs"])  
X_encoded_v2 = encode_data(autoencoder_v2, X_test)  

print("Encoded data from Autoencoder V2:\n", X_encoded_v2)