import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import BernoulliRBM  
from sklearn.pipeline import Pipeline  
from sklearn.linear_model import LogisticRegression  
from tensorflow.keras.layers import Input, Dense  
from tensorflow.keras.models import Model  
from tensorflow.keras import regularizers  

class DeepAutoencoderDBN:  
    def __init__(self, input_dim, encoding_dim=[64, 32, 16, 8], epochs=50, batch_size=32):  
        self.input_dim = input_dim  
        self.encoding_dim = encoding_dim  
        self.epochs = epochs  
        self.batch_size = batch_size  
        
        self.autoencoder, self.encoder = self.build_autoencoder()  
        self.dbn_classifier = None  
        
    def build_autoencoder(self):  
        # Define the input layer  
        input_layer = Input(shape=(self.input_dim,))  

        # Define the autoencoder architecture  
        encoded = input_layer  
        for units in self.encoding_dim:  
            encoded = Dense(units, activation='relu', kernel_regularizer=regularizers.l2(0.01))(encoded)  

        # Decoder architecture  
        decoded = encoded  
        for units in reversed(self.encoding_dim[:-1]):  
            decoded = Dense(units, activation='relu', kernel_regularizer=regularizers.l2(0.01))(decoded)  
        decoded = Dense(self.input_dim, activation='linear')(decoded)  

        # Create autoencoder and encoder models  
        autoencoder = Model(input_layer, decoded)  
        encoder = Model(input_layer, encoded)  
        
        # Compile the autoencoder model  
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')  
        return autoencoder, encoder  

    def train_autoencoder(self, X_train):  
        # Train the autoencoder  
        self.autoencoder.fit(X_train, X_train, epochs=self.epochs, batch_size=self.batch_size)  

    def encode(self, X):  
        # Use the encoder to transform data into lower-dimensional representation  
        return self.encoder.predict(X)  

    def build_dbn_classifier(self, n_components=64):  
        # Define the Deep Belief Network (DBN) classifier  
        rbm = BernoulliRBM(n_components=n_components, learning_rate=0.01,   
                           batch_size=10, n_iter=10, verbose=0, random_state=42)  
        logistic = LogisticRegression(solver='newton-cg', tol=1, multi_class='auto')  
        self.dbn_classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])  

    def train_dbn_classifier(self, X_encoded, y):  
        # Train the DBN classifier on the encoded data  
        self.dbn_classifier.fit(X_encoded, y)  

    def predict(self, X_encoded):  
        # Make predictions using the DBN classifier  
        return self.dbn_classifier.predict(X_encoded)  

# Example usage:  
# Load your data, here assuming X and y are defined  
# X should be the feature data and y the target labels  

# Splitting data into train and test sets  
X_train_1, X_test_1, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# Standardize the data (optional, but recommended)  
scaler = StandardScaler()  
X_train_1 = scaler.fit_transform(X_train_1)  
X_test_1 = scaler.transform(X_test_1)  

# Initialize the DeepAutoencoderDBN  
deep_autoencoder_dbn = DeepAutoencoderDBN(input_dim=X_train_1.shape[1])  

# Train the autoencoder  
deep_autoencoder_dbn.train_autoencoder(X_train_1)  

# Encode the test data  
X_encoded_5 = deep_autoencoder_dbn.encode(X_test_1)  

# Build and train the DBN classifier  
deep_autoencoder_dbn.build_dbn_classifier()  
deep_autoencoder_dbn.train_dbn_classifier(X_encoded_5, y_test)  

# Make predictions  
predictions = deep_autoencoder_dbn.predict(X_encoded_5)  

# Print the predictions  
print("DBN Classifier Predictions:", predictions)