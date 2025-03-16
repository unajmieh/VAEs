import numpy as np  
from sklearn.preprocessing import StandardScaler  
from keras.layers import Input, Dense  
from keras.models import Model  
from keras import regularizers  
from keras.optimizers import Adam  
import keras.backend as K  

class AdversarialAutoencoder:  
    def __init__(self, input_dim, encoding_dim=[64, 32, 16, 8], decoding_dim=[16, 32, 64],   
                 discriminator_units=[16], l2_reg=0.01, epochs=50, batch_size=32):  
        self.input_dim = input_dim  
        self.encoding_dim = encoding_dim  
        self.decoding_dim = decoding_dim  
        self.discriminator_units = discriminator_units  
        self.l2_reg = l2_reg  
        self.epochs = epochs  
        self.batch_size = batch_size  
        
        self.autoencoder = self.build_autoencoder()  
        self.discriminator = self.build_discriminator()  
        self.adversarial_autoencoder = self.build_adversarial_autoencoder()  
        
    def build_autoencoder(self):  
        input_layer = Input(shape=(self.input_dim,))  

        # Encoder  
        encoded = input_layer  
        for units in self.encoding_dim:  
            encoded = Dense(units, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg))(encoded)  

        # Decoder  
        decoded = encoded  
        for units in self.decoding_dim:  
            decoded = Dense(units, activation='relu', kernel_regularizer=regularizers.l2(self.l2_reg))(decoded)  

        decoded = Dense(self.input_dim, activation='linear')(decoded)  

        autoencoder = Model(input_layer, decoded)  
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')  
        return autoencoder  

    def build_discriminator(self):  
        discriminator_input = Input(shape=(self.encoding_dim[-1],))  
        discriminator = Dense(self.discriminator_units[0], activation='relu')(discriminator_input)  
        discriminator_output = Dense(1, activation='sigmoid')(discriminator)  
        discriminator_model = Model(discriminator_input, discriminator_output)  
        discriminator_model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy')  
        return discriminator_model  

    def build_adversarial_autoencoder(self):  
        input_layer = Input(shape=(self.input_dim,))  
        encoder_output = self.autoencoder.layers[-1](self.autoencoder.layers[-2](self.autoencoder.layers[-3](self.autoencoder.layers[-4](self.autoencoder.layers[-5](input_layer)))))  
        discriminator_output = self.discriminator(encoder_output)  
        adversarial_autoencoder = Model(input_layer, [encoder_output, discriminator_output])  
        adversarial_autoencoder.compile(optimizer='adam', loss=['mean_squared_error', 'binary_crossentropy'], loss_weights=[0.999, 0.001])  
        return adversarial_autoencoder  

    def train(self, X):  
        # Standardize the input data  
        scaler = StandardScaler()  
        X_scaled = scaler.fit_transform(X)  
        real_data = X_scaled  

        # Train the adversarial autoencoder  
        for epoch in range(self.epochs):  
            discriminator_loss = []  
            for _ in range(100):  
                # Generate a batch of real and fake data  
                real_indices = np.random.randint(0, X_scaled.shape[0], size=self.batch_size)  
                real_batch = X_scaled[real_indices]  
                
                # Generate fake data  
                fake_data = K.eval(self.autoencoder.predict(X_scaled[np.random.randint(0, X_scaled.shape[0], size=self.batch_size)]))  
                
                # Train the discriminator  
                discriminator_loss.append(self.discriminator.train_on_batch(real_batch, np.ones((self.batch_size, 1))))  
                discriminator_loss.append(self.discriminator.train_on_batch(fake_data, np.zeros((self.batch_size, 1))))  

            # Freeze the autoencoder weights for the discriminator training  
            for layer in self.autoencoder.layers:  
                layer.trainable = False  

            # Train the autoencoder independently  
            autoencoder_loss = self.autoencoder.train_on_batch(X_scaled, X_scaled)  

            # Unfreeze the autoencoder weights for adversarial autoencoder training  
            for layer in self.autoencoder.layers:  
                layer.trainable = True  

            # Train adversarial autoencoder  
            adversarial_autoencoder_loss = self.adversarial_autoencoder.train_on_batch(X_scaled, [X_scaled, np.ones((X_scaled.shape[0], 1))])  

            # Print the loss for each epoch  
            print('Epoch %d: Autoencoder Loss = %f, Discriminator Loss = %f, Adversarial Loss = %f' % (  
                epoch, autoencoder_loss, np.mean(discriminator_loss), adversarial_autoencoder_loss[0]))  

# Example usage:  
# X is your dataset  
# ae = AdversarialAutoencoder(input_dim=X.shape[1])  
# ae.train(X)