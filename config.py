# config.py  
AUTOENCODER_CONFIGS = {  
    "version_1": {  
        "hidden_units": 3,  
        "batch_size": 32,  
        "epochs": 10,  
        "activation": "relu",  
        "optimizer": "adam",  
    },  
    "version_2": {  
        "hidden_units": 5,  
        "batch_size": 32,  
        "epochs": 15,  
        "activation": "relu",  
        "optimizer": "adam",  
    },  
    # Add more versions as needed...  
}