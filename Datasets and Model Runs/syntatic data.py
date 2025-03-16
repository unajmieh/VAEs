import numpy as np
import pandas as pd

# Set the random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generating synthetic feature data
X = pd.DataFrame({
    'feature1': np.random.rand(n_samples),
    'feature2': np.random.rand(n_samples),
    'feature3': np.random.rand(n_samples),
    'feature4': np.random.rand(n_samples),
    'feature5': np.random.rand(n_samples)
})

# Generating synthetic target variable
y = np.random.randint(0, 2, size=n_samples)

# Combine features and target into a single DataFrame
synthetic_data = X.copy()
synthetic_data['target_column'] = y

# Display the first few rows of the synthetic data
print(synthetic_data.head())
