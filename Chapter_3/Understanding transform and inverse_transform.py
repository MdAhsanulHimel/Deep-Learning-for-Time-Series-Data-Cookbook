import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Step 1: Create Dummy Time Series Data
data = np.array([[10], [20], [30], [40], [50], [60], [70], [80], [90], [100]])
df = pd.DataFrame(data, columns=['Original Data'])

# Step 2: Initialize MinMaxScaler (Scaling Data to Range 0 to 1)
scaler = MinMaxScaler(feature_range=(0, 1))

# Step 3: Fit & Transform Training Data (Scale Between 0 and 1)
scaled_data = scaler.fit_transform(data)

# Convert Back to DataFrame for Visualization
df['Scaled Data'] = scaled_data

# Step 4: Apply Inverse Transform (Convert Scaled Data Back to Original Scale)
inversed_data = scaler.inverse_transform(scaled_data)
df['Inversed Data'] = inversed_data

# Print Dataframe
print(df)
