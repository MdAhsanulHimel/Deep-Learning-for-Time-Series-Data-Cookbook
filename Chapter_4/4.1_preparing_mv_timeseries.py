import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

# loading a multivariate time series
mvtseries = pd.read_csv('assets/datasets/time_series_smf1.csv',
                        parse_dates=['datetime'],
                        index_col='datetime')

stat_by_variable = {
    'Incoming Solar': 'sum',
    'Wind Dir': 'mean',
    'Snow Depth': 'sum',
    'Wind Speed': 'mean',
    'Dewpoint': 'mean',
    'Precipitation': 'sum',
    'Vapor Pressure': 'mean',
    'Relative Humidity': 'mean',
    'Air Temp': 'max',
}

mvtseries = mvtseries.resample('D').agg(stat_by_variable)
mvtseries = mvtseries.ffill()

mvtseries.to_csv('assets/daily_multivariate_timeseries.csv')

TARGET = 'Incoming Solar'
N_LAGS = 3
HORIZON = 1

input_data = []
output_data = []
for i in range(N_LAGS, mvtseries.shape[0] - HORIZON + 1):
    input_data.append(mvtseries.iloc[i - N_LAGS:i].values)
    output_data.append(mvtseries.iloc[i:(i + HORIZON)][TARGET])

input_data, output_data = np.array(input_data), np.array(output_data)



# USING TIMESERIESDATASET

mvtseries.T.head(5)  # transpose for viewing purpose
mvtseries.head(5)

mvtseries['time_index'] = np.arange(mvtseries.shape[0])
mvtseries['group_id'] = 0

# create the dataset from the pandas dataframe
dataset = TimeSeriesDataSet(
    data=mvtseries,                # pandas dataframe
    group_ids=["group_id"],        # name of column with group ids
    target="Incoming Solar",       # name of column with target values
    time_idx="time_index",         # name of column with time index
    max_encoder_length=3,          # maximum length of encoder (input)
    max_prediction_length=1,       # maximum length of prediction (output)
    time_varying_unknown_reals=[
        'Incoming Solar',
        'Wind Dir',
        'Snow Depth',
        'Wind Speed',
        'Dewpoint',
        'Precipitation',
        'Vapor Pressure',
        'Relative Humidity',
        'Air Temp',
    ],                             # names of columns with time-varying but unknown reals
)

# Create DataLoader
data_loader = dataset.to_dataloader(batch_size=1, shuffle=False)

# Extract One Batch of Data from DataLoader
x, y = next(iter(data_loader))  # Get first batch

print(mvtseries)

# Print Extracted Data from DataLoader
# Past numerical features (e.g., sales, day_of_week)
print("\n📌 Encoder Continuous Features (Past Data):\n", x['encoder_cont'])

# The ground truth sales values for future prediction
print("\n📌 Target Values for Forecasting (Future Sales):\n", y)

# ✅ Explanation of Extracted Data
# - `encoder_cont`: Contains past numerical values used as input (normalized)
# - `encoder_target`: Stores actual past sales values
# - `decoder_target`: The actual sales values we want the model to predict
# - `decoder_cont`: The features for the prediction period (e.g., future day_of_week)

# This setup enables the model to learn from past sequences and predict future values efficiently. 🚀🔥
