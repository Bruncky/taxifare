import pandas as pd

from sklearn.model_selection import train_test_split

def get_data(line_count = 100):
    url = 's3://wagon-public-datasets/taxi-fare-train.csv'
    data = pd.read_csv(url, nrows = line_count)
    
    return data

def clean_data(data):
    # Drop NaNs
    data = data.dropna(how = 'any', axis = 'rows')
    
    # Keep lat and lon if not zero
    data = data[(data.dropoff_latitude != 0) | (data.dropoff_longitude != 0)]
    data = data[(data.pickup_latitude != 0) | (data.pickup_longitude != 0)]
    
    # Keep fare amounts between 0 and 4K
    if 'fare_amount' in list(data):
        data = data[data.fare_amount.between(0, 4000)]
    
    # Keep passengers between 1 and 8
    data = data[data.passenger_count < 8]
    data = data[data.passenger_count >= 1]

    # Keep lat and lon between a certain range
    data = data[data['pickup_latitude'].between(left = 40, right = 42)]
    data = data[data['pickup_longitude'].between(left = -74.3, right = -72.9)]
    data = data[data['dropoff_latitude'].between(left = 40, right = 42)]
    data = data[data['dropoff_longitude'].between(left = -74, right = -72.9)]

    return data

def holdout(data):
    X_train = data.drop('fare_amount', axis = 1)
    y_train = data['fare_amount']

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.1)

    return (X_train, X_test, y_train, y_test)