import pandas as pd




def load_data(file_path, months_in_adv):
    """Loading data from csv and preparing it for modeling."""
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data['revolving_credit'] = data['revolving_credit'].fillna(0)
    data['pce_future'] = data['pce'].shift(-months_in_adv)

    return data


def add_lag_features(df, months_lag, features):
    """Adding lag features to the dataframe, based on the number of months in advance the model is trained for."""
    df_copy = df.copy()
    for feature in features:
        for i in range(1, months_lag + 1):
            df_copy[f'{feature}_lag_{i}'] = df_copy[feature].shift(i)
    return df_copy


