
# changed funciton some to make graph for presentation work correctly

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def train_eval_model_vis(X, y, model, time_series_split, og_dataset, months_in_adv):
    """ Model training and evaluation for time series data """

    mse_scores = []
    r2_scores = []
    test_results = []

    scaler = StandardScaler()

    for i, (train_index, test_index) in enumerate(time_series_split.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        train_dates_min = og_dataset.iloc[train_index]['date'].min().strftime('%Y-%m')
        train_dates_max = og_dataset.iloc[train_index]['date'].max().strftime('%Y-%m')

        prediction_dates = og_dataset.iloc[test_index]['date']

        actual_pce_dates = prediction_dates.reset_index(drop=True) + pd.DateOffset(months=months_in_adv)

        test_dates_min = actual_pce_dates.min().strftime('%Y-%m')
        test_dates_max = actual_pce_dates.max().strftime('%Y-%m')

 
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

        baseline_values = og_dataset.iloc[test_index]['pce'].reset_index(drop=True) 

        split_tests = pd.DataFrame({
            'date': actual_pce_dates, 
            'actual_pce': y_test.reset_index(drop=True), 
            'predicted_pce': y_pred, 
            'baseline_pce': baseline_values, 
            'split': i + 1
        })
        test_results.append(split_tests)
        last_prediction = split_tests.iloc[-1]

    return mse_scores, r2_scores, test_results