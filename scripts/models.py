import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def train_eval_model(X, y, model, time_series_split, og_dataset, model_name): 
    """ Model training and evaluation for time series data """    
    
    mse_scores = []
    r2_scores = []
    test_results = []

    print(f"*** {model_name} ***")

    for i, (train_index, test_index) in enumerate(time_series_split.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        train_dates_min = og_dataset.iloc[train_index]['date'].min().strftime('%Y-%m')
        train_dates_max = og_dataset.iloc[train_index]['date'].max().strftime('%Y-%m')
        test_dates_min = og_dataset.iloc[test_index]['date'].min().strftime('%Y-%m')
        test_dates_max = og_dataset.iloc[test_index]['date'].max().strftime('%Y-%m')
        test_dates = og_dataset.iloc[test_index]['date']

        print(f"\n--- Split {i + 1} ---")
        print(f"  TRAIN: {len(train_index)} months ({train_dates_min} to {train_dates_max})")
        print(f"  TEST: {len(test_index)} months ({test_dates_min} to {test_dates_max})")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

        split_tests = pd.DataFrame({
            'date': test_dates.reset_index(drop=True),
            'actual_pce': y_test.reset_index(drop=True),
            'predicted_pce': y_pred,
            'split': i + 1
        })
        test_results.append(split_tests)
        last_prediction = split_tests.iloc[-1]

        print("Model Performance:")
        print(f"    Mean Squared Error: {mse : .2f}")
        print(f"    R-squared: {r2 : .2f}")
        print(f"    Actual PCE: {last_prediction['actual_pce']:.2f}") 
        print(f"    Predicted PCE: {last_prediction['predicted_pce']:.2f}")

    return mse_scores, r2_scores, test_results
