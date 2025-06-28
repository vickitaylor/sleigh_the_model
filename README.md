# 🎄 Sleigh the Model: Predicting Holiday Spending 🎁

## Project Overview

This project, "Sleigh the Model," develops and evaluates a machine learning model to predict holiday spending for December 2025. Using diverse of economic indicators and historical spending patterns (with data from 1959 to April 2025), this model aims to provide valuable insights for businesses to make data driven decisions for holiday season.

The primary goal is to identify key economic drivers influencing holiday spending and provide a robust forecasting tool.

[View my Capstone Presentation](visuals/Capstone%20Presentation.pptx)

## Motivation
I thought it would be interesting to see if economic historical factors can predict future spending, in an unusual economic climate that there has been in the past few years (high inflation, supply chain issues, and post pandemic recovery). Holiday spending can be a good indicator or economic health and consumer confidence. 


## Key Steps

* Data Collection & Cleaning 
* Feature Engineering 
* Machine Learning Model Development 
* Model Evaluation & Optimization 
* Future Spending Forecasting 


## Methodology

1.  **Data Collection & Wrangling:** 
    - Gathering time-series economic indicators and historical holiday spending data.
    - Date manipulation to have dates aligned due to all sources had different date formats
    - Missing values: For revolving credit (credit cards) data was unavailable for the first 10 years, due to credit cards not being as popular in that time frame. Replaced NaN values with 0 as it is realistic that most consumers did not use that form of credit at the time. 

2.  **Feature Engineering:** 
    - Created lag features based on the model forecasting value for all indicators (PCE, inflation, unemployment, and debt) to capture historical trends

3.  **Model Selection:** 
    - Tested and evaluated multiple regression models to find the best performing model including: 
        - Baseline (previous year values)
        - Lasso Regression
        - Ridge Regression
        - Linear Regression
        - XG Boost Regressor
        - Random Forest Regressor 
        - Cat Boost Regressor

4.  **Training & Validation:** 
    - Used TimeSeriesSplit for cross validation, using 5 year for the splits
    - Evaluated performance based on r squared and mean square error values

5.  **Forecasting:** 
    - Using the trained model to predict December 2025 holiday spending, using recursive forecasting. 


## Results and Insights
* **Prediction for December 2025:** 
    - The model predicts an estimated $45 trillion consumer spending for December 2025. The amount the model found seems to be over estimating the amount. Assuming the exponential estimate is due to the recursive forecast used.
* **Key Drivers:** 
    - Found there to be an inverse relationship with consumer spending and the unemployment rate. 
    - All other factors used had a strong positive correlation with each other. 
* **Model Performance:** 
    - Used a Lasso Regression model. The average R2 value was 0.83, with an average mean squared error of 151,706.30 which was the best preforming model used on the test set


## Future Enhancements
* Explore additional machine learning models (e.g. ARIMA and SARIMA)
* Incorporate more granular spending data or regional economic indicators
* Develop an interactive dashboard with Streamlit to visualize predictions and allow for custom scenario analysis


## Languages Used 
* Python
* Pandas
* scikit-learn
* Matplotlib
* Seaborn


## Data Sources

* [Inflation: Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/series/CPIAUCSL)
* [Unemployment: US Bureau of Labor Statistics](https://data.bls.gov/pdq/SurveyOutputServlet)
* [Consumer Debt: Federal Reserve](https://www.federalreserve.gov/datadownload/Download.aspx?rel=G19&series=be2df920f30707fd397c306408143a6c&lastObs=&from=&to=&filetype=csv&label=include&layout=seriescolumn&type=package)
* [Personal Consumption Expenditures: Bureau of Economic Analysis](https://apps.bea.gov/iTable/)



