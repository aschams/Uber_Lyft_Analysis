# Uber/Lyft Analysis
#### By Anthony Schams and Phoebe Wong

The purpose of this analysis is to explore the best conditions on driving for Uber and Lyft. Models for prices, predicting price surges, and basic time series analysis were performed. 

## Data
Data was obtained from [Kaggle](https://www.kaggle.com/ravi72munde/uber-lyft-cab-prices). Both the 'cab_rides.csv' and 'weather.csv' are used. Cleaning of the data was performed using the code in the file cleaning_data.ipynb. 

## Analysis
Analysis and some discussion can be found in the analysis.ipynb file. It includes a walkthrough of the most important models we built, alongside examination of statistical assumptions for the analysis performed. 

We performed LASSO to find a linear model to predict prices from a subset of features selected through random forest. 

We perform logistic regression in an attempt to predict when a surge multiplier will be applied to the price of a ride. Because the dataset includes only surge multipliers for Lyft rides, we were forced to limit our analysis to those. 

We perform time series analysis on the average price of rides by hour of the day.

Additional functions used can be found in the extra_functions.py file

## Conclusions
We found that the best predictors for price are log(distance) and the starting neighborhood of the driver.

Time series analysis revealed no form of trend or seasonality in the data; the average price of a trip was essentially random.
