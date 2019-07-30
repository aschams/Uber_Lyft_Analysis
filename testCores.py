#!/usr/bin/python

import pandas as pd
import re
import math
import numpy as np
import scipy
import os
import time
import seaborn as sns
from scipy.stats import binom, hypergeom
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


merged_df = pd.read_csv('merged_df.csv')

merged_df.head()

# # Random Forest Regressor

dataSet_df = merged_df[['distance', 'surge_multiplier', 'temp', 'clouds', 'pressure',
                        'rain', 'humidity', 'wind', 'name_Black', 'name_Black SUV',
                        'name_Lux', 'name_Lux Black', 'name_Lux Black XL', 'name_Lyft',
                        'name_Lyft XL', 'name_Shared', 'name_Taxi', 'name_UberPool', 'name_UberX',
                        'name_UberXL', 'name_WAV', 'destination_Back Bay', 'destination_Beacon Hill',
                        'destination_Boston University', 'destination_Fenway', 'destination_Financial District',
                        'destination_Haymarket Square', 'destination_North End', 'destination_North Station',
                        'destination_Northeastern University', 'destination_South Station',
                        'destination_Theatre District', 'destination_West End', 'source_Back Bay',
                        'source_Beacon Hill', 'source_Boston University', 'source_Fenway',
                        'source_Financial District', 'source_Haymarket Square',
                        'source_North End', 'source_North Station',
                        'source_Northeastern University', 'source_South Station',
                         'source_Theatre District', 'source_West End','price']]





XX= dataSet_df[['distance', 'surge_multiplier', 'temp', 'clouds', 'pressure',
                        'rain', 'humidity', 'wind', 'name_Black', 'name_Black SUV',
                        'name_Lux', 'name_Lux Black', 'name_Lux Black XL', 'name_Lyft',
                        'name_Lyft XL', 'name_Shared', 'name_Taxi', 'name_UberPool', 'name_UberX',
                        'name_UberXL', 'name_WAV', 'destination_Back Bay', 'destination_Beacon Hill',
                        'destination_Boston University', 'destination_Fenway', 'destination_Financial District',
                        'destination_Haymarket Square', 'destination_North End', 'destination_North Station',
                        'destination_Northeastern University', 'destination_South Station',
                        'destination_Theatre District', 'destination_West End', 'source_Back Bay',
                        'source_Beacon Hill', 'source_Boston University', 'source_Fenway',
                        'source_Financial District', 'source_Haymarket Square',
                        'source_North End', 'source_North Station',
                        'source_Northeastern University', 'source_South Station',
                         'source_Theatre District', 'source_West End']]
YY = dataSet_df['price']


X_train, X_test, y_train, y_test = train_test_split(XX.values, YY.values, test_size=0.33, random_state=42)

regr = RandomForestRegressor(max_depth=4, random_state=137, n_estimators=1000)
regr.fit(X_train, y_train)

y_rf = regr.predict(X_test)


for fimp in list(zip(XX.columns, regr.feature_importances_)):
    print(fimp)



#plt.hist(y_test-y_rf, bins=50)

