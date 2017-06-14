# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 00:47:23 2017

@author: zjgsw
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_path = 'Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)
#print (rides.head())

dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    #print (dummies)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
#print (data.head())

quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std
    
    # Save data for approximately the last 21 days 
test_data = data[-21*24:]
#
## Now remove the test data from the data set 
data = data[:-21*24]
#
# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]

#print (features)
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
#
## Hold out the last 60 days or so of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]
input_nodes =3
hidden_nodes = 4
output_nodes = 1
weights_input_to_hidden = np.random.normal(0.0, input_nodes**-0.5, 
                                      (input_nodes, hidden_nodes))
#
weights_hidden_to_output = np.random.normal(0.0, hidden_nodes**-0.5, 
                                      (hidden_nodes, output_nodes))

print(weights_input_to_hidden)
def sigmoid(x):
    return 1/(1+np.exp(-x))
n_records = features.shape[0]
#print (n_records)
delta_weights_i_h = np.zeros(weights_input_to_hidden.shape)
delta_weights_h_o = np.zeros(weights_hidden_to_output.shape)

for X, y in zip(features.values, targets):
    print(X)
    hidden_inputs =  np.dot(X,weights_input_to_hidden)
    print (hidden_inputs)
    