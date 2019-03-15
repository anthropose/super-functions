# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 10:52:13 2019

@author: chari
"""

#%%
# import libraries
import MyFuncs as mf
import PreprocessingFuncs as pf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV


#%%
# import data
combo = mf.loadDF("cust_call_sale.csv", "csv")


#%%
# view df
combo.info()

# subset df to only consider those from OBCallHistory
combo.dropna(subset = ["Contacted"], inplace = True)


#%%
# build a model to describe the relationships between an outbound call and the
# probability of sale, controlling for any confounding attributes

# set up data
X = combo.drop(["sold"], axis = 1)
y = combo["sold"]

# is target variable imbalanced?
combo["sold"].value_counts() # yes

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, stratify = y)

# ensure we are working with a copy of the data and not a view
Xtrain = Xtrain.copy()
Xtest = Xtest.copy()
ytrain = ytrain.copy()
ytest = ytest.copy()


#%%
# preprocessing functions for Xtrain that don't work in a pipeline

# drop unnecessary features
# isolate features of interest
drop_cols = ["userid", "PhoneNumber_hashed"]
pf.drop_preprocessing(Xtrain, drop_cols)

# preprocess date-time features
# isolate features of interest
date_time_cols = ["Date_Time_account", "Date_Time_call", "Date_Time_sale"]
pf.datetime_preprocessing(Xtrain, date_time_cols)

# preprocess time features 
# isolate features of interest
time_cols = ["Call_Duration"]
pf.time_preprocessing(Xtrain, time_cols)

# preprocess currency features
# isolate features of interest
currency_cols = ["StickerPrice"]
pf.currency_preprocessing(Xtrain, currency_cols)

# preprocess strongly correlated numeric features
# isolate features of interest
correlated_cols = [["CreditScore1", "CreditScore2"], ["BureauIncome", "Income"]]
pf.correlated_preprocessing(Xtrain, correlated_cols)
    


#%%
# fit_transform missing values
# isolate features of interest
missing_cols = list(Xtrain.columns.values)
si = pf.fill_missing_constant_preprocessor(Xtrain, missing_cols)


#%%
# fit_transform numeric features with uniform distribution using KBinsDiscretizer
# isolate features of interest
numeric2kbins_uniform_cols = ["StickerPrice", "ClicksQuintile"]

# verify distribution
mf.Numplots(Xtrain[numeric2kbins_uniform_cols], "hist") # uniform distribution
kbd_uniform = pf.numeric2kbins_preprocessor(Xtrain, numeric2kbins_uniform_cols, strategy = "uniform", nbins = 10)


#%%
# fit_transform numeric features with normal distribution using KBinsDiscretizer
# isolate features of interest
numeric2kbins_norm_cols = ["TotalInboundCallLength", 
                           "FraudScore",
                           "MedianVehicleFuelEcon",
                           "MedianVehiclePrice",
                           "TradeInValueAmount",
                           "MedianVehicleMileage",
                           "NumberSavedVehicles",
                           "NumberSearches",
                           "UniqueModels",
                           "VehicleVelocity1",
                           "VehicleVelocity15",
                           "VehicleVelocity30",
                           "DeviceCount",
                           "Odometer",
                           "Call_Duration_totalsec",
                           "CreditScore1_CreditScore2_avg",
                           "BureauIncome_Income_avg"]

# verify distributions
mf.Numplots(Xtrain[numeric2kbins_norm_cols], "hist") # mostly normal dists
kbd_norm = pf.numeric2kbins_preprocessor(Xtrain, numeric2kbins_norm_cols, strategy = "quantile", nbins = 4)


#%%
# fit_transform numeric features requiring binarization
# isolate features of interest
numeric2binary_cols = ["InboundCallCount", 
                       "InboundChatCount", 
                       "InboundEmailCount",
                       "OutboundCallCount",
                       "OutboundEmailCount"]

# verify distributions
mf.Numplots(Xtrain[numeric2binary_cols], "hist") # discrete distributions
binarizer = pf.numeric2binary_preprocessor(Xtrain, numeric2binary_cols, threshold = 0.0)


#%%
# fit_transform categorical features requiring oridinal encoding
# isolate features of interest
cat2ordinal_cols = ["PhoneType", 
                   "LeadSource", 
                   "CensusGeoRegionAddress",
                   "ModeDeviceType",
                   "ModeVehicleType",
                   "ServiceProvider",
                   "AgeCategory",
                   "InboundCallContact"]

ordinal_encoders = pf.cat2ordinal_preprocessor(Xtrain, cat2ordinal_cols)


#%%
# build and fit the model
model = LogisticRegressionCV(cv = 10, solver = "liblinear", fit_intercept = False, penalty = "l1").fit(Xtrain, ytrain)


#%%
# transform Xtest data using preprocessing functions
pf.drop_preprocessing(Xtest, drop_cols)
pf.datetime_preprocessing(Xtest, date_time_cols)
pf.time_preprocessing(Xtest, time_cols)
pf.currency_preprocessing(Xtest, currency_cols)
pf.correlated_preprocessing(Xtest, correlated_cols)


#%%
# transform Xtest using transformers fit on Xtrain
# fill missing
pf.fill_missing_constant_transformer(Xtest, missing_cols = missing_cols, transformer = si)

# kbinsdiscretize uniform and normally distributed numeric features   
pf.numeric2kbins_transformer(Xtest, continuous_cols = numeric2kbins_uniform_cols, transformer = kbd_uniform)
pf.numeric2kbins_transformer(Xtest, continuous_cols = numeric2kbins_norm_cols, transformer = kbd_norm)

# numeric features to binary
pf.numeric2binary_transformer(Xtest, binary_cols = numeric2binary_cols, transformer = binarizer)

# categorical features to ordinal
pf.cat2ordinal_transformer(Xtest, ordinal_cols = cat2ordinal_cols, transformer = ordinal_encoders)


#%%
# evaluate best_estimator_ on test data
ypred = model.predict(Xtest)

# get R^2 score (coefficient of determination); the higher, the better
model.score(Xtest, ytest)

# get coefficients of model
model_coefficients = model.coef_
model_coefficients = np.reshape(model_coefficients, newshape = (63, 1))
mylist = map(lambda x: x[0], model_coefficients)
model_coefficients = pd.Series(mylist)
features = pd.DataFrame(data = Xtest.columns.values)
features["Coefficients"] = model_coefficients
features["log-odds"] = features["Coefficients"].apply(lambda x: np.log10(x))
features["probability"] = features["log-odds"].apply(lambda x: (np.exp(x)/(1+(np.exp(x)))))

model.get_params()
probabilities_Xtrain = pd.DataFrame(model.predict_proba(Xtrain))
probabilities_Xtest = pd.DataFrame(model.predict_proba(Xtest))


#%%
# get predictions for original data set X using the above model
# re-import full dataframe
all_data = mf.loadDF("cust_call_sale.csv", "csv")

# drop sold column
all_data.drop("sold", axis = 1, inplace = True)

# save userid column for later
user_id = all_data["userid"]

# must first transform the data as above
pf.drop_preprocessing(all_data, drop_cols)
pf.datetime_preprocessing(all_data, date_time_cols)
pf.time_preprocessing(all_data, time_cols)
pf.currency_preprocessing(all_data, currency_cols)
pf.correlated_preprocessing(all_data, correlated_cols)
pf.fill_missing_constant_transformer(all_data, missing_cols = missing_cols, transformer = si)
pf.numeric2kbins_transformer(all_data, continuous_cols = numeric2kbins_uniform_cols, transformer = kbd_uniform)
pf.numeric2kbins_transformer(all_data, continuous_cols = numeric2kbins_norm_cols, transformer = kbd_norm)
pf.numeric2binary_transformer(all_data, binary_cols = numeric2binary_cols, transformer = binarizer)
pf.cat2ordinal_transformer(all_data, ordinal_cols = cat2ordinal_cols, transformer = ordinal_encoders)

prediction_labels = pd.DataFrame(model.predict(all_data))
# combine probabilities with all_data
all_data = pd.concat([all_data, prediction_labels], axis = 1)
# rename columns to something more intuitive
all_data.rename(columns = {0: "Predicted_Sale"}, inplace = True)

# add userid back into dataframe
all_data["userid"] = user_id


#%%
# sort dataframe by those who are predicted to buy, those who have high income,
# and those with a referral discount
sort_cols = ["Predicted_Sale", "high_income", "ReferralAttachment"]
all_data.sort_values(by = sort_cols, ascending = False, inplace = True)

# extract just the relevant columns as a new dataframe
call_order = pd.DataFrame()
call_order["userid"] = all_data["userid"]
call_order["Call?"] = all_data["Predicted_Sale"]


#%%
# export dataframe
call_order.to_csv("customer_call_order.csv", index = False)














