# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:59:33 2019

@author: chari
"""

#%%
# import libraries
import MyFuncs as mf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics 

#%%
# import data
combo = mf.loadDF("cust_call_sale.csv", "csv")


#%%
# view df
combo.info()


#%%
# build a model to estimate probability that a customer will purchase a car
# within 45 days of account creation

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
def drop_preprocessing(df):
    for name in drop_cols:
        df.drop(name, axis = 1, inplace = True)
    return df
drop_preprocessing(Xtrain)

# preprocess date-time features
# isolate features of interest
date_time_cols = ["Date_Time_account", "Date_Time_call", "Date_Time_sale"]
def datetime_preprocessing(df):
    # convert all date-time features into date-time objects
    for name in date_time_cols:
        df[name] = pd.to_datetime(df[name]) 
    # parse all date-time features into new features for month, day, hour
        df[name + "_month"] = df[name].apply(lambda x: x.month)
        df[name + "_day"] = df[name].apply(lambda x: x.day)
        df[name + "_hour"] = df[name].apply(lambda x: x.hour) 
    # drop date-time objects
        df.drop(name, axis = 1, inplace = True)
    return df
datetime_preprocessing(Xtrain)

# preprocess time features 
# isolate features of interest
time_cols = ["Call_Duration"]
def time_preprocessing(df):
    # convert time feature into time object
    for name in time_cols:
        df[name] = pd.to_datetime(
                df[name], errors = "raise", format = "%H:%M:%S").dt.time
    # calculate total seconds into new feature
        df[name + "_totalsec"] = df[name].apply(
                lambda x: x.hour * 3600 + x.minute * 60 + x.second)
    # drop time object
        df.drop(name, axis = 1, inplace = True)
    return df
time_preprocessing(Xtrain)

# preprocess currency features
# isolate features of interest
currency_cols = ["StickerPrice"]
def currency_preprocessing(df):
    for name in currency_cols:
        df[name] = df[name].str.replace("$", "")
        df[name] = df[name].str.replace(",", "")
        df[name] = df[name].astype("float64")
    return df
currency_preprocessing(Xtrain)


#%%
# preprocessing pipeline for numeric features with uniform distribution
# requiring KBinsDiscretizer
    
# isolate features of interest
numeric2kbins_uniform_cols = ["StickerPrice", 
                              "ClicksQuintile"]

# verify distribution
mf.Numplots(Xtrain[numeric2kbins_uniform_cols], "hist") # uniform distribution

# pipeline to transform numeric feature to a kbinsdiscretized feature that is
# one hot encoded
numeric2kbins_uniform_transformer = Pipeline(steps = [
        ("si", SimpleImputer(missing_values = np.nan, 
                             strategy = "constant", 
                             fill_value = -9999)),
        ("kbd", KBinsDiscretizer(
                n_bins = 5, 
                encode = "ordinal", 
                strategy = "uniform")),
        ("ohe", OneHotEncoder(
                categories = "auto",
                sparse = False,
                dtype = int,
                handle_unknown = "ignore"))])


#%%
# preprocessing pipeline for numeric features with normal distribution
# requiring KBinsDiscretizer

# isolate features of interest
numeric2kbins_norm_cols = ["TotalInboundCallLength", 
                           "BureauIncome", 
                           "CreditScore1", 
                           "CreditScore2",
                           "FraudScore",
                           "Income",
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
                           "Call_Duration_totalsec"]

# verify distributions
mf.Numplots(Xtrain[numeric2kbins_norm_cols], "hist") # mostly normal dists

# pipeline to transform numeric feature to a kbinsdiscretized feature that is
# one hot encoded
numeric2kbins_norm_transformer = Pipeline(steps = [
        ("si", SimpleImputer(
                missing_values = np.nan, 
                strategy = "constant", 
                fill_value = -9999)),
        ("kbd", KBinsDiscretizer(
                n_bins = 10, 
                encode = "ordinal", 
                strategy = "quantile")),
        ("ohe", OneHotEncoder(
                categories = "auto",
                sparse = False,
                dtype = int,
                handle_unknown = "ignore"))])


#%%
# preprocessing pipeline for numeric features requiring binarization

# isolate features of interest
numeric2binary_cols = ["InboundCallCount", 
                       "InboundChatCount", 
                       "InboundEmailCount",
                       "OutboundCallCount",
                       "OutboundEmailCount"]

# verify distributions
mf.Numplots(Xtrain[numeric2binary_cols], "hist") # discrete distributions

# pipeline to transform numeric features to binarized features
numeric2binary_transformer = Pipeline(steps = [
        ("bin", Binarizer(
                threshold = 0.0,
                copy = False))])


#%%
# preprocessing pipeline for binary or numerical feature requiring simple 
# imputation using an indicator value and one hot encoding

# isolate feature of interest
numeric2ohe_cols = ["Contacted", 
                   "advocate_id",
                   "account_sale_days",
                   "Date_Time_account_month",
                   "Date_Time_account_day",
                   "Date_Time_account_hour",
                   "Date_Time_call_month",
                   "Date_Time_call_day",
                   "Date_Time_call_hour",
                   "Date_Time_sale_month",
                   "Date_Time_sale_day",
                   "Date_Time_sale_hour"]

numeric2ohe_transformer = Pipeline(steps = [
        ("si", SimpleImputer(
                missing_values = np.nan,
                strategy = "constant",
                fill_value = -9999)),
        ("ohe", OneHotEncoder(
                categories = "auto",
                sparse = False,
                dtype = int,
                handle_unknown = "ignore"))])


#%%
# preprocessing pipeline for categorical features requiring oridinal encoding
# and binarization

# isolate features of interest
cat2binary_cols = ["PhoneType", 
                   "LeadSource"]

# verify distribution
mf.describeCat(Xtrain, include = cat2binary_cols)
# PostPaid vs Other, Website vs Other

# pipeline to transform categorical features to ordinal/binary features
cat2binary_transformer = Pipeline(steps = [
        ("oe", OrdinalEncoder(
                categories = "auto",
                dtype = int)),
        ("bin", Binarizer(
                threshold = 0,
                copy =False))])


#%%
# preprocessing pipeline for categorical features requiring ordinal encoding

# isolate features of interest
cat2ohe_cols = ["CensusGeoRegionAddress",
                "ModeDeviceType",
                "ModeVehicleType",
                "ServiceProvider",
                "AgeCategory",
                "InboundCallContact"]

# pipeline to transform categorical features to one hot encoded features
cat2ohe_transformer = Pipeline(steps = [
        ("ohe", OneHotEncoder(
                categories = "auto",
                sparse = False,
                dtype = int,
                handle_unknown = "ignore"))])


#%%
# columntransformer for pipelines
preprocessor = ColumnTransformer(
        transformers = [
                ("num1", numeric2kbins_uniform_transformer, numeric2kbins_uniform_cols),
                ("num2", numeric2kbins_norm_transformer, numeric2kbins_norm_cols),
                ("num3", numeric2binary_transformer, numeric2binary_cols),
                ("num4", numeric2ohe_transformer, numeric2ohe_cols),
                ("cat1", cat2binary_transformer, cat2binary_cols),
                ("cat2", cat2ohe_transformer, cat2ohe_cols)],
                remainder = "passthrough")


#%%
# build the preprocessing to model pipeline
clf = Pipeline(steps = [
        ("pp", preprocessor),
        ("rfc", RandomForestClassifier())])


#%%
# transform Xtest data using non-pipeline preprocessing
drop_preprocessing(Xtest)
datetime_preprocessing(Xtest)
time_preprocessing(Xtest)
currency_preprocessing(Xtest)


#%%
# cross-validate using 5-fold cross-validation and use grid search
# setup grid search
param_grid = {
        "rfc__n_estimators": [100, 200, 300, 400, 500, 1000],
        "rfc__criterion" : ["gini", "entropy"]}

gscv = GridSearchCV(clf, param_grid, iid = False, cv = 5, return_train_score = False)

# search for best parameters
gscv.fit(Xtrain, ytrain)
print(gscv.best_estimator_, "\n")
print(gscv.best_score_, "\n")
print(gscv.best_params_, "\n")
print(gscv.cv_results_, "\n")


#%%
# evaluate best_estimator_ on test data
ypred = gscv.best_estimator_.predict(Xtest)
print(metrics.accuracy_score(ytest, ypred))
print(metrics.roc_auc_score(ytest, ypred))
print(metrics.confusion_matrix(ytest, ypred))
print(metrics.classification_report(ytest, ypred))





















