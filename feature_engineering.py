# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:51:22 2019

@author: chari
"""

#%%
# feature engineering: add columns calculating percent error between forecast and demand?
# add columns for 3-month, 6-month, 9-month, and 12-month demand totals; or first-quarter, second-quarter, etc.; or lag-columns
# these new features, plus all the old ones (but maybe not the forecast columns), could be used in a clustering analysis
# clustering yields a final feature called cluster ID



#%%
# import libraries
import numpy as np
import pandas as pd
import MyFuncs as mf
import EngineeringFuncs as ef


#%%
# load data into df
comps = mf.loadDF("data_comps.csv", "csv")


#%%
# view data
mf.QView(comps)


#%%
# create column for percent error between a demand and a forecast
demand_cols = ["DMD_01_2018", 
               "DMD_02_2018", 
               "DMD_03_2018", 
               "DMD_04_2018", 
               "DMD_05_2018", 
               "DMD_06_2018", 
               "DMD_07_2018",
               "DMD_08_2018", 
               "DMD_09_2018", 
               "DMD_10_2018", 
               "DMD_11_2018", 
               "DMD_12_2018"]
forecast_cols = ["FCST_01_2018", 
                 "FCST_02_2018", 
                 "FCST_03_2018", 
                 "FCST_04_2018", 
                 "FCST_05_2018", 
                 "FCST_06_2018", 
                 "FCST_07_2018",
                 "FCST_08_2018", 
                 "FCST_09_2018", 
                 "FCST_10_2018", 
                 "FCST_11_2018", 
                 "FCST_12_2018"]

#test = ef.absolute_difference(comps, demand_cols, forecast_cols) 
        

#%%
# create lag variables      
test = ef.buildLaggedFeatures(comps)
        