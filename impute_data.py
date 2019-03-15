# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 17:32:38 2019

@author: chari
"""

#%%
import pandas as pd
import numpy as np
import MyFuncs as mf
from sklearn.impute import SimpleImputer

#%%
# load data into df
comps = mf.loadDF("Data\Forecast POC Components.xlsx", "excel")

#%%
# count missing values in df
comps.isna().sum()

# identify demand features
demand_cols = ["DMD_12", "DMD_11", "DMD_10", "DMD_9", "DMD_8", "DMD_7", "DMD_6", 
         "DMD_5", "DMD_4", "DMD_3", "DMD_2", "DMD_1"]

# impute missing demand data with constant "0"
imp_demand = SimpleImputer(missing_values = np.nan, 
                           strategy = "constant", 
                           fill_value = 0)

for col in demand_cols:
    comps[col] = pd.DataFrame(
            imp_demand.fit_transform(comps[[col]]), 
            index = comps.index)


# verify that all missing values for demand columns are filled
comps.isna().sum()


#%%
# identify features to be used in pivot table
pivot_cols = ["COST", "LEADTIME", "SAFETY_STK", "ONHANDNEW"]

test = pd.pivot_table(comps, values = pivot_cols, index = "ABC", columns = [comps[comps["ASL" == "y"]]], aggfunc = np.mean)

#print(mf.elimNan(comps).pivot_table(values = pivot_cols, 
#          index = "ABC", 
#          columns = "ASL", 
#          aggfunc = np.mean))

# filter out where ASL is "n"

