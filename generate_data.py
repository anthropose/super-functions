# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 14:51:01 2019

@author: chari
"""


#%%
import MyFuncs as mf
import GenerateFuncs as genfuncs
import pandas as pd

#%%
# load data into df
comps = mf.loadDF("Data\Forecast POC Components.xlsx", "excel")

#%%
# first, groupby ABC and find mean of other features
comps_groupABC = comps.groupby("ABC").mean()
# index of comps_group is already set to ABC codes

# take transpose of comps_group
comps_groupABC_transpose = comps_groupABC.transpose()
comps_groupABC_transpose.index

# keep only desired rows of new df
keep = ["DMD_12", "DMD_11", "DMD_10", "DMD_9", "DMD_8", "DMD_7", "DMD_6", 
         "DMD_5", "DMD_4", "DMD_3", "DMD_2", "DMD_1"]
comps_groupABC = comps_groupABC_transpose.loc[keep]

#%%
# groupby ABC and filter by ASL "y"
# generate df without any rows where ASL is "n" or NaN
comps_y = comps.loc[comps["ASL"] == "y"]
# reset the index
comps_y = comps_y.reset_index(drop = True)

# groupby ABC and find mean of other features
comps_groupABCy = comps_y.groupby("ABC").mean()

# take transpose of comps_groupABCy
comps_groupABCy_transpose = comps_groupABCy.transpose()
# verify index of transposed df
comps_groupABCy_transpose.index

# keep only desired rows of new df
comps_groupABCy = comps_groupABCy_transpose.loc[keep]

#%%
# create dataframe(s) of newly generated demands for each ABC group
# number of years of new data
num_years = 2

# number of random variates; this is 12, because we have 12 months of demand
size = 12

# initialize an empty list to hold new dataframe(s)
new_years = []

# calls the genfuncs.generate_df function, which calls the 
# genfuncs.GeneratedData class
for year in range(num_years):
    yeardf = genfuncs.generate_df(comps_groupABCy, size = size)
    yeardf.set_index(keys = pd.Series(keep), drop = True, inplace = True)
    new_years.append(yeardf)

#%%
# simulate 2017 demands
year_2017 = new_years[0]

# simulate 2016 demands
year_2016 = new_years[1]
    

#%%
# compare simulated demand data histograms to original histograms
# original data
mf.Numplots(comps_groupABCy, "hist")

# new data
mf.Numplots(year_2017, "hist")
mf.Numplots(year_2016, "hist")












