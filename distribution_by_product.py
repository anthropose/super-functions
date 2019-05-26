# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 17:05:30 2019

@author: chari
"""

#%%
import MyFuncs as mf
import DistributionFuncs as distfuncs
import matplotlib.pyplot as plt
import scipy.stats as st

#%%
# load data into df
comps = mf.loadDF("Data\Forecast POC Components.xlsx", "excel")

#%%
# create list of desired columns
keep = ["HOSTPARTID", "DMD_12", "DMD_11", "DMD_10", "DMD_9", "DMD_8", "DMD_7", "DMD_6",
        "DMD_5", "DMD_4", "DMD_3", "DMD_2", "DMD_1"]

# create new df of desired columns and set index as HOSTPARTID
comps_demand = comps[keep].set_index(keys = "HOSTPARTID", drop = True)

# transpose the df
comps_transpose = comps_demand.transpose()

# create a smaller df using just a sample of the possible HOSTPARTID columns
comps_tiny = comps_transpose.sample(frac = 0.005, axis = 1)

#%%
# generate histograms using myfuncs.Numplots (because it can be used with NaNs)
mf.Numplots(comps_tiny, "hist")

# THE NUMBER OF UNIQUE HOSTPARTIDS AND NaNs MAKE THIS APPROACH INFEASIBLE

#%%
# try to simplify histograms by using groupby
# first, groupby ABC and find mean of other features
comps_groupABC = comps.groupby("ABC").mean()
# index of comps_group is already set to ABC codes

# take transpose of comps_group
# no need to sample the df
comps_groupABC_transpose = comps_groupABC.transpose()
comps_groupABC_transpose.index

# keep only desired rows of new df
keep2 = ["DMD_12", "DMD_11", "DMD_10", "DMD_9", "DMD_8", "DMD_7", "DMD_6", 
         "DMD_5", "DMD_4", "DMD_3", "DMD_2", "DMD_1"]
comps_groupABC = comps_groupABC_transpose.loc[keep2]

# generate histograms
mf.Numplots(comps_groupABC, "hist")

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
comps_groupABCy = comps_groupABCy_transpose.loc[keep2]

# generate histograms
mf.Numplots(comps_groupABCy, "hist")

#%%
# find candidate distribution(s) for annual demand by each ABC group using
# p-values from a Kolmogorov-Smirnoff goodness of fit test
# The K-S test statistic measures the largest distance between the empirical
# distribution function of the observed data and the cdf of the 
# theoretical/candidate function
# H0: observed data has the same underlying distribution as the candidate distribution
# Ha: observed data does not have the same underlying distribution

# use DistributionsByColumn from distfuncs.py to loop over all columns and
# generate a dictionary of column names and their associated fitted 
# distributions, parameters, and SSEs
column_distributions = distfuncs.DistributionsByColumn(comps_groupABC)

#column_distributions["A1"][0].Plot(data)
#
#for distribution in column_distributions["A1"]:
#    distribution.Plot(data)
#
## distribution(s) for ABC group A1
#A1_dists_ABC = distfuncs.GetDistributions(comps_groupABC["A1"], distfuncs.scipy_distributions)
#
## explore candidate distribution(s) where p-value > 0.05
#for distribution in A1_dists_ABC:
#    distribution.Plot(comps_groupABC["A1"])
#
## explore SSEs
## save all distributions and their SSEs to a sorted list
#A1_dists_ABC_SSE = sorted(A1_dists_ABC, key = lambda x: x.SSE, reverse = False)
#
## print out the names of the sorted distributions and their SSEs
#for distribution in A1_dists_ABC_SSE:
#    print("%s: %f" % (distribution.name, round(distribution.SSE, 5)))
#
## plot the observed data and the PDF
#for distribution in A1_dists_ABC_SSE:
#    distribution.PlotPDF(comps_groupABC["A1"])
#
#
## distribution(s) for ABC group A2
#A2_dists_ABC = distfuncs.GetDistributions(comps_groupABC["A2"], distfuncs.scipy_distributions)
#
## explore candidate distribution(s) where p-value > 0.05
#for distribution in A2_dists_ABC:
#    distribution.Plot(comps_groupABC["A2"])
#    
## explore SSEs
## save all distributions and their SSEs to a sorted list
#A2_dists_ABC_SSE = sorted(A2_dists_ABC, key = lambda x: x.SSE, reverse = False)
#
## print out the names of the sorted distributions and their SSEs
#for distribution in A2_dists_ABC_SSE:
#    print("%s: %f" % (distribution.name, round(distribution.SSE, 5)))
#
#print("The best fit distribution is %s with an SSE of %f:" % (A2_dists_ABC_SSE[0].name, A2_dists_ABC_SSE[0].SSE))
#
## plot the observed data and the PDF
#for distribution in A2_dists_ABC_SSE:
#    distribution.PlotPDF(comps_groupABC["A2"])


#%%
# groupby ASL and find mean of other features
comps_groupASL = comps.groupby("ASL").mean()
# index of comps_group is already set to ASL codes

# take transpose of comps_group
# no need to sample the df
comps_groupASL_transpose = comps_groupASL.transpose()
comps_groupASL_transpose.index

# keep only desired rows of new df
keep2 = ["DMD_12", "DMD_11", "DMD_10", "DMD_9", "DMD_8", "DMD_7", "DMD_6", 
         "DMD_5", "DMD_4", "DMD_3", "DMD_2", "DMD_1"]
comps_groupASL = comps_groupASL_transpose.loc[keep2]

# generate histograms
mf.Numplots(comps_groupASL, "hist")

#%%
# find candidate distribution(s) for annual demand by ASL group "y" using
# p-values from a Kolmogorov-Smirnoff goodness of fit test
# The K-S test statistic measures the largest distance between the empirical
# distribution function of the observed data and the cdf of the 
# theoretical/candidate function
# H0: observed data has the same underlying distribution as the candidate distribution
# Ha: observed data does not have the same underlying distribution

comps_dists_ASLy = distfuncs.GetDistributions(comps_groupASL["y"], distfuncs.scipy_distributions)

# explore candidate distribution(s) where p-value > 0.05 by plotting the hists
for distribution in comps_dists_ASLy:
    distribution.Plot(comps_groupASL["y"])

# explore SSEs
# save all distributions and their SSEs to a sorted list
comps_dists_ASLy_SSE = sorted(comps_dists_ASLy, 
                              key = lambda x: x.SSE, 
                              reverse = False)

# print out the names of the sorted distributions and their SSEs
for distribution in comps_dists_ASLy_SSE:
    print("%s: %f" % (distribution.name, round(distribution.SSE, 5)))

# plot the observed data and the PDF
for distribution in comps_dists_ASLy_SSE:
    distribution.PlotPDF(comps_groupASL["y"])


















