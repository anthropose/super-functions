# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:05:36 2019

@author: chari
"""

#%%
# import libraries
import MyFuncs as mf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as multi
from statsmodels.graphics.api import interaction_plot


#%%
# import data
cust_hist = mf.loadDF("cust_hist.csv", "csv")
call_hist = mf.loadDF("call_hist.csv", "csv")
sale_hist = mf.loadDF("sale_hist.csv", "csv")


#%%
# join three dataframes together
# must use outer join because union of all dfs are necessary for machine learning

# start by merging cust_hist and call_hist using outer join and 
# PhoneNumber_hashed
cust_call = pd.merge(cust_hist, 
                     call_hist, 
                     how = "outer", 
                     on = "PhoneNumber_hashed")

# view merged df and confirm joins
cust_call.info()
cust_call.head()
cust_call.tail()


# merge cust_call and sale_hist using outer join and userid
cust_call_sale = pd.merge(cust_call, sale_hist, how = "outer", on = "userid")

# view merged df and confirm joins
cust_call_sale.info()
cust_call_sale.head()


#%%
# convert Date_Time_account, Date_Time_call, Date_Time_sale to datetime objects
cust_call_sale["Date_Time_account"] = pd.to_datetime(
        cust_call_sale["Date_Time_account"])
cust_call_sale["Date_Time_call"] = pd.to_datetime(
        cust_call_sale["Date_Time_call"])
cust_call_sale["Date_Time_sale"] = pd.to_datetime(
        cust_call_sale["Date_Time_sale"])


#%%
# create new binary variable "sold" to indicate whether a customer purchased a 
# vehicle within 45 days of the time they created their account

# first, create new variable with number of days between sale and account creation
cust_call_sale["account_sale_days"] = (
        cust_call_sale["Date_Time_sale"] - cust_call_sale["Date_Time_account"]).dt.days

# view new variable
cust_call_sale["account_sale_days"].describe()

# create "sold" variable
cust_call_sale["sold"] = cust_call_sale["account_sale_days"].apply(
        lambda x: 1 if np.isnan(x) == False and 0 <= x <= 45 else 0)

# view "sold" variable
cust_call_sale["sold"].describe()


#%%
# create new binary variable "high_credit" to indicate whether a customer's
# CreditScore1 is above the median customer in the dataset

# find median creditscore1
creditscore1_median = np.median(cust_call_sale["CreditScore1"])

# create "high_credit" variable
cust_call_sale["high_credit"] = cust_call_sale["CreditScore1"].apply(
        lambda x: 1 if np.isnan(x) == False and x > creditscore1_median else 0)

# view "high_credit" variable
cust_call_sale["high_credit"].describe()


#%%
# create new binary variable "high_income" to indicate whether a customer's
# income is above the 80th percentile

# find 80th percentile of income
# a small number of users have a negative income, which may affect the validity
# of this calculation
percentile_80 = np.percentile(a = cust_call_sale["Income"], q = 0.80)

# create "high_income" variable
cust_call_sale["high_income"] = cust_call_sale["Income"].apply(
        lambda x: 1 if np.isnan(x) == False and x > percentile_80 else 0)

# view "high_income" variable
cust_call_sale["high_income"].describe()


#%%
# compute sale conversion rates of customers who were contacted, attempted to
# be contacted, or not attempted to be contacted

# total number of visitors who purchased a vehicle within 45 days
total_visitors = len(cust_call_sale)

# total number contacted who also purchased a vehicle within 45 days
contacted = cust_call_sale["Contacted"][
        (cust_call_sale["Contacted"] == 1) & (cust_call_sale["sold"] == 1)].count()

# total number attempted to be contacted who also purchased a vehicle within 45
# days
attempted = cust_call_sale["Contacted"][
        (cust_call_sale["Contacted"] == 0) & (cust_call_sale["sold"] == 1)].count()

# total number not attempted to be contacted who also purchased a vehicle within
# 45 days
not_attempted = cust_call_sale["Contacted"][
        (cust_call_sale["Contacted"].isnull()) & (cust_call_sale["sold"] == 1)].count()

# calculate sales conversion rates of each
contacted_conv = contacted/total_visitors # 0.235
attempted_conv = attempted/total_visitors # 0.045
not_attempted_conv = not_attempted/total_visitors # 0.000


#%%
# use 1-way ANOVA to check for differences in conversion rates for campaign levels
anova1_vars = ["Contacted", "sold"]

# extract just variables of interest and create new df
anova1_df = cust_call_sale[anova1_vars]

# change contacted values into their respective levels: "contacted", 
# "attempted", "not-attempted"
# define a function to map the levels
def campaign(x):
    if np.isnan(x) == True:
        result = "not_attempted"
    elif x == 0:
        result = "attempted"
    elif x == 1:
        result = "contacted"
    return result

# apply lambda of function for each factor
anova1_df["Contacted"] = cust_call_sale["Contacted"].apply(lambda x: campaign(x))
anova1_df.info()

# define samples for each campaign level/group
contacted_anova = anova1_df["sold"][anova1_df["Contacted"] == "contacted"]
attempted_anova = anova1_df["sold"][anova1_df["Contacted"] == "attempted"]
not_attempted_anova = anova1_df["sold"][anova1_df["Contacted"] == "not_attempted"]

# perform one-way ANOVA to test null hypothesis
# H0: no difference between means; mu1 = mu2 = mu3
# Ha: difference between means exist somewhere
# assumes: samples are independent, each sample is from a normally distributed
# population, the population standard deviations of the groups are all equal
# sample sizes for each group/level differ quite a bit, so Kruskal-Wallis may be
# an alternative option
stats.f_oneway(contacted_anova, attempted_anova, not_attempted_anova) 
# F-value = 135.58, p-value = 0.000
# but, we don't know if all, or just one, level is significantly different
# because ANOVA is an omnibus test

# ANOVA is, in some sense, a GLM, and thus can be fit using OLS
results1 = ols("sold ~ C(Contacted)", data = anova1_df).fit()
results1.summary()
# in this case, the intercept term is for the control group "attempted"

# view ANOVA table
aov_table1 = sm.stats.anova_lm(results1, typ=2)
aov_table1
# co-efficients for "attempted" and "contacted" are statistically significant

# but what is the effect size?
# define function that calculates effect size
# https://pythonfordatascience.org/anova-python/
def anova_table(aov):
    aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']
    
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])
    
    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    aov = aov[cols]
    return aov

# view ANOVA table with effect size metrics, eta-squared and omega-squared
anova_table(aov_table1)

# Assumptions checking
# check for homogeneity of variances
stats.levene(contacted_anova, attempted_anova, not_attempted_anova)
# groups are not homoscedastic

# check for normality
stats.shapiro(results1.resid)
# groups are not normally distributed


# given violations of ANOVA assumptions, perform non-parametric method too
# try Kruskal-Wallis H-test
# tests the null hypothesis that the population median of all groups are equal
stats.kruskal(contacted_anova, attempted_anova, not_attempted_anova)
# H-value = 265.22, p-value = 0.000
# much like the ANOVA, the null hypothesis is rejected


# Post-hoc testing
# perform post-hoc comparisons using Tukey's HSD
# controls for type-I error and maintains familywise error rate at 0.05
mc1 = multi.MultiComparison(anova1_df["sold"], anova1_df["Contacted"])
mc_results1 = mc1.tukeyhsd()
print(mc_results1)

# perform post-hoc comparisons using Bonferroni correction
# calculate corrected p-value
corrected_pvalue = 0.05/3 # 0.017
# t-test between contacted and attempted
stats.ttest_ind(contacted_anova, attempted_anova)
# t-value = 9.93, p-value = 0.000
# t-test between contacted and not-attempted
stats.ttest_ind(contacted_anova, not_attempted_anova)
# t-value = 15.27, p-value = 0.000
stats.ttest_ind(attempted_anova, not_attempted_anova)
# t-value = 1.23, p-value = 0.219

# results would suggest that mean difference in conversion rate between 1)
# attempted and contacted is statistically significant; and 2) not-attempted 
# and contacted is statistically significant
# there is not a statistically significant difference in conversion rate between
# attempted and not_attempted


#%%
# use 2-way ANOVA to check for differences in conversion rates for factors and 
# levels of campaign and high_credit
anova2_vars = ["Contacted", "high_credit", "sold"]

# extract just variables of interest and create new df
anova2_df = cust_call_sale[anova2_vars]

# apply lambda of function for each campaign level
anova2_df["Contacted"] = cust_call_sale["Contacted"].apply(lambda x: campaign(x))

# define a function to map the levels of high_credit
def credit_level(x):
    if x == 1:
        result = "high"
    else:
        result = "low"
    return result

# apply lambda of function for each credit level
anova2_df["high_credit"] = cust_call_sale["high_credit"].apply(lambda x: credit_level(x))
anova2_df.info()


# define samples for each factor and level
contacted_high1_anova = anova2_df["sold"][
        (anova2_df["Contacted"] == "contacted") & (anova2_df["high_credit"] == "high")]
contacted_low1_anova = anova2_df["sold"][
        (anova2_df["Contacted"] == "contacted") & (anova2_df["high_credit"] == "low")]
attempted_high1_anova = anova2_df["sold"][
        (anova2_df["Contacted"] == "attempted") & (anova2_df["high_credit"] == "high")]
attempted_low1_anova = anova2_df["sold"][
        (anova2_df["Contacted"] == "attempted") & (anova2_df["high_credit"] == "low")]
not_attempted_high1_anova = anova2_df["sold"][
        (anova2_df["Contacted"] == "not_attempted") & (anova2_df["high_credit"] == "high")]
not_attempted_low1_anova = anova2_df["sold"][
        (anova2_df["Contacted"] == "not_attempted") & (anova2_df["high_credit"] == "low")]


# perform two-way ANOVA to test the following:
# Factor Contacted x Factor high_credit
    # H0: there is no interaction
    # Ha: there is an interaction
# main effect of factor Contacted
    # H0: mu1 = mu2 = mu3
    # Ha: not all of the mu are equal
# main effect of factor high_credit
    # H0: mu1 = mu2
    # Ha: mu1 != mu2
# assumes: samples are independent, each sample is from a normally distributed
# population, the population standard deviations of the groups are all equal

# fit 2-way ANOVA using OLS
results2 = ols("sold ~ C(Contacted)*C(high_credit)", data = anova2_df).fit()
results2.summary()
# the interaction between not_attempted and low credit is not significant
# looks like there are some violations of normality

# view ANOVA table
aov_table2 = sm.stats.anova_lm(results2, typ=2)
aov_table2
# interaction term between contacted and high_credit is significant
# main effects of contacted and high_credit are not interpretable by themselves
# because the effect of the factor contacted depends upon the level of contacted
# and the level of factor high_credit (and vice-versa)

# view interaction plots
fig = interaction_plot(
        anova2_df["Contacted"], 
        anova2_df["high_credit"], 
        anova2_df["sold"], 
        colors = ['red','blue'], 
        markers=['D','^'], 
        ms=10)
plt.show()
# this would imply that individuals with a credit level of high will have high
# sales conversion rates regardless of campaign level; although, there is a
# slight dip from attempted to contacted and from attempted to not-attempted
# where the interaction becomes interesting is when those customers with low 
# credit are contacted, they too have much higher sales conversion rates than 
# for other campaign levels and even higher than for those with high credit who 
# are contacted

# view ANOVA table with effect size metrics, eta-squared and omega-squared
anova_table(aov_table2)


# Assumptions checking
# check for homogeneity of variances
stats.levene(contacted_high1_anova, 
             contacted_low1_anova, 
             attempted_high1_anova, 
             attempted_low1_anova, 
             not_attempted_high1_anova, 
             not_attempted_low1_anova)
# groups are not homoscedastic

# check for normality
stats.shapiro(results2.resid)
# groups are not normally distributed

# violations of assumptions imply need for non-parametric version of 2-way ANOVA
# permutation test?


# Post-hoc testing
# cannot be interpreted in a meaningful way because the interaction between the
# factors is significant
# perform post-hoc comparisons using Tukey's HSD
# controls for type-I error and maintains familywise error rate at 0.05

# tukey between sold and contacted is same as mc1 above
mc2 = multi.MultiComparison(anova2_df["sold"], anova2_df["high_credit"])
mc_results2 = mc2.tukeyhsd()
print(mc_results2)


#%%
# use 2-way ANOVA to check for differences in conversion rates for factors and 
# levels of campaign and high_income
anova3_vars = ["Contacted", "high_income", "sold"]

# extract just variables of interest and create new df
anova3_df = cust_call_sale[anova3_vars]

# apply lambda of function for each campaign level
anova3_df["Contacted"] = cust_call_sale["Contacted"].apply(lambda x: campaign(x))

# define a function to map the levels of high_credit
def income_level(x):
    if x == 1:
        result = "high"
    else:
        result = "low"
    return result

# apply lambda of function for each credit level
anova3_df["high_income"] = cust_call_sale["high_income"].apply(lambda x: income_level(x))
anova3_df.info()


# define samples for each factor and level
contacted_high2_anova = anova3_df["sold"][
        (anova3_df["Contacted"] == "contacted") & (anova3_df["high_income"] == "high")]
contacted_low2_anova = anova3_df["sold"][
        (anova3_df["Contacted"] == "contacted") & (anova3_df["high_income"] == "low")]
attempted_high2_anova = anova3_df["sold"][
        (anova3_df["Contacted"] == "attempted") & (anova3_df["high_income"] == "high")]
attempted_low2_anova = anova3_df["sold"][
        (anova3_df["Contacted"] == "attempted") & (anova3_df["high_income"] == "low")]
not_attempted_high2_anova = anova3_df["sold"][
        (anova3_df["Contacted"] == "not_attempted") & (anova3_df["high_income"] == "high")]
not_attempted_low2_anova = anova2_df["sold"][
        (anova3_df["Contacted"] == "not_attempted") & (anova3_df["high_income"] == "low")]


# perform two-way ANOVA to test the following:
# Factor Contacted x Factor high_income
    # H0: there is no interaction
    # Ha: there is an interaction
# main effect of factor Contacted
    # H0: mu1 = mu2 = mu3
    # Ha: not all of the mu are equal
# main effect of factor high_income
    # H0: mu1 = mu2
    # Ha: mu1 != mu2
# assumes: samples are independent, each sample is from a normally distributed
# population, the population standard deviations of the groups are all equal

# fit 2-way ANOVA using OLS
results3 = ols("sold ~ C(Contacted)*C(high_income)", data = anova3_df).fit()
results3.summary()
# looks like there are some violations of normality and independence

# view ANOVA table
aov_table3 = sm.stats.anova_lm(results3, typ=2)
aov_table3
# interaction term between contacted and high_income is not significant

fig = interaction_plot(
        anova3_df["Contacted"], 
        anova3_df["high_income"], 
        anova3_df["sold"], 
        colors = ['red','blue'], 
        markers=['D','^'], 
        ms=10)
plt.show()

# re-run ANOVA without interaction term
results4 = ols("sold ~ C(Contacted) + C(high_income)", data = anova3_df).fit()
results4.summary()
# violations of normality; but issue of multicollinearity seems to have resolved

# view ANOVA table
aov_table4 = sm.stats.anova_lm(results4, typ=2)
aov_table4
# income factor is not significant; only the contacted factor seems to matter

# view ANOVA table with effect size metrics, eta-squared and omega-squared
anova_table(aov_table4)
# again, contacted factor is most significant; income seems to have no real effect


# Post-hoc testing
# perform post-hoc comparisons using Tukey's HSD
# controls for type-I error and maintains familywise error rate at 0.05

# tukey between sold and contacted is same as mc1 above
mc3 = multi.MultiComparison(anova3_df["sold"], anova3_df["high_income"])
mc_results3 = mc3.tukeyhsd()
print(mc_results3)
# as expected, cannot reject the null hypothesis for differences between high 
# and low income levels


#%%
# export cust_call_sale for model building
cust_call_sale.to_csv("cust_call_sale.csv", index = False)














