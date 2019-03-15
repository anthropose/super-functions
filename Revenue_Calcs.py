# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 07:57:00 2019

@author: chari
"""

#%%
# import libraries
import MyFuncs as mf
import PreprocessingFuncs as pf
import pandas as pd
import numpy as np


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
# convert stickerprice to a float
currency_cols = ["StickerPrice"]
pf.currency_preprocessing(cust_call_sale, currency_cols)

# fill Contacted NaN with dummy value
missing_cols = ["Contacted"]
pf.fill_missing_constant_preprocessor(cust_call_sale, missing_cols)

# calculate total seconds for call duration
time_cols = ["Call_Duration"]
pf.time_preprocessing(cust_call_sale, time_cols)


#%%
# estimate revenue and profit under the current campaign

# mean stickerprice grouped by campaign level
revenue1 = cust_call_sale.groupby("Contacted").mean()["StickerPrice"]
# number of vehicles sold by campaign level; count of stickerprice is used as a
# proxy
counts1 = cust_call_sale.groupby("Contacted").count()["StickerPrice"]
# mean call duration by campaign level; necessary for cost calculation
phone_sec1 = cust_call_sale.groupby("Contacted").mean()["Call_Duration_totalsec"]
# count of referral discounts by campaign level; necessary for cost calculation
referral_counts1 = cust_call_sale.groupby("Contacted").count()["ReferralAttachment"]

profit1 = pd.DataFrame()
profit1["Mean_StickerPrice"] = revenue1
profit1["Sale_Counts"] = counts1
profit1["Call_Duration"] = phone_sec1
profit1["Referral_Counts"] = referral_counts1

# set index values to campaign levels
profit1["Contacted"] = ["Not_Attempted", "Attempted", "Contacted"]
profit1.set_index("Contacted", drop = True, inplace = True)

# calculate total revenue by campaign
profit1["Total_Revenue"] = profit1["Mean_StickerPrice"] * profit1["Sale_Counts"]
total_revenue1 = profit1["Total_Revenue"].sum()

# to facilitate calculations, fill NaNs
profit1.isna().sum()
profit1["Call_Duration"].fillna(value = 0, inplace = True)

# calculate individual costs
profit1["Dial_Fixed_Cost"] = 0.10 # connection cost
profit1["Dial_Variable_Cost"] = 0.02 # cost per second
profit1["Referral_Discount_Cost"] = 500
profit1["Holding_Cost"] = (profit1["Mean_StickerPrice"].apply(lambda x: x * 0.05)) * profit1["Sale_Counts"]

# calculate total cost by campaign
profit1["Total_Cost"] = (profit1["Sale_Counts"] * profit1["Dial_Fixed_Cost"]) + (profit1["Call_Duration"] * profit1["Dial_Variable_Cost"]) + (profit1["Referral_Discount_Cost"] * profit1["Referral_Counts"]) + (profit1["Holding_Cost"])

# calculate total profit by campaign
profit1["Total_Profit"] = profit1["Total_Revenue"] - profit1["Total_Cost"]

ProfitDelta_ContactedvsNotAttempted = profit1.iloc[2, 10] - profit1.iloc[0, 10]
ProfitDelta_ContactedvsAttempted = profit1.iloc[2, 10] - profit1.iloc[1, 10]
ProfitDelta_AttemptedvsNotAttempted = profit1.iloc[1, 10] - profit1.iloc[0, 10]

# calculate total overall profit
total_profit1 = profit1["Total_Profit"].sum() # $123,617,632.00


#%%
# based on logistic regression model, assume those customers marked as high
# income are all contacted; then estimate revenue and profit under these 
# conditions and the associated probability of a sale

model_efficacy = cust_call_sale.copy()
model_efficacy["Contacted_byIncome"] = model_efficacy["high_income"]

# mean stickerprice grouped by campaign level
revenue2 = model_efficacy.groupby("Contacted_byIncome").mean()["StickerPrice"]
# number of vehicles sold by campaign level; count of stickerprice is used as a
# proxy
counts2 = model_efficacy.groupby("Contacted_byIncome").count()["StickerPrice"]
# mean call duration by campaign level; necessary for cost calculation
phone_sec2 = model_efficacy.groupby("Contacted_byIncome").mean()["Call_Duration_totalsec"]
# count of referral discounts by campaign level; necessary for cost calculation
referral_counts2 = model_efficacy.groupby("Contacted_byIncome").count()["ReferralAttachment"]

profit2 = pd.DataFrame()
profit2["Mean_StickerPrice"] = revenue2
profit2["Sale_Counts"] = counts2

# modify sale counts by associated probability for high income
sale_probability_income = 0.65
profit2["Modified_Sale_Counts"] = profit2["Sale_Counts"]
profit2.iloc[1, 2] = profit2.iloc[1, 2] * sale_probability_income

profit2["Call_Duration"] = phone_sec2
profit2["Referral_Counts"] = referral_counts2

# set index values to income levels
profit2["Contacted_byIncome"] = ["Low", "High"]
profit2.set_index("Contacted_byIncome", drop = True, inplace = True)

# calculate total revenue by income
profit2["Total_Revenue"] = profit2["Mean_StickerPrice"] * profit2["Modified_Sale_Counts"]
total_revenue2 = profit2["Total_Revenue"].sum()

# to facilitate calculations, fill NaNs
profit2.isna().sum()
profit2["Call_Duration"].fillna(value = 0, inplace = True)

# calculate individual costs
profit2["Dial_Fixed_Cost"] = 0.10 # connection cost
profit2["Dial_Variable_Cost"] = 0.02 # cost per second
profit2["Referral_Discount_Cost"] = 500
profit2["Holding_Cost"] = (profit2["Mean_StickerPrice"].apply(lambda x: x * 0.05)) * profit2["Modified_Sale_Counts"]

# calculate total cost by income
profit2["Total_Cost"] = (profit2["Sale_Counts"] * profit2["Dial_Fixed_Cost"]) + (profit2["Call_Duration"] * profit2["Dial_Variable_Cost"]) + (profit2["Referral_Discount_Cost"] * profit2["Referral_Counts"]) + (profit2["Holding_Cost"])

# calculate total profit by campaign
profit2["Total_Profit"] = profit2["Total_Revenue"] - profit2["Total_Cost"]

ProfitDelta_HighvsLow = profit2.iloc[1, 10] - profit2.iloc[0, 10]

# calculate total overall profit
total_profit2 = profit2["Total_Profit"].sum() # $78,511,853.00

# sales conversion rate by level
total_customers = len(model_efficacy)

# total number customers with high income who were contacted and who also 
# purchased a vehicle within 45 days
total_high_income_sales = profit2.iloc[1, 2]

# total number attempted to be contacted who also purchased a vehicle within 45
# days
total_low_income_sales = profit2.iloc[0,2]

# calculate sales conversion rates of each
high_income_conv = total_high_income_sales/total_customers # 0.302
low_income_conv = total_low_income_sales/total_customers # 0.003


#%%
# based on logistic regression model, assume those customers marked as having a
# referral are all contacted; then calculate total profit under these conditions
# and the associated probability of a sale

model_efficacy["Contacted_byReferral"] = model_efficacy["ReferralAttachment"]

# mean stickerprice grouped by campaign level
revenue3 = model_efficacy.groupby("Contacted_byReferral").mean()["StickerPrice"]
# number of vehicles sold by campaign level; count of stickerprice is used as a
# proxy
counts3 = model_efficacy.groupby("Contacted_byReferral").count()["StickerPrice"]
# mean call duration by campaign level; necessary for cost calculation
phone_sec3 = model_efficacy.groupby("Contacted_byReferral").mean()["Call_Duration_totalsec"]
# count of referral discounts by campaign level; necessary for cost calculation
referral_counts3 = model_efficacy.groupby("Contacted_byReferral").count()["ReferralAttachment"]

profit3 = pd.DataFrame()
profit3["Mean_StickerPrice"] = revenue3
profit3["Sale_Counts"] = counts3

# modify sale counts by associated probability for high income
sale_probability_referral = 0.41
profit3["Modified_Sale_Counts"] = profit3["Sale_Counts"]
profit3.iloc[1, 2] = profit3.iloc[1, 2] * sale_probability_referral

profit3["Call_Duration"] = phone_sec3
profit3["Referral_Counts"] = referral_counts3

# set index values to income levels
profit3["Contacted_byIncome"] = ["No_Referral", "Yes_Referral"]
profit3.set_index("Contacted_byIncome", drop = True, inplace = True)

# calculate total revenue by income
profit3["Total_Revenue"] = profit3["Mean_StickerPrice"] * profit3["Modified_Sale_Counts"]
total_revenue3 = profit3["Total_Revenue"].sum()

# to facilitate calculations, fill NaNs
profit3.isna().sum()
profit3["Call_Duration"].fillna(value = 0, inplace = True)

# calculate individual costs
profit3["Dial_Fixed_Cost"] = 0.10 # connection cost
profit3["Dial_Variable_Cost"] = 0.02 # cost per second
profit3["Referral_Discount_Cost"] = 500
profit3["Holding_Cost"] = (profit3["Mean_StickerPrice"].apply(lambda x: x * 0.05)) * profit3["Modified_Sale_Counts"]

# calculate total cost by income
profit3["Total_Cost"] = (profit3["Sale_Counts"] * profit3["Dial_Fixed_Cost"]) + (profit3["Call_Duration"] * profit3["Dial_Variable_Cost"]) + (profit3["Referral_Discount_Cost"] * profit3["Referral_Counts"]) + (profit3["Holding_Cost"])

# calculate total profit by campaign
profit3["Total_Profit"] = profit3["Total_Revenue"] - profit3["Total_Cost"]

ProfitDelta_YesvsNo = profit3.iloc[1, 10] - profit3.iloc[0, 10]

# calculate total overall profit
total_profit3 = profit3["Total_Profit"].sum() # $117,571,599.00

# sales conversion rate by level
# total number customers with high income who were contacted and who also 
# purchased a vehicle within 45 days
total_yes_referral_sales = profit3.iloc[1, 2]

# total number attempted to be contacted who also purchased a vehicle within 45
# days
total_no_referral_sales = profit3.iloc[0,2]

# calculate sales conversion rates of each
yes_referral_conv = total_yes_referral_sales/total_customers # 0.015
no_referral_conv = total_no_referral_sales/total_customers # 0.430


#%%
# calculate total profit and revenue if both campaigns (income and referral) 
# were adopted
total_revenue_income_referral = total_revenue2 + total_revenue3
total_profit_income_referral = total_profit2 + total_profit3
# this amount is likely inflated because there are 965 customers with high 
# income who also have a referral attachment
# my logistic regression model did not assess interaction effects, so the 
# probability of a sale for these types of customers is unknown

ProfitDelta_CurrentvsModel = total_profit_income_referral - total_profit1
# the difference in profit between the current campaign model and the logistic
# regression model (using just the top two coefficients) is $72,411,819.00

ProfitPercentIncrease = ((total_profit_income_referral - total_profit1) / total_profit_income_referral) * 100 
# this represents a 36.9% increase in profit over the current model

