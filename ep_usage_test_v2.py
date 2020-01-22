# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:55:52 2019

@author: jwillert
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import string
from regression import fit_model

valid_chars = valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
def make_valid(txt):
    
    return ''.join(c for c in txt if c in valid_chars)

MULTIPLIER = 1.0
MAX = 20.0
THRESHOLD_DICT = {}
THRESHOLD_DICT["0 - 1,000 kWh"] = MULTIPLIER*(MAX/np.log10(100))
THRESHOLD_DICT["1,000 - 10,000 kWh"] = MULTIPLIER*(MAX/np.log10(1000))
THRESHOLD_DICT["10,000 - 50,000 kWh"] = MULTIPLIER*(MAX/np.log10(25000))
THRESHOLD_DICT["50,000 - 500,000 kWh"] = MULTIPLIER*(MAX/np.log10(100000))
THRESHOLD_DICT["500,000 - 5,000,000 kWh"] = MULTIPLIER*(MAX/np.log10(1000000))
THRESHOLD_DICT["> 5,000,000 kWh"] = MULTIPLIER*(MAX/np.log10(5000000))
THRESHOLD_DICT["Supplier (> 50,000 kWh)"] = MULTIPLIER*(MAX/np.log10(1000000))
THRESHOLD_DICT["Supplier (0 - 50,000 kWh)"] = MULTIPLIER*(MAX/np.log10(25000))
THRESHOLD_DICT["Other"] = MULTIPLIER*(MAX/np.log10(5000000))
THRESHOLD_DICT["Outdoor Lighting Only"] = MULTIPLIER*(MAX/np.log10(100000))

TARGET = "Daily_Usage"

REMOVE_OUTLIERS = True
ALLOW_QUICK_TEST = False
UNIFORM_WEIGHTS = False
WEIGHT_DECREASE_CONST = 0.5
TEST_TYPE = "Absolute"

def ep_usage_test(data,account_,PLOT_ON):
    
    THRESHOLD = THRESHOLD_DICT[account_['Variance_Consumption_Level'].iloc[0]]
    
    # Convert string dates to pandas datetimes and sort
    data["Month"] = pd.to_datetime(data["Service_Month"])
    data["Billing_Start_Dt"] = pd.to_datetime(data["Billing_Start_Dt"])
    data["Billing_End_Dt"] = pd.to_datetime(data["Billing_End_Dt"])
    data.sort_values(by="Month",inplace=True)
    
    # Transform total usage into daily usage
    data["Daily_Usage"] = data["Usage_KWH"]/data["Billing_Days"]

    # Rename columns (unnecessary, aside from clarity)
    data["Daily CDD"] = data.pop("MeanDailyCDD")
    data["Daily HDD"] = data.pop("MeanDailyHDD")
    
    # Select the appropriate field for testing ("Total Usage" or "Daily Usage")
    usage = data[TARGET]
    
    # Drop any rows with missing values
    data.dropna(inplace=True)

    # If the account changes minimally from the previous month, just pass
    if ALLOW_QUICK_TEST:
        if usage.iloc[-1]/usage.iloc[-2] - 1.0 < 0.05:
            return "Pass - Minor change from previous month", []

    ####################################################
    #                                                  # 
    #  Apply LinReg testing for long-history accounts  #
    #                                                  # 
    ####################################################
    
    out = fit_model(data,
                    ['Time','Daily HDD','Daily CDD'],
                    TARGET,
                    REMOVE_OUTLIERS,
                    TEST_TYPE,
                    THRESHOLD,
                    UNIFORM_WEIGHTS,
                    WEIGHT_DECREASE_CONST)
    
    # Split tuple into individual components
    preds, errors, IQR, low, high, unex_var, outlier_notice = out

    print("Fraction of variance unexplained (FVU) = ",unex_var)
    
    ##############################################
    #                                            #
    #  Apply threshold test and package results  #
    #                                            #
    ##############################################

    # Test for normality of errors
    stat, pval = stats.normaltest(errors.iloc[:-1])
    appender = outlier_notice
    if pval < 0.01:
        appender += ", Possible non-normality"

    if IQR <= 0.001:
        return 'Pass - constant usage', []

    if TEST_TYPE == "RPD" and IQR >= 3.99:
        return 'Pass - constant usage', []

    # Do billing days appear normal?
    bill_days = data["Billing_Days"].iloc[-1]
    if bill_days >= 35:
        appender += ", Long Billing Period ({} days).  ".format(bill_days)
    elif bill_days <= 24:
        appender += ", Short Billing Period ({} days).  ".format(bill_days)

    # How well does the current month match our prediction?
    pred_dev_ = errors.iloc[-1]/IQR

    # Determine if we flag a variance and return response
    if abs(pred_dev_) > THRESHOLD:
        result = 'Fail'
    else:
        result = 'Pass'
        
    if PLOT_ON:
        plt.figure(figsize=[8,6])
        plt.title("EP Usage\n{}\n{}\n{}".format(account_["Client_Name"].iloc[0],
                                  account_["Account_Number"].iloc[0],
                                  account_["Variance_Consumption_Level"].iloc[0]))
        plt.fill_between(data["Service_Month"],low,high,alpha=0.3)
        plt.plot(data["Service_Month"],preds)
        plt.plot(data["Service_Month"],data[TARGET])
        plt.legend(['Predictions','Observations'])
        plt.xticks(rotation=45)
        plt.savefig(".\\Final_Versions\\{}\\epu_{}_{}_{}.png".format(result,
                                                         make_valid(str(account_["Client_Name"].iloc[0])),
                                                         make_valid(str(account_["Account_Number"].iloc[0])),
                                                         make_valid(str(data['Service_Month'].iloc[-1])[:10])))
        plt.close()
    
    return result, preds