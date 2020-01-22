# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 08:12:48 2019

@author: jwillert
"""

import numpy as np
from sklearn.linear_model import LinearRegression

def return_credible_region(obs, preds, preds_all, THRESHOLD, TEST_TYPE):
    """ Given the observations and the predictions,
    return the credible region for each data point"""

    if TEST_TYPE == "Absolute":

        errors = obs - preds
        desc = errors.iloc[:-1].describe()
        p25 = desc["25%"]
        p75 = desc["75%"]
        IQR_ = p75 - p25

        low = preds_all - THRESHOLD*IQR_
        high = preds_all + THRESHOLD*IQR_

    elif TEST_TYPE == "Relative":

        errors = (obs -preds)/(1.0 + obs)
        desc = errors.iloc[:-1].describe()
        p25 = desc["25%"]
        p75 = desc["75%"]
        IQR_ = p75 - p25

        #print(p25,p75)

        #print(errors1,IQR)
        low = preds_all*(1 - THRESHOLD*IQR_)
        high = preds_all*(1 + THRESHOLD*IQR_)
        #print(low,high)

    elif TEST_TYPE == "RPD":

        errors = 2*(obs - preds)/(abs(obs) + abs(preds))
        desc = errors.iloc[:-1].describe()
        p25 = desc["25%"]
        p75 = desc["75%"]
        IQR_ = p75 - p25

        cnst1 = THRESHOLD*IQR_*abs(preds_all) - 2*preds_all
        cnst2 = -THRESHOLD*IQR_*abs(preds_all) - 2*preds_all
        low = -cnst1/(THRESHOLD*IQR_ + 2)
        high = cnst2/(THRESHOLD*IQR_ - 2)

    return errors, IQR_, low, high

def fraction_unexplained(true,prediction,wgts):
    
    mean = true.mean()
    
    SS_err = (pow(true - prediction,2.0)*wgts).sum()
    SS_tot = (pow(true - mean,2.0)*wgts).sum()
    
    unex_frac = SS_err/SS_tot
    
    return unex_frac


def fit_model(data,
              dep_vars,
              ind_var,
              REMOVE_OUTLIERS=True,
              TEST_TYPE='Absolute',
              THRESHOLD=3.5,
              UNIFORM_WEIGHTS=False,
              WEIGHT_DECREASE_CONST=0.5):

    # We'll convert month labels into a trailing 'time prior to target month'
    data["Time"] = ((data["Month"] - data["Month"].iloc[0]).dt.days)/365.25

    # For our data, we'll select the provided dependent variables
    LR_data = data[dep_vars].copy(deep=True)

    # The training data will be all data aside from the target month
    LR_train_data = LR_data.iloc[:-1]

    # We'll extract the target variable for training and testing
    LR_output = data[ind_var]
    LR_train_output = LR_output.iloc[:-1]

    # Instantiate a linear regression model
    LR_model = LinearRegression()

    #############################################
    #
    #   COMPUTE WEIGHTS FOR LINEAR REGRESSION
    #
    if UNIFORM_WEIGHTS:
        # Create uniform weights
        wgts = [1.0 for obs in range(len(LR_train_output))]
        
    else:
        # Determine weights, focusing on recency and seasonality
        date_diff = (data["Month"].max() - data["Month"].iloc[:-1]).dt.days/365

        # Decreasing by recency
        rwgts = np.exp(-WEIGHT_DECREASE_CONST*date_diff)

        # Decreasing by off-seasonality
        seasonal_dates = 2.0*(0.5 - abs(date_diff.mod(1) - 0.5))
        swgts = np.exp(-seasonal_dates)

        # Compute final weights
        wgts = rwgts*swgts

    # Fit the model
    LR_model.fit(LR_train_data, 
                 LR_train_output, 
                 sample_weight=wgts)

    # Use the model to make predictions
    preds = LR_model.predict(LR_data)
    preds_all = preds

    # Determine how closely our predictions match our observations
    errors, IQR, low, high = return_credible_region(LR_output, 
                                                    preds,
                                                    preds,
                                                    THRESHOLD,
                                                    TEST_TYPE)

    # Find the outliers, remove them, retrain (same flow as above - uncommented)
    if REMOVE_OUTLIERS:
        
        # Determine which observations are outliers
        outliers_ = (abs(errors) > THRESHOLD*IQR).values
        
        # Don't allow current month or previous month to be labeled outliers
        outliers_[-2] = False
        outliers_[-1] = False

        total_outliers = np.sum(outliers_)
        if total_outliers > 0:
            outlier_notice = "We have removed {} historical outliers.  ".format(total_outliers)
        else:
            outlier_notice = "There were no outliers historically"
        
        # Remove the outliers
        LR_data = LR_data[~outliers_]
        data2 = data[~outliers_].copy(deep=True)
        wgts = wgts[~outliers_[:-1]]

        # Select training data
        LR_train_data = LR_data.iloc[:-1]

        # Select target column
        LR_output = data2[ind_var]

        # Select training output
        LR_train_output = LR_output.iloc[:-1]

        # Fit the model
        LR_model.fit(LR_train_data, 
                     LR_train_output, 
                     sample_weight=wgts)
    
        # Execute predictions
        preds = LR_model.predict(LR_data)
        preds_all = LR_model.predict(data[dep_vars])
        
        # Calculate errors and allowable values
        errors, IQR, low, high = return_credible_region(LR_output, 
                                                        preds,
                                                        preds_all,
                                                        THRESHOLD,
                                                        TEST_TYPE)

    # Compute fraction of variance unexplained
    unex_var = fraction_unexplained(LR_train_output,LR_model.predict(LR_train_data),wgts)    
    
    return (preds_all, errors, IQR, low, high, unex_var, outlier_notice)