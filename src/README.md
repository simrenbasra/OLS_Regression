# Project Utils

## Overview

There are two util python files:
    1. data_preprocess.py - To be used for preprocessing of data in order to prepare the data for OLS regression model building.
    2. linear_utils.py - To be used to build and assess the OLS regression model

## data_preprocess.py

### Class: Data_Preprocess()

### Objects:

1. init 
   **Description:** Initialises the class with a dataframe 
   **Inputs:**  a dataframe
   **Outputs:** data_raw - dataframe (not to be modified in other objects)
                data - dataframe (to be modified in other objects)

2. get_dummies 
   **Description:** To encode categorical columns
   **Inputs:**    threshold - default set to None 
                            - in case there are categorical columns with a large number of categories
                            - prevents number of columns from getting too large 
                 drop_first - default set to True
                            - drops one category from each dummied variable, required to drop multi-collinearity
                            - if choose not to drop-first (random), user must drop one category for each dummy varaible themselves
   **Outputs:**  Dataframe where categorical columns are dummified

3. std_scaling
   **Description:** To scale numerical columns
   **Inputs:** None
   **Outputs:** Dataframe where numerical columns are scaled

## linear_utils.py

### Class: Linear_Model_Builder 

### Objects:

1. init 
   **Description:** Initialises the class with a dataframe and target variable (y)
   **Inputs:**  dataframe - dataframe 
                   target - target variable (y) as a string 
   **Outputs:**  data_raw - dataframe (not to be modified in other objects)
                     data - dataframe (to be modified in other objects)
                   target - target column (as series datatype)
              target_name - name of target variable
                        X - defining X (independent variables)

2. get_heatmap 
   **Description:** Produce heatmap to assess co-linearity in addition to performing vif to remove multi-colinearity
   **Inputs:**vif_cutoff - default value of 5
                         - removes multi-colinearity
                 figsize - default value of (50,50)
                         - size of correlation matrix
   **Outputs:** Correlation matrix showing correlation between X variables 

3. x_modelling 
   **Description:**  To make modifications to X variables.
   **Inputs:**   to_drop - default set to None
                         - list of columns to drop from X/data

   **Outputs:** Returns a dataframe with specified columns removed

4. build_OLS_model 
   **Description:** Instatitates and fits OLS regression model to X and y
   **Inputs:** None
   **Outputs:** OLS regression model

5. assess_accuracy 
   **Description:** To assess check assumption 3 and 4 hold for the ols regression, we will investigate:

                    - Histogram of residuals: Checking residuals follow a normal distribution (Assumption 3)
                    - Shapiro-Wilk: Hypothesis test to test for normal distribution in residuals (Assumption 3)
                    - QQ Plot: Compares actual model residuals against residuals which follow a normal distribution (Assumption 3)
                    - Homoschedasticity: Plots the spread of residuals across predicted y values, homoschedasticity is where spread is constant (Assumption 4)
   **Inputs:** final_model - 'best' OLS model
   **Outputs:** Outputs the following four plots:
                    1. Hist of residuals
                    2. Shapiro-Wilk Hypothesis Test 
                    3. QQ Plot
                    4. Homo/Hetero-scedasticity of Residuals 

## Using classes

To use classes in notebooks run the following code at the start of the notebook to access class and objects:

    sys.path.append('../../src')
    from linear_utils import Linear_Model_Builder
    from data_preprocess import Data_Preprocess 

## Authors

Contributors names:

Simren Basra

## Version History

* 0.1
    * Initial Release

