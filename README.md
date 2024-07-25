----
# Stats Linear Regression 
----
This project is a simple the implementation of an Ordinary Least Squares (OLS) regression model, including preprocessing the data and exploring the effects of Principal Component Analysis (PCA) on the model's performance.

## Project Overview

This project aims to:

- Data Preprocessing: Prepare the data set for OLS regression by making all columns numeric, scaling and transforming any data.
- Implement OLS Regression: Build a OLS regression model
- Explore PCA: Assess effect PCA has on OLS regression model
- Evaluate Model Performance: Analyse the performance of the model using R-squared, MSE (mean squared error) and MAE (mean absolute error).

## Dataset 

Dataset is from Kaggle, please refer to link below:

- https://www.kaggle.com/datasets/hellbuoy/car-price-prediction/data


## Detailed Steps

1. **Data Preprocessing**

    - **Convert All Columns to Numeric:** Ensure all columns in the dataset are numeric by encoding categorical variables.
    - **Scaling Data:** Standardise the data using standard scaler so all columns are on the same scale and all features are on a 'level playing field'.
    - **Data Transformation:** Apply any transforamtions to ensure data is ready for OLS regression modelling.


2. **Implement OLS Regression**
    
    - Build the Model: Use statsmodels to build the OLS regression model.
    - Assess model meets the 4 assumptions of linear regression:
        - Assumption 1: A linear relationship between X (independent variable/s) and y (dependent variable)
        - Assumption 2: Independent variables are independent to each other, there is no collinearity or multicollinearity.
        - Assumption 3: Residuals should be normally distributed.
        - Assumption 4: Residuals should show homoscedasticity.

3. **Explore PCA**

    - Apply PCA to data: Determine optimal number of principal components and perform PCA to reduce the number of features and multicollinearity. 
    - Assess impace of PCA on OLS regression: Re-build the OLS regression model using principal components and compare the perfomance of the model to the base model.

4. **Evaluate Performance of Each Model**

    - R-squared: Measures the amount of variance in the target variable explained by the independent variables.
    - Mean Squared Error (MSE): Measure the average squared difference between the estimated values and the actual value.
    - Mean Absolute Error (MAE): Measure the average of the errors in a set of predictions to the actual values.

## Files and Directories

- data/: Directory containing the dataset.
- notebooks/: Notebooks for data preprocessing, model building and analysis.
- src/: Python scripts for data preprocessing and linear model building.

## How to run

1. Clone the Repository:

        git clone https://github.com/simrenbasra/Stats_Linear_Regression.git

2. Install dependencies

        conda env create -f conda_env.yml

3. Run the notebooks 
    Open the notebooks in the notebooks/directory and follow the instructions
    

## Authors
Contributors name/s: Simren Basra

## Version History
* 0.1
    * Initial Release
