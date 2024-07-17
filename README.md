# Linear-Regression Class Builder

This project is a simple implementation of a linear regression model class builder applied to a car dataset. The class allows users to easily create a linear regression model, train it on the dataset and make predictions. The car dataset typically includes features such as carbody type, aspiration and engine type which are used to predict price of cars.

## Getting Started

### Class Methods

* __init__(self, dataframe, target)
    - Initialises the linear regression model with the provided dataframe and name of target variable.
    - dataframe : of type pandas DataFrame
    - target : name of target column in dataset 

* get_dummies(self,threshold=None, drop=True)
    - Produces dummy variable for categorical columns
    - threshold : int, if number of dummies greater than threshold value, category column is to be dropped. Optional argument.
    - drop: if True drops the first value when creating dummies. Default value is True.
 
* x_modelling(self,to_drop=None)
    - Returns all feature variables
    - to_drop : inputs name of columns to drop. Optional argument.

* get_heatmap(self,vif_cutoff=5)
    - generates correlation heatmap to show correlation between independent features
    - calculates VIF values before generating correlation heatmap, if VIF values are greater than vut_off columns are deleted from the dataframe.
    - vif_cutoff : cut off for vif values, default value is 5

* build_OLS_model(self)
    - Builds and fits linear regression model using statsmodel.api 
    - Adds constant to all independent features

* assess_accuracy(self,final_model)
    - Assesses accuracy of model through 
        * Hist of residuals
        * Shapiro-Wilk Hypothesis Test 
        * QQ Plot
        * Homo/Hetero-scedasticity of Residuals
    - returns graphs 
    - final_model: pass final model for assessment

### Installing

### Executing program


## Authors

Contributors names:

Simren Basra

## Version History

* 0.1
    * Initial Release
