##########################################################################
# Imports
##########################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


class Linear_Model_Builder():

##########################################################################
# Init
##########################################################################

    def __init__(self, dataframe, target):
        ''' 
            Description:
                Initialises the class with a dataframe and target variable (y)

            Inputs:
                dataframe - dataframe 
                   target - target variable (y) as a string 

            Outputs:
                data_raw - dataframe (not to be modified in other objects)
                    data - dataframe (to be modified in other objects)
                  target - target column (as series datatype)
             target_name - name of target variable
                       X - defining X (independent variables)
        '''
        self.data_raw = dataframe
        self.data = dataframe 
        self.target = self.data[target]
        self.target_name = target
        self.X = self.data.loc[:,self.data.columns!=target]


##########################################################################
# Get Heatmap (with VIF)
##########################################################################
        
    def get_heatmap(self,vif_cutoff=5, figsize = (50,50)):
        ''' 
            Description:
                Produce heatmap to assess co-linearity in addition to performing vif to remove multi-colinearity

            Inputs:

              vif_cutoff - default value of 5
                         - removes multi-colinearity

                 figsize - default value of (50,50)
                         - size of correlation matrix

            Outputs:
                Correlation matrix showing correlation between X variables 
        '''

        # Choosing to ignore constant in VIF calculation since it is causing warnings
        # Not sure why the VIF for constant is 0 - to be further investigated
        self.X_with_c = sm.add_constant(self.X)  

        vif_values = pd.Series([variance_inflation_factor(self.X_with_c,i) for i in range(1,self.X_with_c.shape[1])])
        vif_values.index = self.X_with_c.columns[1:]
        vv_df = vif_values.reset_index()

        cols_to_drop = []
        for vif in vv_df.values[1:]: # ignoring const
            if vif[1] >=vif_cutoff:
                cols_to_drop.append(vif[0])
        if len(cols_to_drop) != 0:
            self.X = self.X.drop(cols_to_drop,axis=1)
        else:
            pass

        plt.figure(figsize=figsize)
        corr = self.X.corr()
        # Using mask to remove duplicated values 
        mask = np.triu(corr)
        # Using vmax/vmin to keep scale between 1 and -1 
        sns.heatmap(corr, cmap="coolwarm", annot=True, mask=mask, annot_kws={"size":figsize[1]/2},vmax=1,vmin=-1)
        plt.show()

##########################################################################
# Choosing Modelling Variables (X)
##########################################################################

    def x_modelling(self,to_drop=None): 
        ''' 
            Description:
                To make modifications to X variables.

            Inputs:
                to_drop - default set to None
                        - list of columns to drop from X/data

            Outputs:
                Returns a dataframe with specified columns removed
        '''
        if to_drop != None:
            self.data = self.data.drop(columns=to_drop, axis=1)
            self.X = self.X.drop(columns=to_drop, axis=1)
        else:
            pass      
        return self.X
    
##########################################################################
# Building OLS model
##########################################################################

    def build_OLS_model(self): 
        ''' 
            Description:
                Instatitates and fits OLS regression model to X and y

            Inputs:
                None

            Outputs:
                OLS regression model
                
        '''
        self.X_with_c = sm.add_constant(self.X)      
        model = sm.OLS(self.target,self.X_with_c)
        fit_model = model.fit()
        return fit_model

    
##########################################################################
# Assess Accuracy of OLS Regression
##########################################################################
#   1. Hist of residuals
#   2. Shapiro-Wilk Hypothesis Test 
#   3. QQ Plot
#   4. Homo/Hetero-scedasticity of Residuals
##########################################################################
#  
    def assess_accuracy(self,final_model):
        ''' 
            Description:
                To assess check assumption 3 and 4 hold for the ols regression, we will investigate:
                    - Histogram of residuals: Checking residuals follow a normal distribution (Assumption 3)
                    - Shapiro-Wilk: Hypothesis test to test for normal distribution in residuals (Assumption 3)
                    - QQ Plot: Compares actual model residuals against residuals which follow a normal distribution (Assumption 3)
                    - Homoschedasticity: Plots the spread of residuals across predicted y values, homoschedasticity is where spread is constant (Assumption 4)

            Inputs:
                final_model - 'best' OLS model

            Outputs:
                Outputs the following four plots:
                   1. Hist of residuals
                   2. Shapiro-Wilk Hypothesis Test 
                   3. QQ Plot
                   4. Homo/Hetero-scedasticity of Residuals 
        '''
        # Get residuals of model using.resid
        residuals = final_model.resid
        
        # Plotting histogram of residuals
        plt.figure(figsize=(8,5))
        plt.hist(residuals, bins=50 ,edgecolor='black', color='teal')
        plt.axvline(x=residuals.mean(), color='hotpink', label='Mean')
        plt.axvline(x=residuals.median(), color='yellow', label='Median')
        plt.title('Histogram of Residuals',fontsize=10, fontweight='bold')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

        # Shapiro-Wilk Hypothesis Test
        print('Shapiro-Wilk Hypothesis Test')
        shap_wilk = stats.shapiro(residuals)

        print('Null Hypothesis : Residuals ARE normally distributed')
        print('Alternate Hypothesis : Residuals are NOT normally distributed')

        if shap_wilk[1] < 0.05:
            print(f'''\nShapiro Wilk Results:\np-value of {round(shap_wilk[1],4)} is less than 0.05, therefore we can reject the null hypothesis and assume the alternate hypothesis holds true
                  ''')
        
        else:
            print(f'''\nShapiro Wilk Results:\np-value of {round(shap_wilk[1],4)} is greater than or equal to 0.05, therefore we can reject the null hypothesis and assume the alternate hypothesis holds true
                  ''')
        
        # QQ Plot
        plt.figure()
        stats.probplot(residuals, plot=plt,dist='norm')
        plt.title('QQ Residual Plot',fontsize=10, fontweight='bold')
        plt.show()

        # Homo/Hetero-scedasticity of Residuals
        pred_y = final_model.predict(self.X_with_c)
        plt.figure()
        plt.scatter(x= pred_y, y=residuals, alpha=0.5, color='teal') 
        plt.xlabel('Predicted Price')
        plt.ylabel('Residuals')
        plt.title('Homo/Hetero-scedasticity of Residuals',fontsize=10, fontweight='bold')
        plt.show()