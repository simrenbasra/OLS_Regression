# import relevant packages
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
    def __init__(self, dataframe, target):
        self.data_raw = dataframe
        self.data = dataframe 
        self.target = self.data[target]
        self.target_name = target
        self.x = self.data.loc[:,self.data.columns!=target]

    def get_dummies(self,threshold=None, drop=True):

        cat_cols = self.x.select_dtypes(include='object').columns
        
        if threshold != None:
            for col in cat_cols:
                if len(self.data[col].value_counts().index) > threshold:
                    self.data = self.data.drop(col, axis= 1)
                    self.x = self.x.drop(col,axis=1)           
        else:
            pass 
        
        cat_cols = self.x.select_dtypes(include='object').columns
        self.dum_data = pd.get_dummies(data=self.data, columns=cat_cols, dtype= int, drop_first= drop)      
        return self.dum_data
 
    def x_modelling(self,to_drop=None): 
        self.x = self.dum_data.loc[:,self.dum_data.columns!=self.target_name]
        if to_drop != None:
            self.dum_data = self.dum_data.drop(columns=to_drop, axis=1)
            self.x = self.x.drop(columns=to_drop, axis=1)
        else:
            pass      
        return self.x
        
    def get_heatmap(self,vif_cutoff=5):

        self.x_with_c = sm.add_constant(self.x)  

        vif_values = pd.Series([variance_inflation_factor(self.x_with_c,i) for i in range(self.x_with_c.shape[1])])
        vif_values.index = self.x_with_c.columns
        vv_df = vif_values.reset_index()

        cols_to_drop = []
        for vif in vv_df.values[1:]: # ignore const
            if vif[1] >=vif_cutoff:
                cols_to_drop.append(vif[0])
        if len(cols_to_drop) != 0:
            self.x.drop(cols_to_drop,axis=1, inplace=True)
            self.dum_data.drop(columns=cols_to_drop, axis=1,inplace=True)
        else:
            pass
        
        plt.figure(figsize=(50,50))
        corr = self.x.corr()
        mask = np.triu(corr)
        sns.heatmap(corr, cmap="coolwarm", annot=True, mask=mask, annot_kws={"size":20},vmax=1,vmin=-1)
        plt.show()

    def build_OLS_model(self): 
        self.x_with_c = sm.add_constant(self.x)      
        model = sm.OLS(self.target,self.x_with_c)
        fit_model = model.fit()
        return fit_model
    
    def assess_accuracy(self,final_model):
        # Three things to check :
            # 1. Hist of residuals
            # 2. Shapiro-Wilk Hypothesis Test 
            # 3. QQ Plot
            # 4. Homo/Hetero-scedasticity of Residuals

        # Get residuals of model using.resid
        residuals = final_model.resid
        
        # Plotting histogram of residuals
        plt.figure(figsize=(8,5))
        plt.hist(residuals, bins=50 ,edgecolor='black', color='teal')
        plt.axvline(x=residuals.mean(), color='hotpink', label='Mean')
        plt.axvline(x=residuals.median(), color='yellow', label='Median')
        plt.title('Residual Histogram',fontsize=10, fontweight='bold')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

        # Shapiro-Wilk Hypothesis Test
        shap_wilk = stats.shapiro(residuals)
        # p value less than 0.05 so therefore we can reject hypothesis 0 and can assume alternate hypothesis holds true
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
        pred_y = final_model.predict(self.x_with_c)
        plt.figure()
        plt.scatter(x= pred_y, y=residuals, alpha=0.5, color='teal') 
        plt.xlabel('Predicted Price')
        plt.ylabel('Residuals')
        plt.title('Homo/Hetero-scedasticity of Residuals',fontsize=10, fontweight='bold')
        plt.show()