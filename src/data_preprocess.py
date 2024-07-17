##########################################################################
# Imports
##########################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

class Data_Preprocess():

##########################################################################
# Init
##########################################################################
    def __init__(self, dataframe):
        self.data_raw = dataframe
        self.data = dataframe

##########################################################################
# Get Dummies (for categorical columns)
##########################################################################

    def get_dummies(self,threshold=None, drop_first=True):
        cat_cols = self.data.select_dtypes(include='object').columns
        
        # cases where the number of dummies is too large
        if threshold != None:
            for col in cat_cols:
                if len(self.data[col].value_counts().index) > threshold:
                    self.data = self.data.drop(col, axis= 1)      
        else:
            pass 
        
        cat_cols = self.data.select_dtypes(include='object').columns
        self.dummy_data = pd.get_dummies(data=self.data, columns=cat_cols, dtype= int, drop_first= drop_first)      
        return self.dummy_data


##########################################################################
# Scaling (numerical columns)
##########################################################################

    def std_scaling(self):
        num_cols = self.data.select_dtypes(exclude='object').columns

        scaler = StandardScaler()

        for col in num_cols:
            scaler.fit(self.data[[col]])
            self.data[col] = scaler.transform(self.data[[col]])

        return self.data   
