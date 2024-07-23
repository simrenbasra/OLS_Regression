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
        ''' 
            Description:
                Initialises the class with a dataframe 

            Inputs:
                dataframe - dataframe 

            Outputs:
                data_raw - dataframe (not to be modified in other objects)
                    data - dataframe (to be modified in other objects)
        '''
        self.data_raw = dataframe
        self.data = dataframe

##########################################################################
# Get Dummies (for categorical columns)
##########################################################################

    def get_dummies(self,threshold=None, drop_first=True):
        ''' 
            Description:
                To encode categorical columns

            Inputs:
                threshold - default set to None 
                          - in case there are categorical columns with a large number of categories
                          - prevents number of columns from getting too large

               drop_first - default set to True
                          - drops one category from each dummied variable, required to drop multi-collinearity
                          - if choose not to drop-first (random), user must drop one category for each dummy varaible themselves

            Outputs:
                dummy_data - returns dataframe where categorical columns are dummified
        '''
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
        ''' 
            Description:
                To scale numerical columns

            Inputs:
                None

            Outputs:
                Dataframe where numerical columns are scaled
        '''
                
        num_cols = self.data.select_dtypes(exclude='object').columns
        # using std scaling (z score) not to distort correlations in data
        scaler = StandardScaler()

        for col in num_cols:
            scaler.fit(self.data[[col]])
            self.data[col] = scaler.transform(self.data[[col]])

        return self.data   
