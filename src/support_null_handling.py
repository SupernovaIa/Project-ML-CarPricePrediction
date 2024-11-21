# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Imputación de nulos usando métodos avanzados estadísticos
# -----------------------------------------------------------------------
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer


class MissingValuesHandler:
    """
    Class for handling missing values in a dataframe
    """

    def __init__(self, dataframe):

        self.dataframe = dataframe
    

    def get_missing_values_percentages(self):
        df_missing_values = (self.dataframe.isna().sum() / self.dataframe.shape[0]) * 100
        return df_missing_values[df_missing_values > 0]
    

    def select_missing_values_columns(self):

        filter_missing_values = self.dataframe.columns[self.dataframe.isna().any()]
        
        cols = self.dataframe[filter_missing_values].select_dtypes(include=np.number).columns
        return cols


    def use_knn(self, list_columns=None, n=5):

        if list_columns == None:
            list_columns = self.select_missing_values_columns().to_list()

        imputer = KNNImputer(n_neighbors= n)
        imputed = imputer.fit_transform(self.dataframe[list_columns])

        new_columns = [col + "_knn" for col in list_columns]
        self.dataframe[new_columns] = imputed

        return self.dataframe
    

    def use_imputer(self, list_columns):

        imputer = IterativeImputer(max_iter=20, random_state=42)
        imputed = imputer.fit_transform(self.dataframe[list_columns])

        new_columns = [col + "_iterative" for col in list_columns]
        self.dataframe[new_columns] = imputed

        return self.dataframe
    

    def comparar_metodos(self):

        columns = self.dataframe.columns[self.dataframe.columns.str.contains("_knn|_iterative")].tolist() + self.select_missing_values_columns.tolist()
        results = self.dataframe.describe()[columns].reindex(sorted(columns), axis=1)
        return results