import pandas as pd
from pandas import DataFrame
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
"""
Created on Sun Feb 19 13:47:23 2023

@author: Guillermo Furlan, Luis Pedro Gonzalez Aldana
"""

def remove_columns_w_value_in_any_column(df: DataFrame, value: str) -> DataFrame:
    """In a dataframe removes all rows that has the "value" in at least one of its columns. 

    Args:
        df (DataFrame): Base dataframe
        value (str): Target value

    Returns:
        DataFrame: Modified dataframe without the rows with ?
    """
    return df[~df.astype(str).apply(lambda x: x.str.contains(f'\{value}', na=False)).any(axis=1)]


def findBestK(X_entreno, y_entreno, X_prueba, y_prueba, max_k = 40) -> int:
    """Find the bestk for the dataset. 

    Args:
        X_entreno (Dataframe): _description_
        y_entreno (Serie): _description_
        X_prueba (Dataframe): _description_
        y_prueba (Serie): _description_
        max_k (int, optional): _description_. Defaults to 40.

    Returns:
        int: _description_
    """
    tasa_error = []
    for i in range(1, max_k):
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(X_entreno, y_entreno)
        pred_i = knn.predict(X_prueba)
        tasa_error.append(np.mean(pred_i != y_prueba))

    return tasa_error.index(min(tasa_error))+1
