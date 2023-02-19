import pandas as pd
from pandas import DataFrame

def remove_columns_w_value_in_any_column(df: DataFrame, value: str) -> DataFrame:
    """In a dataframe removes all rows that has the "value" in at least one of its columns. 

    Args:
        df (DataFrame): Base dataframe
        value (str): Target value

    Returns:
        DataFrame: Modified dataframe without the rows with ?
    """
    return df.drop(df[df == "value"].index)