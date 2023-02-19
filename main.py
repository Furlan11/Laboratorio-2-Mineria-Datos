import lpfdata as lpf
import pandas as pd
def main():
    df = pd.read_csv('./data/breast-cancer-wisconsin.csv')
    df = lpf.remove_columns_w_value_in_any_column(df, '?')
    
if __name__ == "__main__":
    main()