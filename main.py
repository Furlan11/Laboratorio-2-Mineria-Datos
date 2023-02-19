import lpfdata as lpf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def main():
    df = pd.read_csv('./data/breast-cancer-wisconsin.csv')
    df = lpf.remove_columns_w_value_in_any_column(df, '?')
    df = df.drop(['id','cell_shape','marginal_Adhesion','Epithehtial_Cell_Size','Bland_Chromatin' ], axis=1)
    
    print("Logical regression")
    # Separa dataframe en varriables dependientes e independientes 
    X = df.iloc[:, 0:4].values
    y = df.iloc[:, -1].values
    
    X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(X, y, test_size = 0.25, random_state = 0)
    
    # Logical regression
    modeloLog = LogisticRegression(max_iter = 500)
    modeloLog.fit(X_entreno,y_entreno)
    predicciones = modeloLog.predict(X_prueba)
    accuracyLR = classification_report(y_prueba, predicciones)
    print("------------------Precision------------------")
    print(accuracyLR)
    print("KNN")
    best_k = lpf.findBestK(X_entreno, y_entreno, X_prueba, y_prueba)
    knn = KNeighborsClassifier(n_neighbors = best_k)
    knn.fit(X_entreno, y_entreno)
    pred = knn.predict(X_prueba)
    print(f'Con k = {best_k}')
    accuracyKNN = classification_report(y_prueba, pred)
    print("------------------Precision------------------")
    print(accuracyKNN)

    
if __name__ == "__main__":
    main()