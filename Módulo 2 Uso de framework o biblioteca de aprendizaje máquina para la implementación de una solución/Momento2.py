# Common imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd


def split_datasets(df):
    #Variables hardcorded
    #data frame
    #print(f'{"Dataframe":=^50}')
    #print(df)
    print(f'\n{"columnas del dataframe":=^50}\n\n')
    print(df.columns.values)
    print(f'\n{"":=^50}')
    colDes = int(input('\nTe gustaría poner las columnas que vas a usar ( escribe "1") ó las columnas que NO quieres usar ( escribe "2"): '))
    if (colDes == 1):
        columnsnum = int(input('\nCuantas columnas te gustaría usar (Número): '))
    else:
        columnsnum = int(input('\nCuantas columnas te gustaría quitar (Número): '))
    columnas = []
    for i in range(columnsnum):
        columna = input(f'\nIngresa el nombre de la columna {i+1}: ')
        columnas.append(columna)
    if (colDes == 1):
        X = df[columnas]
    else:
        X = df.drop(columnas, axis=1)
    ycolumn = input('\nCúal quieres que sea tu Y: ')
    print(f'\n{"Posibles clases de tu y":=^50}\n')
    y = df[[ycolumn]]
    print(y[ycolumn].unique())
    print(f'\n{"Entrenamiento del modelo":=^50}')
    PEntrenamiento = int(input("\nPorcentaje que te gustaría para entrenar (número entero): "))
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=(100-PEntrenamiento)/100) 
    X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
    modelo (X_train, X_test, y_train, y_test,columnas,ycolumn)

def modelo(X_train, X_test, y_train, y_test,columnas,ycolumn):
    #n_estimators is the number of trees in the forest
    #n_jobs is the number of jobs to run in parallel
    print(f'\n{"Híper parámetros árbol":=^50}')
    NEstimadoes = int(input("\nCuantos estimadores quieres: "))
    NHojas = int(input("\nCuantas hojas máximas quieres: "))
    rnd_clf = RandomForestClassifier(n_estimators=NEstimadoes, max_leaf_nodes=NHojas, n_jobs=-1)
    rnd_clf.fit(X_train, y_train.values.ravel())
    #Calculate accuracy
    y_pred_rf= rnd_clf.predict(X_test)
    print(f'\n{"Accuracy":=^50}')
    print("\nrandom forest accuracy:", accuracy_score(y_test[ycolumn].reset_index(drop=True), y_pred_rf))
    plt.title("Random Forest Real vs Predicción")
    plt.plot(y_test[ycolumn].reset_index(drop=True),label="Y real",color='red')
    plt.plot(y_pred_rf,label="Predicción",color='green')
    plt.legend(["Valor Real", "Predicción"], loc ="upper right")
    plt.show()
    valores = []
    print(f'\n{"Valores de columnas":=^50}')
    for columna in columnas:
        valor = input(f'\nValor a asignar en la columna {columna}: ')
        valor = eval(valor)
        valores.append(valor)
    probs = rnd_clf.predict_proba([valores])
    print(f'\n{"Probabilidades":=^50}')
    print(f'Valores de Y: {rnd_clf.classes_}')
    print("\nprobabilidad de las clases",[valores],probs)

    pred =  rnd_clf.predict([valores])
    print(f'\n{"Predicción final":=^50}')
    print(f'tu predicción de la columna {ycolumn} tiene un valor de: {pred}')


def read_file(filename):
    df = pd.read_csv(f'{filename}.csv')
    split_datasets(df)

def main():
    print(f'\n{"Comenzemos":=^50}')
    filename = input('\nNombre del archivo: ')
    read_file(filename)

main()