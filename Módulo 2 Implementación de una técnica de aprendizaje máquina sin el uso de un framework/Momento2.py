# Common imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.evaluate import bias_variance_decomp
import statsmodels.api as sm


def split_datasets(df):
    #Variables hardcorded
    #data frame
    #print(f'{"Dataframe":=^50}')
    #print(df)
    print(f'\n{"columnas del dataframe":=^50}\n\n')
    print(df.columns.values)
    print(f'\n{"":=^50}')
    #colDes = int(input('\nTe gustaría poner las columnas que vas a usar ( escribe "1") ó las columnas que NO quieres usar ( escribe "2"): '))
    colDes = 1
    print(f'\nTe gustaría poner las columnas que vas a usar ( escribe "1") ó las columnas que NO quieres usar ( escribe "2"): {colDes}')
    if (colDes == 1):
        #columnsnum = int(input('\nCuantas columnas te gustaría usar (Número): '))
        columnsnum = 2
        print(f'\n\nCuantas columnas te gustaría usar (Número): {columnsnum}')
    else:
        columnsnum = int(input('\nCuantas columnas te gustaría quitar (Número): '))
    columnas = []
    '''
    for i in range(columnsnum):
        columna = input(f'\nIngresa el nombre de la columna {i+1}: ')
        columnas.append(columna)
    if (colDes == 1):
        X = df[columnas]
    else:
        X = df.drop(columnas, axis=1)
        '''
    columnas =['OverallQual', 'OverallCond']
    X = df[columnas]
    #ycolumn = input('\nCúal quieres que sea tu Y: ')
    ycolumn = "MSSubClass"
    print(f'\n{"Posibles clases de tu y":=^50}\n')
    y = df[[ycolumn]]
    print(y[ycolumn].unique())
    print(f'\n{"Entrenamiento del modelo":=^50}')
    #PEntrenamiento = int(input("\nPorcentaje que te gustaría para entrenar (número entero): "))
    PEntrenamiento = 50
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=(100-PEntrenamiento)/100,random_state=42)
    NEstimadores = 3
    NHojas = 4
    rnd_clf = RandomForestClassifier(n_estimators=NEstimadores, max_leaf_nodes=NHojas, n_jobs=12,max_features=4,min_samples_split=4)
    avg_expected_loss, bias, var = bias_variance_decomp(rnd_clf, X_train.to_numpy(), y_train.to_numpy().ravel(), X_test.to_numpy(), y_test.to_numpy().ravel(), loss='mse', num_rounds=200, random_seed=42) 
    X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
    print('avg expected loss: %.3f' % avg_expected_loss)
    print('Bias: %.3f' % bias)
    print('Variance: %.3f' % var)
    modelo (X_train, X_test, y_train, y_test,columnas,ycolumn,rnd_clf)

def modelo(X_train, X_test, y_train, y_test,columnas,ycolumn,rnd_clf):
    #n_estimators is the number of trees in the forest
    #n_jobs is the number of jobs to run in parallel
    print(f'\n{"Híper parámetros árbol":=^50}')
    #NEstimadores = int(input("\nCuantos estimadores quieres: "))
    #NHojas = int(input("\nCuantas hojas máximas quieres: "))
    # summarize results
    rnd_clf.fit(X_train, y_train.values.ravel())
    #Calculate accuracy
    y_pred_rf= rnd_clf.predict(X_test)
    y_pred_train= rnd_clf.predict(X_train)
    print(f'\n{"Accuracy":=^50}')
    print("\nrandom forest accuracy:", accuracy_score(y_test[ycolumn].reset_index(drop=True), y_pred_rf))
    plt.title("Random Forest Real vs Predicción (Train)")
    plt.plot(y_train[ycolumn].reset_index(drop=True),label="Y real",color='red')
    plt.plot(y_pred_train,label="Predicción",color='green')
    plt.legend(["Valor Real", "Predicción"], loc ="upper right")
    plt.show()
    plt.title("Random Forest Real vs Predicción")
    plt.plot(y_test[ycolumn].reset_index(drop=True),label="Y real",color='red')
    plt.plot(y_pred_rf,label="Predicción",color='green')
    plt.legend(["Valor Real", "Predicción"], loc ="upper right")
    plt.show()
    valores = []
    print(f'\n{"Valores de columnas":=^50}')
    '''for columna in columnas:
        valor = input(f'\nValor a asignar en la columna {columna}: ')
        valor = eval(valor)
        valores.append(valor)'''
    valores = [5,11]
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
    #filename = input('\nNombre del archivo: ')
    filename = 'train'
    read_file(filename)

main()