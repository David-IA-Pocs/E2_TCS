import pandas as pd
import joblib
import polars as pl
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

def cargar_modelo():
    # Cargar el modelo guardado
    modelo_cargado = joblib.load("modelo_arbol (1).pkl")
    
    return modelo_cargado
def scaler():
    standarscaler = StandardScaler()
    df_resampled = pd.DataFrame(pd.read_csv("data/df_resampled.csv"))
    ## Separando nuestros datos en prueba y entrenamiento
    df_resampled =df_resampled.drop(columns='Unnamed: 0')
    y = df_resampled['isFraud']
    x = df_resampled.drop(columns='isFraud')
    #df_resampled.to_csv("df_resampled.csv")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state = 77)
    x_train = pd.DataFrame(x_train)
    standarscaler.fit(x_train)
    return standarscaler
def prediccion_model(modelo_cargado,standarscaler,dato_nuevo):
    # Realizar la predicción
    x_test_Sce = standarscaler.transform(dato_nuevo)
    y_pred = modelo_cargado.predict(x_test_Sce)
    
    return y_pred
def cargar_datos():
    df_banco_clean = pl.read_csv("data/datos50.csv") 
    df_banco_clean = df_banco_clean.drop('isFraud')
    return df_banco_clean
#registro normal
# 0,1,4,5,6
#registro fraud
#  2,3,251,252,680
# Cargar el modelo guardado

modelo_cargado = cargar_modelo()
standarscaler = scaler()
df_banco_clean = cargar_datos()

X=df_banco_clean[2]
#X=X.drop("isFraud")
#X=X.drop(X.columns[0])
#X=df_banco_clean.with_row_index().filter(pl.col("isFraud") == 1)
prediccion = prediccion_model(modelo_cargado,standarscaler,X)
print(prediccion)
X=df_banco_clean[1]
prediccion = prediccion_model(modelo_cargado,standarscaler,X)
print(prediccion)