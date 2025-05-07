import pandas as pd
import joblib
from google import genai
from google.genai import types
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
    df_resampled = pd.DataFrame(pd.read_csv("data1/df_resampled.csv"))
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
pl.Config.set_tbl_cols(-1)
def cargar_datos():
    df_banco_clean = pl.read_csv("data/datos50.csv") 
    df_banco_clean = df_banco_clean.drop('isFraud')
    return df_banco_clean
#registro normal
# 0,1,4,5,6
#registro fraud
#  2,3,251,252,680
client = genai.Client(api_key="AIzaSyD83oAyLmHEnIj-emaz3CaciuniNoqYNDg")

def gem(prediccion, datos):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=f"""Eres un asistente de IA que ayuda a los usuarios a entender una tabla que tiene los siguientes campos
            type_CASH_IN,type_CASH_OUT,type_PAYMENT,type_TRANSFER,amount,type_2_CC,type_2_CM,day,part_of_the_day_madrugada,
            part_of_the_day_mañana,part_of_the_day_noche,part_of_the_day_tarde,isFraud.
            los registros estan en dumies, es decir, en 0 y 1.
            las columnas  type_CASH_IN,type_CASH_OUT,type_PAYMENT,type_TRANSFER pertenece a la transaccion, es decir, si es un ingreso, un egreso, un pago o una transferencia.
            las columnas type_2_CC,type_2_CM, son el tipo de tarjeta, es decir, si es una tarjeta de credito o una tarjeta de debito.
            las columnas day,part_of_the_day_madrugada,part_of_the_day_mañana,part_of_the_day_noche,part_of_the_day_tarde son el dia y la hora de la transaccion, es decir, si es un lunes, martes, miercoles, jueves, viernes, sabado o domingo.
            usa el registro amount, que es el monto de la transaccion, para ayudar al analisis determinar si es un fraude o no.
            con esta informacion genera una respuesta en español, que explique si la transaccion es fraude o no, y porque es fraude o no.
            si es 1 es fraude, si es 0 no es fraude.
            no analices el modelo, solo la tabla y los datos que te doy.
            no uses las variables que te doy, usa palabras mas claras y entendibles para el usuario.
            """
),
        contents=f"fraude: {prediccion}, datos: {datos}",
    )
    return response.text