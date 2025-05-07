from fastapi import FastAPI
from pydantic import BaseModel
import met as mt
import polars as pl
from google.genai import types
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import modelo as mo

pl.Config.set_tbl_cols(-1)

app = FastAPI()

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
modelo_cargado = mt.cargar_modelo()
standarscaler = mt.scaler()
#df_banco_clean = mt.cargar_datos()
# Definir modelo de datos
class BD_embedings(BaseModel):
    Monto: float
    ubicacion: str
    Método_Pago: str
    Hora_Transaccion: int
    Intentos_Fallidos: int

class BD_Model(BaseModel):
    type_CASH_IN: float
    type_CASH_OUT: float
    type_PAYMENT: float
    type_TRANSFER: float
    amount: float
    type_2_CC: float
    type_2_CM: float
    day: int
    part_of_the_day_madrugada: float
    part_of_the_day_mañana: float
    part_of_the_day_noche: float
    part_of_the_day_tarde: float

@app.post("/ollama")
async def read_item(transaccion: BD_Model):
    transaccion = str(transaccion)
    data_dict = {kv.split("=")[0].strip(): kv.split("=")[1].strip("'") for kv in transaccion.split()}

    # Crear un DataFrame de Polars
    df = pl.DataFrame([data_dict])
    #g= pl.DataFrame(transaccion)
    print (df)

    #return {"mensaje": "Transacción recibida", "datos": transaccion}
    X=df
    g = pl.DataFrame({"type_CASH_IN" :1, "type_CASH_OUT" :0, "type_PAYMENT" :0, "type_TRANSFER" :0, "amount"   :6000.56, "type_2_CC" :1, "type_2_CM" :0,
                   "day" :1, "part_of_the_day_madrugada" :1, "part_of_the_day_mañana" :0, "part_of_the_day_noche" :0, "part_of_the_day_tarde": 0})

    prediccion = mt.prediccion_model(modelo_cargado,standarscaler,X)
    a = int(prediccion[0])
    print(a)
    if a == 0:
        a = "No es fraude"
        print("No es fraude")
    else:
        a = "Es fraude"
        print("Es fraude")

    a = mt.gem(prediccion, X)

    return {"prediccion": str(a)}

#uso de embedings con gemini
@app.post("/gemini/Embedings")
# no fraude
#2443,Tokio,Transferencia,22,0,0
#5321,Nueva York,Criptomoneda,2,0,0
# fraude
#5061,Buenos Aires,Criptomoneda,9,4,1
#1194,Madrid,Criptomoneda,22,2,1
async def recibir_transaccion_geminiEmb(transaccion: BD_embedings):
        # Convertir el string en un diccionario
    transaccion = str(transaccion)
    data_dict = {kv.split("=")[0].strip(): kv.split("=")[1].strip("'") for kv in transaccion.split()}

    # Crear un DataFrame de Polars
    df = pl.DataFrame([data_dict])
    #g= pl.DataFrame(transaccion)
    print (df)
    question = f"""
la transaccion {df} es fraude ? 
"""
    response = mo.rag_pipeline(question, vectorstore)
    return {"mensaje": "Transacción recibida", "datos": str(response)}
#uso de embedings con ollama
@app.post("/ollama/Embedings")
async def recibir_transaccion_ollamaEmb(transaccion: BD_embedings):
        # Convertir el string en un diccionario
    transaccion = str(transaccion)
    data_dict = {kv.split("=")[0].strip(): kv.split("=")[1].strip("'") for kv in transaccion.split()}

    # Crear un DataFrame de Polars
    df = pl.DataFrame([data_dict])
    #g= pl.DataFrame(transaccion)
    print (df)
    return {"mensaje": "Transacción recibida", "datos": transaccion}

#uso de gemini y modelo arboles de desicion
@app.post("/gemini/ModeloEn")
async def recibir_transaccion_geminiModel(transaccion: BD_Model):
    # data fraude
    # 0,1,0,0,40433.84,1,0,17,0,0,0, 1
    # 1,0,0,0,254693.1,1,0,9,0,0,0, 1
    # data no fraude
    # 0,0,1,0,34799.63,0,1,11,0,1,0, 0
    # 0,1,0,0,53782.87,1,0,6,0,0,1, 0
    # Convertir el string en un diccionario
    transaccion = str(transaccion)
    data_dict = {kv.split("=")[0].strip(): kv.split("=")[1].strip("'") for kv in transaccion.split()}

    # Crear un DataFrame de Polars
    df = pl.DataFrame([data_dict])
    X= df
    #g= pl.DataFrame(transaccion)
    print (df)
    prediccion = mt.prediccion_model(modelo_cargado,standarscaler,X)
    a = int(prediccion[0])
    print(a)
    if a == 0:
        a = "No es fraude"
        print("No es fraude")
    else:
        a = "Es fraude"
        print("Es fraude")

    a = mt.gem(prediccion, X)

    return {"prediccion": str(a)}
    return {"mensaje": "Transacción recibida", "datos": transaccion}

""" import requests

url = "http://127.0.0.1:8000/transaccion/"
data = {
    "usuario": "Ana",
    "monto": 500.75,
    "ubicacion": "Madrid",
    "fraude": False
}

response = requests.post(url, json=data)
print(response.json()) """