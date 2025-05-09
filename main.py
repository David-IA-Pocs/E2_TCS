from fastapi import FastAPI
from pydantic import BaseModel
import met as mt
import polars as pl
from google.genai import types
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import modelo as mo
#PY version 3.10.11
pl.Config.set_tbl_cols(-1)

app = FastAPI()
#carga de recursos necesarios para predicciones
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
modelo_cargado = mt.cargar_modelo()
standarscaler = mt.scaler()

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

#uso de embedings con ollama
@app.post("/ollama")
async def read_item(transaccion: BD_Model):
    #limpeza de datos de transaccion
    transaccion = str(transaccion)
    data_dict = {kv.split("=")[0].strip(): kv.split("=")[1].strip("'") for kv in transaccion.split()}

    # Crear un DataFrame de Polars
    df = pl.DataFrame([data_dict])
    #g= pl.DataFrame(transaccion)
    print (df)
    X=df
    question = f""" {X} """
    response = mo.rag_pipeline_ollama(question, vectorstore)

    return {"prediccion": str(response)}

#uso de embedings con gemini
@app.post("/gemini/Embedings")
async def recibir_transaccion_geminiEmb(transaccion: BD_embedings):
    #limpeza de datos de transaccion
    transaccion = str(transaccion)
    data_dict = {kv.split("=")[0].strip(): kv.split("=")[1].strip("'") for kv in transaccion.split()}

    # Crear un DataFrame de Polars
    df = pl.DataFrame([data_dict])
    #g= pl.DataFrame(transaccion)
    print (df)
    question = f""" {df} """
    response = mo.rag_pipeline_gemini(question, vectorstore)
    return {"mensaje": "Transacción recibida", "datos": str(response)}

#uso de gemini y modelo arboles de desicion
@app.post("/gemini/ModeloEn")
async def recibir_transaccion_geminiModel(transaccion: BD_Model):
    #limpeza de datos de transaccion
    transaccion = str(transaccion)
    data_dict = {kv.split("=")[0].strip(): kv.split("=")[1].strip("'") for kv in transaccion.split()}

    # Crear un DataFrame en Polars
    df = pl.DataFrame([data_dict])
    X= df
    print (df)
    #prediccion 1= fraude ,0= no fraude
    prediccion = mt.prediccion_model(modelo_cargado,standarscaler,X)
    a = int(prediccion[0])
    print(a)
    if a == 0:
        a = "No es fraude"
        print("No es fraude")
    else:
        a = "Es fraude"
        print("Es fraude")
    #uso de gemini para explicacion
    a = mt.gem(prediccion, X)

    return {"prediccion": str(a)}

