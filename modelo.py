import ollama
from google import genai
from google.genai import types
import chardet
from langchain.text_splitter import CharacterTextSplitter
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import polars as pl
client = genai.Client(api_key="")

""" 
file_path = "transacciones_bancarias.csv"
with open(file_path, "r",  errors='replace') as file:
    text = file.read()
 """
# Función personalizada para dividir el texto en fragmentos manejables
def custom_text_splitter(text, chunk_size, chunk_overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

# Dividir el texto en fragmentos manejables
#chunks = custom_text_splitter(text, chunk_size=700, chunk_overlap=100)

# Configurar embeddings usando Ollama con el modelo 'nomic-embed-text'
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Crear la base de datos vectorial FAISS
#vectorstore = FAISS.from_texts(chunks, embeddings)
#vectorstore.save_local("faiss_index")

# Cargar la base de datos vectorial FAISS existente
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Función para consultar con gemini
def query_gemini_with_context(context, question):
    #print("CONTEXTO", context)
    #print("FIN CONTEXTO")
    if not context.strip():  # Si el contexto está vacío, no tenemos información suficiente
        return "Con los datos de la base de datos no es posible predecir fraude o no ."

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=f"""
        usa el siguiente contexto para responder si la transaccion puede ser fraude o no.
        El contexto contiene información sobre transacciones y si fueron fraudes o no.
        El contexto contiene Monto ┆ Ubicación    ┆ Método_Pago   ┆ Hora_Transaccion ┆ Intentos_Fallidos | fraude.         
        Respuesta:
        responde siempre si la transacción es fraude o no en menos de 100 palabras con una explicación breve. 
        responde en el siguiente formato en texto plano:
        
       "explicacion": "no mas de 100 palabras"
        omite la palabra text o json y las comillas
        ------------------------"""

),
        contents=f"Contexto: {context}, transaccion: {question}",
    )
    #print ("RESPONSE", response.text)
    return str(response.text)

def query_ollama_with_context(context, question):
    #print("CONTEXTO", context)
    #print("FIN CONTEXTO")
    if not context.strip():  # Si el contexto está vacío, no tenemos información suficiente
        return "Con los datos de la base de datos no es posible predecir fraude o no ."

    prompt = f"""
        Eres un asistente de IA que ayuda a los usuarios a entender una tabla que tiene los siguientes campos
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
          Contexto: {context}
        transaccion: {question}  
        ------------------------

        
    """

  

    response = ollama.chat(
        model='llama3.2:1b',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content']

# Recuperar fragmentos relevantes
def retrieve_relevant_chunks(query, vectorstore):
    results = vectorstore.similarity_search(query, k=3)
    return results if results else []
   
# Pipeline RAG 
def rag_pipeline_gemini(question, vectorstore):
    results = retrieve_relevant_chunks(question, vectorstore)
    
    if not results:
        return  print("Esa información no está en el documento cargado, pero basado en la información que tengo la siguiente respuesta te puede ser útil\n")
    
    context = "\n".join([result.page_content for result in results])
    
    response = query_gemini_with_context(context, question)
    return response
def rag_pipeline_ollama(question, vectorstore):
    results = retrieve_relevant_chunks(question, vectorstore)
    
    if not results:
        return  print("Esa información no está en el documento cargado, pero basado en la información que tengo la siguiente respuesta te puede ser útil\n")
    
    context = "\n".join([result.page_content for result in results])
    
    response = query_ollama_with_context(context, question)
    return response
