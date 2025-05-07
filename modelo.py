import ollama
from google import genai
from google.genai import types
import chardet
from langchain.text_splitter import CharacterTextSplitter
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import polars as pl

""" # Detectar codificación
file_path = "transacciones.csv"
#file_path = "doc.txt"
with open(file_path, "rb") as raw_file:
    result = chardet.detect(raw_file.read())
    detected_encoding = result['encoding']
    print(f"Codificación detectada: {detected_encoding}")

# Leer usando la codificación detectada
with open(file_path, "r", encoding=detected_encoding, errors='replace') as file:
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
#embeddings = OllamaEmbeddings(model="all-minilm:33m")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Crear la base de datos vectorial FAISS
#vectorstore = FAISS.from_texts(chunks, embeddings)
#vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

#print("Fragmentos almacenados en la base de datos vectorial FAISS.")
client = genai.Client(api_key="AIzaSyD83oAyLmHEnIj-emaz3CaciuniNoqYNDg")
# Función para consultar con Mistral, asegurando que solo use el contexto proporcionado
def query_ollama_with_context(context, question):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=f"""
        usa el siguiente contexto para responder la pregunta.
        El contexto contiene información sobre transacciones y si fueron fraudes o no.
        El contexto contiene Monto ┆ Ubicación    ┆ Método_Pago   ┆ Hora_Transaccion ┆ Intentos_Fallidos | fraude.         
        usa solo la información del contexto para responder, no uses información externa.
        Respuesta:
        responde siempre si la transacción es fraude o no en menos de 100 palabras con una explicación breve. 
        responde en el siguiente formato en texto plano:
        
       "explicacion": "no mas de 100 palabras"
        omite la palabra text o json y las comillas
        ------------------------"""

),
        contents=f"Contexto: {context}, Pregunta: {question}",
    )
    #print ("RESPONSE", response.text)
    return str(response.text)

def query_ollama_with_context1(context, question):
    #print("CONTEXTO", context)
    #print("FIN CONTEXTO")
    if not context.strip():  # Si el contexto está vacío, evitar respuestas fuera del documento
        return "Esa información no está en nuestra base de datos."

    prompt = f"""
        usa el siguiente contexto para responder la pregunta.
        El contexto contiene información sobre transacciones y si fueron fraudes o no.
        El contexto contiene Monto ┆ Ubicación    ┆ Método_Pago   ┆ Hora_Transaccion ┆ Intentos_Fallidos | fraude.         
        usa solo la información del contexto para responder, no uses información externa.
        Contexto: {context}
        Pregunta: {question}
        Respuesta:
        responde siempre si la transacción es fraude o no en menos de 100 palabras con una explicación breve. 
        responde siempre en el siguiente formato:
        explicacion: "no mas de 100 palabras"
        omite la palabra text o json y las comillas
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
    #print("RESULTS", results)
    #results = vectorstore.similarity_search_with_score(query=question, k=3)
    #filtered_context = [doc.page_content for doc, score in results if score > 0.5]


    return results if results else []
    



# Pipeline RAG con validación estricta de relevancia
def rag_pipeline(question, vectorstore):
    results = retrieve_relevant_chunks(question, vectorstore)
    
    if not results:
        return  print("Esa información no está en el documento cargado, pero basado en la información que tengo la siguiente respuesta te puede ser útil\n")
    
    context = "\n".join([result.page_content for result in results])
    
    response = query_ollama_with_context(context, question)
    return response
g = pl.DataFrame({"type_CASH_IN" :1, "type_CASH_OUT" :0, "type_PAYMENT" :0, "type_TRANSFER" :0, "amount"   :6000.56, "type_2_CC" :1, "type_2_CM" :0,
                   "day" :1, "part_of_the_day_madrugada" :1, "part_of_the_day_mañana" :0, "part_of_the_day_noche" :0, "part_of_the_day_tarde": 0})

# Ejemplo de consulta válida
#53,3162,Buenos Aires,Transferencia,14,2,
#7926,Buenos Aires,Tarjeta Débito,21,3
h = pl.DataFrame({
    "Monto": [5098],
    "Ubicación": ["BuenosAires"],
    "Método_Pago": ["Criptomoneda"],
    "Hora_Transaccion": [9],
    "Intentos_Fallidos": [4],
    
})

question = f"""
la transaccion {h} es fraude ? 
"""
response = rag_pipeline(question, vectorstore)

print(response)