# --- app.py (Versión con RAG y LangChain) ---

import streamlit as st
from huggingface_hub import InferenceClient
import pypdf
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub # Para integrar el LLM en la cadena

# --- 1. Definición del Rol y Configuración Inicial ---
# Mantenemos la definición del rol para guiar al LLM
MASTER_PROMPT_ROLE = """
[INICIO DE LA DEFINICIÓN DEL ROL]
**Nombre del Rol:** Investigador Doctoral IA (IDA)
**Objetivo Principal:** Asistir en la investigación y redacción de una tesis doctoral, basando tus respuestas estrictamente en el contexto proporcionado.
**Personalidad:** Eres un asistente de investigación post-doctoral; preciso, metódico, objetivo y riguroso. Siempre que sea posible, cita la fuente de tu información dentro del contexto.
**REGLA CRÍTICA:** NO inventes información. Si la respuesta no se encuentra en el contexto proporcionado, declara explícitamente: "La información solicitada no se encuentra en los documentos proporcionados."
[FIN DE LA DEFINICIÓN DEL ROL]
"""

# --- 2. Funciones del Pipeline RAG ---

# Usamos el cache de Streamlit para no tener que procesar los PDFs cada vez que el usuario interactúa.
# El decorador cachea el resultado de la función. Si los archivos PDF no cambian, se reutiliza el resultado.
@st.cache_resource
def create_vector_store(pdf_files):
    """
    Toma una lista de archivos PDF, extrae el texto, lo divide en trozos,
    genera embeddings y crea una base de datos vectorial (vector store).
    """
    if not pdf_files:
        return None
    
    with st.spinner("Procesando documentos... Este proceso puede tardar un poco."):
        # a. Extraer texto de todos los PDFs
        all_text = ""
        for pdf_file in pdf_files:
            try:
                pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_file.getvalue()))
                text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
                all_text += text + "\n\n"
            except Exception as e:
                st.error(f"Error leyendo el archivo {pdf_file.name}: {e}")
        
        if not all_text:
            st.warning("No se pudo extraer texto de los PDFs.")
            return None

        # b. Dividir el texto en trozos (chunks) manejables
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Tamaño de cada trozo de texto
            chunk_overlap=200, # Solapamiento para no perder contexto entre trozos
            length_function=len
        )
        chunks = text_splitter.split_text(all_text)
        
        st.info(f"Se han dividido los documentos en {len(chunks)} trozos de texto.")

        # c. Crear embeddings (convertir texto a vectores)
        # Usamos un modelo de embeddings eficiente y popular. Se descargará la primera vez.
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # d. Crear la base de datos vectorial (Vector Store) con FAISS
        # Esta es la base de conocimiento que usaremos para buscar información relevante.
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
        
        return vector_store

# --- Función para llamar a la API de Hugging Face (Modificada para LangChain) ---
# En lugar de usarla directamente, la integramos en el ecosistema de LangChain
def get_hf_llm(api_key, model, temperature):
    """Crea una instancia del LLM de Hugging Face para usar con LangChain."""
    if not api_key or not api_key.startswith("hf_"):
        return None
    try:
        llm = HuggingFaceHub(
            repo_id=model,
            huggingfacehub_api_token=api_key,
            model_kwargs={"temperature": temperature, "max_new_tokens": 4096}
        )
        return llm
    except Exception as e:
        st.error(f"Error al inicializar el modelo de Hugging Face: {e}")
        return None

# --- 3. Interfaz de Streamlit ---
st.set_page_config(layout="wide", page_title="Asistente de Tesis Doctoral IA (con RAG)")

# --- Estado de la sesión ---
# Usamos st.session_state para mantener la conversación y la base de datos vectorial.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# --- UI Principal ---
st.title("🎓 Asistente de Tesis Doctoral IA (con RAG)")
st.markdown("""
**Bienvenido a tu asistente de investigación potenciado por RAG.**
1.  **Configura** tu API Key y el modelo en la barra lateral.
2.  **Sube** uno o más artículos en PDF que formen tu base de conocimiento.
3.  **Chatea** con tus documentos: haz preguntas, pide resúmenes, o solicita la generación de código basado en ellos.
""")

# --- Configuración en la barra lateral ---
with st.sidebar:
    st.header("Configuración")
    api_key_value = st.secrets.get("HF_API_KEY", "")
    hf_api_key_input = st.text_input(
        "Hugging Face API Key", 
        type="password", 
        value=api_key_value
    )
    
    st.subheader("Parámetros del Modelo")
    model_reasoning = st.selectbox(
        "Selección de Modelo",
        # Qwen2 es una excelente opción. He añadido otros por si los prefieres.
        ["mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-8B-Instruct", "Qwen/Qwen2-7B-Instruct", "google/gemma-7b-it"]
    )
    temp_slider = st.slider(
        "Temperatura",
        min_value=0.1, max_value=1.0, value=0.3, step=0.1,
        help="Valores bajos = respuestas más factuales y predecibles. Valores altos = más creativas."
    )

    st.subheader("Base de Conocimiento (PDFs)")
    uploaded_files = st.file_uploader(
        "Sube tus archivos PDF aquí", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Procesar Documentos"):
            vector_store = create_vector_store(uploaded_files)
            if vector_store:
                # Convertimos el vector_store en un "retriever", que es el componente que busca los documentos.
                # 'k=5' significa que buscará los 5 trozos más relevantes.
                st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                st.success(f"¡Documentos procesados! Ya puedes chatear con ellos.")
                
                # Crear la cadena de RAG una vez que el retriever esté listo
                llm = get_hf_llm(hf_api_key_input, model_reasoning, temp_slider)
                if llm:
                    # Este es el prompt que "aumentaremos" con el contexto recuperado
                    prompt_template = ChatPromptTemplate.from_template(
                        f"{MASTER_PROMPT_ROLE}\n\n"
                        "**Contexto recuperado de los documentos:**\n"
                        "---------------------\n"
                        "{context}\n"
                        "---------------------\n\n"
                        "**Pregunta del usuario:** {input}\n\n"
                        "**Respuesta del Asistente (basada SÓLO en el contexto):**"
                    )
                    
                    # Creamos la cadena que combina los documentos recuperados en el prompt
                    document_chain = create_stuff_documents_chain(llm, prompt_template)
                    
                    # Creamos la cadena principal que primero recupera y luego genera
                    st.session_state.rag_chain = create_retrieval_chain(st.session_state.retriever, document_chain)
                else:
                    st.error("No se pudo inicializar el modelo de lenguaje. Verifica tu API Key.")
            else:
                st.error("No se pudo crear la base de conocimiento a partir de los PDFs.")

# --- 4. Interfaz de Chat ---

st.header("Chat con tus Documentos")

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input del usuario
if prompt := st.chat_input("Haz una pregunta sobre tus documentos..."):
    # Añadir y mostrar el mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Procesar la respuesta si la cadena RAG está lista
    if st.session_state.rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("Buscando en tus documentos y generando respuesta..."):
                try:
                    # ¡Aquí ocurre la magia de RAG!
                    response = st.session_state.rag_chain.invoke({"input": prompt})
                    
                    answer = response.get("answer", "No se pudo generar una respuesta.")
                    st.markdown(answer)
                    
                    # Opcional: Mostrar las fuentes (los trozos de texto usados para responder)
                    with st.expander("Ver fuentes utilizadas"):
                        for i, doc in enumerate(response["context"]):
                            st.write(f"**Fuente {i+1} (página aprox. basada en contenido):**")
                            st.info(doc.page_content)

                    # Añadir la respuesta del asistente al historial
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"Ocurrió un error al generar la respuesta.")
                    st.exception(e)

    else:
        st.warning("Por favor, sube y procesa tus documentos en la barra lateral antes de chatear.")
