import streamlit as st
import pypdf
import io
import os

# --- Importaciones actualizadas para LangChain v0.2+ ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuración General ---
MASTER_PROMPT_ROLE = """
[INICIO DE LA DEFINICIÓN DEL ROL]
**Nombre del Rol:** Investigador Doctoral IA (IDA)
**Objetivo Principal:** Asistir en la investigación y redacción de una tesis doctoral, basando tus respuestas estrictamente en el contexto proporcionado.
**Personalidad:** Eres un asistente de investigación post-doctoral; preciso, metódico, objetivo y riguroso. Siempre que sea posible, cita la fuente de tu información dentro del contexto.
**REGLA CRÍTICA:** NO inventes información. Si la respuesta no se encuentra en el contexto proporcionado, declara explícitamente: "La información solicitada no se encuentra en los documentos proporcionados."
[FIN DE LA DEFINICIÓN DEL ROL]
"""

# --- Funciones del Núcleo (Backend) ---

@st.cache_resource
def create_vector_store(pdf_files):
    """
    Procesa los PDFs subidos: extrae texto, divide en chunks, genera embeddings y crea el índice FAISS.
    Se cachea para no repetir el proceso si los archivos no cambian.
    """
    if not pdf_files:
        return None
    
    with st.spinner("⚙️ Procesando documentos... (Esto puede tardar un poco la primera vez)"):
        # 1. Extraer texto de todos los PDFs
        all_text = ""
        for pdf_file in pdf_files:
            try:
                pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_file.getvalue()))
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"
            except Exception as e:
                st.error(f"Error leyendo el archivo {pdf_file.name}: {e}")
                return None
        
        if not all_text:
            st.warning("No se pudo extraer texto utilizable de los PDFs.")
            return None

        # 2. Dividir el texto en trozos manejables (Chunks)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,    # Tamaño del chunk en caracteres
            chunk_overlap=200,  # Solapamiento para mantener contexto
            length_function=len
        )
        chunks = text_splitter.split_text(all_text)
        st.info(f"📚 Documentos procesados en {len(chunks)} fragmentos de información.")

        # 3. Generar Embeddings y Vector Store
        # Usamos un modelo ligero y rápido para CPU
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
        
        return vector_store

def get_llm_chain(vector_store, api_key, model_name, temperature):
    """
    Configura y devuelve la cadena RAG completa lista para usarse (compatible con LangChain >=0.2).
    """
    try:
        # 1. Configurar el LLM (Modelo de Lenguaje)
        llm = HuggingFaceHub(
            repo_id=model_name,
            huggingfacehub_api_token=api_key,
            model_kwargs={
                "temperature": temperature,
                "max_new_tokens": 1024,
                "repetition_penalty": 1.1,
                "return_full_text": False
            }
        )

        # 2. Configurar el Retriever
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # 3. Crear el Prompt Template
        system_prompt = (
            f"{MASTER_PROMPT_ROLE}\n\n"
            "Utiliza los siguientes fragmentos de contexto recuperado para responder a la pregunta.\n"
            "Si no sabes la respuesta basándote en el contexto, dilo explícitamente. No inventes.\n\n"
            "Contexto:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # 4. Función para formatear los documentos recuperados
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 5. Construir la cadena RAG manualmente (nueva API)
        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain

    except Exception as e:
        st.error(f"Error al configurar la cadena RAG: {e}")
        return None

# --- Función Principal (Frontend) ---

def main():
    st.set_page_config(page_title="Asistente Tesis RAG", page_icon="🎓", layout="wide")
    st.title("🎓 Asistente de Tesis Doctoral (RAG)")

    # --- Inicialización del Estado de Sesión ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    # --- Barra Lateral de Configuración ---
    with st.sidebar:
        st.header("🔧 Configuración")
        
        # API Key
        api_key = st.text_input("Hugging Face API Token", type="password", help="Empieza por 'hf_'")
        if not api_key and "HF_API_KEY" in st.secrets:
             api_key = st.secrets["HF_API_KEY"]

        # Selección de Modelo
        model_options = [
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "Qwen/Qwen2-7B-Instruct",
            "google/gemma-1.1-7b-it"
        ]
        selected_model = st.selectbox("Modelo LLM", model_options, index=0)
        temperature = st.slider("Temperatura (Creatividad)", 0.0, 1.0, 0.3, 0.1)

        st.divider()
        st.header("📄 Base de Conocimiento")
        uploaded_files = st.file_uploader("Sube tus PDFs", type=["pdf"], accept_multiple_files=True)
        
        # Botón para procesar
        if st.button("🔄 Procesar Documentos", use_container_width=True):
            if uploaded_files and api_key:
                st.session_state.vector_store = create_vector_store(uploaded_files)
                if st.session_state.vector_store:
                    st.session_state.rag_chain = get_llm_chain(
                        st.session_state.vector_store, api_key, selected_model, temperature
                    )
                    if st.session_state.rag_chain:
                        st.success("✅ ¡Sistema listo para chatear!")
            elif not api_key:
                 st.warning("⚠️ Necesitas una API Key de Hugging Face.")
            else:
                 st.warning("⚠️ Por favor, sube al menos un PDF.")

        # Botón para limpiar historial
        if st.button("🗑️ Limpiar Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # --- Área Principal de Chat ---
    
    # Si no hay sistema RAG listo, mostrar bienvenida
    if not st.session_state.rag_chain:
        st.info("👈 Configura tu API Key y sube tus documentos en la barra lateral para comenzar.")
        return

    # Mostrar historial de mensajes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input del usuario
    if prompt := st.chat_input("Haz una pregunta sobre tus documentos..."):
        # Añadir mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analizando documentos..."):
                try:
                    response = st.session_state.rag_chain.invoke({"input": prompt})
                    answer = response['answer']
                    
                    # Limpieza básica si el modelo devuelve el prompt por error (pasa a veces con HF Hub)
                    if "System:" in answer or "Human:" in answer:
                         answer = answer.split("Respuesta del Asistente:")[-1].strip()

                    st.markdown(answer)
                    
                    # Opcional: Mostrar fuentes en un desplegable
                    with st.expander("🔍 Ver fuentes consultadas"):
                        for i, doc in enumerate(response['context']):
                            st.markdown(f"**Fuente {i+1}**")
                            st.caption(doc.page_content[:500] + "...") # Muestra solo los primeros 500 caracteres

                    # Guardar respuesta en historial
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"Error generando la respuesta: {e}")

# --- Punto de Entrada ---
if __name__ == "__main__":
    main()
