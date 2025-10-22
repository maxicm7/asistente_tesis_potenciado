import streamlit as st
import pypdf
import io
import os
import logging

# Configura logging para ver errores en los logs de Streamlit Cloud
logging.basicConfig(level=logging.INFO)

# --- Importaciones seguras ---
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.llms import HuggingFaceHub
    from langchain_community.vectorstores import Chroma
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError as e:
    st.error(f"Error al importar dependencias: {e}")
    st.stop()

# --- Prompt del rol ---
MASTER_PROMPT_ROLE = """
[INICIO DE LA DEFINICIÓN DEL ROL]
**Nombre del Rol:** Investigador Doctoral IA (IDA)
**Objetivo Principal:** Asistir en la investigación y redacción de una tesis doctoral, basando tus respuestas estrictamente en el contexto proporcionado.
**Personalidad:** Eres un asistente de investigación post-doctoral; preciso, metódico, objetivo y riguroso.
**REGLA CRÍTICA:** NO inventes información. Si la respuesta no se encuentra en el contexto proporcionado, declara: "La información solicitada no se encuentra en los documentos proporcionados."
[FIN DE LA DEFINICIÓN DEL ROL]
"""

# --- Procesamiento de PDFs ---
@st.cache_resource
def create_vector_store(pdf_files):
    if not pdf_files:
        return None

    with st.spinner("⚙️ Procesando documentos..."):
        all_text = ""
        for pdf_file in pdf_files:
            try:
                pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_file.getvalue()))
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"
            except Exception as e:
                st.error(f"Error leyendo {pdf_file.name}: {e}")
                return None

        if not all_text.strip():
            st.warning("No se extrajo texto de los PDFs.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(all_text)
        st.info(f"📚 Procesados {len(chunks)} fragmentos.")

        # Usamos Chroma con embedding function directa (más estable)
        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        vector_store = Chroma.from_texts(
            texts=chunks,
            embedding_function=embedding_function
        )
        return vector_store

# --- Cadena RAG ---
def get_llm_chain(vector_store, api_key, model_name, temperature):
    try:
        # Verifica API key
        if not api_key or not api_key.startswith("hf_"):
            st.error("La API Key de Hugging Face es inválida o está vacía.")
            return None

        # Configura el LLM
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

        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"{MASTER_PROMPT_ROLE}\n\nContexto:\n{{context}}"),
            ("human", "{{input}}"),
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain

    except Exception as e:
        st.error(f"⚠️ Error al crear la cadena RAG: {e}")
        logging.exception("Error detallado en get_llm_chain")
        return None

# --- App principal ---
def main():
    st.set_page_config(page_title="Asistente Tesis RAG", page_icon="🎓", layout="wide")
    st.title("🎓 Asistente de Tesis Doctoral (RAG)")

    # Inicializar estado
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuración")
        api_key = st.text_input("Hugging Face API Token", type="password", help="Empieza por 'hf_'")
        if not api_key and "HF_API_KEY" in st.secrets:
            api_key = st.secrets["HF_API_KEY"]

        model = st.selectbox("Modelo", [
            "google/gemma-1.1-7b-it",
            "Qwen/Qwen2-7B-Instruct",
            "meta-llama/Meta-Llama-3-8B-Instruct"
        ])
        temp = st.slider("Temperatura", 0.0, 1.0, 0.3)

        st.divider()
        st.header("📄 Documentos")
        st.caption("ℹ️ Solo PDFs con texto (no escaneados)")
        uploaded = st.file_uploader("Sube PDFs", type=["pdf"], accept_multiple_files=True)

        if st.button("🔄 Procesar", use_container_width=True):
            if uploaded and api_key:
                vs = create_vector_store(uploaded)
                if vs:
                    chain = get_llm_chain(vs, api_key, model, temp)
                    if chain:
                        st.session_state.rag_chain = chain
                        st.session_state.messages = []
                        st.success("✅ Listo para usar")
            else:
                st.warning("Sube PDFs y configura la API Key")

        if st.button("🗑️ Limpiar chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Chat
    if not st.session_state.rag_chain:
        st.info("👈 Sube documentos y configura tu API Key para comenzar.")
        return

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    if prompt := st.chat_input("Pregunta sobre tus documentos..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Analizando..."):
                    response = st.session_state.rag_chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"❌ Error: {e}")
                logging.exception("Error al generar respuesta")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Error crítico en la app: {e}")
        logging.exception("Error crítico")
