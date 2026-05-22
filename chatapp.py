import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "google")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash-exp")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "")

PROVIDER_KEYS = {
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# Validate required API keys
for provider, key_name in [(LLM_PROVIDER, "LLM"), (EMBEDDING_PROVIDER, "Embedding")]:
    if provider in PROVIDER_KEYS:
        if not os.getenv(PROVIDER_KEYS[provider]):
            st.error(f"{PROVIDER_KEYS[provider]} not found for {key_name} provider '{provider}'. "
                     f"Please set it in your .env file.")
            st.stop()

# Google SDK requires explicit init; OpenAI/Anthropic read API key from env automatically
if LLM_PROVIDER == "google" or EMBEDDING_PROVIDER == "google":
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def create_llm():
    if LLM_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {"model": LLM_MODEL, "temperature": LLM_TEMPERATURE}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        return ChatOpenAI(**kwargs)
    elif LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kwargs = {"model": LLM_MODEL, "temperature": LLM_TEMPERATURE}
        if ANTHROPIC_BASE_URL:
            kwargs["base_url"] = ANTHROPIC_BASE_URL
        return ChatAnthropic(**kwargs)
    elif LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE, base_url=OLLAMA_BASE_URL)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. Use: google, openai, anthropic, ollama")


def create_embeddings():
    if EMBEDDING_PROVIDER == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    elif EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        kwargs = {"model": EMBEDDING_MODEL}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        return OpenAIEmbeddings(**kwargs)
    elif EMBEDDING_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER}. Use: google, openai, ollama")


def get_pdf_text(pdf_docs):
    pages = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for i, page in enumerate(pdf_reader.pages):
            extracted = page.extract_text()
            if extracted:
                pages.append({"text": extracted, "source": pdf.name, "page": i + 1})
    return pages


def get_text_chunks(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    all_chunks = []
    for page in pages:
        chunks = text_splitter.split_text(page["text"])
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "metadata": {"source": page["source"], "page": page["page"]},
            })
    return all_chunks


def get_vector_store(chunks):
    embeddings = get_embeddings()
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local("faiss_index")


def get_embeddings():
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = create_embeddings()
    return st.session_state.embeddings


def get_conversational_chain():
    if "chain" not in st.session_state:
        prompt_template = """
        Answer the question as detailed as possible from the provided context.
        If the answer is not in the provided context, just say "answer is not available in the context", don't provide the wrong answer.
        When possible, cite the source document and page number for each piece of information.

        Context:\n {context}\n
        Question: \n{question}\n

        Answer:
        """

        model = create_llm()

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        st.session_state.chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return st.session_state.chain


def user_input(user_question):
    try:
        embeddings = get_embeddings()
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question, k=4)

        chain = get_conversational_chain()

        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True,
        )

        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"Error getting answer: {e}")


def main():
    st.set_page_config("Multi PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF's \U0001f4da - Chat Agent \U0001f916 ")

    if "processed" not in st.session_state:
        st.session_state.processed = False

    with st.sidebar:
        st.image("img/Robot.jpg")
        st.write("---")

        st.title("\U0001f4c1 PDF File's Section")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files & \n Click on the Submit & Process Button ",
            accept_multiple_files=True,
        )
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    try:
                        pages = get_pdf_text(pdf_docs)
                        if not pages:
                            st.error("Could not extract text from the uploaded PDFs.")
                        else:
                            chunks = get_text_chunks(pages)
                            get_vector_store(chunks)
                            st.session_state.processed = True
                            st.success("Done")
                    except Exception as e:
                        st.error(f"Error processing PDFs: {e}")

        st.write("---")

    if st.session_state.processed:
        user_question = st.text_input(
            "Ask a Question from the PDF Files uploaded .. ✍️\U0001f4dd",
            key="user_question_input",
        )
        if user_question:
            user_input(user_question)
    else:
        st.info("Please upload PDF files and click 'Submit & Process' before asking questions.")

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            &copy; <a href="https://github.com/gurpreetkaurjethra" target="_blank">Gurpreet Kaur Jethra</a> | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
