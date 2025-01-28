import bs4
import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import chromadb
import tempfile

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


def load_and_process_data(url):
    # Create a persistent directory for Chroma
    persist_directory = os.path.join(tempfile.gettempdir(), 'chroma_db')

    # Load Documents
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    # Split Documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Initialize Chroma with persistence
    embeddings = OpenAIEmbeddings()

    # Clean up any existing DB
    if os.path.exists(persist_directory):
        import shutil
        shutil.rmtree(persist_directory)

    # Create new vectorstore
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    # Persist the database
    vectorstore.persist()
    return vectorstore.as_retriever()


def generate_answer(retriever, question):
    # Set up prompt and LLM
    template = """Answer the question based only on the following context:
{context}

Question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    chain = prompt | llm

    # Retrieve relevant documents and generate answer
    retrieved_docs = retriever.get_relevant_documents(question)
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    response = chain.invoke(
        {"context": formatted_context, "question": question})
    return response.content


def main():
    st.title("RAG Application with Streamlit")

    # Input fields
    url = st.text_input("Enter website URL:",
                        "https://lilianweng.github.io/posts/2023-06-23-agent/")
    question = st.text_input("Ask a question:", "What is Task Decomposition?")

    # Session state to track if data is loaded
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'is_data_loaded' not in st.session_state:
        st.session_state.is_data_loaded = False

    # Fetch button to load data
    if st.button("Fetch Data"):
        with st.spinner("Loading and processing data..."):
            try:
                st.session_state.retriever = load_and_process_data(url)
                st.session_state.is_data_loaded = True
                st.success("Data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.session_state.is_data_loaded = False

    # Generate answer only if data is loaded
    if st.session_state.is_data_loaded:
        if st.button("Generate Answer"):
            with st.spinner("Generating answer..."):
                try:
                    answer = generate_answer(
                        st.session_state.retriever, question)
                    st.subheader("Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")


if __name__ == "__main__":
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    main()
