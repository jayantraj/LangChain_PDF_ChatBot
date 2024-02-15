# LangChain components to use
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter

# With CassIO, the engine powering the Astra DB integration in LangChain,
# you will also initialize the DB connection:
import cassio
from PyPDF2 import PdfReader
import streamlit as st


ASTRA_DB_APPLICATION_TOKEN = st.secrets.ASTRA_DB_APPLICATION_TOKEN
ASTRA_DB_ID = st.secrets.ASTRA_DB_ID

OPENAI_API_KEY = st.secrets.OPENAI_API_KEY 




cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Create the LangChain embedding and LLM objects for later usage:

llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Create your LangChain vector store ... backed by Astra DB!
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="Pdf_QA",
    session=None,
    keyspace=None,
)

def get_pdf_text(pdf_documents):

    # total_text will have text from all the pdfs
    total_text = ''
    for pdf in pdf_documents:
        reader = PdfReader(pdf)
        for page in reader.pages:
            total_text+=page.extract_text()

    return total_text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    return chunks

# convert the chunks to vectors

def get_vector_store(text_chunks):
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    astra_vector_store.add_texts(text_chunks)
    
    

def user_input(user_question):
    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
    answer = astra_vector_index.query(user_question, llm=llm).strip()
    # print("ANSWER: \"%s\"\n" % answer)
    st.write('Answer: \n',answer)
    st.session_state['chat_history'].append(("Bot", answer))
    

def main():
    st.header("Chat with your PDF")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    with st.sidebar:
        st.title("PDFs Here:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit Button", accept_multiple_files=True)

        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    user_question = st.text_input("Ask a Question from your PDFs")

    if user_question:
        st.session_state['chat_history'].append(("You", user_question))
        user_input(user_question)

st.set_page_config(page_title="PDF CHAT BOT")

main()
st.subheader("Chat History :")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")






