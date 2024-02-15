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



ASTRA_DB_APPLICATION_TOKEN = "AstraCS:ZyXaJYoKMbgwEfStXlxHxnNa:73926f4265893c3b5d21f765b14a960fd87de5267652da6743bd1388bc57578e" # enter the "AstraCS:..." string found in in your Token JSON file
ASTRA_DB_ID = "a156cc4c-7263-4686-aebf-fdfd61428182" # enter your Database ID

OPENAI_API_KEY = "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" # enter your OpenAI key


cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Create the LangChain embedding and LLM objects for later usage:

llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Create your LangChain vector store ... backed by Astra DB!
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo",
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

    # print("FIRST DOCUMENTS BY RELEVANCE:")
    # top 4 matches
    st.write('\nFirst 2 Matching Documents by Relevance')
    for doc, score in astra_vector_store.similarity_search_with_score(user_question, k=2):
        # print("    [%0.4f] \"%s ...\"" % (score, doc.page_content))
        st.write("    [%0.4f] \"%s ...\"" % (score, doc.page_content))
    

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






