import streamlit as st
import langchain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
import os
import time
import torch
from langchain.storage import InMemoryStore, LocalFileStore, RedisStore, UpstashRedisStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings

#to turn off parallelism errors
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the quantized llm from local dir(only non relative path)
llm = CTransformers(model="/Users/user/Documents/testing_local_gpt/Run_llama2_local_cpu_upload/models/llama-2-7b-chat.ggmlv3.q2_K.bin",
                  model_type="llama",
                  config={'max_new_tokens':140,
                          'temperature':0.01,
                          })

# Create a PromptTemplate for the QA Chain
template = """Use the following pieces of information to answer the user's question.
If you dont know the answer just say you know, don't try to make up an answer.

Context:{context}
Question:{question}

Only return the helpful answer below and nothing else
Helpful answer
"""

qa_prompt = PromptTemplate(template=template, input_variables=['context', 'question'])



# Create a Streamlit app
st.set_page_config(page_title="Retrieval QA App", page_icon="🤖")

# Create a sidebar
sidebar = st.sidebar

#Create an option selectbox
option =  st.selectbox(
    'Do you want to upload file?',
    ('Yes', 'No'))

if option =='Yes':

# Add a file uploader to the sidebar
    theme = st.text_input('Please enter the theme of the document').replace(' ','_').lower()
    pdf_file = sidebar.file_uploader("Upload PDF file:")
    

# do you want to upload a file


# If the user has uploaded a PDF file, process it
    if pdf_file is not None:

        with open(f"tmp/{theme}.pdf","wb") as file:
            file.write(pdf_file.getvalue())
            
        loader = PyPDFLoader(f"tmp/{theme}.pdf")
        text_splitter = RecursiveCharacterTextSplitter(
                                             chunk_size=500,
                                             chunk_overlap=50,
                                             )
        #text_chunks = text_splitter.split_documents([pdf_file])
        # Extract the text from the PDF file
        text = loader.load_and_split()
        # load_and_split(pdf_file)

        # Split the text into chunks
        #
        text_chunks = text_splitter.split_documents(text)

        # Convert the text chunks into embeddings
        #embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'})
        underlying_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'})
        fs = LocalFileStore("./cache/")

        cached_embedder = CacheBackedEmbeddings.from_bytes_store(underlying_embeddings, fs)

        # Create a FAISS vector store from the embeddings
        #vec = FAISS()
        vector_store = FAISS.from_documents(text_chunks, cached_embedder)

        #vector_store.save_local(f"data/{theme}")

        # Create the RetrievalQA Chain
        chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type='stuff',
                                    retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
                                    return_source_documents=True,
                                    chain_type_kwargs={'prompt': qa_prompt})

    # Update the chain retriever
    #chain.retriever = vector_store.as_retriever(search_kwargs={'k': 2})
else:
    themes = [fname.split(".")[0] for fname in  os.listdir('data/')]
    option_n =  st.selectbox(
    'What do you want to ask about?',
    themes)
    # Convert the text chunks into embeddings
    st_1 = time.time()
    underlying_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'})
    fs = LocalFileStore("./cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(underlying_embeddings, fs)

    st_2 = time.time() - st_1
    st.write(st_2)


    st_3 = time.time()
    vector_store = FAISS.from_documents(f"data/{option_n}", cached_embedder)

    #vector_store = FAISS.load_local(f"data/{option_n}", embeddings)
    st_4 = time.time() - st_3
    st.write(f"Done loading vector {st_4} ")
    st_5 = time.time()
    chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type='stuff',
                                    retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
                                    return_source_documents=True,
                                    chain_type_kwargs={'prompt': qa_prompt})
    st_6 = time.time() - st_5
    st.write(f"Done parsing the chain{st_6}")






def generate_response(query):
    """Generates a response to a query using the LangChain chain."""

    # Get the top 3 documents for the query
    docs = vector_store.similarity_search(query)

    # Generate a response based on the retrieved documents
    response = chain({'query': query, 'docs': docs})

    return response['result']

# Allow the user to ask questions
query = sidebar.text_input("Query:")

# If the user has entered a query, generate a response
if query:
    # Display a processing indicator
    with st.spinner("Generating response..."):
        st_7 = time.time()
        response = generate_response(query)
        st_8 = time.time() - st_7


    # Print the response
    st.write(f"Answer:{response}")
    st.write(f"The vector similarity search took  {st_8}")



