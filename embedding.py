# scripts to begin analysis


import matplotlib.pyplot as plt 
import sklearn as sk 
import statsmodels as sm
# Here I'm going to create class objects from what I'm importing to langchain
import os
import pandas as pd
import numpy as np
# import chromadb
import uuid
import langchain
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import Pinecone, Chroma, FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
# from chromadb.config import Settings
from dotenv import load_dotenv
import pinecone
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

class AILLM:

    # we need to make lambda calls as the dictionary values are functions. 
    EMBEDDINGS = {
                'OpenAIEmbeddings': lambda: OpenAIEmbeddings(),
                'HuggingFaceInstructEmbeddings': lambda model_name: HuggingFaceInstructEmbeddings(model_name= model_name),
                "SentenceTransformerEmbeddings": lambda model_name: SentenceTransformerEmbeddings(model_name = model_name)}
    
    VECTORSTORES = {"Pinecone": lambda texts, embedding, index_name: Pinecone.from_documents(texts, embedding, index_name),
                    "ChromaDB": lambda texts, embedding: Chroma.from_documents(texts, embedding),
                    "FAISS": lambda texts, embedding: FAISS.from_texts(texts, embedding)}


    CHAT_MODELS = {"ChatOpenAI": lambda: ChatOpenAI()}
    LLMS = {"HuggingFaceHub": lambda: HuggingFaceHub()}


    def __init__(self, openai_api_key:str = None, pinecone_api_key:str = None, huggingfacehub_api_token:str = None):

        """
        This is the class object to run the pdf reader
        
        To add chromaDB Functionality Q1 2024

        """
        self.OPENAI_API_KEY = openai_api_key if openai_api_key is not None else os.getenv('OPENAI_API_KEY')
        self.PINECONE_API_KEY = pinecone_api_key if pinecone_api_key is not None else os.getenv('PINECONE_API_KEY')
        self.HUGGINGFACEHUB_API_TOKEN = huggingfacehub_api_token if huggingfacehub_api_token is not None else os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.index_name = None

        if self.PINECONE_API_KEY:
            self.setup_pinecone()
    
    def setup_pinecone(self, index_name:str = 'Langchain_Index,', dimension:int = 1536, metric:str='cosine'):
        
        """
        Sets up Pinecone environment:

        I set three default parameters that can be changed.

        """
        #set index name value
        self.index_name = index_name
        pinecone.init(api_key = self.PINECONE_API_KEY)
        #Check if the index already exists. If not, create it. 
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                                    name=self.index_name,
                                    metric= metric,
                                    dimension = dimension
                                )

    def get_pdf_text(self,pdf_docs):
        
        """
        This reads the pdf file.
        """
        
        pdf_text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                #this keeps concatonating the text
                pdf_text += page.extract_text()
        return pdf_text
    
    def get_text_chunks(self, text, recursive: bool = True,
                        chunk_size: int = 1000, chunk_overlap: int = 200, length_function=len):
        
        """
        This gets the chunks for the embeddings
        Users can alter the chunk size and overlap
        
        """
        if recursive:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap, 
                separators=[" ", ",", "\n"]
            )
        else:
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=length_function
            )
        chunks = text_splitter.split_text(text)
        return chunks
    
    def get_vectorstore(self, text_chunks, embedding: str = "OpenAIEmbeddings", vectorstore:str = "FAISS"):
        
        """
        you need to feed the text chunks.
        Default params are OpenAI's embeddings and FAISS VectorStore
        
        Need a name
        
        """
        
        embeddings = AILLM.EMBEDDINGS[embedding]()
        if vectorstore == 'Pinecone':
            vectorstore = AILLM.VECTORSTORES[vectorstore](docs=text_chunks, embeddings=embedding, name=self.index_name)
        
        elif vectorstore == "ChromaDB":
            vectorstore = AILLM.VECTORSTORES[vectorstore](docs=text_chunks, embedding_function=embeddings)
        else:
            vectorstore = AILLM.VECTORSTORES[vectorstore](texts=text_chunks, embedding=embeddings)
        
        return vectorstore
    
    def get_conversation_chain(self, vectorstore, memory_key:str = "chat_history", chat:str = "ChatOpenAI"):
        
        """
        YOu need to feed the vectorstore
        params given are memory_key and OpenAI's ChatOpenAI for the llm version. 
        This most likely will not be altered however LLAMA  or a hugging face model could be used for 
        other projects. 
        https://python.langchain.com/docs/use_cases/question_answering/how_to/chat_vector_db
        
        
        """
        
        llm = AILLM.CHAT_MODELS[chat]()
        memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
                                                                    llm=llm, 
                                                                    retriever=vectorstore.as_retriever(), 
                                                                    memory=memory
                                                                   )
        return conversation_chain
    
    def run():
        print("This file is backend class object for the pdfReader.")
