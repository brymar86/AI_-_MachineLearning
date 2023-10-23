import os
import api_keys
import streamlit as st 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
# this will be erased when we close the app==>eventually move to Pinecone. 
from langchain.vectorstores import FAISS, Chroma, Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
#Ricardo CSS
from htmlTemplates import css,bot_template, user_template


#

# os.environ['OPENAI_API_KEY'] = api_keys.open_api()

def get_pdf_text(pdf_docs):

    """
    This is going to read the pdf we feed the GUI
    Returns a string of texts

    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):

    # text_splitter = CharacterTextSplitter(
    #                                     separator="\n", 
    #                                     chunk_size=1000,
    #                                     chunk_overlap = 200,
    #                                     length_function = len,
    #                                     )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0, 
        separators=[" ", ",", "\n"]
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):

    embeddings = OpenAIEmbeddings()
    #this is where we move to Pinecone 
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """
    Add memory to your conversations.
    I think this would be for the given instance
    We need to put the memory for longerterm memory in another fuction
    """
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(model_name = "hkunlp/instructor-xl" )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
        )
    return conversation_chain

def handle_userinput(user_question):
    
    """
    Handles user questions in the GUI

    """
  
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        # st.write(response)
        st.session_state.chat_history = response['chat_history']
        
        # Reverse the conversation so newest messages appear on top
        reversed_chat_history = list(reversed(st.session_state.chat_history))
        
        # Then switch the order of each Human/AI message so we get questions on top of answers
        for i in range(1, len(reversed_chat_history), 2):
            reversed_chat_history[i-1], reversed_chat_history[i] = reversed_chat_history[i], reversed_chat_history[i-1]
        
        for message in reversed_chat_history:
            # Message types are HumanMessage and AIMessage
            if 'Human' in str(type(message)):
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else: 
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        print("No conversation")

def main():
    """
    This is the GUI for our pdf reader using.

    Run 'CTRL + C' to turn it off.

    """ 
    load_dotenv()

    st.set_page_config(
                        page_title="ChatFusion: Meld Conversations with PDFs",
                        page_icon="😎"
                        )
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None 
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    st.header("Merge, Chat, Conquer! :books:")
    user_question = st.text_input("Ask a question you'd like to know from the documents")
    if user_question:
        handle_userinput(user_question)

    # put user template here
    st.write(user_template.replace("{{MSG}}", "Bryan is Right"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Ricardo is Wrong"), unsafe_allow_html=True)

    # this is the line of code that lets users upload their pdf documents
    with st.sidebar:
        #DO NO ADD PARENTHESE'S ABOVE
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
                                    "Upload your pdfs here & click 'Process'", accept_multiple_files=True
                                    )
        if st.button("Process"):
            #spinner
            with st.spinner("Processing"):
                #get pdf text: 
                raw_text = get_pdf_text(pdf_docs)
                #get the text chunks
                text_chunks = get_text_chunks(raw_text)
                #create vectorstoring/vectorstore with the embeddings (We use OPENAI's)
                vectorstore = get_vectorstore(text_chunks)
                # Create conversation chain (takes history of conversation and returns new element of conversation)
                #In Streamlit, Streamlit may reload its entire code. We must make sure the app doesn't reload this session
                st.session_state.conversation = get_conversation_chain(vectorstore)
                #here is a method for Ricardo to improve.

#This can be available outside of the main code if we need it. 
# st.session_state.conversation


if __name__ =="__main__":

    main()