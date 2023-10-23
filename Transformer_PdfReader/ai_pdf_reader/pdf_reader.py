import streamlit as st
from dotenv import load_dotenv
import sys

path = "/home/user/Documents/GitKraken_pulls/Transformers_MachineLearning_Public/"
sys.path.append(f'{path}')
print(sys.executable)
from .templates import css, bot_template, user_template
from embedding import AILLM

class PDFReaderGUI(AILLM):
    
    def __init__(self, user_query=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_query = user_query

    def handle_userinput(self, user_question):
        """
        Handles user questions in the GUI
        """
        if st.session_state.conversation:
            response = st.session_state.conversation({'question': user_question}) 
            st.session_state.chat_history = response['chat_history']
            # reverse the conversation so newest messages appear on top
            reversed_chat_history = list(reversed(st.session_state.chat_history))
            # then switch the order of each Human/AI message so we get question on the top
            for ii in range(1, len(reversed_chat_history), 2):
                reversed_chat_history[ii - 1], reversed_chat_history[ii] = reversed_chat_history[ii], reversed_chat_history[ii - 1]

            for message in reversed_chat_history:
                # message types are human messages and AI messages
                if 'Human' in str(type(message)):
                    st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            print("No Conversation")
            
    def run(self):
        """
        This is the GUI for our pdf reader using.
        Run 'CTRL + C' to turn it off.
        """

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
            self.handle_userinput(user_question)

        st.write(user_template.replace("{{MSG}}", "BlackSheep Quant"), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", "I'm Here to Answer Your Questions Blacksheep. How can I help?"), unsafe_allow_html=True)

        with st.sidebar:
            st.subheader("Your Documents")
            pdf_docs = st.file_uploader("Upload your pdfs here & click 'Process'", accept_multiple_files=True)
            if st.button("Process"):
                with st.spinner("Processing"):
                    raw_text = self.get_pdf_text(pdf_docs)
                    text_chunks = self.get_text_chunks(raw_text)
                    vectorstore = self.get_vectorstore(text_chunks)
                    st.session_state.conversation = self.get_conversation_chain(vectorstore)
