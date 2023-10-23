import os
import dotenv
from ai_pdf_reader.pdf_reader import PDFReaderGUI

if __name__ == '__main__':
    # Load environment variables
    dotenv.load_dotenv()

    # Initiate reader
    reader = PDFReaderGUI(
        openai_api_key=os.environ['OPENAI_API_KEY'],
        pinecone_api_key=os.environ['PINECONE_API_KEY'],
        huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN']
    )

    reader.run()
