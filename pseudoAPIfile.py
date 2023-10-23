# Fake apikey file

import pandas as pd
import numpy as np
import json
import requests



def open_api():

    """
    This script returns open_api key.

    """
    open_api ='YourOpenAPIKEY'

    return open_api


def pinecone_api():
    
    """
    This script returns pincone_api key. 

    """
    pinecone = "YourPineconeAPIKEY"


    return pinecone

def ChromaDb():
    
    """
    This script returns pincone_api key. 

    """
    pinecone = "YourChromaDB_APIKey"


    return pinecone



if __name__ == '__main__':
    OpenAPI = open_api()
    print(f"Open API Key: {OpenAPI}")
    
    pincone = pinecone_api()
    print(f"Pincone api: {pincone}.")

    Chroma = ChromaDb()
    print(f"ChromaDb api: {Chroma}")

