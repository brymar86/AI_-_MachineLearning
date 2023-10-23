import os
import dotenv
from embedding import AILLM

if __name__ == '__main__':
    # Load environment variables
    dotenv.load_dotenv()
    
    ai = AILLM(
        openai_api_key=os.environ['OPENAI_API_KEY'],
        pinecone_api_key=os.environ['PINECONE_API_KEY'],
        huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN']
    )
    # test Langchain Functionality
    text_chunks = ai.get_text_chunks("""
    In the experimental sciences (physics, chemistry, biology, etc.), it is relatively straightforward to
    propose and falsify causal mechanisms through interventional studies (Fisher [1971]). This is not
    generally the case in financial economics. Researchers cannot reproduce the financial conditions
    of the Flash Crash of May 6, 2010, remove some traders, and observe whether stock market prices
    still collapse. This has placed the field of financial economics at a disadvantage when compared
    with experimental sciences. A direct consequence of this limitation is that, for the past 50 years,
    factor investing researchers have focused on publishing associational claims, without theorizing
    and subjecting to falsification the causal mechanisms responsible for the observed associations. In
    absence of a falsifiable theory, the scientific method calls for a skeptical stance, and the
    presumption that observed associations are spurious. The implication is that the factor investing
    literature remains in a “pre-scientific,” associational (phenomenological) stage.
    """)
    # test vectorstor
    vectorstore = ai.get_vectorstore(text_chunks)
    
    print(vectorstore)
