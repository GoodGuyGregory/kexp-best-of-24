from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import pprint
from music_parser_service import MusicParserService
import uuid


def main():

    # loads the .env file with LLM API Keys
    load_dotenv()

    # sets API access keys
    OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

    if not OPEN_AI_API_KEY:
        raise ValueError("OPEN_AI_API_KEY not found, please add your API key in a .env")
    
    # LLM:
    #====================================
    open_ai_gpt_4o_mini = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=200, 
        api_key=OPEN_AI_API_KEY,
        timeout=30
    )


    # mistral embedding model
    embedding_model = OpenAIEmbeddings(
                            api_key=OPEN_AI_API_KEY,
                            model="text-embedding-3-small" 
                        )
    
    # Chroma Vector DB
    # ======================

    if "chroma_db" not in os.listdir():
        music_parser = MusicParserService()
        music_documents = music_parser.load_kexp_album_documents()
        # supply the chunked documents and the embedding model chroma db
        vector_store = Chroma.from_documents(     
                            documents=music_documents,   
                            ids=[str(uuid.uuid4()) for _ in music_documents],   
                            embedding=embedding_model,
                            collection_name="KEXP-24-Embeddings", 
                            persist_directory='chroma_db',
                            )
    
    else:
        # load the existing vector db with lang_chain.Chroma
        vector_store = Chroma(
            embedding_function=embedding_model,
            persist_directory='chroma_db',
            collection_name="KEXP-24-Embeddings"
        )

        # there might have been errors from the previous kexp parse
        music_parser = MusicParserService()
        music_documents = music_parser.load_kexp_album_documents()

        if len(music_documents) != 0:            
            vector_store.add_documents(documents=music_documents, ids=[str(uuid.uuid4()) for _ in music_documents])
    
    # # set up chroma to be the retriever for related documents based on the embedding
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )


    template = """
            Use the following pieces of context to answer the question about the musical groups, answer questions about albums and releases.
            include dates, and genres as well as record labels if the information is available to best answer the questions.
        
            
            {context}

            Question: {question}
        """
    
    prompt = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    qa_with_source = RetrievalQA.from_chain_type(
        llm=open_ai_gpt_4o_mini,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt, },
        return_source_documents=True,
    )


    pprint.pprint(
        qa_with_source("Tell me about Beak> in 2024.")
    )

main()