from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import WikipediaLoader
from langchain.prompts import PromptTemplate
import re
import os
import colorama
from dotenv import load_dotenv
import pprint
from requests.exceptions import ConnectionError
import time
import random
import uuid


def progress_bar(progress, total, color=colorama.Fore.GREEN):
    '''Progress bar for running time display'''
    percent = 100 * (progress/ float(total))
    #alt + 219
    prog_bar = '█' * int(percent) + '-' * int(100 - percent)
    # \r ensures the same line is used '\repeat'
    print(color + f"\r|{prog_bar}| {percent:.2f}%", end="\r")

    if percent == total:
        print(colorama.Fore.GREEN + f"\r|{prog_bar}| {percent:.2f}%", end="\r")

def chunk_best_of_24_list(page_content):
    '''chunks each line of the albums listed within the webpage'''
    splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n"],      
                chunk_size=50,
                chunk_overlap=5
                ) 

    album_data = splitter.split_documents(page_content)

    return album_data

def load_kexp_album_documents():
    documents = []

    loader = BSHTMLLoader("./Vote for KEXP's Best of 2024.html")
    kexp_best_of_24 = loader.load()

    page_content = kexp_best_of_24[0].page_content

    # extract the artists and their albums
    found_albums = extract_artists(page_content=page_content)
    
    # get articles about the artists
    band_related_articles = wiki_search_bands(album_list=found_albums)

    # get articles about the album
    album_related_articles = wiki_search_albums(album_list=found_albums)

    # supply the html page content to be embedded
    album_data = chunk_best_of_24_list(page_content=kexp_best_of_24)
    documents.extend(album_data)

    # supply the band information to be embedded
    for band_article in band_related_articles:
        documents.extend(band_article)

    # # supply the album information to be embedded
    for album_article in album_related_articles:
        documents.extend(album_article)

    return documents


# determine artists to wiki walk:
def extract_artists(page_content):
    '''determines artists list for wikipedia extraction'''
    artist_album_regex = r'(.* - .*)'

    matches = re.findall(artist_album_regex, page_content)

    # see matches:
    # for match in enumerate(matches):
    #     print(f'Artist - Album: {match}' )

    if matches:
        return matches
    else:
        return []

def get_artist_album(artist_album, artist=False, album=False):
    '''snags either artist or band data depending on the boolean supplied'''
    album_regex = r'(.*) - (.*)'

    match_found = re.search(album_regex, artist_album)

    if match_found:
        groups = match_found.groups()
        if artist:
            return groups[0]
        if album:
            return groups[1]
        
    return -1


def wiki_search_bands(album_list):
    '''queries wikipedia for band details'''
    
    back_off_throttle = 15

    band_related_articles = []
    
    print('Wiki Searching Band Details from 2024 List...')

    if "bands.txt" not in os.listdir():
        searched_bands = 0
        total_albums = len(album_list)-1
        # open the file and create a running list of parsed albums
        with open("bands.txt","a") as bands_file:
            for band in album_list:
                # {something-artist} - {something-album}
                progress_bar(progress=searched_bands, total=total_albums)
                band = get_artist_album(artist_album=band, artist=True)
                band = band.strip()
                # query for album information
                retries = 3
                for attempt in range(retries):
                    try:
                        articles = WikipediaLoader(query=band, load_max_docs=3).load()
                    except ConnectionError as e:
                        if attempt < retries:
                            print("ConnectionError: pulling back on request time")
                            time.sleep(back_off_throttle * (2 ** attempt))
                        else:
                            # long nap.. :(
                            # wait a random time between 100-600 seconds
                            time.sleep(100 * random.randint(1,6))

                searched_bands += 1
                for article in articles:
                    article_title = article.metadata['title']

                    if article_title in band:
                        article = chunk_wiki_content(article=article)
                        band_related_articles.append(article)
                        bands_file.write(f'{band} \n')
                    else:
                        # case where it's an ambiguous name
                        band_regex = r'\(band\)'
                        # search articles for band_regex
                        if re.search(band_regex, article_title):
                            article = chunk_wiki_content(article=article)
                            band_related_articles.append(article)
                            bands_file.write(f'{band}\n')
                        else:
                            continue
            print(colorama.Fore.RESET)
            print('Finished Wiki Searching Bands...')
            # close the file.
            bands_file.close()
            return band_related_articles
    else:
        # need to get the latest album found and skip the line 
        with open("bands.txt","r") as bands:
            read_band_list = bands.readlines()
            bands.close()

        # reopen the file to append more bands.
        with open("bands.txt", "a") as bands_file:
            # get the last artist searched.
            last_band_searched = get_artist_album(artist_album=read_band_list[-1], artist=True)
            last_band = last_band_searched.strip()

            band_restart_idx = 0
            for idx, band in enumerate(album_list):
                # strips for band/artist
                band = get_artist_album(artist_album=band, artist=True)
                band = band.strip()

                if band != last_band:
                    continue
                else:
                    # found your starting point.
                    band_restart_idx = idx
                    break
            
            remaining_bands = album_list[band_restart_idx]
            # start where you left off.
            for band in remaining_bands:
                # {something-artist} - {something-album}
                progress_bar(progress=searched_bands, total=total_albums)
                band = get_artist_album(artist_album=band, artist=True)
                band = band.strip()
                # query for album information
                retries = 3
                for attempt in range(retries):
                    try:
                        articles = WikipediaLoader(query=band, load_max_docs=3).load()
                    except ConnectionError as e:
                        if attempt < retries:
                            print("ConnectionError: pulling back on request time")
                            time.sleep(back_off_throttle * (2 ** attempt))
                        else:
                            # long nap.. :(
                            # wait a random time between 100-600 seconds
                            time.sleep(100 * random.randint(1,6))

                searched_bands += 1
                for article in articles:
                    article_title = article.metadata['title']

                    if article_title in band:
                        article = chunk_wiki_content(article=article)
                        band_related_articles.append(article)
                        bands_file.write(f'{band} \n')
                    else:
                        # case where it's an ambiguous name
                        band_regex = r'\(band\)'
                        # search articles for band_regex
                        if re.search(band_regex, article_title):
                            article = chunk_wiki_content(article=article)
                            band_related_articles.append(article)
                            bands_file.write(f'{band}\n')
                        else:
                            continue
            print(colorama.Fore.RESET)
            print('Finished Wiki Searching Bands...')
            # close the file.
            bands_file.close()
            return band_related_articles


def wiki_search_albums(album_list):
    '''queries wikipedia for album details and band details'''
    
    back_off_throttle = 15

    album_related_articles = []
    print('Wiki Searching Album Details from 2024 List...')
    if "albums.txt" not in os.listdir():
        searched_albums = 0
        total_albums = len(album_list)-1
        # create a albums file list for a running tally
        with open("albums.txt","a") as albums_file:
            for album in album_list:
                # remove white space
                progress_bar(progress=searched_albums, total=total_albums)
                album = get_artist_album(artist_album=album,album=True)
                album = album.strip()
                # query for album information
                retries = 3
                for attempt in range(retries):
                    try: 
                        articles = WikipediaLoader(query=album.strip(), load_max_docs=3).load()
                    except ConnectionError as e:
                        if attempt < retries:
                            print(f"Connection error on attempt: {attempt + 1}: {e}")
                            time.sleep(back_off_throttle * (2 ** attempt))
                        else:
                            # long nap.. :(
                            # wait a random time between 100-600 seconds
                            time.sleep(100 * random.randint(1,6))
                
                searched_albums += 1
                for article in articles:
                    article_title = article.metadata['title']
                    # search articles for band_regex
                    if article_title in album:
                        article = chunk_wiki_content(article=article)
                        album_related_articles.append(article)
                        # add the band to the list
                        albums_file.write(f'{album}\n')
                    else:
                        # case where it's an ambiguous name
                        album_regex = r'\(album\)'
                        # search articles for band_regex
                        if re.search(album_regex, article_title):
                            article = chunk_wiki_content(article=article)
                            album_related_articles.append(article)
                            # add the band to the list
                            albums_file.write(f'{album}\n')
                        else:
                            continue

            albums_file.close()

            print(colorama.Fore.RESET)
            print('Finished Chunking ALbums...')
            return album_related_articles
    else:
        searched_albums = 0
        total_albums = len(album_list)-1
        # assume the file exists and check the list
        with open("albums.txt","r") as albums:
            read_albums_list = albums.readlines()
            albums.close()

        # reopen file to add to it.
        with open("albums.txt", "a") as albums_file:
            # get the last read band
            last_album_searched = get_artist_album(artist_album=read_albums_list[-1], album=True)
            last_album = last_album_searched.strip()

            album_restart_indx = 0
            for idx, album in enumerate(album_list):
                album = get_artist_album(artist_album=album, album=True)
                # remove wite space
                album = album.strip()

                if album != last_album:
                    continue
                else:
                    album_restart_indx = idx
                    searched_albums = idx
                    break

            remaining_albums = album_list[album_restart_indx]
            # iterate through the remaining items
            for album in remaining_albums:
                # remove white space
                progress_bar(progress=searched_albums, total=total_albums)
                album = get_artist_album(artist_album=album,album=True)
                album = album.strip()
                # query for album information
                retries = 3
                for attempt in range(retries):
                    try: 
                        articles = WikipediaLoader(query=album.strip(), load_max_docs=3).load()
                    except ConnectionError as e:
                        if attempt < retries:
                            print(f"Connection error on attempt: {attempt + 1}: {e}")
                            time.sleep(back_off_throttle * (2 ** attempt))
                        else:
                            # long nap.. :(
                            # wait a random time between 100-600 seconds
                            time.sleep(100 * random.randint(1,6))
                
                searched_albums += 1
                for article in articles:
                    article_title = article.metadata['title']
                    # search articles for band_regex
                    if article_title in album:
                        article = chunk_wiki_content(article=article)
                        album_related_articles.append(article)
                        # add the band to the list
                        albums_file.write(album)
                    else:
                        # case where it's an ambiguous name
                        album_regex = r'\(album\)'
                        # search articles for band_regex
                        if re.search(album_regex, article_title):
                            article = chunk_wiki_content(article=article)
                            album_related_articles.append(article)
                            # add the band to the list
                            albums_file.write(album)
                        else:
                            continue

        print(colorama.Fore.RESET)
        print('Finished Chunking ALbums...')
        return album_related_articles


def chunk_wiki_content(article):
    '''takes article text as input and chunks the results for embedding'''

    splitter = RecursiveCharacterTextSplitter(
                    separators=['\n','\n\n', '.', ' '],
                    chunk_size=100, 
                    chunk_overlap=20,
                    length_function=len,
                    is_separator_regex=False
                ) 
    
    # Wrap the article in a list if it's a single Document object
    if not isinstance(article, list):
        article = [article]
    
    # Ensure the article is a Document object
    data = splitter.split_documents(article)

    return data



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
        max_tokens=45, 
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
        documents = load_kexp_album_documents()
        # supply the chunked documents and the embedding model chroma db
        vector_store = Chroma.from_documents(     
                            documents=documents,   
                            ids=[str(uuid.uuid4()) for _ in documents],   
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

    
    
    # # set up chroma to be the retriever for related documents based on the embedding
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
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
        qa_with_source("")
    )

main()