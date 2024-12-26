from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import WikipediaLoader
import re
import os
import colorama
from dotenv import load_dotenv
from requests.exceptions import ConnectionError
from requests.exceptions import ReadTimeout
from file_empty import FileEmpty
import time
import random

class MusicParserService():
    def __init__(self):
        # default timeout parameter
        self.wiki_timeout = False    

    def progress_bar(self, progress, total, color=colorama.Fore.GREEN):
        '''Progress bar for running time display'''
        percent = 100 * (progress/ float(total))
        #alt + 219
        prog_bar = '█' * int(percent) + '-' * int(100 - percent)
        # \r ensures the same line is used '\repeat'
        print(color + f"\r|{prog_bar}| {percent:.2f}%", end="\r")

        if percent == total:
            print(colorama.Fore.GREEN + f"\r|{prog_bar}| {percent:.2f}%", end="\r")

    def chunk_best_of_24_list(self, page_content):
        '''chunks each line of the albums listed within the webpage'''
        splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n"],      
                    chunk_size=50,
                    chunk_overlap=5
                    ) 

        album_data = splitter.split_documents(page_content)

        return album_data

    def load_BSHTML(self):
        loader = BSHTMLLoader("./Vote for KEXP's Best of 2024.html")
        kexp_best_of_24 = loader.load()

        return kexp_best_of_24

    def extract_albums_from_kexp(self):
        kexp_best_of_24 = self.load_BSHTML()

        page_content = kexp_best_of_24[0].page_content

        # extract the artists and their albums
        found_albums = self.extract_artists(page_content=page_content)

        if "kexp_albums_24.txt" not in os.listdir():
            with open("kexp_albums_24.txt", "a") as kexp_albums:
                # write the content of the found_albums to a file.
                for album in found_albums:
                    kexp_albums.write(f'{album}\n')
                print("written to kexp_albums_24.txt")
        
        # open a slice for scraping.
        with open("kexp_albums_24.txt", "r") as kexp_albums:
            # take the first slice of albums 
            albums = kexp_albums.readlines()
            slice_index = len(albums) // 4
            if slice_index < 100:
                # just take the remaining albums
                sliced_albums = albums
            else:
                # slice the albums needed for scraping
                sliced_albums = albums[:slice_index]

        # remove that slice from the file.
        with open("kexp_albums_24.txt", "w") as kexp_albums:
            # iterate through the albums
            for album in albums:
                if album not in sliced_albums:
                    kexp_albums.write(f'{album}')
        
        # return a quarter of the albums remaining
        return sliced_albums

    def load_kexp_album_documents(self):
        documents = []

        found_albums = self.extract_albums_from_kexp()
        
        # get articles about the artists
        band_related_articles = self.wiki_search_bands(album_list=found_albums)

        if self.wiki_timeout != True:
            # get articles about the album
            album_related_articles = self.wiki_search_albums(album_list=found_albums)
        else:
            print("Wiki-Timeout: await a few hours, and then run the list again.")

        kexp_best_of_24 = self.load_BSHTML()

        # supply the html page content to be embedded
        album_data = self.chunk_best_of_24_list(page_content=kexp_best_of_24)
        documents.extend(album_data)

        # supply the band information to be embedded
        for band_article in band_related_articles:
            documents.extend(band_article)

        # # supply the album information to be embedded
        for album_article in album_related_articles:
            documents.extend(album_article)

        return documents


    # determine artists to wiki walk:
    def extract_artists(self, page_content):
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

    def get_artist_album(self, artist_album, artist=False, album=False):
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


    def wiki_search_bands(self, album_list):
        '''queries wikipedia for band details'''
        
        back_off_throttle = 15

        band_related_articles = []
        
        print('Wiki Searching Band Details from 2024 List...')

        if "bands.txt" not in os.listdir():
            searched_bands = 0
            total_albums = len(album_list)-1
            # open the file and create a running list of parsed albums
            with open("bands.txt",'a', encoding='utf-8') as bands_file:
                for band in album_list:
                    # {something-artist} - {something-album}
                    self.progress_bar(progress=searched_bands, total=total_albums)
                    band = self.get_artist_album(artist_album=band, artist=True)
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
                        except ReadTimeout as e:
                            print("ReadTimeout occurred returning everything captured thus far.")
                            # return everything captured thus far
                            self.wiki_timeout = True
                            bands_file.close()
                            return band_related_articles

                    searched_bands += 1
                    for article in articles:
                        article_title = article.metadata['title']

                        if article_title in band:
                            article = self.chunk_wiki_content(article=article)
                            band_related_articles.append(article)
                            bands_file.write(f'{band} \n')
                        else:
                            # case where it's an ambiguous name
                            band_regex = r'\(band\)'
                            # search articles for band_regex
                            if re.search(band_regex, article_title):
                                article = self.chunk_wiki_content(article=article)
                                band_related_articles.append(article)
                                bands_file.write(f'{band}\n')
                            else:
                                # case where (singer) is in the name
                                # case where it's an ambiguous name
                                singer_regex = r'\(singer\)'
                                # search articles for band_regex
                                if re.search(singer_regex, article_title):
                                    article = self.chunk_wiki_content(article=article)
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
            with open("bands.txt", 'r', encoding='utf-8') as bands_file:
                read_band_list = bands_file.readlines()
                if len(read_band_list) == 0:
                    raise FileEmpty("bands.txt is empty. delete bands.txt")

            # reopen the file to append more bands.
            with open("bands.txt", "a") as bands_file:
                # get the last artist searched.
                last_band_searched = read_band_list[-1]
                last_band = last_band_searched.strip('\n')

                band_restart_idx = 0
                for idx, band in enumerate(album_list):
                    # strips for band/artist
                    band = self.get_artist_album(artist_album=band, artist=True)
                    band = band.strip()

                    if band != last_band:
                        continue
                    else:
                        # found your starting point.
                        band_restart_idx = idx
                        break
                # slice from slice_index:EoL
                remaining_bands = album_list[band_restart_idx:]
                # start where you left off.
                searched_bands = 0
                # collect remaining total
                remaining_albums = len(remaining_bands)
                for remaining_band in remaining_bands:
                    # {something-artist} - {something-album}
                    self.progress_bar(progress=searched_bands, total=remaining_albums)
                    band = self.get_artist_album(artist_album=remaining_band, artist=True)
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
                        except ReadTimeout as e:
                            print("ReadTimeout occurred returning everything captured thus far.")
                            # return everything captured thus far
                            bands_file.close()
                            return band_related_articles

                    searched_bands += 1
                    for article in articles:
                        article_title = article.metadata['title']

                        if article_title in band:
                            article = self.chunk_wiki_content(article=article)
                            band_related_articles.append(article)
                            bands_file.write(f'{band} \n')
                        else:
                            # case where it's an ambiguous name
                            band_regex = r'\(band\)'
                            # search articles for band_regex
                            if re.search(band_regex, article_title):
                                article = self.chunk_wiki_content(article=article)
                                band_related_articles.append(article)
                                bands_file.write(f'{band}\n')
                            else:
                                # case where (singer) is in the name
                                # case where it's an ambiguous name
                                singer_regex = r'\(singer\)'
                                # search articles for band_regex
                                if re.search(singer_regex, article_title):
                                    article = self.chunk_wiki_content(article=article)
                                    band_related_articles.append(article)
                                    bands_file.write(f'{band}\n')
                                else:
                                    continue
                print(colorama.Fore.RESET)
                print('Finished Wiki Searching Bands...')
                # close the file.
                bands_file.close()
                return band_related_articles


    def wiki_search_albums(self, album_list):
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
                    self.progress_bar(progress=searched_albums, total=total_albums)
                    album = self.get_artist_album(artist_album=album,album=True)
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
                        except ReadTimeout as e:
                            print("ReadTimeout occurred returning everything captured thus far.")
                            self.wiki_timeout = True
                            # return everything captured thus far
                            albums_file.close()
                            return album_related_articles
                    
                    searched_albums += 1
                    for article in articles:
                        article_title = article.metadata['title']
                        # search articles for band_regex
                        if article_title in album:
                            article = self.chunk_wiki_content(article=article)
                            album_related_articles.append(article)
                            # add the band to the list
                            albums_file.write(f'{album}\n')
                        else:
                            # case where it's an ambiguous name
                            album_regex = r'\(album\)'
                            # search articles for band_regex
                            if re.search(album_regex, article_title):
                                article = self.chunk_wiki_content(article=article)
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
            # assume the file exists and check the list
            with open("albums.txt","r") as albums:
                read_albums_list = albums.readlines()
                if len(read_albums_list) == 0:
                    raise FileEmpty("albums.txt is empty. delete albums.txt")

            # reopen file to add to it.
            with open("albums.txt", "a") as albums_file:
                # get the last read band
                last_album_searched = read_albums_list[-1]
                last_album = last_album_searched.strip('\n')

                album_restart_indx = 0
                for idx, album in enumerate(album_list):
                    # remove wite space
                    album = self.get_artist_album(artist_album=album, album=True)
                    album = album.strip()

                    if album != last_album:
                        continue
                    else:
                        album_restart_indx = idx
                        break

                remaining_albums = album_list[album_restart_indx:]
                # start where you left off.
                searched_albums = 0
                # collect th remaining total
                albums_left = len(remaining_albums)
                # iterate through the remaining items
                for remaining_album in remaining_albums:
                    # remove white space
                    self.progress_bar(progress=searched_albums, total=albums_left)
                    album = self.get_artist_album(artist_album=remaining_album,album=True)
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
                        except ReadTimeout as e:
                            print("ReadTimeout occurred returning everything captured thus far.")
                            # return everything captured thus far
                            albums_file.close()
                            return album_related_articles
                    
                    searched_albums += 1
                    for article in articles:
                        article_title = article.metadata['title']
                        # search articles for band_regex
                        if article_title in album:
                            article = self.chunk_wiki_content(article=article)
                            album_related_articles.append(article)
                            # add the band to the list
                            albums_file.write(f'{album}\n')
                        else:
                            # case where it's an ambiguous name
                            album_regex = r'\(album\)'
                            # search articles for band_regex
                            if re.search(album_regex, article_title):
                                article = self.chunk_wiki_content(article=article)
                                album_related_articles.append(article)
                                # add the band to the list
                                albums_file.write(f'{album}\n')
                            else:
                                continue

            print(colorama.Fore.RESET)
            print('Finished Chunking ALbums...')
            return album_related_articles


    def chunk_wiki_content(self,article):
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