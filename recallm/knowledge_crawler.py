# Given a URL this program loads the web page text and provides it to
# the Recall system as ground truth knowledge

# For dataset creation
import os
import pickle
import json
import argparse

from datasets import load_dataset

# URL libs
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup # For extracting contents of html pages
from readability import Document

from datetime import datetime



class DatasetElement:
    def __init__(self, source, source_title, source_content) -> None:
        self.source = source
        self.source_title = source_title
        self.source_content = source_content
    
    def __repr__(self) -> str:
        return f'{self.source_title}\n\tlen: {len(self.source_content)}\n\tbody text:{self.source[0:min(len(self.source), 150)]}'

class DatasetCollector:
    def __init__(self, scrapeops_api_key) -> None:
        self.SCRAPEOPS_API_KEY = scrapeops_api_key
        self.USE_SCAPEOPS = True # Without scrapeops some website might block simple python requests

    def fetch_html_source(self, url):
        # User agent generator - https://www.useragentstring.com/pages/useragentstring.php
        request = Request(url = url, headers={'User-Agent': 'Opera/9.80 (X11; Linux i686; Ubuntu/14.10) Presto/2.12.388 Version/12.16.2'})
        html = ""

        try: # Try simple request
            html = str(urlopen(request).read())

        except HTTPError as e:
            if e.code == 403:
                # Connection refused by host - Try proxy using scrape ops
                if self.USE_SCAPEOPS:
                    reponse = requests.get(
                        url='https://proxy.scrapeops.io/v1/', 
                        params={'api_key': self.SCRAPEOPS_API_KEY, 'url': url}
                    )
                    html = reponse.content

                else:
                    print(f'Failed to fetch html source (Scrape Ops disabled)')
        
        return html

    def get_clean_text_from_url(self, url):
        html = self.fetch_html_source(url)

        doc = Document(html)

        # This function should be called from try catch block
        # Dereferencing doc might fail if document is empty for some reason
        title = doc.title()
        body = doc.summary(html_partial=True)

        soup = BeautifulSoup(body, 'html.parser')
        body = soup.get_text()  # Clean the text of html tags

        return title, body




class TruthfulQADatasetCollector(DatasetCollector):
    def __init__(self, percentage_of_dataset, scrapeops_api_key) -> None:
        super().__init__(scrapeops_api_key)

        self.percentage_of_dataset = percentage_of_dataset
        self.hf_dataset = []
        self.dataset = []

    # Load the truthful_qa dataset
    def load_tqa_dataset(self):
        raw_dataset = load_dataset('truthful_qa', 'generation', split=f'validation[:{self.percentage_of_dataset}%]')
        print(f'Loaded {self.percentage_of_dataset}% of truthful_qa with {raw_dataset.num_rows} rows')
        print(f'\t features: [{raw_dataset.features.keys()}]')
        dataset = []
        for element in raw_dataset:
            if element['source'].startswith("http"):
                dataset.append(element)
        print(f'\t Using {len(dataset)} elements in dataset')
        return dataset

    
    def fetch_sources(self):
        self.hf_dataset = self.load_tqa_dataset()
        
        loaded_pages = {} # To check so that we don't load the same page twice for different elements
        for element in self.hf_dataset:
            url = urlparse(element['source'])
            url = f'{url.scheme}://{url.hostname}{url.path}' # This remove URL tags/queries at the end so that we don't load duplicate pages

            body = ""

            if url in loaded_pages:
                pass
                # If we wanted to load cached articles - but this would create duplicates
                # print(f'Loading cached article for {url}\n')
                # body = loaded_pages['url']['body']
                # title = loaded_pages['url']['title']

                # element = DatasetElement(
                #     source=url,
                #     source_title=title,
                #     source_content=body)
                
                # self.dataset.append(element)

            else:
                print(f'Fetching article for {url}')
                try: # get_clean_text_from_url might raise an error
                    title, body = self.get_clean_text_from_url(url)

                    element = DatasetElement(
                        source=url,
                        source_title=title,
                        source_content=body)
                    
                    self.dataset.append(element)

                    loaded_pages[url] = {'body':body, 'title':title}

                    print(f'\tFinished fetching article: {title}\n')
                except:
                    print(f'\n\nERROR: FAILED TO EXTRACT CONTENT FROM SOURCE!\n\n')
                    pass


def load_truthqa_dataset(api_keys, print_results=False):
    # Load Dataset
    dataset_collector_path = f'datasets/truthful_qa/truthful_qa'
    if os.path.exists(dataset_collector_path):
        # Try load from pickle
        dataset_collector_file = open(dataset_collector_path, 'rb')
        dataset_collector = pickle.load(dataset_collector_file)

        print(f'\nLOADED DATASET FROM FILE!\n')
    else:
        # Fetch sources using webscraper
        dataset_collector = TruthfulQADatasetCollector(
            percentage_of_dataset=10,
            scrapeops_api_key=api_keys['scrapeops'])
        
        dataset_collector.fetch_sources()

        # save using pickle
        dataset_collector_file = open(dataset_collector_path, 'wb')
        pickle.dump(dataset_collector, dataset_collector_file)

        print(f'\nFINISHED CREATING DATASET!\n')

    if print_results:
        for element in dataset_collector.dataset:
            print(f'\n\nelement:\n{element}')

    return dataset_collector.hf_dataset, dataset_collector.dataset




# if __name__ == '__main__':
#     from datastore_gen import *
#     from recall import TextColor

#     print(f'Please wait - Initializing...', end="", flush=True)

#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument('--reset', action='store_true', help='Reset the database')
#     args = parser.parse_args()

#     with open('api_keys.json') as f:
#         api_keys = json.load(f)


#     # Load content from truthfulqa sources
#     # knowledge_update_dataset is a list of DatasetElement
#     hf_dataset, knowledge_update_dataset = load_truthqa_dataset(api_keys)

#     datastore_handler = DatastoreHandler(api_keys=api_keys,
#                                          reset_collection=args.reset)

#     print(f'\r                                     ')


#     start_time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     print(f'\nSTARTING KNOWLEDGE UPDATE!\n')

#     knowledge_update_progress_file_path = f'datasets/truthful_qa/knowledge_update_progress.txt'
#     with open(knowledge_update_progress_file_path, 'a+') as progress_file:
#         progress_file.seek(0)
#         loaded_elements = []
#         for line in progress_file.readlines():
#             loaded_elements.append(line.strip()) # .strip() removes any leading or trailing whitespace

#         for element in knowledge_update_dataset:
#             if element.source in loaded_elements: # Do not re-load elements that have already been loaded
#                 continue

#             print(f'{TextColor.CYAN}Updating knowledge for: {element.source_title}\t {element.source}{TextColor.RESET}')
#             print(f'\t{len(knowledge_update_dataset) - len(loaded_elements)} elements remaining', end="", flush=True)
#             # Load knowledge to Recall LM system
#             text = f'{element.source_title}\n\n{element.source_content}'
#             datastore_handler.knowledge_update_pipeline(text)
            
#             progress_file.write(f'{element.source}\n')
#             progress_file.flush()

#             print(f'\r                                           ', end="", flush=True)
#             print(f'\r   {TextColor.MAGENTA}Knowledge updated{TextColor.RESET}\n')
    
#     end_time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     print(f"TruthfulQA knowledge crawl complete!!\n\nStart time:\t {start_time_string}\nEnd time:\t {end_time_string}")

#     datastore_handler.close_datastore()
