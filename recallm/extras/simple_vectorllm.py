# Copyright 2023 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import sys
sys.path.append('..') 

from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI



# Chroma Database
from langchain.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
from langchain.embeddings import HuggingFaceEmbeddings

CHROMA_COLLECTION_NAME = "collection"
CHROMA_DB_PERSIST_DIR = 'database/chroma_LLMContext'






class SimpleVectorLLM:
    def __init__(self, api_keys) -> None:
        self.collection = get_chroma_collection(reset_collection=True)
        self.collection_id_counter = 1

        llm = OpenAI(
            temperature=0,
            openai_api_key=api_keys['open_ai'],
            model_name="gpt-3.5-turbo"
        )

        self.chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm,
            chain_type="stuff",
            retriever=self.collection.as_retriever()
        )
    
    def ask_question(self, question):
        return self.chain(
            {"question": question},
            return_only_outputs=True
        )['answer']

    def load_knowledge(self, text):
        self.collection.add_texts(
            texts=[text],
            metadatas=[{"source": f"k-{self.collection_id_counter}"}],
            ids=[f'{self.collection_id_counter}']
        )
        self.collection_id_counter += 1





def get_chroma_collection(reset_collection = False):
    chroma_client_settings = Settings(chroma_api_impl="rest",
                                    chroma_server_host="localhost",
                                    chroma_server_http_port="8000")
    
    if reset_collection:
        reset_chroma_client(chroma_client_settings)

    embedding_function = HuggingFaceEmbeddings()


    chroma_collection = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=CHROMA_DB_PERSIST_DIR,
        client_settings=chroma_client_settings
    )

    if reset_collection:
        add_texts_to_collection(chroma_collection, texts=["Sample"]) # TODO: Fix this, this was a workaround so that the collection is actually created immediately instead of remaining empty
    
    return chroma_collection


def reset_chroma_client(client_settings):
    client = chromadb.Client(settings=client_settings)
    client.reset()
    print("\n!!!\tChroma client reset\t!!!")


def add_texts_to_collection(collection, texts):
    for i, sample in enumerate(texts):
        print(f'\rAdding document: {i+1}/{len(texts)}', end="")
        collection.add_texts(
            texts=[sample],
            metadatas=[{"source":f'chunk-{i}'}],
            ids=[f'{i}']
        )
    print(f'\n\n')