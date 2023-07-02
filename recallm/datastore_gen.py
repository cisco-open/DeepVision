import config

from knowledge_crawler import *

# Misc
from typing import List
from langchain.text_splitter import CharacterTextSplitter
import re
from datetime import datetime
from deprecated import deprecated
from enum import Enum
import time
from collections import deque

# Graph Database
from neo4j import GraphDatabase

# Chroma Database
from langchain.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
from langchain.embeddings import HuggingFaceEmbeddings

# Summarization chain
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

# NLTK Pipeline
import stanza
from nltk.stem import PorterStemmer

# NER Pipeline (Deprecated)
from transformers import AutoModelForTokenClassification
# from transformers import pipeline





NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"

CHROMA_COLLECTION_NAME = "collection"
CHROMA_DB_PERSIST_DIR = 'database/chroma_LLMContext'

TEXT_CHUNK_SIZE = 2500
TEXT_CHUNK_OVERLAP = 1000

# CONTEXT_SIZE is only used for NER pipeline and is deprecated
CONTEXT_SIZE = 50               # Size of context window for each update

CONTEXT_REVISION_PERIOD = 4     # Number of iterations before context is revised
CONTEXT_SUMMARIZATION_DESIRED_SIZE = 80 # Size in number of words (Only for OpenAI chain)

NEIGHBORING_CONCEPT_DISTANCE_FOR_KNOWLEDGE_UPDATE = 1 # Number of related concepts to fetch -> Warning this causes exponential growth


class DatastoreHandler:
    def __init__(self, api_keys, reset_collection=False) -> None:
        # self.driver = create_neo4j_driver()
        self.driver = NeoDriverWrapper()

        # self.chroma_collection = get_chroma_collection(reset_collection=reset_collection)
        if reset_collection:
            reset_graph_db(driver=self.driver)

        self.knowledge_crawler = DatasetCollector(api_keys['scrapeops'])

        # self.ner_pipeline = create_ner_pipeline()
        self.nltk_pipeline = create_nltk_pipeline()
        self.word_stemmer = PorterStemmer()
        self.summarization_chain = SummarizationChain(type=SummarizationType.OPENAI,
                                                      api_keys=api_keys)

    def reset_datastore(self):
        reset_graph_db(self.driver)

    def close_datastore(self):
        self.driver.close()

    def question_system(self, question, default_chain, vanilla_chain):
        context = fetch_contexts_for_question(
            question=question,
            nltk_pipeline=self.nltk_pipeline,
            word_stemmer=self.word_stemmer,
            driver=self.driver)
        
        if context.had_graph_db_entity:
            response = default_chain(
                {
                    "question": question,
                    "info":context.context
                },
                return_only_outputs=True
            )
        else:
            response = vanilla_chain(
                {
                    "question": question,
                    "info":context.context
                },
                return_only_outputs=True
            )
        
        return response['text']

    def perform_knowledge_update_from_url(self, url):
        title, body = self.knowledge_crawler.get_clean_text_from_url(url)
        text = f"{title}\n\n{body}"
        self.knowledge_update_pipeline(text)
    
    def perform_knowledge_update_from_file(self, file):
        text = load_texts(filename=file)
        self.knowledge_update_pipeline(text)

    def knowledge_update_pipeline(self, text):
        # Deprecated approach using NER
        # ner_results = self.ner_pipeline(texts)
        # concepts_chunks = fetch_ner_concepts_from_texts(texts, ner_results)

        texts = split_text(text)   # Convert to batches
        concepts_chunks = fetch_concepts_from_texts(texts, self.nltk_pipeline, self.word_stemmer)

        # Deprecated approach using chromaDB
        # store_concepts_to_database(
        #     concepts_chunks,
        #     self.summarization_chain,
        #     self.driver,
        #     self.chroma_collection)

        update_knowledge_graph(
            concepts_chunks,
            self.summarization_chain,
            self.driver)








def create_nltk_pipeline():
    stanza.download('en')
    nlp = stanza.Pipeline('en')
    return nlp

class NeoDriverWrapper:
    def __init__(self) -> None:
        self.driver = NeoDriverWrapper.create_neo4j_driver()
        self.query_count = 0

        self.QUERY_RESET_PERIOD = 1000

        self.error_log_name = "NEO4J_ERROR_LOG.txt"

    def create_neo4j_driver():
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()

        driver.execute_query("""
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Concept) REQUIRE n.name IS UNIQUE
        """)

        driver.execute_query("MERGE (n:Meta)")

        return driver

    def close(self):
        self.driver.close()

    def restart_connection(self):
        print(f'\n\nRestarting Neo4J connection to prevent memory overflow from Neo4J garbage collection issues. Please wait.\n\n')
        self.driver.close()
        time.sleep(10)
        self.driver = NeoDriverWrapper.create_neo4j_driver()
        self.query_count = 0
        print(f'Neo4J driver connection has been reset')

    def execute_query(self, query):
        try:
            result = self.driver.execute_query(query)
            self.query_count += 1

            if self.query_count > 0 and self.query_count % self.QUERY_RESET_PERIOD == 0:
                self.restart_connection()
        
        except Exception as e:
            self.write_to_log(filename=self.error_log_name,
                              text=f'{e}\nQUERY:\n{query}')
            print(e)

            time.sleep(5)
            self.restart_connection()
            self.query_count = 0
            return self.execute_query(query) # Try again
        
        return result
    
    def write_to_log(self, filename, text):
        time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"{time_string}\n{text}"
        try:
            with open(filename, 'a') as file:
                # Append the text to the file
                file.write(text)
                file.write('\n\n')
        except FileNotFoundError:
            # If the file doesn't exist, create a new file and append the text
            with open(filename, 'w') as file:
                file.write(text)
                file.write('\n\n')

def reset_graph_db(driver):
    driver.execute_query("""
MATCH (n)
OPTIONAL MATCH (n)-[r]-()
DELETE n, r
""")
    
    driver.execute_query("""
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Concept) REQUIRE n.name IS UNIQUE
    """)

    driver.execute_query("MERGE (n:Meta)")

def reset_chroma_client(client_settings):
    client = chromadb.Client(settings=client_settings)
    client.reset()
    print("\n!!!\tChroma client reset\t!!!")

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
        add_texts_to_collection(chroma_collection, texts=["Sample"]) # TODO: Fix this, this was a workaround to so that the collection is actually created immediately instead of remaining empty
    
    return chroma_collection

class SummarizationType(Enum):
    OPENAI = 1
    LOCAL_DISTIL_BART = 2

class SummarizationChain:
    def __init__(self, type: SummarizationType, api_keys) -> None:
        self.type = type
        self.chain = create_summarization_chain(api_keys, self.type)

def create_summarization_chain(api_keys, summarization_type):
    if summarization_type == SummarizationType.OPENAI:
        llm = OpenAI(
            temperature=0.01,
            openai_api_key=api_keys['open_ai'],
            model_name="gpt-3.5-turbo"
        )
        
        prompt = PromptTemplate(
            input_variables=["context", "max_words"],
            template="""Each sentence in the folllowing texts are true in chronological order, summarize the text in less than {max_words} words while retaining all relevant information and details:
            
            Source: Set in the 21st century, Mars has been 23% terraformed. Set in the second half of the 22nd century, Mars has been 84% terraformed, allowing humans to walk on the surface without pressure suits. Martian society has become matriarchal, with women in most positions of authority. Arriving at the remote mining town, Ballard finds all of the people missing. She learns that they had discovered an underground doorway created by an ancient Martian civilization.
            Summary: In the 22nd century, 84% of Mars has been terraformed, allowing humans to walk the surface without pressure suits. Women hold most positions of power in Martian society. Ballard finds missing people in remote mining town, these people discovered an underground doorway created by an ancient Martian civilization
            
            Source: {context}
            Summary: """,
        )

        chain = LLMChain(llm=llm, prompt=prompt)

        return chain

    elif summarization_type == SummarizationType.LOCAL_DISTIL_BART:
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

        model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

        summarization_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        
        return summarization_pipeline

    else:
        raise Exception(f"Summarization method not implemented: {summarization_type}")

def load_texts(filename):
    with open(filename, encoding="utf-8") as f:
        file = f.read()
    return file

def split_text(text, text_chunk_size=TEXT_CHUNK_SIZE, chunk_overlap=TEXT_CHUNK_OVERLAP):
    texts = []
    start = 0
    while start < len(text):
        end = start + text_chunk_size
        end = min(len(text)-1, end)
        if end < len(text)-1:
            # Find last full stop
            while text[end] not in ['.', '\n']:
                end -= 1
        texts.append(text[start:end])
        start += text_chunk_size - chunk_overlap
    return texts

def add_texts_to_collection(collection, texts):
    for i, sample in enumerate(texts):
        print(f'\rAdding document: {i+1}/{len(texts)}', end="")
        collection.add_texts(
            texts=[sample],
            metadatas=[{"source":f'chunk-{i}'}],
            ids=[f'{i}']
        )
    print(f'\n\n')






# What is a concept?
#   Anything that we can ask questions about, has properties, or related 
#   beliefs about that concept. Eg. a person, a place, an entity,
#   a noun etc.
class Concept:
    def __init__(self, name, type, start_index, end_index, chunk_index) -> None:
        name = re.sub(r'[^\w\s]', '', name) # Remove special symbols
        name = re.sub(r'\s+', '', name) # Remove whitespace

        self.name = name
        self.type = type

        # These variables need to be overwritten later
        self.context = ""
        self.related_concepts = []
        self.unique_id = 0  # This id is only unique to the source text that the concept was extracted from. It is not unique in the entire concept knowledge base.
        self.chroma_id = name # This id is unique in the chroma database.

        # These variables are only relevant when first extracting context from source and adding to databases
        self.start_index = start_index # Refers to position in source text
        self.end_index = end_index
        self.chunk_index = chunk_index

        self.MERGES_BEFORE_REVISION = CONTEXT_REVISION_PERIOD
        self.merge_count = 0
        self.revision_count = 0 # Number of times that context has been updated, context is revised every n updates

        # Variables for when questioning the system
        self.t_index = 0 # t_index is only set and used when reconstructing the graph for graph traversal in fetch_contexts_for_question

        self.sort_val = 0

    def __lt__(self, other):
        return self.start_index < other.start_index

    def __eq__(self, other) -> bool:
        if isinstance(other, Concept):
            return self.name == other.name
        return False

    # Merge two different concepts to create one concept
    #   when full_merge is true -> merge contexts and related concepts as well
    def merge_with(self, concept, full_merge=False):
        if self.end_index <= concept.start_index:
            if self.name != concept.name:
                self.name = f'{self.name}{concept.name}'
            self.end_index = concept.end_index
        else:
            if self.name != concept.name:
                self.name = f'{concept.name}{self.name}'
            self.start_index = concept.start_index
        
        if self.type != concept.type:
            if self.type[0] < concept.type[0]: # Concept is just a string so we want to make sure we always concate in alphabetical order so we don't get duplicates
                self.type = f'{self.type}|{concept.type}'
            else:
                self.type = f'{concept.type}|{self.type}'

        if full_merge:
            self.merge_count += 1

            self.context = f'{self.context} {concept.context}'

            for related_concept in concept.related_concepts:
                if related_concept not in self.related_concepts:
                    self.related_concepts.append(related_concept)
    
    def should_revise(self) -> bool:
        if self.revision_count > 0 and self.revision_count % CONTEXT_REVISION_PERIOD == 0:
            return True
        
        if self.merge_count >= self.MERGES_BEFORE_REVISION:
            return True
        
        return False

    def revise_context(self, summarization_chain):
        if summarization_chain.type == SummarizationType.OPENAI:
            self.context = summarization_chain.chain({
                                "context": self.context,
                                "max_words": f'{CONTEXT_SUMMARIZATION_DESIRED_SIZE}'
                            },
                            return_only_outputs=True)['text']
        
        elif summarization_chain.type == SummarizationType.LOCAL_DISTIL_BART:
            self.context = summarization_chain.chain(self.context)[0]['generated_text']

        else:
            raise Exception("Summarization type not implemented")

        self.merge_count = 0
        

    def __repr__(self) -> str:
        # return f'{self.name}:{self.type}\t{self.start_index}'
        return f'{self.name}:\t{self.context}'
        # return f'{self.name}: \t{[c.name for c in self.related_concepts]}'


# Used to represent a relationship between two concepts, specifically for 'fetch_contexts_for_question()'
#   when reconstructing the subgraph
class TemporalRelation:
    def __init__(self, to:Concept, t_index, strength) -> None:
        self.to = to    # Which concept is it related to - We don't store a 'from', because this instance is contained in the from instance
        self.t_index = t_index
        self.strength = strength
    
    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, TemporalRelation):
            return self.to.name == __value.to.name
        return False







# Fetch the text surrounding this concept in the source text to form the 
# context for this concept
def fetch_contexts_for_concepts(concepts, texts, context_size):
    if len(texts) == 0:
        return
    if type(texts) == str:      # There was no batching in texts
        texts = [texts]         # So we put it in batch format to work with rest of code

    for concept in concepts:
        current_text = texts[concept.chunk_index]
        concept.context = current_text[
            max(0, concept.start_index - context_size):
            min(len(current_text), concept.end_index + context_size)]
        concept.context = re.sub(r'[^\w\s]', '', concept.context) # Remove special symbols
        
# Given a list of concepts, find the neighbouring concepts for all concepts
# by relative position in source text
def fetch_neigbouring_concepts(concepts, distance):
    concepts = sorted(concepts)
    for i in range(len(concepts)):
        concepts[i].related_concepts = []
        # concepts[i].chroma_id = ""
        for j in range(-distance, distance + 1, 1): # If distance from current concept is less than parameter distance
            if i + j >= 0 and i + j < len(concepts): # If index is in bounds
                # concepts[i].chroma_id = f'{concepts[i].chroma_id} {concepts[i+j].name}'
                
                if j == 0:
                    continue

                if concepts[i].name < concepts[i+j].name: # Ensure that we only create one connection between nodes in Neo4J graph
                    concepts[i].related_concepts.append(concepts[i+j])

# Given a list of concepts, merge all concepts that share the same name (Regardless
#   of their location in the source text)
def merge_concepts(concepts):
    result_concepts = []
    for i in range(len(concepts)):
        if concepts[i] not in result_concepts:
            result_concepts.append(concepts[i])
        
        for j in range(i+1, len(concepts), 1):
            if concepts[i].name == concepts[j].name:
                concepts[i].merge_with(concept=concepts[j],
                                       full_merge=True)

    return result_concepts

def assign_unique_ids_to_concepts(concepts, chunk_index):
    for (i, concept) in enumerate(concepts):
        concept.unique_id = f'c{chunk_index}{i}'

# Root function to fetch concepts from a source text
def fetch_concepts_from_texts(texts, nltk_pipeline, word_stemmer):
    if len(texts) == 0:
        return []
    if type(texts) != list:     # There was no batching in texts input
        texts = [texts]         # So we put it in batch format to work with rest of code

    concepts_chunks = [] # Array of chuhks/batches
    for (i, text) in enumerate(texts):
        new_concepts = fetch_concepts_from_single_batch(text=text,
                                                        nltk_pipeline=nltk_pipeline,
                                                        word_stemmer=word_stemmer,
                                                        batch_index=i,
                                                        concept_type=ConceptType.MULTIPLE_PER_SENTENCE)
        
        concepts_chunks.append(new_concepts)

    for i in range(len(concepts_chunks)):
        assign_unique_ids_to_concepts(
            concepts=concepts_chunks[i],
            chunk_index=i
        )

        fetch_neigbouring_concepts(
            concepts=concepts_chunks[i],
            distance=NEIGHBORING_CONCEPT_DISTANCE_FOR_KNOWLEDGE_UPDATE
        )

        concepts_chunks[i] = merge_concepts(concepts=concepts_chunks[i])
    
    return concepts_chunks

class ConceptType(Enum):
    MERGED = 1
    MULTIPLE_PER_SENTENCE = 2

def fetch_concepts_from_single_batch(text,
                                     nltk_pipeline,
                                     word_stemmer,
                                     batch_index,
                                     concept_type):
    doc = nltk_pipeline(text)

    concepts = []
    for sentence in doc.sentences:
        sentence_concepts = {}
        sentence_concept_string = ""

        index = 0
        for word in sentence.words:
            if word.upos in ["NOUN", "PROPN", "NUM"]:
                word_concept_string = word_stemmer.stem(word.text).lower()
                sentence_concepts[word_concept_string] = {'pos':word.upos, 'index':index}
                index += 1

                if concept_type == ConceptType.MERGED:
                    sentence_concept_string = f'{sentence_concept_string} {word_concept_string}'

        if len(sentence_concepts.keys()) > 3:   # Number of concepts in sentence is too much, we need to reduce it
            # If it contains proper noun -> Reduce to proper noun and surrounding nouns
            if "PROPN" in [value['pos'] for value in sentence_concepts.values()]:
                propn_index = [element['index'] for element in sentence_concepts.values() if element['pos'] == "PROPN"]
                propn_index = propn_index[0]

                sentence_concepts_new = []
                for text, meta in zip(sentence_concepts.keys(), sentence_concepts.values()):
                    if abs(meta['index'] - propn_index) <= 1:
                        sentence_concepts_new.append(text)

                if concept_type == ConceptType.MERGED:
                    sentence_concept_string = ""
                    for concept in sentence_concepts_new:
                        sentence_concept_string = f'{sentence_concept_string} {concept}'
            # Else just use middle nouns
            else:
                sentence_concepts_new = []
                for text, meta in zip(sentence_concepts.keys(), sentence_concepts.values()):
                    middle_index = len(sentence_concepts.keys()) // 2
                    if abs(meta['index'] - middle_index) <= 1:
                        sentence_concepts_new.append(text)

                if concept_type == ConceptType.MERGED:
                    sentence_concept_string = ""
                    for concept in sentence_concepts_new:
                        sentence_concept_string = f'{sentence_concept_string} {concept}'

            # Update sentence_concepts dictionary
            keys = list(sentence_concepts.keys())
            for key in keys:
                if key not in sentence_concepts_new:
                    del sentence_concepts[key]
        
        if len(sentence_concepts.values()) == 0:
            continue

        # Get original sentence string back from [stanfordnlp.pipeline.word]
        sentence_context = sentence.words[0].text
        for word in sentence.words[1:]:
            if word.upos == "PUNCT":
                sentence_context = f'{sentence_context}{word.text}'
            else:
                sentence_context = f'{sentence_context} {word.text}'
        

        if concept_type == ConceptType.MULTIPLE_PER_SENTENCE:
            for concept_key, concept_value in zip(sentence_concepts.keys(), sentence_concepts.values()):
                concept_index = concept_value['index'] + (batch_index * TEXT_CHUNK_SIZE)
                sentence_concept_string = sentence_concept_string[1:]
                new_concept = Concept(
                    name=concept_key,
                    type=concept_value['pos'],
                    start_index=concept_index,
                    end_index=concept_index,
                    chunk_index=batch_index
                )
                new_concept.context = sentence_context

                concepts.append(new_concept)
        
        if concept_type == ConceptType.MERGED:
            concept_index = list(sentence_concepts.values())[0]['index'] + (batch_index * TEXT_CHUNK_SIZE)
            sentence_concept_string = sentence_concept_string[1:]
            new_concept = Concept(
                name=sentence_concept_string,
                type="",
                start_index=concept_index,
                end_index=concept_index,
                chunk_index=batch_index
            )
            new_concept.context = sentence_context

            concepts.append(new_concept)

    return concepts

def update_knowledge_graph(concepts_chunks, summarization_chain, driver, use_mini_batches=True):
    # It is assumed that all information in the knowledge update is provided to the system at the
    # same point in time. Hence the knowledge update only counts for one increment in the global
    # temporal counter

    ### UPDATE THE GLOBAL TEMPORAL INDEX COUNTER ###
    temporal_update_query = """
MATCH (n:Meta)
WITH n, COALESCE(n.temporal_index, 0) + 1 as n_temporal_index
SET n.temporal_index = n_temporal_index
RETURN n_temporal_index
    """
    global_temporal_index = driver.execute_query(temporal_update_query)
    global_temporal_index = int(global_temporal_index[0][0]['n_temporal_index'])
    ### UPDATE THE GLOBAL TEMPORAL INDEX COUNTER ###

    for i in range(len(concepts_chunks)):
        if len(concepts_chunks[i]) == 0:
            continue


        ### CREATE/MERGE CONCEPT NODES AND CREATE RELATIONS WITH UPDATED STRENGTHS AND  ###
        ### TEMPORAL INDICES                                                            ###

        #################################################################################
        #   SAMPLE QUERY:                                                               #
        # MERGE (c00:Concept {name: 'brandon'})                                     (1) #
        # MERGE (c03:Concept {name: 'cisco'})                                           #
        # WITH c00, c03                                                             (2) #
        # MERGE (c00)-[rc00c03:RELATED]->(c03)                                      (3) #
        # WITH c00, c03, rc00c03, COALESCE(rc00c03.strength, 0) + 1 AS rc00c03ic    (4) #
        # SET c00.t_index = {global_temporal_index}                                 (5) #
        # SET c03.t_index = {global_temporal_index}                                     #
        # SET rc00c03.strength = rc00c03ic                                          (6) #
        # SET rc00c03.t_index = {global_temporal_index}                             (7) #
        #################################################################################

        if use_mini_batches:    # Mini-batches can help when Neo4J is running out of memory, mini-batches also seem to be faster
            for mini_batch_i, concept in enumerate(concepts_chunks[i]):
                # (1)
                neo_query = f"MERGE ({concept.unique_id}:Concept {{name: \'{concept.name}\'}})\n"
                if len(concept.related_concepts) > 0:
                    # (1)
                    for related in concept.related_concepts:
                        neo_query += f"MERGE ({related.unique_id}:Concept {{name: \'{related.name}\'}})\n"
                    # (2)
                    neo_query += f"WITH {concept.unique_id}"
                    for related in concept.related_concepts:
                        neo_query += f", {related.unique_id}"
                    neo_query += f"\n"
                    # (3)
                    for related in concept.related_concepts:
                        neo_query += f"MERGE ({concept.unique_id})-[r{concept.unique_id}{related.unique_id}:RELATED]->({related.unique_id})\n"
                    # (4)
                    neo_query += f"WITH {concept.unique_id}"
                    for related in concept.related_concepts:
                        neo_query += f", {related.unique_id}"
                    for related in concept.related_concepts:
                        neo_query += f", r{concept.unique_id}{related.unique_id}"
                    for related in concept.related_concepts:
                        neo_query += f", COALESCE(r{concept.unique_id}{related.unique_id}.strength, 0) + 1 AS r{concept.unique_id}{related.unique_id}ic"
                    neo_query += f"\n"
                    # (5)
                    neo_query += f"SET {concept.unique_id}.t_index = {global_temporal_index}"
                    for related in concept.related_concepts:
                        neo_query += f", {related.unique_id}.t_index = {global_temporal_index}"
                    neo_query += f"\n"
                    # (6)
                    neo_query += f"SET "
                    for related in concept.related_concepts:
                        neo_query += f"r{concept.unique_id}{related.unique_id}.strength = r{concept.unique_id}{related.unique_id}ic, "
                    neo_query = neo_query[:len(neo_query)-2]
                    neo_query += f"\n"
                    # (7)
                    neo_query += f"SET "
                    for related in concept.related_concepts:
                        neo_query += f"r{concept.unique_id}{related.unique_id}.t_index = {global_temporal_index}, "
                    neo_query = neo_query[:len(neo_query)-2]

                    print(f'\rAdding concept relations for chunk:\t {i+1}/{len(concepts_chunks)}\t mini-batch: {mini_batch_i}/{len(concepts_chunks[i])}', end="", flush=True)
                    driver.execute_query(neo_query)
                

        else:
            # TODO:
            #   There is something wrong with the query below (I.e. When not using mini-batches) that is
            #   creating empty nodes. Fix this!!
            print(f'Warning, knowledge update without mini-batches is deprecated!', flush=True)

            neo_query = ''
            # (1)
            for concept in concepts_chunks[i]:
                neo_query = f'{neo_query}\nMERGE ({concept.unique_id}:Concept {{name: \'{concept.name}\'}})'

            # (2)
            # Create relations between concepts, if relation already exists we increase it's strength
            # This is synonymous to a synapse developing a stronger connection
            relation_string = f'\nWITH {concepts_chunks[i][0].unique_id}'
            for j in range(1, len(concepts_chunks[i]), 1):
                relation_string = f'{relation_string}, {concepts_chunks[i][j].unique_id}'
            
            # (3)
            relations_exist = False
            for concept in concepts_chunks[i]:
                for related_concept in concept.related_concepts:
                    relations_exist = True
                    relation_string = f'{relation_string}\nMERGE({concept.unique_id})-[r{concept.unique_id}{related_concept.unique_id}:RELATED]->({related_concept.unique_id})'

            if relations_exist: # This flag will be false if no connections exist
                # (4)
                relation_string = f'{relation_string}\nWITH'
                for concept in concepts_chunks[i]:
                    relation_string = f'{relation_string} {concept.unique_id},'
                    for related_concept in concept.related_concepts:
                        relation_string = f'{relation_string} r{concept.unique_id}{related_concept.unique_id},'
                relation_string = relation_string[0:len(relation_string)-1] # Remove extra comma at the end

                for concept in concepts_chunks[i]:
                    for related_concept in concept.related_concepts:
                        relation_string = f'{relation_string}, COALESCE(r{concept.unique_id}{related_concept.unique_id}.strength, 0) + 1 AS r{concept.unique_id}{related_concept.unique_id}ic'

                # (5)
                relation_string = f'{relation_string}\nSET'
                for concept in concepts_chunks[i]:
                    relation_string = f"{relation_string} {concept.unique_id}.t_index = {global_temporal_index},"
                relation_string = relation_string[0:len(relation_string)-1] # Remove extra comma at the end

                # (6)
                relation_string = f'{relation_string}\nSET'
                for concept in concepts_chunks[i]:
                    for related_concept in concept.related_concepts:
                        relation_string = f'{relation_string} r{concept.unique_id}{related_concept.unique_id}.strength = r{concept.unique_id}{related_concept.unique_id}ic,'
                relation_string = relation_string[0:len(relation_string)-1] # Remove extra comma at the end
                
                # (7)
                relation_string = f'{relation_string}\nSET'
                for concept in concepts_chunks[i]:
                    for related_concept in concept.related_concepts:
                        relation_string = f'{relation_string} r{concept.unique_id}{related_concept.unique_id}.t_index = {global_temporal_index},'
                relation_string = relation_string[0:len(relation_string)-1] # Remove extra comma at the end

                neo_query = f'{neo_query}{relation_string}'
            
            print(f'\rAdding concept relations for chunk:\t {i+1}/{len(concepts_chunks)}')
            # print(f'Batch neo query:\n{neo_query}\n\n')
            driver.execute_query(neo_query)

        print('')




        ### GET EXISTING CONTEXTS FROM DATABASE ###

        #################################################################
        #   SAMPLE QUERY:                                               #
        # MATCH (n:Concept)                                             #
        # WHERE n.name IN ['seed', 'two', 'watermelon']                 #
        # RETURN n.name, n.context AS matches                           #
        #################################################################

        neo_query = "MATCH (n:Concept)"
        neo_query += f"\nWHERE n.name IN {[concept.name for concept in concepts_chunks[i]]}"
        neo_query += f"\nRETURN n.name, n.context, n.revision_count"

        print(f'\rFetching concept contexts for chunk:\t {i+1}/{len(concepts_chunks)}')

        neo_results_object = driver.execute_query(neo_query)

        # Result structure [<result array>, <neo4j._work.summary.ResultSummary>, <keys>]

        neo_results = {}
        for result in neo_results_object[0]:
            neo_results[result['n.name']] = {
                "context" : result['n.context'],
                "revision_count" : result['n.revision_count'] if ('n.revision_count' in result.keys()) else 0
            }

        for concept in concepts_chunks[i]:
            if concept.name in neo_results:     # If concept exists in database with a context
                database_context = neo_results[concept.name]['context']
                if f"{database_context}" == "None":
                    database_context = ""

                if concept.context != "":
                    concept.context = f"{database_context}. {concept.context}"
                else:
                    concept.context = database_context

                if neo_results[concept.name]['revision_count'] != None:
                    concept.revision_count = int(neo_results[concept.name]['revision_count']) + concept.revision_count + 1





        ### REVISE CONTEXTS ###

        # TODO: Use local summarization model -> Compute in parallel across multiple GPU's
        for j, concept in enumerate(concepts_chunks[i]):
            print(f'\rRevising context {j+1}/{len(concepts_chunks[i])} for chunk:\t {i+1}/{len(concepts_chunks)}', end="", flush=True)
            if concept.should_revise():
                concept.revise_context(summarization_chain)

        print('')
    



        ### SET CONTEXTS AND REVISION_COUNT ###

        #################################################################
        #   SAMPLE QUERY:                                               #
        # MATCH (n:Concept)                                             #
        # WHERE n.name IN ['b', 'seeds']                                #
        # WITH n,                                                       #
        #     CASE n.name                                               #
        #         WHEN 'b' THEN 'Context for b'                         #
        #         WHEN 'seeds' THEN 'Context for seeds'                 #
        #         ELSE n.context                                        #
        #     END AS newContext,                                        #
        #     CASE n.name                                               #
        #         WHEN 'b' THEN 3                                       #
        #         WHEN 'seeds' THEN 5                                   #
        #         ELSE 0                                                #
        #     END AS revisionCount                                      #
        # SET n.context = newContext                                    #
        # SET n.revision_count = revisionCount                          #
        #################################################################

        neo_query = "MATCH (n:Concept)"
        neo_query += f"\nWHERE n.name IN {[concept.name for concept in concepts_chunks[i]]}"
        neo_query += f"\nWITH n,"
        neo_query += f"\n\tCASE n.name"

        for concept in concepts_chunks[i]:  # WHEN 'b' THEN 'Context for b'
            context = concept.context.replace('\\ \'', '\\\'')    # Text formatting fixes for Neo4J query
            context = context.replace(' \'', '\'')
            context = context.replace('\\\'', '\'')
            context = context.replace('\'', '\\\'')
            if context[-1] == "\\":
                context = context[0:len(context)-1]
            neo_query += f"\n\t\tWHEN '{concept.name}' THEN '{context}'"
        neo_query += f"\n\t\tELSE n.context"
        neo_query += f"\n\tEND AS newContext,"
        neo_query += f"\n\tCASE n.name"
        for concept in concepts_chunks[i]:  # WHEN 'b' THEN 3
            neo_query += f"\n\t\tWHEN '{concept.name}' THEN {concept.revision_count}"
        neo_query += f"\n\t\tELSE 0"
        neo_query += f"\n\tEND AS revisionCount"

        neo_query += f"\nSET n.context = newContext"
        neo_query += f"\nSET n.revision_count = revisionCount"

        print(f'\rAdding concept context for chunk:\t {i+1}/{len(concepts_chunks)}')
        # print(f'\n\n{neo_query}\n\n', flush=True)
        driver.execute_query(neo_query)
        print('')

class ContextResponse:
    def __init__(self) -> None:
        self.context = ""
        self.had_graph_db_entity = False

def fetch_contexts_for_question(question,
                                nltk_pipeline,
                                word_stemmer,
                                driver,
                                temporal_window_size=15,
                                max_relation_distance=2,
                                max_contexts_count=10):
    context_result = ContextResponse()

    # Identify key concepts in question
    q_concepts_chunks = fetch_concepts_from_texts(question, nltk_pipeline, word_stemmer)

    q_concepts_strings = []
    for q_concept_chunk in q_concepts_chunks:
        for q_concept in q_concept_chunk:
            q_concepts_strings.append(q_concept.name)

    if config.verbose:
        print(f'Main concepts in question: {q_concepts_strings}')


    ###     RECONSTRUCT SUBGRAPH IN PYTHON      #########################################################################
    NODE_LIMIT_PER_CONCEPT_QUERY = 800  # This is just an upper bound, fewer results are actually returned due to duplicates before using 'DISTINCT'
    graph_concept_nodes = {}

    # For each concept identified in question -> Fetch related concepts using temporal_index
    # 
    # We begin by recreating the relevant subgraph in python so that we perform more complex concept extraction in python 
    #       We have to execute this query once per concept in question, so that the spanning tree finds nodes
    #       within the correct temporal context window
    for q_concept_name in q_concepts_strings:
        neo_query = f"""
MATCH (startNode:Concept{{name: '{q_concept_name}'}})
CALL apoc.path.spanningTree(startNode, {{relationshipFilter: "", minLevel: 0, maxLevel: {max_relation_distance}}}) YIELD path
WITH path, nodes(path) as pathNodes, startNode.t_index as current_t
UNWIND range(0, size(pathNodes)-1) AS index
WITH path, pathNodes[index] as node, current_t
ORDER BY node.t_index DESC
WHERE node.t_index <= current_t AND node.t_index >= current_t - {temporal_window_size}
WITH DISTINCT node LIMIT {NODE_LIMIT_PER_CONCEPT_QUERY}
MATCH ()-[relation]->()
RETURN node, relation
"""

        # query_result[0] -> type: list of results
        # query_result[1] -> type: neo4j._work.summary.ResultSummary
        # query_result[2] -> type: list of returned result names ie. ['node', 'relation']
        query_result = driver.execute_query(neo_query)

        # Add Concepts
        for result in query_result[0]:
            node = result.get('node') 
            node_id = node.element_id # Eg. '4:4c33fa63-9fcc-4078-9a2b-4bb3f4aa6a2a:106'
            node = node._properties # Eg. {'name': 'collector', 'context': 'However, collectors often ... '}

            if node_id not in graph_concept_nodes.keys():
                new_concept = Concept(name=node['name'],
                                      type=ConceptType.MULTIPLE_PER_SENTENCE,
                                      start_index=0,
                                      end_index=0,
                                      chunk_index=0)
                new_concept.context = node['context']
                new_concept.t_index = node['t_index']
                new_concept.unique_id = node_id
                graph_concept_nodes[new_concept.name] = new_concept

        # Populate relations
        for result in query_result[0]:
            relation = result.get('relation')
            # relation_type = relation.type # Eg. 'RELATED'
            relation_properties = relation._properties # Eg. {'strength': 1, 't_index': 1}
            relation_left_node, relation_right_node = relation.nodes
            relation_left_node = relation_left_node._properties
            relation_right_node = relation_right_node._properties

            if len(relation_left_node.keys()) > 0 and len(relation_right_node.keys()) > 0:   # For some reason there are results in query result with only one node, we should just ignore these
                # relation_left_t_index = int(relation_left_node["t_index"])
                # relation_right_t_index = int(relation_right_node["t_index"])
                # relation_left_strength = int(relation_left_node["strength"])
                # relation_right_strength = int(relation_right_node["strength"])

                relation_left_node_name = relation_left_node["name"] # String
                relation_right_node_name = relation_right_node["name"] # String

                relation_t_index = int(relation_properties['t_index'])
                relation_strength = int(relation_properties['strength'])


                left_concept = graph_concept_nodes[relation_left_node_name]
                right_concept = graph_concept_nodes[relation_right_node_name]

                # Relations are bi-directional so we create 2
                relation_to_left = TemporalRelation(to=left_concept,
                                                    t_index=relation_t_index,
                                                    strength=relation_strength)
                
                relation_to_right = TemporalRelation(to=right_concept,
                                                     t_index=relation_t_index,
                                                     strength=relation_strength)

                # Check that relation hasn't already been added (This could have happened from previous iteration)
                if relation_to_right not in left_concept.related_concepts:
                    left_concept.related_concepts.append(relation_to_right)
                
                if relation_to_left not in right_concept.related_concepts:
                    right_concept.related_concepts.append(relation_to_left)

    ###     (END) RECONSTRUCT SUBGRAPH IN PYTHON      #########################################################################



    
    # To extract the most relevant context from the subgraph (Because in most cases, the sub graph will still
    # be too large to use the entire subgraph):
    #   Every concept in 'graph_concept_nodes' should already be within the correct temporal window, so we 
    #   don't worry about that.
    #   Although, there might still be too many concepts to create a prompt
    #   so we sort by the sum of 't_index' on all relations of a concept

    # Set sort val
    graph_concepts = graph_concept_nodes.values()
    for concept in graph_concepts:
        concept.sort_val = 0
        for relation in concept.related_concepts:
            concept.sort_val += (relation.t_index * 3) + relation.strength     # TODO: 3 is a hyperparameter, move this to header
    graph_concepts = sorted(graph_concepts, key=lambda c: c.sort_val)
    graph_concepts.reverse()    # We want them in descending order, so that highest temporal relations are first


    # Concept objects for all concepts found directly in question
    # We use these first regardless of their sortval
    essential_concepts = [concept for concept in graph_concepts if concept.name in q_concepts_strings]

    context_concepts = set()
    context = ""
    for concept in essential_concepts:
        context_concepts.add(concept.name)
        context = f'{concept.context}\n\n{context}' # The order here is very important, concepts with higher temporal index should appear after in the context string

    context_count = len(context_concepts)
    for concept in graph_concepts:
        if concept.name not in context_concepts and context_count < max_contexts_count:
            context_count += 1
            context_concepts.add(concept.name)
            context = f'{concept.context}\n\n{context}' # The order here is very important, concepts with higher temporal index should appear after in the context string

    context = context[:len(context)-2] # Remove extra new lines


    if config.verbose:
        print(f'Related concepts: {context_concepts}')

    if len(context_concepts) > 0:
        context_result.had_graph_db_entity = True

    if config.verbose:
        print(f'Context:\n\t{context}\n')

    context_result.context = context

    return context_result



























@deprecated(version='1.0', reason='Concept extraction using NER is depreciated, use nltk pipeline instead')
def create_ner_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
    return ner_pipeline

@deprecated(version='1.0', reason='Joint storage using ChromaDB is deprecated')
def fetch_chroma_contexts_for_question(question, ner_pipeline, driver, chroma_collection, max_relation_distance = 1):
    context_result = ContextResponse()
    
    # Identify key concepts in question
    ner_results = ner_pipeline(question)
    q_concepts_chunks = fetch_ner_concepts_from_texts(question, ner_results)

    q_concepts_strings = []
    for q_concept_chunk in q_concepts_chunks:
        for q_concept in q_concept_chunk:
            q_concepts_strings.append(q_concept.name)

    if config.verbose:
        print(f'Main concepts in question: {q_concepts_strings}')

    # For each key concept in question -> Fetch closest/connected concepts
    neo_query = f"""
MATCH (startNode: Concept)
WHERE startNode.name IN {q_concepts_strings}
CALL apoc.path.spanningTree(startNode, {{relationshipFilter: "", minLevel: 0, maxLevel: {max_relation_distance}}}) YIELD path
RETURN nodes(path) AS connectedNodes LIMIT 10
"""
    query_response = driver.execute_query(neo_query)
    
    # query_response[0] -> type: list of results
    # query_response[1] -> type: neo4j._work.summary.ResultSummary
    # query_response[2] -> type: list of returned result names ie. ['connectedNodes']
    query_results = query_response[0]
    q_concepts_related = []
    for q_concept in query_results:
        q_concepts_related.append(q_concept.get('connectedNodes')[len(q_concept.get('connectedNodes')) - 1].get('name'))
    
    if config.verbose:
        print(f'Related concepts: {q_concepts_related}')

    if len(q_concepts_related) > 0:
        context_result.had_graph_db_entity = True

    # Add identified concepts from question even if they could not be found in graphDB
    #   Because we might be able to find similar concepts from similarity search in chromaDB
    for q_concept_chunk in q_concepts_chunks:
        for q_concept in q_concept_chunk:
            if q_concept.name not in q_concepts_related:
                q_concepts_related.append(q_concept.name)

    if len(q_concepts_related) == 0:
        return context_result

    # Query chromadb using concepts to get releveant context
    chroma_results = chroma_collection.query(    # Returns len(q_concepts_related) * n_results array
        query_texts=q_concepts_related,
        n_results=1 # Return single closest match for each concept in q_concepts_related
    )

    context = ""
    for result in chroma_results['metadatas']:
        for n_result in result:
            if "context" in n_result: # check that key exists - If the database contains a record without the 'context' key it will return an error
                context = f'{context}{n_result["context"]}'
                if context[len(context)-1] == '.':
                    context = f'{context} '
                else:
                    context = f'{context}. '
            else:
                if config.verbose:
                    print(f'WARNING: A record exists in chromaDB without the \'context\' key and was excluded from a query.')
    context = context[0:len(context)-1] # Remove whitespace at the end

    if config.verbose:
        print(f'Context:\n\t{context}\n')

    context_result.context = context

    return context_result

# Join neighbouring words to form a single concept
# Eg. Given the text "... from Johannesburg, South Africa ..." this would
#   become three seperate concepts {Johannesburg}, {South}, {Africa}
#   we merge them to become one concept {JohannesburgSouthAfrica}
@deprecated(version='1.0', reason='\'concatenate_concepts\' is part of the NER pipeline which is deprecated')
def concatenate_concepts(concepts, max_distance) -> List[Concept]:
    concepts = sorted(concepts)
    i = 0
    while i < len(concepts)-1:
        if concepts[i].end_index >= concepts[i+1].start_index-1 - max_distance:
            concepts[i].merge_with(concepts[i+1])
            del concepts[i+1]
        else:
            i += 1
    return concepts

@deprecated(version='1.0', reason='Fetching concepts using Named Entity Recognition is deprecated, us \'fetch_concepts_from_texts\' to perform concept exctraction more efficiently using dependency parsing.')
def fetch_ner_concepts_from_texts(texts, ner_results):
    if len(ner_results) == 0:
        return []
    if type(ner_results[0]) != list:    # There was no batching in ner_results
        ner_results = [ner_results]     # So we put it in batch format to work with rest of code

    concepts_chunks = [] # Array of chunks/batches
    for (i, ner_chunk) in enumerate(ner_results):
        concepts_chunks.append([])
        for ner in ner_chunk:
            concepts_chunks[i].append(Concept(
                name=ner['word'],
                type=ner['entity'],
                start_index=ner['start'],
                end_index=ner['end'],
                chunk_index=i))

    for i in range(len(concepts_chunks)):
        concepts_chunks[i] = concatenate_concepts(concepts_chunks[i], max_distance=1)

        assign_unique_ids_to_concepts(
            concepts=concepts_chunks[i],
            chunk_index=i
        )

        fetch_contexts_for_concepts(
            concepts=concepts_chunks[i],
            texts=texts,
            context_size=CONTEXT_SIZE
        )

        fetch_neigbouring_concepts(
            concepts=concepts_chunks[i],
            distance=NEIGHBORING_CONCEPT_DISTANCE_FOR_KNOWLEDGE_UPDATE
        )
    
    return concepts_chunks

@deprecated(version='1.0', reason='Joint storage using ChromaDB is deprecated')
def store_concepts_to_database(concepts_chunks, summarization_chain, driver, chroma_collection):
    # Store relations in graph database
    for i in range(len(concepts_chunks)):
        if len(concepts_chunks[i]) == 0:
            continue

        neo_query = ''
        for concept in concepts_chunks[i]:
            neo_query = f'{neo_query}\nMERGE ({concept.unique_id}:Concept {{name: \'{concept.name}\', type: \'{concept.type}\'}})'

        # Create relations between concepts, if relation already exists we increase it's strength
        # This is synonymous to a synapse developing a stronger connection
        relation_string = f'\nWITH {concepts_chunks[i][0].unique_id}'
        for j in range(1, len(concepts_chunks[i]), 1):
            relation_string = f'{relation_string}, {concepts_chunks[i][j].unique_id}'
        relations_exist = False
        for concept in concepts_chunks[i]:
            for related_concept in concept.related_concepts:
                relations_exist = True
                relation_string = f'{relation_string}\nMERGE({concept.unique_id})-[r{concept.unique_id}{related_concept.unique_id}:RELATED]->({related_concept.unique_id})'

        if relations_exist: # This flag will be false if no connections exist
            relation_string = f'{relation_string}\nWITH'
            for concept in concepts_chunks[i]:
                for related_concept in concept.related_concepts:
                    relation_string = f'{relation_string} r{concept.unique_id}{related_concept.unique_id},'
            relation_string = relation_string[0:len(relation_string)-1] # Remove extra comma at the end

            for concept in concepts_chunks[i]:
                for related_concept in concept.related_concepts:
                    relation_string = f'{relation_string}, COALESCE(r{concept.unique_id}{related_concept.unique_id}.strength, 0) + 1 AS r{concept.unique_id}{related_concept.unique_id}ic'

            relation_string = f'{relation_string}\nSET'
            for concept in concepts_chunks[i]:
                for related_concept in concept.related_concepts:
                    relation_string = f'{relation_string} r{concept.unique_id}{related_concept.unique_id}.strength = r{concept.unique_id}{related_concept.unique_id}ic,'

            relation_string = relation_string[0:len(relation_string)-1] # Remove extra comma at the end
            
            neo_query = f'{neo_query}{relation_string}'
        
        print(f'\rAdding concept relations for chunk: {i+1}/{len(concepts_chunks)}', end="")
        # print(f'Batch neo query:\n{neo_query}\n\n')
        driver.execute_query(neo_query)

    print('')

    # Store contexts in ChromaDB
    total_revisions = 0
    for i in range(len(concepts_chunks)):
        # For knowledge updates
        chroma_update_ids = []
        chroma_update_metadatas = []
        chroma_update_documents = []

        # For creating new knowledge
        chroma_create_texts = []
        chroma_create_metadatas = []
        chroma_create_ids = []

        for j, concept in enumerate(concepts_chunks[i]):
            print(f'\rAdding context:\tchunk: {i+1}/{len(concepts_chunks)}\t concept: {j+1}/{len(concepts_chunks[i])}', end="")
            
            # Check if context for concept id exists
            db_record = chroma_collection._collection.get(ids=[concept.chroma_id])

            # ChromaDB embedding is by done texts/documents
            # We use the chroma_id string for embedding so that if we query the concept name or
            # neighbouring concept names we still get this record as a result

            if len(db_record['ids']) > 0:       # If it does exist -> update context for same concept id (Using summarization method)
                if config.verbose:
                    print(f'\n\tUpdating context for {concept.name} (ID: {concept.chroma_id}):\t {concept.context}')
                
                # Combine previous context with new context
                current_context = db_record['metadatas'][0]['context']
                if current_context[len(current_context)-1] == '.':
                    new_context = f'{current_context} {concept.context}'
                else:
                    new_context = f'{current_context}. {concept.context}'

                # Periodically revise the context to shorten it and maintain key concepts while forgetting irrelevant info
                update_count = db_record['metadatas'][0]['update_count']
                update_count += 1
                if update_count > 0 and update_count % CONTEXT_REVISION_PERIOD == 0:
                    total_revisions += 1
                    new_context = summarization_chain.chain({
                            "context": new_context,
                            "max_words": f'{CONTEXT_SUMMARIZATION_DESIRED_SIZE}'
                        },
                        return_only_outputs=True)['text']
                    
                    if config.show_revisions:
                        print(f'\nContext revision for: {concept.name}\n\t{new_context}')

                chroma_update_ids.append(concept.chroma_id)
                chroma_update_metadatas.append({"context": new_context, "update_count": update_count})
                chroma_update_documents.append(concept.chroma_id)

                concept.context = new_context

            else:                               # Else create new record
                if config.verbose:
                    print(f'\n\tAdding context for {concept.name} (ID: {concept.chroma_id}):\t {concept.context}')

                chroma_create_texts.append(concept.chroma_id)
                chroma_create_metadatas.append({"context":concept.context, "update_count":1})
                chroma_create_ids.append(f'{concept.chroma_id}')

        if len(chroma_update_ids) > 0:
            try:
                chroma_collection._collection.update(
                    ids=chroma_update_ids,
                    metadatas=chroma_update_metadatas,
                    documents=chroma_update_documents)
            except Exception as e:
                print(f'Failed to update ids: {chroma_update_ids}\n\n{e}')

        if len(chroma_create_ids) > 0:
            chroma_collection.add_texts(
                texts=chroma_create_texts,
                metadatas=chroma_create_metadatas,
                ids=chroma_create_ids)

    print(f'\nFinished storing concepts to ChromaDB with {total_revisions} revisions')



