import string
import contextlib
import io
from typing import Any

from datastore_gen import *

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain.schema import BaseRetriever
from langchain.schema import Document as LangDocument






def create_vanilla_chain(api_keys):
    chain = ChatOpenAI(temperature=0,
                       openai_api_key=api_keys['open_ai'],
                       model_name="gpt-3.5-turbo")
    
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template) 

    chat_prompt = ChatPromptTemplate.from_messages(
        [human_message_prompt]
    )

    return chain, chat_prompt

def create_chain(api_keys):
    chain = ChatOpenAI(temperature=0,
                       openai_api_key=api_keys['open_ai'],
                       model_name="gpt-3.5-turbo")
    
    template = (
        """Using the following statements when necessary, answer the question that follows. Each sentence in the following statements is true when read in chronological order: 

        statements:
        {info}

        question:
        {question}

        Answer:
        """
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt]
    )

    return chain, chat_prompt


class RecallM:
    def __init__(self, openai_key='', reset_on_init=False) -> None:
        """
        Initialize the RecallM system and connect to the Docker database.

        Parameters:
        openai_key (string): OpenAI api key
        reset_on_init (bool): Should the database be reset when initialized? This will delete all data in the database.

        Returns:
        RecallM object
        """

        if openai_key == '':
            with open('./api_keys.json') as f:
                api_keys = json.load(f)
            if 'scrapeops' not in api_keys:
                api_keys['scrapeops'] = ""
        else:
            api_keys = {
                "open_ai": openai_key,
                "scrapeops": ""
            }
            

        os.environ['OPENAI_API_KEY'] = api_keys['open_ai']
        self.datastore_handler = DatastoreHandler(api_keys=api_keys,
                                                  reset_collection=reset_on_init)

        self.vanilla_chain, self.vanilla_prompt = create_vanilla_chain(api_keys)
        self.default_chain, self.default_prompt = create_chain(api_keys)

    def reset_knowledge(self):
        self.datastore_handler.reset_datastore()

    def close(self):
        self.datastore_handler.close_datastore()

    def question(self, question):
        response = self.datastore_handler.question_system(
            question=question,
            default_chain=self.default_chain,
            default_prompt=self.default_prompt,
            vanilla_chain=self.vanilla_chain,
            vanilla_prompt=self.vanilla_prompt
        )
        return response

    def update(self, knowledge: string):
        with contextlib.redirect_stdout(io.StringIO()):
            self.datastore_handler.knowledge_update_pipeline(text=knowledge)

    def update_from_url(self, url):
        self.datastore_handler.perform_knowledge_update_from_url(url=url)

    def update_from_file(self, file):
        self.datastore_handler.perform_knowledge_update_from_file(file=file)

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        return RecallMRetriever(datastore_handler=self.datastore_handler)




class RecallMRetriever(BaseRetriever):
    def __init__(self, datastore_handler) -> None:
        super().__init__()

        self.datastore_handler = datastore_handler
    
    def get_relevant_documents(self, query: str) -> List[LangDocument]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        context = self.datastore_handler.fetch_contexts_for_question(query)

        docs = []
        # We only use one doc to ensure the temporal order of contexts is correct
        docs.append(LangDocument(page_content=context.context, metadata={"source":query}))

        return docs

    async def aget_relevant_documents(self, query: str) -> List[LangDocument]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        raise Exception("Async get_relevant_documents not implemented yet!")



















# from extras.benchmark import *
# import argparse
# from utils import TextColor
# import config
# import json
# import os



# def check_input_valid(user_input, recallm_obj):
#     valid_input = True
        
#     if user_input in ["exit", "Exit", "EXIT"]:
#         recallm_obj.close()
#         exit(0)
    
#     if len(user_input) < 5:
#         print(f'Input length is too short')
#         valid_input = False
#     else:
#         if user_input[0:2] not in ['q:', 'f:', 'o:', 't:', 's:', 'b:', 'g:', 'a:']:
#             valid_input = False
    
#     if not valid_input:
#         print(f'Invalid input - Please specify an operator:\n\tq: -> Question\n\tf: -> provide knowledge or fact\n\to: -> Specify file to load more knowledge from')

#     return valid_input


# if __name__ == "__main__":
#     # NOTE: We want to see if we can store info in the chroma database better so
#     #   that LLM has better chance of producing the correct ouput.
#     #   For example the custom response chain produces the correct output with a
#     #   very small llm because it has exactly the right context, where as 
#     #   RetrievalQAWithSourcesChain.from_chain_type fails with small llm because the
#     #   context isn't precise enough

#     # RetrievalQAWithSourcesChain.from_chain_type fails to answer this question with bloom
#     # response = chain(
#     #     {"question": "How old was officer Mora"}, # 27 years old
#     #     return_only_outputs=False
#     # )

#     print(f'Please wait - Initializing...', end="", flush=True)

#     #################################################################################
#     ###     ARG PARSER      #########################################################
#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument('--reset', action='store_true', help='Reset the database')
#     parser.add_argument('--verbose', action='store_true', help='Debugging mode')
#     parser.add_argument('--show_revisions', action='store_true', help='Show concept revisions/summarizations')
#     parser.add_argument('--compare', action='store_true', help='Show the response from raw LLM without any prompting')

#     args = parser.parse_args()
#     ###     ARG PARSER      #########################################################
#     #################################################################################


#     #################################################################################
#     ###     INITIALIZE     ##########################################################
    
#     if args.verbose:
#         config.verbose = True
#     if args.show_revisions:
#         config.show_revisions = True
#     if args.compare:
#         config.compare = True

#     with open('api_keys.json') as f:
#         api_keys = json.load(f)
#         os.environ['OPENAI_API_KEY'] = api_keys['open_ai']

#     recall_obj = RecallM(openai_key=api_keys['open_ai'], reset_on_init=args.reset)

#     if config.compare:
#         empty_chain = create_empty_chain(api_keys)

#     if not args.reset:
#         print(f'\r                                     ')
#     ###     INITIALIZE     ##########################################################
#     #################################################################################

#     print(f"""

# Welcome to Recall LM!

# To interact with the system please type an operator followed by your input:
#     q: -> Question
#     f: -> provide knowledge or fact
#     o: -> Specify a file or url to load more knowledge from
          
#     To benchmark RecallLM:
#     t: -> Load knowledge from file in test format and begin benchmarking. Or provide directory containing multiple benchmark files.
#     s: -> Interactively grade results by hand using blind grading - you cannot see which model generated the output.
        
#     b: -> Automatically perform a full benchmark. This performs the knowledge update on a file or directory, then autogrades results
#     g: -> Automatically grade results using GPT auto-grader
          
#     a: -> compute aggregate stastics for benchmark results in folder
          
#     Eg. 'q: Who is Brandon?'
#         'f: Brandon is an artificial intellignce researcher working for Cisco.'
#         'o: datasets/state_of_the_union.txt'
#         'o: https://en.wikipedia.org/wiki/List_of_common_misconceptions'
          
#         't: datasets/benchmark.txt'
#         't: datasets/duorc'
#         's: benchmarks/benchmark.json'
#         'g: benchmarks/benchmark.json'
#         'a: benchmarks'

          
#     Type 'exit' to terminate the program

#     """)


#     # User interaction loop
#     user_input = ""
#     while True:
#         user_input = input(f"\n\n{TextColor.CYAN}User: ")
#         print(TextColor.RESET)

#         valid_input = check_input_valid(user_input, recall_obj)

#         if valid_input:
#             operator = user_input[0]
#             user_input = user_input[2:]
#             while user_input[0] == ' ':
#                 user_input = user_input[1:] # sometimes users put whitespace at the front
            

#             if operator == 'q': # Question the system
#                 response = recall_obj.question(question=user_input)

#                 print(f'   {TextColor.MAGENTA}Recal LM: {response}{TextColor.RESET}')

#                 if config.compare:
#                     empty_chain_response = empty_chain(
#                         {"question": user_input},
#                         return_only_outputs=True
#                     )
#                     print(f'   {TextColor.RED}Base model: {empty_chain_response["text"]}{TextColor.RESET}')


#             if operator == 'f': # Provide knowledge to the system
#                 recall_obj.knowledge_update_from_string(knowledge=user_input)
#                 print(f'   {TextColor.MAGENTA}Knowledge updated{TextColor.RESET}')


#             if operator == 'o': # Load knowledge from a text file or url
#                 if user_input.startswith("http"):
#                     recall_obj.perform_knowledge_update_from_url(url=user_input)
#                     print(f'   {TextColor.MAGENTA}Knowledge updated{TextColor.RESET}')

#                 elif os.path.exists(user_input):
#                     recall_obj.perform_knowledge_update_from_file(file=user_input)
#                     print(f'   {TextColor.MAGENTA}Knowledge updated{TextColor.RESET}')

#                 else:
#                     print(f'   {TextColor.MAGENTA}Invalid file path or url.{TextColor.RESET}')

            





#             if operator == 't':
#                 benchmark = MasterBenchmarkHandler(api_keys=api_keys,
#                                                     test_file=user_input,
#                                                     datastore_handler=datastore_handler,
#                                                     recall_default_chain=chain,
#                                                     recall_vanilla_chain=vanilla_chain)

#                 benchmark.begin_benchmarking(
#                     question_samples=1,
#                     knowledge_update_loops=1
#                 )


#             if operator == 's':
#                 # Perform manual benchmark grading
#                 benchmark_grader = ManualBenchmarkGrader(results_file=user_input)
#                 benchmark_grader.score_responses()
            
#             if operator == 'g':
#                 # GPT autograder for benchmarking
#                 try:
#                     benchmark_grader = GPTBenchmarkGrader(results_file=user_input, api_keys=api_keys)
#                     benchmark_grader.score_responses()
#                 except Exception as e:
#                     print(e)
                
#             if operator == 'b':
#                 # Perform full benchmark, knowledge update then autograde results using GPT grader (test_hybrid=True)

#                 benchmark = MasterBenchmarkHandler(api_keys=api_keys,
#                                                     test_file=user_input,
#                                                     datastore_handler=datastore_handler,
#                                                     recall_default_chain=chain,
#                                                     recall_vanilla_chain=vanilla_chain,
#                                                     test_hybrid=False,
#                                                     autograde=False)

#                 benchmark.begin_benchmarking(
#                     question_samples=1,
#                     knowledge_update_loops=1
#                 )
            
#             if operator == 'a':
#                 # Compute aggregate stastics
#                 stastics_handler = StasticHandler(user_input)
#                 stastics_handler.compute_aggregate_score()
