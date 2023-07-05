import argparse

from datastore_gen import *
from benchmark import *

from utils import TextColor

import config

from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import json
import os





def check_input_valid(user_input):
    valid_input = True
        
    if user_input in ["exit", "Exit", "EXIT"]:
        datastore_handler.close_datastore()
        exit(0)
    
    if len(user_input) < 5:
        print(f'Input length is too short')
        valid_input = False
    else:
        if user_input[0:2] not in ['q:', 'f:', 'o:', 't:', 's:', 'b:', 'g:', 'a:']:
            valid_input = False
    
    if not valid_input:
        print(f'Invalid input - Please specify an operator:\n\tq: -> Question\n\tf: -> provide knowledge or fact\n\to: -> Specify file to load more knowledge from')

    return valid_input

def create_empty_chain(api_keys):
    llm = OpenAI(
        temperature=0.01,
        openai_api_key=api_keys['open_ai'],
        model_name="gpt-3.5-turbo"
    )

    prompt = PromptTemplate(
        input_variables=["question"],
        template="""{question}""",
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

def create_vanilla_chain(api_keys):
    llm = OpenAI(
        temperature=0.01,
        openai_api_key=api_keys['open_ai'],
        model_name="gpt-3.5-turbo"
    )

    prompt = PromptTemplate(
        input_variables=["question", "info"],
        template="""Statement:{info}\n\n{question}""",
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

def create_chain(api_keys):
    llm = OpenAI(
        temperature=0.01,
        openai_api_key=api_keys['open_ai'],
        model_name="gpt-3.5-turbo"
    )

    prompt = PromptTemplate(
        input_variables=["question", "info"],
        template="""Using the following statements when necessary, answer the question that follow. Each sentence in the following statements is true when read in chronological order: 

        statements: {info}

        question: {question}

        Answer:
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    return chain



if __name__ == "__main__":
    # NOTE: We want to see if we can store info in the chroma database better so
    #   that LLM has better chance of producing the correct ouput.
    #   For example the custom response chain produces the correct output with a
    #   very small llm because it has exactly the right context, where as 
    #   RetrievalQAWithSourcesChain.from_chain_type fails with small llm because the
    #   context isn't precise enough

    # RetrievalQAWithSourcesChain.from_chain_type fails to answer this question with bloom
    # response = chain(
    #     {"question": "How old was officer Mora"}, # 27 years old
    #     return_only_outputs=False
    # )

    print(f'Please wait - Initializing...', end="", flush=True)

    #################################################################################
    ###     ARG PARSER      #########################################################
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--reset', action='store_true', help='Reset the database')
    parser.add_argument('--verbose', action='store_true', help='Debugging mode')
    parser.add_argument('--show_revisions', action='store_true', help='Show concept revisions/summarizations')
    parser.add_argument('--compare', action='store_true', help='Show the response from raw LLM without any prompting')

    args = parser.parse_args()
    ###     ARG PARSER      #########################################################
    #################################################################################


    #################################################################################
    ###     INITIALIZE     ##########################################################
    
    if args.verbose:
        config.verbose = True
    if args.show_revisions:
        config.show_revisions = True
    if args.compare:
        config.compare = True

    # Set API keys
    with open('api_keys.json') as f:
        api_keys = json.load(f)
        os.environ['OPENAI_API_KEY'] = api_keys['open_ai']

    datastore_handler = DatastoreHandler(api_keys=api_keys,
                                         reset_collection=args.reset)

    vanilla_chain = create_vanilla_chain(api_keys)
    chain = create_chain(api_keys)
    if config.compare:
        empty_chain = create_empty_chain(api_keys)

    if not args.reset:
        print(f'\r                                     ')
    ###     INITIALIZE     ##########################################################
    #################################################################################

    print(f"""

Welcome to Recall LM!

To interact with the system please type an operator followed by your input:
    q: -> Question
    f: -> provide knowledge or fact
    o: -> Specify a file or url to load more knowledge from
          
    To benchmark RecallLM:
    t: -> Load knowledge from file in test format and begin benchmarking. Or provide directory containing multiple benchmark files.
    s: -> Interactively grade results by hand using blind grading - you cannot see which model generated the output.
        
    b: -> Automatically perform a full benchmark. This performs the knowledge update on a file or directory, then autogrades results
    g: -> Automatically grade results using GPT auto-grader
          
    a: -> compute aggregate stastics for benchmark results in folder
          
    Eg. 'q: Who is Brandon?'
        'f: Brandon is an artificial intellignce researcher working for Cisco.'
        'o: datasets/state_of_the_union.txt'
        'o: https://en.wikipedia.org/wiki/List_of_common_misconceptions'
          
        't: datasets/benchmark.txt'
        't: datasets/duorc'
        's: benchmarks/benchmark.json'
        'g: benchmarks/benchmark.json'
        'a: benchmarks'

          
    Type 'exit' to terminate the program

    """)


    # User interaction loop
    user_input = ""
    while True:
        user_input = input(f"\n\n{TextColor.CYAN}User: ")
        print(TextColor.RESET)

        valid_input = check_input_valid(user_input)

        if valid_input:
            operator = user_input[0]
            user_input = user_input[2:]
            while user_input[0] == ' ':
                user_input = user_input[1:] # sometimes users put whitespace at the front
            

            if operator == 'q': # Question the system
                response = datastore_handler.question_system(
                    question=user_input,
                    default_chain=chain,
                    vanilla_chain=vanilla_chain
                )

                print(f'   {TextColor.MAGENTA}Recal LM: {response}{TextColor.RESET}')

                if config.compare:
                    empty_chain_response = empty_chain(
                        {"question": user_input},
                        return_only_outputs=True
                    )
                    print(f'   {TextColor.RED}Base model: {empty_chain_response["text"]}{TextColor.RESET}')


            if operator == 'f': # Provide knowledge to the system
                datastore_handler.knowledge_update_pipeline(text=user_input)
                print(f'   {TextColor.MAGENTA}Knowledge updated{TextColor.RESET}')


            if operator == 'o': # Load knowledge from a text file or url
                if user_input.startswith("http"):
                    datastore_handler.perform_knowledge_update_from_url(url=user_input)
                    print(f'   {TextColor.MAGENTA}Knowledge updated{TextColor.RESET}')

                elif os.path.exists(user_input):
                    datastore_handler.perform_knowledge_update_from_file(file=user_input)
                    print(f'   {TextColor.MAGENTA}Knowledge updated{TextColor.RESET}')

                else:
                    print(f'   {TextColor.MAGENTA}Invalid file path or url.{TextColor.RESET}')

            
            if operator == 't':
                benchmark = MasterBenchmarkHandler(api_keys=api_keys,
                                                    test_file=user_input,
                                                    datastore_handler=datastore_handler,
                                                    recall_default_chain=chain,
                                                    recall_vanilla_chain=vanilla_chain)

                benchmark.begin_benchmarking(
                    question_samples=1,
                    knowledge_update_loops=1
                )


            if operator == 's':
                # Perform manual benchmark grading
                benchmark_grader = ManualBenchmarkGrader(results_file=user_input)
                benchmark_grader.score_responses()
            
            if operator == 'g':
                # GPT autograder for benchmarking
                try:
                    benchmark_grader = GPTBenchmarkGrader(results_file=user_input, api_keys=api_keys)
                    benchmark_grader.score_responses()
                except Exception as e:
                    print(e)
                
            if operator == 'b':
                # Perform full benchmark, knowledge update then autograde results using GPT grader

                benchmark = MasterBenchmarkHandler(api_keys=api_keys,
                                                    test_file=user_input,
                                                    datastore_handler=datastore_handler,
                                                    recall_default_chain=chain,
                                                    recall_vanilla_chain=vanilla_chain,
                                                    autograde=True)

                benchmark.begin_benchmarking(
                    question_samples=1,
                    knowledge_update_loops=1
                )
            
            if operator == 'a':
                # Compute aggregate stastics
                stastics_handler = StasticHandler(user_input)
                stastics_handler.compute_aggregate_score()
