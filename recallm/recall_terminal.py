import argparse
import os

from recall import RecallM

from extras.benchmark import *

import config
from utils import TextColor

def check_input_valid(user_input, recallm):
    valid_input = True
        
    if user_input.lower() == "exit":
        recallm.close()
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

if __name__ == "__main__":
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

    recallm = RecallM()

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

        valid_input = check_input_valid(user_input, recallm)

        if valid_input:
            operator = user_input[0]
            user_input = user_input[2:]
            while user_input[0] == ' ':
                user_input = user_input[1:] # sometimes users put whitespace at the front
            
        if operator == 'q': # Question the system
                response = recallm.question(user_input)
                print(f'   {TextColor.MAGENTA}Recal LM: {response}{TextColor.RESET}')

                if config.compare:
                    empty_chain_response = recallm.question_no_context(user_input)
                    print(f'   {TextColor.RED}Base model: {empty_chain_response}{TextColor.RESET}')


        if operator == 'f': # Provide knowledge to the system
            recallm.update(knowledge=user_input)
            print(f'   {TextColor.MAGENTA}Knowledge updated{TextColor.RESET}')


        if operator == 'o': # Load knowledge from a text file or url
            if user_input.startswith("http"):
                recallm.update_from_url(url=user_input)
                print(f'   {TextColor.MAGENTA}Knowledge updated{TextColor.RESET}')

            elif os.path.exists(user_input):
                recallm.update_from_file(file=user_input)
                print(f'   {TextColor.MAGENTA}Knowledge updated{TextColor.RESET}')

            else:
                print(f'   {TextColor.MAGENTA}Invalid file path or url.{TextColor.RESET}')

        
        if operator == 't':
            benchmark = MasterBenchmarkHandler(api_keys=api_keys,
                                               test_file=user_input,
                                               recallm=recallm,
                                               test_hybrid=True,
                                               autograde=False)

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
                                               recallm=recallm,
                                               test_hybrid=True,
                                               autograde=True)

            benchmark.begin_benchmarking(
                question_samples=1,
                knowledge_update_loops=1
            )
        
        if operator == 'a':
            # Compute aggregate stastics
            stastics_handler = StasticHandler(user_input)
            stastics_handler.compute_aggregate_score()