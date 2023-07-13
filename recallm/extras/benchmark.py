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

from datastore_gen import DatastoreHandler
from datastore_gen import split_text
from extras.simple_vectorllm import *
from extras.hybrid_memory_llm import *
from utils import TextColor

from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import os
import json
import shutil
import random
import platform

from abc import ABC, abstractmethod

# Master benchmark handler handles batches of benchmarks
class MasterBenchmarkHandler:
    def __init__(self,
                 api_keys,
                 test_file,
                 datastore_handler:DatastoreHandler,
                 recall_default_chain,
                 recall_vanilla_chain,
                 test_hybrid=False,
                 autograde=False) -> None:
        
        self.datastore_handler = datastore_handler
        self.recall_default_chain = recall_default_chain
        self.recall_vanilla_chain = recall_vanilla_chain
        self.api_keys = api_keys

        self.test_hybrid = test_hybrid
        self.autograde = autograde

        self.hyrid_memory_llm = HybridMemoryLLM(api_keys)

        self.files = []
        if os.path.isdir(test_file):
            for file in os.listdir(test_file):
                if file.endswith(".txt"):
                    self.files.append(os.path.join(test_file, file))
        else:
            self.files.append(test_file)

    def begin_benchmarking(self, question_samples=1, knowledge_update_loops=1):
        for file in self.files:
            benchmark_file = os.path.join(os.path.join(os.getcwd(), 'benchmarks'), f'{os.path.splitext(os.path.basename(file))[0]}.json')

            # If file has not been benchmarked on recall and vectordb
            if not os.path.exists(benchmark_file):
                benchmark_handler = BenchmarkHandler(
                    api_keys=self.api_keys,
                    test_file=file,
                    datastore_handler=self.datastore_handler,
                    recall_default_chain=self.recall_default_chain,
                    recall_vanilla_chain=self.recall_vanilla_chain
                )

                benchmark_handler.begin_benchmarking(question_samples=question_samples,
                                                    knowledge_update_loops=knowledge_update_loops)
            
                # Get Hybrid result
                if self.test_hybrid:
                    self.get_hyrid_responses(benchmark_file)

                if self.autograde:
                    benchmark_grader = GPTBenchmarkGrader(results_file=benchmark_handler.results_file(),
                                                        api_keys=self.api_keys)
                    benchmark_grader.score_responses()
    
    def get_hyrid_responses(self, results_file):
        with open(results_file, 'r') as rf:
            json_results = json.load(rf)

        num_samples = int(json_results['question_samples'])
        num_questions = int(json_results['num_questions'])

        for sample_i in range(num_samples):
            for question_i in range(num_questions):
                print(f'\rComputing hybrid response for question {question_i+1}/{num_questions}\t\tsample: {sample_i+1}/{num_samples}', end='')

                qa_pair = json_results[f'sample_{sample_i}'][f'q{question_i}']

                question = qa_pair['question']
                recall_answer = qa_pair['recall']
                vectordb_answer = qa_pair['vectordb']

                hybrid_answer = self.hyrid_memory_llm.combine(question=question,
                                                                recall_answer=recall_answer,
                                                                vector_answer=vectordb_answer)
                
                qa_pair['hybrid'] = hybrid_answer

                json_results[f'sample_{sample_i}'][f'q{question_i}'] = qa_pair
        
        with open(results_file, 'w') as rf:
            json.dump(json_results, rf)
        
        print(f'\n')
        

class BenchmarkHandler:
    def __init__(self,
                 api_keys,
                 test_file,
                 datastore_handler:DatastoreHandler,
                 recall_default_chain,
                 recall_vanilla_chain) -> None:
        
        self.benchmark_name = os.path.splitext(os.path.basename(test_file))[0]
        
        # Dictionary keys are group names
        self.statements = load_statements_from_test_file(test_file) # dictionary with values of type --> [(truth, statement_string)]
        self.questions = load_questions_from_test_file(test_file) # dictionary with values of type --> [(question, answer)]

        self.datastore_handler = datastore_handler
        self.datastore_handler.reset_datastore()

        self.recall_default_chain = recall_default_chain
        self.recall_vanilla_chain = recall_vanilla_chain

        self.vector_lm = SimpleVectorLLM(api_keys=api_keys)     # Initializing new SVLLM resets chromaDB
    
    def results_file(self, question_group_name):
        return f'benchmarks/{question_group_name}_{self.benchmark_name}.json'

    def begin_benchmarking(self, question_samples=1, knowledge_update_loops=1):
        if 'INITIAL' in self.statements:
            # Load initial knowledge only once
            self.load_statement_group_for_recall(self.statements['INITIAL'],
                                                 group_name='INITIAL')
            
            self.load_statement_group_for_vectordb(self.statements['INITIAL'],
                                                   group_name='INITIAL',
                                                   ensure_database_sample_count=False)

        for _ in range(knowledge_update_loops):
            statement_group = self.statements['LOOP']

            self.load_statement_group_for_recall(statement_group, 'LOOP')
            
            self.load_statement_group_for_vectordb(statement_group,
                                                   group_name='LOOP',
                                                   ensure_database_sample_count=False)
        
        print(f'\n########################################################################')
        print(f'######      KNOWLEDGE LOADING COMPLETE       ###########################')
        print(f'########################################################################\n')

        # Begin questioning the system and save results to file
        for question_group in self.questions.keys():
            self.test_question_group(self.questions[question_group],
                                     group_name=question_group,
                                     question_samples=question_samples)


    def load_statement_group_for_recall(self, statements, group_name):
        print(f'{TextColor.WHITE}Loading {group_name} knowledge for RecallLM{TextColor.RESET}')
        for i, statement in enumerate(statements):
            statement_truth, statement_string = statement
            self.datastore_handler.knowledge_update_pipeline(text=statement_string)
            print(f'   {TextColor.MAGENTA}Knowledge updated for {i}/{len(statements)}{TextColor.RESET}')
        print(f'\n\n')

    def load_statement_group_for_vectordb(self, statements, group_name, ensure_database_sample_count=True):
        print(f'{TextColor.WHITE}Loading {group_name} knowledge for vectorDB + LLM{TextColor.RESET}')
        vectordb_statement_count = 0
        if not ensure_database_sample_count:
            vectordb_statement_count = 4

        while vectordb_statement_count <= 4: # Vectordb needs at least 4 samples to work with document retriever
            for i, statement in enumerate(statements):
                statement_truth, statement_string = statement
                statements_split = split_text(statement_string) # Split paragraph statements in chunks in the same way as RecallLM
                for c_statement_split in statements_split:
                    self.vector_lm.load_knowledge(c_statement_split)
                    vectordb_statement_count += 1
                print(f'\r   {TextColor.MAGENTA}Knowledge updated for {i}/{len(statements)}{TextColor.RESET}', end="")
        print(f'\n\n')

    def test_question_group(self, questions, group_name, question_samples):
        results_data = {
            "question_samples":question_samples,
            "num_questions":len(questions)
        }
        for i in range(question_samples):
            print(f'\n{TextColor.GREEN}########################################################################{TextColor.RESET}')
            print(f'{TextColor.GREEN}######       BEGINNING QUESTION FOR {group_name} - SAMPLE {i}         ##########{TextColor.RESET}')
            print(f'{TextColor.GREEN}########################################################################{TextColor.RESET}\n')

            sample = {}
            for j, qa_pair in enumerate(questions):
                question, answer = qa_pair

                print(f'\n{TextColor.GREEN}Q: {question}{TextColor.RESET}\n\t A: {answer}')

                qa_json = {}
                try:
                    recall_answer = self.get_answer_from_recall(question)
                    vectorlm_answer = self.get_answer_from_vectordbLLM(question)
                except:
                    recall_answer = "Exception raised while getting answers"
                    vectorlm_answer = "Exception raised while getting answers"

                print(f'\t{TextColor.CYAN}Recall: {recall_answer}{TextColor.RESET}')
                print(f'\t{TextColor.MAGENTA}VectorDB: {vectorlm_answer}{TextColor.RESET}')

                qa_json['question'] = question
                qa_json['answer'] = answer
                qa_json['recall'] = recall_answer
                qa_json['vectordb'] = vectorlm_answer

                sample[f'q{j}'] = qa_json

            results_data[f'sample_{i}'] = sample

            with open(self.results_file(question_group_name=group_name), 'w') as json_output:
                json.dump(results_data, json_output)



    def get_answer_from_recall(self, question):
        response = self.datastore_handler.question_system(
            question=question,
            default_chain=self.recall_default_chain,
            vanilla_chain=self.recall_vanilla_chain)
        return response
    
    def get_answer_from_vectordbLLM(self, question):
        return self.vector_lm.ask_question(question)






class BenchmarkGrader(ABC):
    def __init__(self, results_file, grader_name) -> None:
        self.benchmark_name = os.path.splitext(os.path.basename(results_file))[0]
        self.results_file = os.path.join(os.path.dirname(results_file), f'{self.benchmark_name}_copy.json')
        self.score_file = os.path.join(os.path.dirname(results_file), f'{self.benchmark_name}_score_{grader_name}.json')
        self.grader_name = grader_name

        # Duplicate the file so that we don't edit the original results
        self.continuing_grading = True
        if not os.path.exists(self.results_file):   # If the file already exists, do nothing so that we can continue grading where we left off
            shutil.copy(src=results_file, dst=self.results_file)
            self.continuing_grading = False

        with open(self.results_file, 'r') as rf:
            self.data = json.load(rf)
        
        if not self.continuing_grading:
            self.initialize_score_json()

    # Copy meta data to score json, and set score to zero for all questions    
    def initialize_score_json(self):
        samples = int(self.data['question_samples'])
        num_questions = int(self.data['num_questions'])

        score_json = {}

        score_json['grader_name'] = self.grader_name
        score_json['question_samples'] = samples
        score_json['num_questions'] = num_questions
        score_json['graded_questions'] = 0

        for i in range(num_questions):
            current = {}
            current['quesiton'] = self.data['sample_0'][f'q{i}']['question']
            current['answer'] = self.data['sample_0'][f'q{i}']['answer']
            current['recall'] = 0       # Number of samples correct
            current['vectordb'] = 0     # Number of samples correct
            current['hybrid'] = 0       # Number of samples correct

            score_json[f'q{i}'] = current
        
        with open(self.score_file, 'w') as sf:
            json.dump(score_json, sf)

    def score_responses(self):
        # Score responses in random order of model while showing the truth value

        # Create array of questions to grade in random order
        grading_pairs = []
        for sample_i in range(int(self.data['question_samples'])):
            for question_i in range(int(self.data['num_questions'])):
                try:
                    # For recall
                    grading_pairs.append({
                        "sample_key":f'sample_{sample_i}',
                        "question_key":f'q{question_i}',
                        "question":self.data[f'sample_{sample_i}'][f'q{question_i}']['question'],
                        "answer":self.data[f'sample_{sample_i}'][f'q{question_i}']['answer'],
                        "model_answer":self.data[f'sample_{sample_i}'][f'q{question_i}']['recall'],
                        "model":"recall"
                    })
                except:
                    pass

                try:
                    # For vectordb
                    grading_pairs.append({
                        "sample_key":f'sample_{sample_i}',
                        "question_key":f'q{question_i}',
                        "question":self.data[f'sample_{sample_i}'][f'q{question_i}']['question'],
                        "answer":self.data[f'sample_{sample_i}'][f'q{question_i}']['answer'],
                        "model_answer":self.data[f'sample_{sample_i}'][f'q{question_i}']['vectordb'],
                        "model":"vectordb"
                    })
                except:
                    pass

                try:
                    # For hybrid
                    grading_pairs.append({
                        "sample_key":f'sample_{sample_i}',
                        "question_key":f'q{question_i}',
                        "question":self.data[f'sample_{sample_i}'][f'q{question_i}']['question'],
                        "answer":self.data[f'sample_{sample_i}'][f'q{question_i}']['answer'],
                        "model_answer":self.data[f'sample_{sample_i}'][f'q{question_i}']['hybrid'],
                        "model":"hybrid"
                    })
                except:
                    pass
        
        random.shuffle(grading_pairs)

        self.grade_grading_pairs(grading_pairs)

        self.compute_statistics()   

        # Delete duplicate results file
        if os.path.exists(self.results_file):
            os.remove(self.results_file) 


    @abstractmethod
    def grade_grading_pairs(self, grading_pairs):
        # Grade the question/answer pairs using either GPT auto-grader, or manual grading
        pass

    def write_changes(self, sample_key, question_key, recall_score=-1, vectordb_score=-1, hybrid_score=-1):
        if (recall_score >= 0 and vectordb_score >= 0) or (recall_score >= 0 and hybrid_score >= 0) or (vectordb_score >= 0 and hybrid_score >= 0):
            raise Exception("'write_changes' should only take the score for one model at a time")

        with open(self.score_file, 'r') as sf:
            score_json = json.load(sf)
        
        # Increment total graded questions count
        graded_questions = int(score_json['graded_questions']) + 1
        score_json['graded_questions'] = graded_questions

        # Increment counters
        if recall_score >= 0:
            recall_count = int(score_json[question_key]['recall']) + recall_score
            score_json[question_key]['recall'] = recall_count

        if vectordb_score >= 0:
            vectordb_count = int(score_json[question_key]['vectordb']) + vectordb_score
            score_json[question_key]['vectordb'] = vectordb_count

        if hybrid_score >= 0:
            hybrid_count = int(score_json[question_key]['hybrid']) + hybrid_score
            score_json[question_key]['hybrid'] = hybrid_count

        # Write results to score_json
        with open(self.score_file, 'w') as sf:
            json.dump(score_json, sf)

        # Write changes to results_json
        if recall_score >= 0:
            del self.data[sample_key][question_key]['recall']
        if vectordb_score >= 0:
            del self.data[sample_key][question_key]['vectordb']
        if hybrid_score >= 0:
            del self.data[sample_key][question_key]['hybrid']
        
        if 'recall' not in self.data[sample_key][question_key] and 'vectordb' not in self.data[sample_key][question_key] and 'hybrid' not in self.data[sample_key][question_key]:
            del self.data[sample_key][question_key]
        
        with open(self.results_file, 'w') as rf:
            json.dump(self.data, rf)

    def compute_statistics(self):
        with open(self.score_file, 'r') as sf:
            score_json = json.load(sf)
        
        samples = int(score_json['question_samples'])
        num_questions = int(score_json['num_questions'])

        total_samples = int(score_json['graded_questions'])

        if total_samples / 3 != samples * num_questions:
            print("WARNING: Scoring is not complete yet")

        total_samples = samples * num_questions
        
        recall_count = 0
        vectordb_count = 0
        hybrid_count = 0
        for i in range(num_questions):
            qa_result = score_json[f'q{i}']
            recall_count += int(qa_result['recall'])
            vectordb_count += int(qa_result['vectordb'])
            hybrid_count += int(qa_result['hybrid'])
        
        recall_accuracy = float(recall_count) / (total_samples * 2)
        vectordb_accuracy = float(vectordb_count) / (total_samples * 2)
        hybrid_accuracy = float(hybrid_count) / (total_samples * 2)

        score_json['recall_correct'] = recall_count
        score_json['vectordb_correct'] = vectordb_count
        score_json['hybrid_correct'] = hybrid_count
        
        score_json['recall_accuracy'] = recall_accuracy
        score_json['vectordb_accuracy'] = vectordb_accuracy
        score_json['hybrid_accuracy'] = hybrid_accuracy

        with open(self.score_file, 'w') as sf:
            json.dump(score_json, sf)

        


class GPTBenchmarkGrader(BenchmarkGrader):
    def __init__(self, results_file, api_keys) -> None:
        if not os.path.isfile(results_file):
            raise Exception(f'GPT grader not implemented for directories with option \'g\'')
        else:
            super().__init__(results_file, grader_name='gpt')

        self.chain = self.create_chain(api_keys)

    def grade_grading_pairs(self, grading_pairs):
        while len(grading_pairs) > 0:
            gp = grading_pairs.pop()

            print(f'\rAutograding qa pairs, {len(grading_pairs)} remaining', end='')

            question = gp['question']
            reference_answer = gp['answer']
            model_answer = gp['model_answer']

            # print(f'Question:\n\t{question}\n')
            # print(f'Reference answer:\n\t{reference_answer}\n\n')
            # print(f'Model answer:\n\t{model_answer}\n\n')

            score = self.chain({
                                "question": question,
                                "correct_answers": reference_answer,
                                "answer": model_answer
                            },
                            return_only_outputs=True)['text']
            try:
                score = int(score)
            except:
                score = 0

            if gp['model'] == 'recall':
                self.write_changes(sample_key=gp['sample_key'],
                                question_key=gp['question_key'],
                                recall_score=score)
            elif gp['model'] == 'vectordb':
                self.write_changes(sample_key=gp['sample_key'],
                                question_key=gp['question_key'],
                                vectordb_score=score)
            elif gp['model'] == 'hybrid':
                self.write_changes(sample_key=gp['sample_key'],
                                question_key=gp['question_key'],
                                hybrid_score=score)
            else:
                print("unkown model")
            
    
    def create_chain(self, api_keys):
        llm = OpenAI(
            temperature=0,
            openai_api_key=api_keys['open_ai'],
            model_name="gpt-3.5-turbo"
        )

        # Question: Where does Noriko move to?
        # Correct answers: ['Ikedas home', 'tokyo', 'Tokyo']
        # student answer: Noriko moves to Tokyo.
        # score: 2

        prompt = PromptTemplate(
            input_variables=["question", "correct_answers", "answer"],
            template="""Using the question and reference answers, grade the student answers below on a scale of 0-2 where 0 is completely wrong, 1 is partially correct, and 2 is correct. If you are unsure, then give the student answer a score of 0.

            Question: Who does Tetsuzo made to pose as a client for I. C. Corp?
            Correct answers: ['There is no mentin of Tetsuzo or I. C. Corp mentioned in the clip.', 'Ikeda', 'Kumiko', 'Tetsuzo']
            student answer: The statement does not provide information on who Tetsuzo made pose as a client for I. C. Corp.
            score: 2

            Question: When does Noriko run away to Tokyo?
            Correct answers: ['December 12, 2001', 'December 10, 2001', '6am']
            student answer: Noriko impulsively leaves home and meets Ueno Station 54 in Tokyo.
            score: 0

            Question: To beat whom  thugs from the organisation arrive?
            Correct answers: ['Cannibal Club', 'Question does not make sense', 'Tetsuzo']
            student answer: The question is unclear and cannot be answered based on the given content.
            score: 2

            Question: Who refused to participate in the game?
            Correct answers: ['Steve']
            student answer: As an AI language model, I cannot answer this question without additional context. Please provide more information or clarify your question.
            score: 0

            Question: How expensive are Brandon's computer screens?
            Correct answers: ['$50 each', '$50']
            student answer: Brandon's computer screens cost $50 each and Jon's computer screens cost $120 each. There is no information about the cost of LG's ultra wide computer monitor or the specific models of Brandon's Samsung and Dell monitors.
            score: 1

            Question: What is the name of Brandon's favorite song?
            Correct answers: ['Lost in you', 'Alone', 'Elastic Heart']
            student answer: The statements do not provide information about Brandon's favorite song.
            score: 0

            Question: What is the name of Brandon's favorite song?
            Correct answers: ['Lost in you', 'Alone', 'Elastic Heart']
            student answer: Elastic Heart
            score: 2

            Question: {question}
            Correct answers: {correct_answers}
            student answer: {answer}
            score: """,
        )

        chain = LLMChain(llm=llm, prompt=prompt)

        return chain



class ManualBenchmarkGrader(BenchmarkGrader):
    def __init__(self, results_file) -> None:
        super().__init__(results_file, grader_name='manual')
    
    def grade_grading_pairs(self, grading_pairs):
        system = platform.system()

        while len(grading_pairs) > 0:
            gp = grading_pairs.pop()

            # Clear console window
            if system == 'Windows':
                os.system('cls')
            else:
                os.system('clear')

            print(f'Grade the similarity of the model answer to the reference answer on a scale of 0-2')
            print(f'\t0 -> answer is wrong, no correlatation')
            print(f'\t1 -> partially correct, contains unrelated information')
            print(f'\t2 -> correct')

            print(f'\n\n')

            question = gp['question']
            reference_answer = gp['answer']
            model_answer = gp['model_answer']

            print(f'Question:\n\t{question}\n')
            print(f'Reference answer:\n\t{reference_answer}\n\n')
            print(f'Model answer:\n\t{model_answer}\n\n')

            user_input = ""
            while user_input == "":
                new_input = input('Score: ')
                if new_input in ['0', '1', '2']:
                    user_input = new_input

            score = int(user_input)

            if gp['model'] == 'recall':
                self.write_changes(sample_key=gp['sample_key'],
                                question_key=gp['question_key'],
                                recall_score=score)
            elif gp['model'] == 'vectordb':
                self.write_changes(sample_key=gp['sample_key'],
                                question_key=gp['question_key'],
                                vectordb_score=score)
            elif gp['model'] == 'hybrid':
                self.write_changes(sample_key=gp['sample_key'],
                                question_key=gp['question_key'],
                                hybrid_score=score)
            else:
                print("unkown model")







class StasticHandler:
    def __init__(self, score_folder) -> None:
        self.score_folder = score_folder

    def compute_aggregate_score(self):
        # Fetch all json score files
        json_score_files = []
        for file in os.listdir(self.score_folder):
            if file.endswith('.json') and os.path.isfile(os.path.join(self.score_folder, file)):
                json_score_files.append(os.path.join(self.score_folder, file))
        
        recall_correct = 0
        vectordb_correct = 0
        hybrid_correct = 0
        maximum_possible_hybrid = 0
        total_questions = 0

        for file in json_score_files:
            with open(file, 'r') as json_file:
                test_json = json.load(json_file)
                
                if 'grader_name' in test_json:   # Ensure that this json is a score file
                    num_questions = int(test_json['num_questions'])
                    question_samples = int(test_json['question_samples'])
                    total_questions += num_questions * question_samples * 2 # maximum possible correct score -- Multiply by 2 because we grade on a 0-2 scale

                    for i in range(num_questions):
                        answers = test_json[f'q{i}']

                        c_recall_correct = int(answers['recall'])
                        c_vectordb_correct = int(answers['vectordb'])

                        recall_correct += c_recall_correct
                        vectordb_correct += c_vectordb_correct

                        hybrid_correct += int(answers['hybrid'])
                        maximum_possible_hybrid += max(c_recall_correct, c_vectordb_correct)
        
        aggregate_json = {}
        aggregate_json['total_question'] = total_questions

        aggregate_json['recall_correct'] = recall_correct
        aggregate_json['vectordb_correct'] = vectordb_correct
        aggregate_json['hybrid_correct'] = hybrid_correct
        aggregate_json['hybrid_maximum_correct'] = maximum_possible_hybrid

        aggregate_json['recall_accuracy'] = float(recall_correct) / total_questions
        aggregate_json['vectordb_accuracy'] = float(vectordb_correct) / total_questions
        aggregate_json['hybrid_accuracy'] = float(hybrid_correct) / total_questions
        aggregate_json['hybrid_accuracy_maximum'] = float(maximum_possible_hybrid) / total_questions

        print(f'\n{aggregate_json}\n')

        with open(os.path.join(self.score_folder, 'aggregate_score.json'), 'w') as aggregate_json_file:
            json.dump(aggregate_json, aggregate_json_file)






def load_statements_from_test_file(filename):
    with open(filename, encoding="utf-8") as f:
        file_text = f.read()
    
    lines = file_text.split("\n")

    statements = []
    statement_groups = {}
    group_name = ""
    if lines[0].strip().lower() == 'paragraph':
        # Statements are in paragraph format, simply read all

        for line in lines:
            if line.strip().lower() == 'questions':
                break

            if line.strip() != '':
                statements.append(('T', line))
        statement_groups['paragraph'] = statements

    else:
        for line in lines:
            if line.strip().lower() == 'questions':
                break

            # Check for start/end of group
            if len(line.strip()) > 0 and line.strip()[0] == '<':
                if line.strip().lower() == '<end>':
                    statement_groups[group_name] = statements
                else:
                    group_name = line.strip()[1:-1]
                statements = []
                continue

            try:
                split = line.split(':')
                truth = split[0] == 'T'
                statement = split[1].strip()
                statements.append((truth, statement))
            except:
                pass
    
    return statement_groups # dictionary with values of type --> [(truth, statement_string)]


def load_questions_from_test_file(filename):
    with open(filename, encoding="utf-8") as f:
        file_text = f.read()
    
    lines = file_text.split("\n")

    qa_pairs = []
    qa_pair_groups = {}
    group_name = ""

    start_questions = False
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.strip().lower() == 'questions':
            start_questions = True
            i+=1
            continue

        if start_questions and line.strip() != "":
            
            # Check for start/end of group
            if line.strip()[0] == '<':
                if line.strip().lower() == '<end>':
                    qa_pair_groups[group_name] = qa_pairs
                else:
                    group_name = line.strip()[1:-1]
                qa_pairs = []
                i+=1
                continue


            try:
                q_split = line.split(':')
                i+=1
                line = lines[i]
                a_split = line.split(':')
                if len(q_split) != 2 or len(a_split) != 2:
                    raise Exception('Invalid input in benchmark file')
                
                question = q_split[1].strip()
                answer = a_split[1].strip()

                qa_pairs.append((question, answer))
                
            except:
                print(f"WARNING: Invalid input found in question section of benchmark file")

        i+=1
    
    return qa_pair_groups # dictionary with values of type --> [(question, answer)]

