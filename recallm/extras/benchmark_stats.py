import argparse
import os
import json

from benchmark import load_statements_from_test_file


def get_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_paths.append(file_path)
    return file_paths

def load_statements_as_raw_text_from_file(file):
    statements = load_statements_from_test_file(file)
    raw_text = ""
    for truth, statement in statements:
        raw_text = f'{raw_text}\n{statement}'
    raw_text = raw_text[1:]
    return raw_text

def load_benchmark_results(folder):
    files = get_file_paths(folder)
    results = {}
    for file in files:
        file_name = os.path.basename(file).split('.')[0]
        if file_name == "aggregate_score":
            continue
        if "score" not in file_name:
            continue
        
        with open(file, 'r') as f:
            score_json = json.load(f)
            results[file_name.split('_')[0]] = score_json
    
    return results


def get_average_length(stats, key):
    sum = 0
    for test_name, stat in stats:
        sum += stat[key]
    return float(sum) / len(stats)



# For all questions where recall was right but vectordb was wrong:
#   get the type of question, length of passage, length of question on average

if __name__ == "__main__":
    ####       LOAD DATA           ##################################################################
    benchmark_results_folder = "benchmarks/duoRC_50_percent"
    datasets_folder = "datasets/duorc"

    benchmark_results_folder = os.path.join(os.getcwd(), benchmark_results_folder)
    datasets_folder = os.path.join(os.getcwd(), datasets_folder)

    duorc_sources = {}
    for file in get_file_paths(datasets_folder):
        duorc_sources[os.path.basename(file).split('.')[0]] = load_statements_as_raw_text_from_file(file)

    benchmark_results = load_benchmark_results(benchmark_results_folder)
    ####       LOAD DATA           ##################################################################


    recall_superior = []
    vectordb_superior = []

    for test_name in benchmark_results.keys():
        source = duorc_sources[test_name]
        result = benchmark_results[test_name]

        num_questions = int(result['num_questions'])
        for i in range(num_questions):
            qi_result = result[f'q{i}']
            
            recall_score = int(qi_result['recall'])
            vectordb_score = int(qi_result['vectordb'])

            question_stats = {
                "question": qi_result['quesiton'],
                "question_length": len(qi_result['quesiton']),
                "source_length": len(source)
            }

            if recall_score == 2 and vectordb_score == 0:
                recall_superior.append((test_name, question_stats))
            
            if recall_score == 0 and vectordb_score == 2:
                vectordb_superior.append((test_name, question_stats))






    print("Recall superior:")
    recall_questions = []
    for test_name, stat in recall_superior:
        recall_questions.append(stat["question"])
        # recall_questions.append(f'{test_name}\t - {stat["question"]}')
    recall_questions = sorted(recall_questions)
    for question in recall_questions:
        print(question)
    recall_average_source_length = get_average_length(recall_superior, key='source_length')
    recall_average_question_length = get_average_length(recall_superior, key='question_length')
    print(f'RecallM superiority: Average source length {recall_average_source_length}')
    print(f'RecallM superiority: Average question length {recall_average_question_length}')
    
    print(f'\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')

    print("VectorDB superior")
    vectordb_questions = []
    for test_name, stat in vectordb_superior:
        vectordb_questions.append(stat["question"])
        # vectordb_questions.append(f'{test_name}\t - {stat["question"]}')
    vectordb_questions = sorted(vectordb_questions)
    for question in vectordb_questions:
        print(question)
    vectordb_average_source_length = get_average_length(vectordb_superior, key='source_length')
    vectordb_average_question_length = get_average_length(vectordb_superior, key='question_length')
    print(f'VectorDB superiority: Average source length {vectordb_average_source_length}')
    print(f'VectorDB superiority: Average question length {vectordb_average_question_length}')


    # print(f'Source:\n{source}\n\n\n')
    # print(f'Results:\n{result}\n')
