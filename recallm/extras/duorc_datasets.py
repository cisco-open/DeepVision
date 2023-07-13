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

# This program uses the duorc dataset to create knowledge update and QA pairs in
# the custom benchmarking format
from datasets import load_dataset
import os

class DuoRCLoader:
    def __init__(self, percent=100) -> None:
        self.duorc_train = load_dataset('duorc', 'ParaphraseRC', split=f'test[:{percent}%]')

        print(f'Loaded DuoRC ParaphraseRC dataset with {self.duorc_train.num_rows} rows')

        self.tests = {}
        for row in self.duorc_train:
            if row['no_answer'] == True:
                continue

            plot_name = row['title']
            if plot_name not in self.tests:
                self.tests[plot_name] = {}
                self.tests[plot_name]['title'] = plot_name
                self.tests[plot_name]['plot'] = row['plot']
                self.tests[plot_name]['qa_pairs'] = []
            
            self.tests[plot_name]['qa_pairs'].append({'question':row['question'], 'answers':row['answers']})
    
    def print_test(self, test):
        test_name = test['title']
        plot = test['plot']
        plot = plot[:min(200, len(plot)-1)]
        print(f'Test: {test_name}')
        print(f'\tPlot:\n\t\t{plot}')
        print(f'Questions:')
        for qa_pair in test['qa_pairs']:
            question = qa_pair['question']
            answers = qa_pair['answers']
            print(f'\tq: {question}\n\ta: {answers}\n')
    
    def write_benchmark_files(self, destination_folder):
        tests = list(self.tests.values())

        destination_folder = os.path.join(os.getcwd(), destination_folder)
        if not os.path.exists(destination_folder):
            os.mkdir(destination_folder)

        for i, test in enumerate(tests):
            print(f'\rWriting benchmark file {i+1}/{len(tests)}', end='')
            test_file = os.path.join(destination_folder, f'test-{i}.txt')
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(f'PARAGRAPH\n')
                f.write(f'{test["plot"]}\n')
                f.write('\n\n')
                f.write('QUESTIONS\n')
                
                for qa_pair in test['qa_pairs']:
                    question = qa_pair['question']
                    answers = qa_pair['answers']
                    f.write(f'q: {question}\n')
                    f.write(f'a: {answers}\n')
                    f.write('\n')

if __name__ == "__main__":
    duorc_loader = DuoRCLoader(percent=50)

    duorc_loader.write_benchmark_files(destination_folder='datasets/duorc')