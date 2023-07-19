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

from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class HybridMemoryLLM:
    def __init__(self, api_keys) -> None:
        self.api_keys = api_keys

        self.chain = HybridMemoryLLM.create_combine_chain(api_keys)

    # Use GPT to choose the more correct answer between the RecallLM model and
    #   vectordb answer
    def combine(self, question, recall_answer, vector_answer):
        combined_response_label = self.chain({
            "question": question,
            "recall": recall_answer,
            "vectordb": vector_answer
        }, return_only_outputs=True)['text']

        combined_response_label = combined_response_label.lower()
        if combined_response_label == 'a':
            return recall_answer
        if combined_response_label == 'b':
            return vector_answer
        else:
            return vector_answer
    
    def create_combine_chain(api_keys):
        llm = OpenAI(
            temperature=0.01,
            openai_api_key=api_keys['open_ai'],
            model_name="gpt-3.5-turbo"
        )

        prompt = PromptTemplate(
            input_variables=["question", "recall", "vectordb"],
            template="""These questions and answers come from an unseen text. Choose the answer to the question that responds with the highest degree of confidence and most attention to detail. Respond with only 'A' or 'B', if you are unsure, respond with 'N':

            Question: The Silver Surfer is imprisoned where?
            Option A: The Silver Surfer is imprisoned in a hideout in Siberia for further study by General Harding.
            Option B: The Silver Surfer is imprisoned in Siberia for further study.
            Better answer: A

            Question: Who is the only prisoner in the camp?
            Option A: As an AI language model, I do not have access to the context of the statement. Please provide more information or context so I can assist you better.
            Option B: Ballard and Williams are the only survivors in the camp.
            Better answer: B

            Question: Who has colonized Mars 200 years in the future?
            Option A: Humans have colonized Mars 200 years in the future.
            Option B: There is no information about who has colonized Mars 200 years in the future in the given content.
            Better answer: A

            Question: Where did the man, X, claim to have met the woman at?
            Option A: X claimed to have met the woman, A, at a hotel (either the one they were currently in or a different one).
            Option B: The content does not provide information on where the man, X, claimed to have met the woman.
            Better answer: A

            Question: What does Rocinante hate?
            Option A: As an AI language model, I do not have access to the current context of the statement. However, if the statement is referring to Rocinante, the horse from the novel \"Don Quixote\" by Miguel de Cervantes, then it is not mentioned in the book that Rocinante hates anything.
            Option B: Rocinante hates leaving his stable.
            Better answer: B

            Question: What is Max's day job?
            Option A: There is no information about Max's day job in the given content.\nSOURCES:
            Option B: Max's day job is a wedding video cameraman.
            Better answer: B

            Question: {question}
            Option A: {recall}
            Option B: {vectordb}
            Best answer: """,
        )

        chain = LLMChain(llm=llm, prompt=prompt)

        return chain