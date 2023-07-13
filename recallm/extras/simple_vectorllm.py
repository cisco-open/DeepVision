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

# from datastore_gen import get_chroma_collection

# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain import OpenAI

# class SimpleVectorLLM:
#     def __init__(self, api_keys) -> None:
#         self.collection = get_chroma_collection(reset_collection=True)
#         self.collection_id_counter = 1

#         llm = OpenAI(
#             temperature=0,
#             openai_api_key=api_keys['open_ai'],
#             model_name="gpt-3.5-turbo"
#         )

#         self.chain = RetrievalQAWithSourcesChain.from_chain_type(
#             llm,
#             chain_type="stuff",
#             retriever=self.collection.as_retriever()
#         )
    
#     def ask_question(self, question):
#         return self.chain(
#             {"question": question},
#             return_only_outputs=True
#         )['answer']

#     def load_knowledge(self, text):
#         self.collection.add_texts(
#             texts=[text],
#             metadatas=[{"source": f"k-{self.collection_id_counter}"}],
#             ids=[f'{self.collection_id_counter}']
#         )
#         self.collection_id_counter += 1