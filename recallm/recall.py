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