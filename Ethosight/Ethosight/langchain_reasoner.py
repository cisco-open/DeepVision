
# Copyright 2022 Cisco Systems, Inc. and its affiliates
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

from .ReasonerInterface import AbstractReasoner
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser


class LangchainReasoner(AbstractReasoner):
    def initialize(self, use_case_prompt):
        """
        Initialize the interface to Langchain
        """
        # Initialize parameters specific to Langchain API/interface.
        pass

    def reason(self, label_affinities=None, prompt_type='blank_slate'):
        """
        Perform reasoning using Langchain
        :return: new list of labels derived from the Langchain reasoning process
        """
        prompt = self.get_prompt(label_affinities, prompt_type)

        class CommaSeparatedListOutputParser(BaseOutputParser):
            """Parse the output of an LLM call to a comma-separated list."""

            def parse(self, text: str):
                """Parse the output of an LLM call."""
                new_labels = text.strip().split(", ")
                new_labels = [label.strip() for label in new_labels]
                return new_labels

        # Use ChatOpenAI model as LLM. Can be replaced with any other LLMs or Chat Models from langchain.
        chain = LLMChain(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0),
            prompt=PromptTemplate(input_variables=["prompt"],template="{prompt}"),
            output_parser=CommaSeparatedListOutputParser(),
        )
        return chain.run(prompt)
