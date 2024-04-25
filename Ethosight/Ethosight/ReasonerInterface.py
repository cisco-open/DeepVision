
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

from abc import ABC, abstractmethod

class AbstractReasoner(ABC):
    """
    Abstract class that defines the common interface for all reasoner types
    """
    DEFAULT_USE_CASE_PROMPT = "health, safety, security and environment"

    def __init__(self, use_case_prompt = DEFAULT_USE_CASE_PROMPT):
        self.use_case_prompt = use_case_prompt
        self.initialize(self.use_case_prompt)

    def set_use_case_prompt(self, use_case_prompt):
            self.use_case_prompt = use_case_prompt

    @abstractmethod
    def initialize(self, use_case_prompt):
        """
        Abstract method to initialize the reasoner
        :param use_case_prompt: The use case prompt to focus the reasoner
        """
        pass

    def blank_slate_prompt(self):
        """
        Generate the blank slate (bootstrap) prompt
        """
        return f"Create a comma-separated list of 30 important labels we will use to analyze an image via an affinity metric focused on the use case {self.use_case_prompt}. no extraneous text or characters other than the comma separated list."

    def iterative_learning_prompt(self, label_affinities):
        """
        Generate the iterative learning prompt
        """
        assert label_affinities is not None, "Must provide label affinities for iterative reasoning."
        return f"Given the following affinity scores: {label_affinities}, analyze these affinity scores for an image focusing on the use case {self.use_case_prompt}. Based on your analysis, create a comma-separated list of important labels for a more insightful analysis within the context of {self.use_case_prompt}. no extraneous text or characters other than the comma separated list."

    def get_prompt(self, label_affinities=None, prompt_type='blank_slate'):
        """
        Get the prompt for the given prompt type
        """
        if prompt_type == 'blank_slate':
            prompt = self.blank_slate_prompt()
        elif prompt_type == 'iterative':
            prompt = self.iterative_learning_prompt(label_affinities)
        else:
            raise ValueError(f"Unsupported prompt type: {prompt_type}")
        return prompt

    @abstractmethod
    def reason(self, label_affinities=None, prompt_type='blank_slate'):
        """
        Abstract method to perform reasoning on the given input data
        :param label_affinities: The label affinities to use for iterative learning
        :param prompt_type: The type of prompt to use ('blank_slate' or 'iterative_learning')
        :return: new list of labels derived from the reasoning process
        """
        pass


class OpenNARSReasoner(AbstractReasoner):
    def initialize(self, use_case_prompt):
        """
        Initialize the interface to OpenNARS
        """
        # Initialize parameters specific to OpenNARS API/interface.
        pass

    def reason(self, label_affinities=None, prompt_type='blank_slate'):
        """
        Perform reasoning using OpenNARS
        :return: new list of labels derived from the OpenNARS reasoning process
        """
        # Call OpenNARS API/interface with the input_data and the correct prompt.
        pass
