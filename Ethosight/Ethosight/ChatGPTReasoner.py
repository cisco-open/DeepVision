
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

import os
import openai
import requests
from .ReasonerInterface import AbstractReasoner
import time

class ChatGPTReasoner(AbstractReasoner):
    def initialize(self, use_case_prompt=AbstractReasoner.DEFAULT_USE_CASE_PROMPT):
        """
        Initialize the interface to ChatGPT
        """
        self.use_case_prompt = use_case_prompt
        # Load the OpenAI API key from the environment
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("Missing OpenAI API key")

    def reason(self, label_affinities=None, prompt_type='blank_slate', prompt=None, max_tokens=500):
        assert self.use_case_prompt is not None, "Must initialize reasoner with use case prompt before reasoning."
        if prompt is None:
            prompt = self.get_prompt(label_affinities, prompt_type)

        # Call to the OpenAI API
        response = self.call_chat_api(prompt, max_tokens)
        new_labels = response['choices'][-1]['message']['content'].split(", ")
        new_labels = [label.strip() for label in new_labels]  # Remove leading/trailing whitespace

        return new_labels

    def call_chat_api(self, prompt, max_tokens, max_retries=10, retry_delay=1):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}",
        }

        data = {
           # "model": "gpt-3.5-turbo",  # Choose the model based on the balance between cost and quality needed
            "model": "gpt-4",
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        for attempt in range(max_retries):
            try:
                response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    retry_delay *= 2  # Exponential backoff
                    print(
                        f"Retrying OpenAI API call. Attempt {attempt + 1} of {max_retries}. Next retry in {retry_delay} seconds.")
                    time.sleep(retry_delay)
                else:
                    raise e
