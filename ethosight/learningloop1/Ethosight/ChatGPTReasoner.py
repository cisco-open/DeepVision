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

    def reason(self, label_affinities=None, prompt_type='blank_slate',prompt=None):
        assert self.use_case_prompt is not None, "Must initialize reasoner with use case prompt before reasoning."
        if prompt is None:
            prompt = self.get_prompt(label_affinities, prompt_type)

        # Call to the OpenAI API
        response = self.call_chat_api(prompt)
        new_labels = response['choices'][-1]['message']['content'].split(", ")
        new_labels = [label.strip() for label in new_labels]  # Remove leading/trailing whitespace

        return new_labels

    def call_chat_api(self, prompt, max_retries=10, retry_delay=5):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}",
        }

        data = {
            #"model": "gpt-3.5-turbo",
            "model": "gpt-4",
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
                if attempt < max_retries - 1:  # i.e. not the last attempt
                    time.sleep(retry_delay)  # wait for some time before retrying
                    print(f"Retrying OpenAI API call. Attempt {attempt + 1} of {max_retries}.")
                    continue
                else:
                    raise e  # re-raise the exception if this was the last attempt

        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
        response.raise_for_status()

        return response.json()
