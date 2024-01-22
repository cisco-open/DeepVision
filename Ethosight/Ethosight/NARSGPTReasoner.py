import os
import sys
import torch
import openai
import time
import numpy as np
cwd = os.getcwd()
os.chdir('./NARS-GPT/')
sys.path.append('./')
import NarsGPT as NAR
os.chdir(cwd)
os.system('rm -f mem.json')
NAR.AddInput("*volume=0")

from ReasonerInterface import AbstractReasoner

class NARSGPTReasoner(AbstractReasoner):
    def initialize(self, use_case_prompt=""):
        """
        Initialize the interface to NARS-GPT
        use_case_prompt serves as the background knowledge
        """
        # Load the OpenAI API key from the environment
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            raise ValueError("Missing OpenAI API key")

    def promptGPT(self, prompt):
        while True:
            try:
                response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[ {"role": "user", "content": prompt}], max_tokens=200, temperature=0)
                ret = response['choices'][0]['message']['content']
            except Exception as e:
                print("Error: API call failed, will try repeating it in 10 seconds!", str(e))
                time.sleep(10) #wait 10 seconds
                continue
            break
        return ret

    def summarize(self, affinity_scores, top_labels = 3, score_threshold=0.01):
        labels, scores = affinity_scores["labels"], affinity_scores["scores"]
        prompt="Please transform the following labels into a sentences which uses all labels:"
        prompt_ext = ""
        for i in range(len(labels)):
            if i >= top_labels:
                break
            label, score = labels[i].replace(" ", "_"), scores[i]
            if score > score_threshold:
                prompt_ext = f"label={label}\n" + prompt_ext
        prompt = prompt + "\n" + prompt_ext
        summary = self.promptGPT(prompt)
        print("SUMMARY IS:\n", summary)
        return summary

    def labels_from_sentence(self, inp):
        prompt = f'extract category labels from the following sentence: "{inp}", represent the labels as a comma-separated list of strings L=[...], and also include a few members of each category as additional label in L.'
        newLabelCandidates = self.promptGPT(prompt).replace(" ","").split("\n")
        for L in newLabelCandidates:
            if L.startswith("L=[") and "]" in L.split("L=")[1]:
                return [l.strip() for l in L.split("L=[")[1].split("]")[0].split(",")]
        return None

    def reason(self, inp, prompt_type=None):
        ret = NAR.AddInput(inp, PrintAnswer=False, Print=False, PrintInputSentenceOverride=True, PrintInputSentenceOverrideValue=False)
        print("NARS-GPT:", ret["GPT_Answer"])
        return ret

