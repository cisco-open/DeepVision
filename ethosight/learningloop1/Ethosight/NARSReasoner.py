import os
import sys
with open("NARSReasoner_knowledge.nal","r") as f:
    DEFAULT_KNOWLEDGE=f.read()
cwd = os.getcwd()
os.chdir('./NARS-GPT/OpenNARS-for-Applications/misc/Python/')
sys.path.append('./')
import NAR
from ReasonerInterface import AbstractReasoner
os.chdir(cwd)

class NARSReasoner(AbstractReasoner):
    top_labels = 3
    score_threshold = 0.01
    reasoning_steps = 100
    def initialize(self, use_case_prompt):
        """
        Initialize the interface to OpenNARS
        """
        # Initialize parameters specific to OpenNARS API/interface.
        pass

    def labelsFromDerivations(self, ret):
        found_labels = set([])
        derivations = ret["derivations"]
        for derivation in derivations:
            if " --> label>" in derivation["term"]:
                label = derivation["term"].split(" --> label>")[0][1:]
                if " " not in label and "$" not in label and "#" not in label and "<" not in label:
                    found_labels.add(label)
        return found_labels

    def reason(self, label_affinities=None, prompt_type='iterative'):
        """
        Perform reasoning using ONA
        :return: new list of labels derived from the ONA reasoning process
        """
        # Call OpenNARS API/interface with the input_data and the correct prompt.
        NAR.AddInput("*reset")
        for line in DEFAULT_KNOWLEDGE.split("\n"):
            if line.strip() != "" and not line.startswith("//"):
                NAR.AddInput(line, Print=False)
        labels, scores = label_affinities["labels"], label_affinities["scores"]
        found_labels = set(labels)
        for i in range(len(labels)):
            if i >= self.top_labels:
                break
            label, score = labels[i].replace(" ", "_"), scores[i]
            if score > self.score_threshold:
                found_labels = found_labels | self.labelsFromDerivations(NAR.AddInput(f"<{label} --> label>. {{1.0 {score}}}", Print=False))
        found_labels = found_labels | self.labelsFromDerivations(NAR.AddInput(str(self.reasoning_steps), Print=False))
        return list(found_labels)

if __name__ == "__main__":
    #TEST
    reasoner = NARSReasoner()
    label_affinities = {"labels": ["theft", "homicide"], "scores": [0.6, 0.2]}
    print(reasoner.reason(label_affinities))

