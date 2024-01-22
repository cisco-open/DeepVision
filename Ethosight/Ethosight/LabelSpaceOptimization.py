from abc import ABC, abstractmethod
from .ChatGPTReasoner import ChatGPTReasoner
from tqdm import tqdm

class AbstractLabelSpaceOptimization(ABC):
    """
    Abstract class defining the interface for label space optimization.
    """

    @abstractmethod
    def optimize(self, labels):
        """
        Optimize the given label space.
        
        Parameters:
            labels (List[str]): The original labels.

        Returns:
            List[str]: The optimized label space.
        """
        pass

class SemanticRelationsOptimization(AbstractLabelSpaceOptimization):
    """
    Concrete implementation of label space optimization using semantic relations.
    """

    def __init__(self, threshold, max_labels):
        self.threshold = threshold
        self.max_labels = max_labels

    def optimize(self, labels):
        # Implement the optimization logic here.
        pass

class SemanticSimilarityOptimization(AbstractLabelSpaceOptimization):
    """
    Concrete implementation of label space optimization using semantic similarity.
    """

    def __init__(self, threshold, max_labels):
        self.threshold = threshold
        self.max_labels = max_labels
        self.reasoner = ChatGPTReasoner()

    def optimize(self, labels):
        print("Optimizing label space using semantic similarity...")
        print(f"threshold: {self.threshold}")
        print(f"max_labels: {self.max_labels}")
        print(f"labels: {labels}")

        new_labels = labels.copy()
        
        # Wrap the labels with tqdm for progress bar
        for label in tqdm(labels, desc="Optimizing Labels", unit="label"):
#            promptPositive = f"Create a comma-separated list of 10 important labels we will use to compute affinity scores between labels and an image. Please use semantic relations that will help us prove the hypothesis that an image contains <<{label}>> include only positive labels. no extraneous text or characters other than the comma separated list please."
#            promptNegative = f"Create a comma-separated list of 10 important labels we will use to compute affinity scores. Please use semantic relations that will help us disprove the hypothesis that an image contains <<{label}>> include only negative labels. no extraneous text or characters other than the comma separated list please."
            promptPositive = f"Create a comma-separated list of 10 important labels we will use to compute affinity scores between labels and an image. The context is to identify crime behavior versus normal behavior. The camera angles may be overhead or normal more horizontal security camera angles. Please use semantic relations that will help us prove the hypothesis that an image contains <<{label}>> include only positive labels. no extraneous text or characters other than the comma separated list please."
            promptNegative = f"Create a comma-separated list of 10 important labels we will use to compute affinity scores between labels and an image. The context is to identify crime behavior versus normal behavior. The camera angles may be overhead or normal more horizontal security camera angles. Please use semantic relations that will help us disprove the hypothesis that an image contains <<{label}>> include only negative labels. no extraneous text or characters other than the comma separated list please."
            
            # reason about positive cases
            positive_labels = self.reasoner.reason(prompt=promptPositive)
            print(f"positive_labels: {positive_labels}")
            
            # reason about negative cases
            negative_labels = self.reasoner.reason(prompt=promptNegative)
            print(f"negative_labels: {negative_labels}")
            
            new_labels += positive_labels + negative_labels
        
        # optimize
        alllabels = labels + new_labels
        return alllabels

class AnotherOptimizationStrategy(AbstractLabelSpaceOptimization):
    """
    Another concrete implementation of label space optimization.
    """

    def optimize(self, labels):
        # Implement the optimization logic here.
        pass
