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
