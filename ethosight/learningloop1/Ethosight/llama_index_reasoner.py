from .ReasonerInterface import AbstractReasoner
from llama_index import (
    Document,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import OpenAI
from llama_index.llms import HuggingFaceLLM

class LlamaIndexReasoner(AbstractReasoner):
    def initialize(self, use_case_prompt=AbstractReasoner.DEFAULT_USE_CASE_PROMPT):
        """
        Initialize the interface to LlamaIndex
        """
        # todo: make llm model configurable
        model = "text-davinci-003"
        #model = "camel-5b-hf"
        # define LLM
        llm = OpenAI(temperature=0.0, model=model)
        #llm = HuggingFaceLLM(model_name=model)
        service_context = ServiceContext.from_defaults(llm=llm)

        # build index
        document = Document(text=self.blank_slate_prompt())
        index = VectorStoreIndex.from_documents([document], service_context=service_context)

        # build query engine
        self.query_engine = index.as_query_engine()


    def reason(self, label_affinities=None, prompt_type='blank_slate'):
        """
        Perform reasoning using LlamaIndex
        :return: new list of labels derived from the LlamaIndex reasoning process
        """
        prompt = self.get_prompt(label_affinities, prompt_type)

        response = self.query_engine.query(prompt)
        new_labels = response.response[1:-1].split(", ") # Remove '\n' and '.' from response
        new_labels = [label.strip() for label in new_labels]  # Remove leading/trailing whitespace
        return new_labels
