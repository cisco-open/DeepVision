from datastore_gen import get_chroma_collection

from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI

class SimpleVectorLLM:
    def __init__(self, api_keys) -> None:
        self.collection = get_chroma_collection(reset_collection=True)
        self.collection_id_counter = 1

        llm = OpenAI(
            temperature=0,
            openai_api_key=api_keys['open_ai'],
            model_name="gpt-3.5-turbo"
        )

        self.chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm,
            chain_type="stuff",
            retriever=self.collection.as_retriever()
        )
    
    def ask_question(self, question):
        return self.chain(
            {"question": question},
            return_only_outputs=True
        )['answer']

    def load_knowledge(self, text):
        self.collection.add_texts(
            texts=[text],
            metadatas=[{"source": f"k-{self.collection_id_counter}"}],
            ids=[f'{self.collection_id_counter}']
        )
        self.collection_id_counter += 1