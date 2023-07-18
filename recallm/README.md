# RecallM

RecallM attempts to bring us one step closer to achieving Artficial General Intelligence (AGI) by providing Large Language Models (LLMs) with an **adaptable and updatable long-term memory**.

RecallM is intended to work in the same manner as a typical chatbot, while retaining any new information provided to it. Users are able to update the knowledge of the system in natural language.

RecallM offers a complete solution encapsulated into a single python object using the RecallM architecture with GPT via LangChain. However, we also offer a LangChain document retriever implementation of RecallM to allow for complete modularity and flexibility with other LangChain applications.

[Click here](https://arxiv.org/abs/2307.02738) to see complete details about the RecallM architecture.


## Installation
- Clone the repository and submodules:
```
> git clone git@github.com:cisco-open/DeepVision.git
> cd DeepVision/recallm
> git submodule update --init --recursive
```
- Create the api keys json file by executing the following command inside the recallm folder. Replace <API_KEY> with your OpenAI api key:
```
> echo {"open_ai":"<API_KEY>"} > api_keys.json
```
- Ensure that you have Docker installed, then create the Docker container:
```
> docker compose up -d
```
- Pip install required packages: (Ideally in a new environment)
```
> pip install -r requirements.txt
```

## Usage
Ensure that the docker container is running, then use the RecallM system as follows:

```python
from recall import RecallM

# Initialize the RecallM instance
recallm = RecallM()

# Add knowledge from a string
recallm.update(knowledge="Brandon loves coffee")

# Add knowledge from a web page (Warning: this might miss some text that is dynamically loaded)
recallm.update_from_url(url='https://paperswithcode.com/paper/recallm-an-architecture-for-temporal-context')

# Add knowledge from a text file
recallm.update_from_file(file='./datasets/other/state_of_the_union.txt')

question = "What do you know about Brandon?"
answer = recallm.question(question)
print(answer)

# Always close the Docker database connection when finished
recallm.close()
```

### Terminal Interface 
Alternatively, to quickly start using RecallM you can launch the interactive terminal interface using:

```
> python recall_terminal.py
```

### Usage with Custom LangChain Application
RecallM also implements the BaseRetriever class from LangChain so that it can be implemented in other LangChain applications:

```python
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI

from recall import RecallM

OPEN_AI_API_KEY = '....'

recallm = RecallM(openai_key=OPEN_AI_API_KEY)
recallm.update(knowledge="Brandon loves coffee")

chat = ChatOpenAI(temperature=0,
                   openai_api_key=OPEN_AI_API_KEY,
                   model_name="gpt-3.5-turbo")

chain = RetrievalQAWithSourcesChain.from_chain_type(
    chat,
    chain_type="stuff",
    retriever=recallm.as_retriever()
)

question = "What do you know about Brandon?"
answer = chain({"question": question}, return_only_outputs=True)['answer']
print(answer)

# Always close the Docker database connection when finished
recallm.close()
```

### Resetting the System's Knowledge
```python
recallm.reset_knowledge()
```

### Accessing the Knolwedge Graph
Please note that by simply running Neo4J desktop instead of the docker container, you can visualize the knowledge graph being created with RecallM.

## Citation
Please cite our work using:
```
@misc{kynoch2023recallm,
      title={RecallM: An Architecture for Temporal Context Understanding and Question Answering}, 
      author={Brandon Kynoch and Hugo Latapie},
      year={2023},
      eprint={2307.02738},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
***