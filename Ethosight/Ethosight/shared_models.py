# shared_models.py

from pydantic import BaseModel
from typing import List

class LabelEmbeddingInput(BaseModel): 
    labels: List[str]
    batch_size: int
