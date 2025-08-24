from pydantic import BaseModel

class RerankerBase(BaseModel):
    """ A Crossencoder Reranker """

    def __init__(self):
        pass

    def rerank(self, vectors, query):
        raise NotImplementedError
    
