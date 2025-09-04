from abc import ABC

class LLMBaseClient(ABC):
    
    def __init__(self):
        super().__init__()
        self.model = None

    def invoke(self):
        raise NotImplementedError
    
    def _count_tokens(self):
        raise NotImplementedError