from typing import List

from runner.base import Runner
from models import Document


class EmebeddingRunner(Runner):
    
    def __init__(self):
        super().__init__()
        pass
    
    async def run(
            self, 
                        documents: List[Document] 

        ):

        pass