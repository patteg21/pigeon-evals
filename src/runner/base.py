from abc import ABC

class Runner(ABC):

    def __init__(self):
        super().__init__()

    def run(self):
        raise NotImplementedError