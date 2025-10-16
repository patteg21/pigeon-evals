from abc import ABC

class RunnerBase(ABC):

    def __init__(self):
        super().__init__()

    def run(self):
        raise NotImplementedError