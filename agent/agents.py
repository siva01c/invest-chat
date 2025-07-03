from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def process_data(self):
        """Process data and return a result"""
        pass
