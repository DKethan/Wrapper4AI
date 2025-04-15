from abc import ABC, abstractmethod
from typing import List, Dict

class BaseHandler(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def _generate_response(self, messages: List[Dict]) -> str:
        pass

    @abstractmethod
    def _generate_title(self, prompt: str) -> str:
        pass

    @abstractmethod
    def _truncate_history(self, messages: List[Dict]) -> List[Dict]:
        pass

    @abstractmethod
    def _count_tokens(self, messages: List[Dict]) -> int:
        pass
