from abc import ABC, abstractmethod
from typing import List, Dict

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Return top-k retrieved documents with text and score."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
