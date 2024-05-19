from typing import List, Any
from abc import ABC, abstractmethod
from pathlib import Path


class InterfaceModel(ABC):
    '''Interface for Model'''

    @abstractmethod
    def load_model(self, path: Path) -> Any:
        ...

    @abstractmethod 
    def preprocess(self, image: Any) -> Any:
        ...

    @abstractmethod
    def postprocess(self, image: Any) -> Any:
        ...

    @abstractmethod
    def get_features(self, image: Any) -> Any:
        ...

    @abstractmethod
    def forward(self, images: List[Any]) -> List[Any]:
        ...