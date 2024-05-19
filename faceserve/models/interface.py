from typing import List, Any
from abc import abstractmethod
from pathlib import Path
from pydantic import BaseModel


class InterfaceModel(BaseModel):
    '''Interface for Model'''

    @abstractmethod
    def load_model(self, path: Path, **kwargs) -> Any:
        ...

    @abstractmethod 
    def preprocess(self, image: Any, **kwargs) -> Any:
        ...

    @abstractmethod
    def postprocess(self, image: Any, **kwargs) -> Any:
        ...

    @abstractmethod
    def get_features(self, image: Any, **kwargs) -> Any:
        ...

    @abstractmethod
    def forward(self, images: List[Any], **kwargs) -> List[Any]:
        ...