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
    def inference(self, image: Any, **kwargs) -> Any:
        ...

    @abstractmethod
    def batch_inference(self, images: List[Any], **kwargs) -> List[Any]:
        ...