from typing import List, Any
from abc import abstractmethod
from pathlib import Path
from pydantic import BaseModel, PrivateAttr
from PIL.Image import Image


class InterfaceModel(BaseModel):
    '''Interface for Model'''
    _model: Any = PrivateAttr()

    @abstractmethod
    def load_model(self, path: Path, **kwargs) -> Any:
        ...

    @abstractmethod 
    def preprocess(self, image: Image, **kwargs) -> Any:
        ...

    @abstractmethod
    def postprocess(self, prediction: Any, **kwargs) -> Any:
        ...

    @abstractmethod
    def inference(self, image: Image, **kwargs) -> Any:
        ...

    @abstractmethod
    def batch_inference(self, images: List[Image], **kwargs) -> List[Any]:
        ...