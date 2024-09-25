from typing import List, Any, Tuple
from abc import ABC, abstractmethod

class InterfaceService(ABC):
    '''Interface for Face Recognition Service'''

    @abstractmethod
    def get_face_emb(self, images: List[Any], **kwargs) -> List[Any]:
        ...

    @abstractmethod 
    def check_face(self, images: List[Any], **kwargs) -> List[Any]:
        ...

    @abstractmethod
    def register_face(self, images: List[Any], **kwargs) -> List[Any]:
        ...

    @abstractmethod
    def validate_face(self, images: List[Any], **kwargs) -> Tuple[List[Any], List[Any]]:
        ...