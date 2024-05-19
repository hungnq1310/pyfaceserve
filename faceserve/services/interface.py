from typing import List, Any
from abc import ABC, abstractmethod
from pathlib import Path


class InterfaceService(ABC):
    '''Interface for Face Recognition Service'''

    @abstractmethod
    def get_face_emb(self, images: List[Any]) -> List[Any]:
        ...

    @abstractmethod 
    def check_face(self, id: str, images: List[Any]) -> List[Any]:
        ...

    @abstractmethod
    def register_face(self, id: str, images: List[Any], face_folder: Path) -> List[Any]:
        ...

    @abstractmethod
    def validate_face(self, id: str, images: List[Any], thresh: float) -> List[Any]:
        ...