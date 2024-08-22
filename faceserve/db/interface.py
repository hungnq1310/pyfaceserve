from abc import ABC, abstractmethod

class InterfaceDatabase(ABC):
    '''Interface for Database'''

    @abstractmethod
    def connect_client(self, host, port, **kwargs):
        return NotImplemented

    @abstractmethod
    def create_colection(self, collection_name, **kwargs):
        return NotImplemented

    @abstractmethod
    def insert_faces(self, face_embs, **kwargs):
        return NotImplemented

    @abstractmethod
    def list_faces(self, **kwargs):
        return NotImplemented

    @abstractmethod
    def delete_face(self, face_id, **kwargs):
        return

    @abstractmethod
    def check_face(self, face_emb, **kwargs):
        return
