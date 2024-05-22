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
    def insert_person(self, person_id, **kwargs):
        return NotImplemented

    @abstractmethod
    def list_person(self):
        return NotImplemented

    @abstractmethod
    def delete_person(self, person_id, **kwargs):
        return NotImplemented

    @abstractmethod
    def insert_face(self, person_id, face_id, face_emb, **kwargs):
        return NotImplemented

    @abstractmethod
    def list_face(self, person_id, **kwargs):
        return NotImplemented

    @abstractmethod
    def delete_face(self, person_id, face_id, **kwargs):
        return

    @abstractmethod
    def check_face(self, person_id, face_emb, **kwargs):
        return
