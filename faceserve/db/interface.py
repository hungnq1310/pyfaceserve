from abc import ABC, abstractmethod

class InterfaceDatabase(ABC):
    '''Interface for Database'''

    @abstractmethod
    def create_person(self, person_id):
        return NotImplemented

    @abstractmethod
    def list_person(self):
        return NotImplemented

    @abstractmethod
    def delete_person(self, person_id):
        return NotImplemented

    @abstractmethod
    def insert_face(self, person_id, face_id, face_emb):
        return NotImplemented

    @abstractmethod
    def list_face(self, person_id):
        return NotImplemented

    @abstractmethod
    def delete_face(self, person_id, face_id):
        return

    @abstractmethod
    def check_face(self, person_id, face_emb, thresh):
        return
