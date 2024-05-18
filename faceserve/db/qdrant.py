import os
from qdrant_client.http import models
from qdrant_client import QdrantClient
from .abc import FaceDatabaseInterface


class QdrantFaceDatabase(FaceDatabaseInterface):
    def __init__(
        self,
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=os.getenv("QDRANT_PORT", 6333),
    ) -> None:
        self._client = QdrantClient(host=host, port=port)

    def create_person(self, person_id):
        self._client.create_collection(
            collection_name=person_id,
            vectors_config=models.VectorParams(
                size=512, distance=models.Distance.COSINE
            ),
        )

    def list_person(self):
        ids = self._client.get_collections().collections
        ids = [i.name for i in ids]
        return ids

    def delete_person(self, person_id):
        self._client.delete_collection(person_id)

    def insert_face(self, person_id, face_id, face_emb):
        self._client.upsert(
            collection_name=person_id,
            points=[models.PointStruct(id=face_id, vector=face_emb)],
        )

    def list_face(self, person_id):
        return self._client.scroll(collection_name=person_id, limit=1000)

    def delete_face(self, person_id, face_id):
        self._client.delete_vectors(
            collection_name=person_id, points_selector=models.PointIdsList([face_id])
        )

    def check_face(self, person_id, face_emb, thresh):
        res = self._client.search(
            collection_name=person_id, query_vector=face_emb, limit=1
        )
        if len(res) > 0:
            for r in res:
                if r.score > thresh:
                    return True
        return False
