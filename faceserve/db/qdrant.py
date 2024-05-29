import os
from typing import Optional, Any, List, Tuple
from qdrant_client.http import models
from qdrant_client import QdrantClient

from .interface import InterfaceDatabase


class QdrantFaceDatabase(InterfaceDatabase):
    def __init__(
        self,
        collection_name: str,
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=os.getenv("QDRANT_PORT", 6333),
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._client = self.connect_client(host, port, url, api_key)
        self.collection_name = collection_name

    def connect_client(self, host, port, url, api_key):
        if url is not None and api_key is not None:
            # cloud instance
            return QdrantClient(url=url, api_key=api_key)
        elif url is not None:
            # local instance with differ url
            return QdrantClient(url=url)
        else:
            return QdrantClient(host=host, port=port)
        
    def create_colection(self, dimension=512, distance='cosine') -> None:
        # resolve distance
        if distance == 'euclidean':
            distance = models.Distance.EUCLID
        elif distance == 'dot':
            distance = models.Distance.DOT
        elif distance == 'manhattan':
            distance = models.Distance.MANHATTAN
        else: 
            distance = models.Distance.COSINE

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=dimension, distance=distance
            ),
        )

    def insert_person(
        self, 
        data_faces: List[Tuple[str, List[Any]]], 
        person_id: str, 
        group_id: str, 
    ):
        '''Insert list of faces of a person to collection'''
        self._client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=hash_id, 
                    vector=face_emb,
                    payload={
                        'person_id': person_id,
                        'group_id': group_id,
                    }
                ) for hash_id, face_emb in data_faces 
            ],
        )

    #TODO: change method for below functions
    def list_person(self):
        '''List all faces of a given person in collection'''
        return self._client.scroll(
            collection_name=self.collection_name,
            limit=1000
        )

    def delete_person(self, person_id):
        '''Delete all faces of a given person in collection'''
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="person_id",
                            match=models.MatchValue(value=f"{person_id}"),
                        ),
                    ],
                )
            ),
        )

    def insert_face(self, face_emb, face_id, person_id, group_id):    
        '''Insert a face of a person to collection'''
        self._client.upsert(
            collection_name=self.collection_name,
            points=[models.PointStruct(
                id=face_id, 
                vector=face_emb, 
                payload={
                    'person_id': person_id,
                    'group_id': group_id,
                }
            )],
        )

    def list_face(self, person_id):
        '''List all faces of a given person in collection'''
        return self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must_not=[
                    models.Filter(
                        must=[
                            models.FieldCondition(
                                key="person_id", match=models.MatchValue(value=f"{person_id}")
                            ),
                        ],
                    ),
                ],
            ),
        )

    def delete_face(self, face_id):
        self._client.delete(
            collection_name=self.collection_name, 
            points_selector=models.PointIdsList(
                points=[face_id]
            )
        )

    #TODO: consistency with interface class
    def check_face(self, face_emb, thresh):
        res = self._client.search(
            collection_name=self.collection_name, query_vector=face_emb, limit=1
        )
        if len(res) > 0:
            for r in res:
                if r.score > thresh:
                    return True
        return False
