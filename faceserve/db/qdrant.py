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
        if not self._client.collection_exists(collection_name):
            self.create_colection(
                dimension=512, distance='cosine'
            )

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

    def insert_faces(
        self, 
        face_embs: List[Tuple[str, List[Any]]], 
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
                ) for hash_id, face_emb in face_embs 
            ],
        )

    def delete_face(self, face_id: str | None, person_id: str | None, group_id: str | None):
        '''Delete a face of a given person's id or group's id in collection'''
        assert person_id is not None and group_id is not None, "person_id and group_id cannot be None at the same time"
        if group_id is not None:
            self._client.delete(
                collection_name="{collection_name}",
                points_selector=models.FilterSelector(filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="group_id",
                            match=models.MatchValue(value=f"{group_id}"),
                        ),
                    ])
                ),
            )
        elif person_id is not None:
            self._client.delete(
                collection_name="{collection_name}",
                points_selector=models.FilterSelector(filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="person_id",
                            match=models.MatchValue(value=f"{person_id}"),
                        ),
                    ])
                ),
            )
        else:
            self._client.delete(
                collection_name=self.collection_name, 
                points_selector=models.PointIdsList(
                    points=[face_id]
                )
            )

    def list_face(self, person_id: int | None, group_id: str | None):
        '''List all faces of a given person's id or group's id in collection'''
        if person_id is not None and group_id is not None:
            return self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="group_id", match=models.MatchValue(value=f"{group_id}")
                        ),
                        models.FieldCondition(
                            key="person_id", match=models.MatchValue(value=f"{person_id}")
                        ),
                    ],
                ),
            )
        elif person_id is None and group_id is not None:
            return self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="group_id", match=models.MatchValue(value=f"{group_id}")
                        ),
                    ],
                ),
            )
        elif person_id is not None and group_id is None:
            return self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="person_id", match=models.MatchValue(value=f"{person_id}")
                        ),
                    ],
                ),
            )
        return self._client.scroll(
            collection_name=self.collection_name,
            limit=1000,
            with_payload=True,
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
