import hashlib
import numpy as np
from fastapi import HTTPException, status
from pathlib import Path
from PIL import Image
from typing import Tuple, Any, List

from faceserve.db.interface import InterfaceDatabase
from faceserve.models.interface import InterfaceModel

from .interface import InterfaceService


class FaceServiceV1(InterfaceService):
    def __init__(
        self,
        detection: InterfaceModel,
        detection_thresh: float,
        recognition: InterfaceModel,
        recognition_thresh: float,
        facedb: InterfaceDatabase,
    ) -> None:
        self.facedb = facedb
        self.detection = detection
        self.detection_thresh = detection_thresh
        self.recognition = recognition
        self.recognition_thresh = recognition_thresh

    def get_face_emb(self, image: Image.Image) -> Tuple[list, list]:
        """Get face embedding from face image
        
        Args:
            image (Image.Image): face image
            
        Returns:
            Tuple[Any, Any]: face bounding boxes, face embeddings
        """
        image = np.array(image)  # type: ignore
        boxes, _, _, kpts = self.detection.inference(
            image, get_layer="face", det_thres=self.detection_thresh
        )
        if len(boxes) == 1:
            res = self.recognition.inference(image, boxes, kpts) # type: ignore
            return boxes[0], res[0]
        return [], []

    def validate_face(self, images: List[Image.Image], person_id: str|None, group_id: str|None) -> Tuple[List[Any], List[Image.Image]]:
        """Validate face images

        Args:
            id (str): user id
            images (List[Image.Image]): list of face images
        
        Returns:
            Tuple[List[Any], List[Image.Image]]: list of face embeddings, list of face images
        """
        # get temp result
        res, imgs, temp = [], [], [self.get_face_emb(x) for x in images]
        for i in range(len(temp)):
            _, emb = temp[i]
            if len(emb) != 0:
                res.append(emb)
                imgs.append(images[i])

        if len(imgs) >= len(images) / 2:
            return res, imgs

        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Your face images is not valid, please try again.",
        )

    def register_face(self, images: List[Image.Image], id: str, group_id: str, face_folder: Path) -> List[str]:
        """
        Register face images
        
        Args:
            id (str): user id
            images (List[Image.Image]): list of face images
            face_folder (Path): face folder
            
        Returns:
            List[str]: list of face hashes
        """
        # validate face
        embeds, imgs = self.validate_face(images=images, person_id=id, group_id=group_id)
        # create folder
        folder = face_folder.joinpath(f"{group_id}").joinpath(f"{id}")
        folder.mkdir(parents=True, exist_ok=True)
        # get hashes
        hashes = [hashlib.md5(img.tobytes()).hexdigest() for img in imgs]
        # assert and insert
        assert len(embeds) == len(hashes), f"Embedding and hash length mismatch, {len(embeds)} != {len(hashes)}"
        # check if face's image already exists
        filter_hashes, filter_emb = [], []
        for hash, emb in zip(hashes, embeds):
            if self.facedb.get_face_by_id(hash) is not None: # function only for qdrant
                print(f"Face hash {hash} already exists, skipping...")
            filter_hashes.append(hash)
            filter_emb.append(emb)  
        # insert
        self.facedb.insert_faces(
            face_embs=zip(filter_hashes, filter_emb), 
            person_id=id, group_id=group_id
        ) 
        # save images
        for i in range(len(imgs)):
            imgs[i].save(folder.joinpath(f"{hashes[i]}.jpg"))
        return hashes


    def check_face(self, images: List[Image.Image], thresh: float, person_id: str|None, group_id: str|None) -> dict:
        """Check face images

        Args:
            id (str): user id
            images (List[Image.Image]): list of face images
            thresh (float): face threshold
        
        Returns:
            dict: status
        """
        res, imgs = self.validate_face(images=images, person_id=person_id, group_id=group_id)
        #
        check_res = [self.facedb.check_face(x, thresh) for x in res] # type: ignore
        verify_face = [True for x in check_res if len(x) == 1 else False]
        #
        if len(verify_face) >= len(imgs) / 2:
            return {"status": "ok"}
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"Face checking fail (only {len(verify_face)}/{len(images)} passed), please try again.",
        )
    
    def get_face_embs(self, image: Image.Image) -> Tuple[Any, Any]:
        image = np.array(image)  # type: ignore
        boxes, _, _, kpts = self.detection.inference(
            image, get_layer="face", det_thres=self.detection_thresh
        )
        embeds = self.recognition.inference(image, boxes, kpts) # type: ignore
        return boxes, embeds

    def check_faces(self, images: List[Image.Image], thresh: float, group_id: str) -> dict:
        """Check face images
        """
        embeds = []
        for img in images:
            _, res = self.get_face_embs(img)
            if res is not None:
                embeds.extend(res) # face_embed
        checked = [self.facedb.check_face(x, thresh) for x in embeds] # type: ignore
        dict_checked = []
        for i in range(len(checked)):
            if len(checked[i]) == 0:
                continue
            dict_checked.append({
                "image_id": checked[i][0].id,
                "person_id": checked[i][0].payload['person_id'],
                "group_id": checked[i][0].payload['group_id'],
            })
        # extract to csv
        self.dict_to_csv(dict_checked, group_id)

        # N images - M faces
        if len(images) < len(embeds):
            return {"check_group": dict_checked}
        # N images - N faces
        elif len(images) == len(embeds):
            checked = [self.facedb.check_face(x, thresh) for x in embeds] # type: ignore
            return {"check_per_person": dict_checked}
        
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"Face checking fail (only {len(checked)}/{len(images)} passed), please try again.",
        )
    
    def dict_to_csv(self, data: List[dict], group_id: str = 'default') -> None:
        import csv

        keys = data[0].keys()
        with open(f'{group_id}.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(data)
