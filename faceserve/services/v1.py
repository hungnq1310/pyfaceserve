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

    def get_face_emb(self, image: Image.Image) -> Tuple[Any, Any]:
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
        return None, None

    def validate_face(self, images: List[Image.Image], person_id: str | None, group_id: str | None) -> Tuple[List[Any], List[Image.Image]]:
        """Validate face images

        Args:
            id (str): user id
            images (List[Image.Image]): list of face images
        
        Returns:
            Tuple[List[Any], List[Image.Image]]: list of face embeddings, list of face images
        """
        if person_id is not None or group_id is not None:
            if not self.facedb.list_faces(person_id=person_id, group_id=group_id):
                raise HTTPException(status.HTTP_404_NOT_FOUND, "ID not found.")
        # get temp result
        res, imgs, temp = [], [], [self.get_face_emb(x) for x in images]
        for i in range(len(temp)):
            _, emb = temp[i]
            if emb is not None:
                res.append(emb)
                imgs.append(images[i])

        if len(imgs) >= len(images) / 2:
            return res, imgs

        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Your face images is not valid, please try again.",
        )

    def register_face(self, images: List[Image.Image], id: str, group_id: str | None, face_folder: Path) -> List[str]:
        """
        Register face images
        
        Args:
            id (str): user id
            images (List[Image.Image]): list of face images
            face_folder (Path): face folder
            
        Returns:
            List[str]: list of face hashes
        """
        # for test, only id is required when registing
        if group_id is None:
            group_id = "default"
        # validate face
        embeds, imgs = self.validate_face(images=images, person_id=id, group_id=group_id)
        # create folder
        folder = face_folder.joinpath(f"{group_id}", f"{id}")
        folder.mkdir(exist_ok=True)
        # get hashes
        hashes = [hashlib.md5(img.tobytes()).hexdigest() for img in imgs]
        # assert and insert
        assert len(embeds) == len(hashes), f"Embedding and hash length mismatch, {len(embeds)} != {len(hashes)}"
        self.facedb.insert_faces(
            face_embs=zip(embeds, hashes), 
            person_id=id, group_id=group_id
        ) 
        # save images
        for i in range(len(imgs)):
            imgs[i].save(folder.joinpath(f"{hashes[i]}.jpg"))
        return hashes


    def check_face(self, images: List[Image.Image], thresh: float) -> dict:
        """Check face images

        Args:
            id (str): user id
            images (List[Image.Image]): list of face images
            thresh (float): face threshold
        
        Returns:
            dict: status
        """
        res, imgs = self.validate_face(images=images)
        #
        checked = [self.facedb.check_face(x, thresh) for x in res] # type: ignore
        checked = [x for x in checked if x is True]
        #
        if len(checked) >= len(imgs) / 2:
            return {"status": "ok"}
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"Face checking fail (only {len(checked)}/{len(images)} passed), please try again.",
        )
