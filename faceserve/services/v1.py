import hashlib
import numpy as np
from fastapi import HTTPException, status
from pathlib import Path
from PIL import Image
from typing import Tuple, Any, List

from faceserve.db.interface import InterfaceDatabase
from faceserve.models.interface import InterfaceModel

from .interface import InterfaceService


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


class FaceServiceV1(InterfaceService):
    def __init__(
        self,
        detection: InterfaceModel,
        detection_thresh: float,
        spoofing: InterfaceModel,
        spoofing_thresh: float,
        recognition: InterfaceModel,
        recognition_thresh: float,
        facedb: InterfaceDatabase,
    ) -> None:
        self.facedb = facedb
        self.detection = detection
        self.detection_thresh = detection_thresh
        self.spoofing = spoofing
        self.spoofing_thresh = spoofing_thresh
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
            spoof = self.spoofing.inference(image, boxes, kpts) # type: ignore
            spoof = softmax(spoof)[:, 0]
            if spoof[0] > self.spoofing_thresh:  # check fake face
                res = self.recognition.inference(image, boxes, kpts) # type: ignore
                return boxes[0], res[0]
            else:
                return boxes[0], None
        return None, None

    def validate_face(self, id: str, images: List[Image.Image]) -> Tuple[List[Any], List[Image.Image]]:
        """Validate face images

        Args:
            id (str): user id
            images (List[Image.Image]): list of face images
        
        Returns:
            Tuple[List[Any], List[Image.Image]]: list of face embeddings, list of face images
        """
        if not self.facedb.list_face(id):
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

    def register_face(self, id: str, images: List[Image.Image], face_folder: Path) -> List[str]:
        """
        Register face images
        
        Args:
            id (str): user id
            images (List[Image.Image]): list of face images
            face_folder (Path): face folder
            
        Returns:
            List[str]: list of face hashes
        """
        res, imgs = self.validate_face(id, images)
        #
        folder = face_folder.joinpath(f"{id}")
        folder.mkdir(exist_ok=True)
        #
        response = []
        for i in range(len(res)):
            emb, img = res[i], imgs[i]
            hash = hashlib.md5(img.tobytes()).hexdigest()
            #
            self.facedb.insert_face(emb, hash, id, "default") # type: ignore
            #
            img.save(folder.joinpath(f"{hash}.jpg"))
            response.append(hash)
        return response

    def check_face(self, id: str, images: List[Image.Image], thresh: float) -> dict:
        """Check face images

        Args:
            id (str): user id
            images (List[Image.Image]): list of face images
            thresh (float): face threshold
        
        Returns:
            dict: status
        """
        res, imgs = self.validate_face(id, images)
        #
        checked = [self.facedb.check_face(id, x, thresh) for x in res] # type: ignore
        checked = [x for x in checked if x is True]
        #
        if len(checked) >= len(imgs) / 2:
            return {"status": "ok"}
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"Face checking fail (only {len(checked)}/{len(images)} passed), please try again.",
        )
