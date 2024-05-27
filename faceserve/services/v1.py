import hashlib
import numpy as np
from fastapi import HTTPException, status
from pathlib import Path

from faceserve.db.interface import InterfaceDatabase
from faceserve.models.interface import InterfaceModel 

from .interface import InterfaceService

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


class FaceServiceV1(InterfaceService):

    def __init__(
        self,
        detection: InterfaceModel,
        detection_thresh: int,
        spoofing: InterfaceModel,
        spoofing_thresh: int,
        recognition: InterfaceModel,
        recognition_thresh: int,
        facedb: InterfaceDatabase,
    ) -> None:
        self.facedb = facedb
        self.detection = detection
        self.detection_thresh = detection_thresh
        self.spoofing = spoofing
        self.spoofing_thresh = spoofing_thresh
        self.recognition = recognition
        self.recognition_thresh = recognition_thresh

    def get_face_emb(self, images):
        #
        if isinstance(images, list):
            res, imgs, temp = list(), list(), [self.get_face_emb(x) for x in images]
            for i in range(len(temp)):
                box, emb = temp[i]
                if emb is not None:
                    res.append(emb)
                    imgs.append(images[i])
            return res, imgs
        #
        else:
            images = np.array(images)
            boxes, scores, _, kpts = self.detection.detect(
                images, get_layer="face", conf_det=self.detection_thresh
            )
            if len(boxes) == 1:
                spoof = self.spoofing.get_features(images, boxes, kpts)
                spoof = softmax(spoof)[:, 0]
                if spoof[0] > self.spoofing_thresh:
                    res = self.recognition.get_features(images, boxes, kpts)
                    return boxes[0], res[0]
                else:
                    return boxes[0], None
            return None, None

    def validate_face(self, id: str, images):
        if id not in self.facedb.list_person():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "ID not found.")
        #
        res, imgs = self.get_face_emb(images)
        if len(imgs) >= len(images) / 2:
            return res, imgs
        else:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "Your face images is not valid, please try again.",
            )

    def register_face(self, id: str, images, face_folder: Path):
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
            self.facedb.insert_face(id, hash, emb)
            #
            img.save(folder.joinpath(f"{hash}.jpg"))
            response.append(hash)
        return response

    def check_face(self, id: str, images, thresh):
        res, imgs = self.validate_face(id, images)
        #
        checked = [self.facedb.check_face(id, x, thresh) for x in res]
        checked = [x for x in checked if x is True]
        #
        if len(checked) >= len(imgs) / 2:
            return {"status": "ok"}
        else:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Face checking fail (only {len(checked)}/{len(images)} passed), please try again.",
            )
