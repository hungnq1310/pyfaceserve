import os
import json
import base64
import pathlib
from PIL import Image
from io import BytesIO
from fastapi import APIRouter, HTTPException, status
from fastapi.staticfiles import StaticFiles

from faceserve.services.v1 import FaceServiceV1
from faceserve.models import HeadFace, GhostFaceNet
from faceserve.db.qdrant import QdrantFaceDatabase
from faceserve.schema.face_request import FaceRequest

"""
Load models and thresh.
"""
# Model
DETECTION = HeadFace(os.getenv("DETECTION_MODEL_PATH", default="weights/yolov7-hf-v1.onnx"))
RECOGNITION = GhostFaceNet(os.getenv("RECOGNITION_MODEL_PATH", default="weights/ghostnetv1.onnx"))
# Threshold
DETECTION_THRESH = os.getenv("DETECTION_THRESH", default=0.7)
RECOGNITION_THRESH = os.getenv("RECOGNITION_THRESH", default=0.4)
# Face db storage.
FACES = QdrantFaceDatabase(
    collection_name="faces_collection",
)
FACES_IMG_DIR = pathlib.Path(os.getenv("IMG_DIR", default="face_images"))
FACES_IMG_DIR.mkdir(exist_ok=True)

"""
Initialize Services
"""
service = FaceServiceV1(
    detection=DETECTION,
    detection_thresh=DETECTION_THRESH,
    recognition=RECOGNITION,
    recognition_thresh=RECOGNITION_THRESH,
    facedb=FACES,
)

"""
Router
"""
router = APIRouter(prefix="/v1")
router.mount("/imgs", StaticFiles(directory=FACES_IMG_DIR), name="imgs")


@router.post("/register")
async def register(id: str, request: FaceRequest, groups_id: str|None = None):
    images = [base64.b64decode(x) for x in request.base64images]
    images = [Image.open(BytesIO(x)) for x in images]
    hash_imgs = service.register_face(images=images, id=id, group_id=groups_id, face_folder=FACES_IMG_DIR)
    return [f"/imgs/{id}/{x}.jpg" for x in hash_imgs]


@router.get("/faces")
async def get_face_image(id: str|None = None, group_id: str|None = None):
    if not FACES.list_faces(person_id=id, group_id=group_id)[0]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"{group_id}/{id} is not founded"
        )
    res = ["".join(x.id.split("-")) 
        for x in FACES.list_faces(person_id=id, group_id=group_id)[0] if x is not None
    ]
    res = [f"/imgs/{id}/{x}.jpg" for x in res]
    return res


@router.delete("/delete")
async def delete_face(face_id: str|None = None, person_id: str|None = None, group_id: str|None = None):
    return FACES.delete_face(face_id=face_id, person_id=person_id, group_id=group_id)


@router.post("/check")
async def check_face_images(request: FaceRequest, person_id: str|None = None, group_id: str|None = None):
    images = [base64.b64decode(x) for x in request.base64images]
    images = [Image.open(BytesIO(x)) for x in images]
    return service.check_face(images, RECOGNITION_THRESH, person_id=person_id, group_id=group_id)