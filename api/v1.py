import os
import json
import base64
import pathlib
from PIL import Image
from io import BytesIO
from fastapi import APIRouter, HTTPException, status

from faceserve.services.v1 import FaceServiceV1
from faceserve.models import HeadFace, SpoofingNet, GhostFaceNet
from faceserve.db.qdrant import QdrantFaceDatabase
from faceserve.schema.face_request import FaceRequest

"""
Load models and thresh.
"""
# Model
DETECTION = HeadFace(os.getenv("DETECTION_MODEL_PATH", default="weights/yolov7-hf-v1.onnx"))
SPOOFING = SpoofingNet(os.getenv("SPOOFING_MODEL_PATH", default="weights/OCI2M.onnx"))
RECOGNITION = GhostFaceNet(os.getenv("RECOGNITION_MODEL_PATH", default="weights/ghostnetv1.onnx"))
# Threshold
DETECTION_THRESH = os.getenv("DETECTION_THRESH", default=0.5)
SPOOFING_THRESH = os.getenv("SPOOFING_THRESH", default=0.6)
RECOGNITION_THRESH = os.getenv("RECOGNITION_THRESH", default=0.3)
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
    spoofing=SPOOFING,
    spoofing_thresh=SPOOFING_THRESH,
    recognition=RECOGNITION,
    recognition_thresh=RECOGNITION_THRESH,
    facedb=FACES,
)

"""
Router
"""
router = APIRouter(prefix="/v1")


@router.get("/ids")
async def get_id():
    return json.dumps(FACES.list_person())


# @router.post("/id")
# async def create_id(id: str, files: List[UploadFile]):
#     try:
#         FACES.insert_person(person_id=id)
#     except:
#         raise HTTPException(
#             status_code=status.HTTP_406_NOT_ACCEPTABLE, detail="Cannot create ID"
#         )


@router.post("/register")
async def register(id: str, request: FaceRequest):
    images = [base64.b64decode(x) for x in request.base64images]
    images = [Image.open(BytesIO(x)) for x in images]
    images = service.register_face(id, images, FACES_IMG_DIR)
    images = [f"/imgs/{id}/{x}.jpg" for x in images]
    return images


@router.get("/person-faces/{id}")
async def get_face_image(id: str):
    if not FACES.list_face(id)[0]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"Person ID {id} is not founded"
        )
    # res = [f'/imgs/{id}/{k}' for k in os.listdir(FACES_IMG_DIR.joinpath(f'{id}')) if k.endswith(".jpg")]
    res = ["".join(x.id.split("-")) for x in FACES.list_face(id)[0] if x is not None]
    res = [f"/imgs/{id}/{x}.jpg" for x in res]
    return res


@router.post("/check")
async def check_face_images(id: str, request: FaceRequest):
    images = [base64.b64decode(x) for x in request.base64images]
    images = [Image.open(BytesIO(x)) for x in images]
    return service.check_face(id, images, RECOGNITION_THRESH)
