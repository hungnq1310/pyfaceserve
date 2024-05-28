import os
import json
import base64
import pathlib
from PIL import Image
from fastapi import *
from pydantic import BaseModel
from faceserve.services.v1 import FaceServiceV1
from faceserve.models import HeadFace, SpoofingNet, GhostFaceNet
from faceserve.db.qdrant import QdrantFaceDatabase
from io import BytesIO

"""
Load models and thresh.
"""
# For face storage.
FACES = QdrantFaceDatabase(
    collection_name="faces_collection",
)
FACES_IMG_DIR = pathlib.Path(os.getenv("IMG_DIR", default="face_images"))
FACES_IMG_DIR.mkdir(exist_ok=True)
# For detection.
DETECTION = HeadFace(
    os.getenv("DETECTION_MODEL_PATH", default="weights/yolov7-hf-v1.onnx")
)
DETECTION_THRESH = os.getenv("DETECTION_THRESH", default=0.5)
# For anti-spoofing.
SPOOFING = SpoofingNet(os.getenv("SPOOFING_MODEL_PATH", default="weights/OCI2M.onnx"))
SPOOFING_THRESH = os.getenv("SPOOFING_THRESH", default=0.6)
# For recognition.
RECOGNITION = GhostFaceNet(
    os.getenv("RECOGNITION_MODEL_PATH", default="weights/ghostnetv1.onnx")
)
RECOGNITION_THRESH = os.getenv("RECOGNITION_THRESH", default=0.3)
#
service = FaceServiceV1(
    DETECTION,
    DETECTION_THRESH,
    SPOOFING,
    SPOOFING_THRESH,
    RECOGNITION,
    RECOGNITION_THRESH,
    FACES,
)


#
router = APIRouter(prefix="/v1")


class FaceRequest(BaseModel):
    base64images: list[str] = None


@router.get("/ids")
async def get_id():
    return json.dumps(FACES.list_person())


@router.post("/id")
async def create_id(id: str):
    try:
        FACES.insert_person(person_id=id)
    except:
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE, detail="Cannot create ID"
        )


@router.post("/register")
async def register(id: str, request: FaceRequest):
    images = [base64.b64decode(x) for x in request.base64images]
    images = [Image.open(BytesIO(x)) for x in images]
    images = service.register(id, images, FACES_IMG_DIR)
    images = [f"/imgs/{id}/{x}.jpg" for x in images]
    return images


@router.get("/register")
async def get_face_image(id: str):
    if id not in FACES.list_person():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"ID {id} is not founded"
        )
    # res = [f'/imgs/{id}/{k}' for k in os.listdir(FACES_IMG_DIR.joinpath(f'{id}')) if k.endswith(".jpg")]
    res = ["".join(x.id.split("-")) for x in FACES.list_face(id)[0] if x is not None]
    res = [f"/imgs/{id}/{x}.jpg" for x in res]
    return res


@router.post("/check")
async def check_face_images(id: str, request: FaceRequest):
    images = [base64.b64decode(x) for x in request.base64images]
    images = [Image.open(BytesIO(x)) for x in images]
    return service.check(id, images, RECOGNITION_THRESH)
