import os
import base64
import pathlib
from PIL import Image
from io import BytesIO
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import APIRouter, HTTPException, status, Request, File, UploadFile
from fastapi.responses import JSONResponse
from faceserve.services.v2 import FaceServiceV2
from faceserve.db.qdrant import QdrantFaceDatabase
from faceserve.schema.face_request import FaceRequest

"""
Load models and thresh.
"""
load_dotenv()
# Model
TRITON_URL = os.getenv("TRITON_URL", default="localhost:6000")
IS_GRPC = bool(os.getenv("IS_GRPC", default=False))
DETECTION_NAME= os.getenv("DETECTION_NAME", default="face_detection")
SPOOFING_NAME = os.getenv("SPOOFING_NAME", default="face_spoofing")
RECOGNITION_NAME = os.getenv("RECOGNITION_NAME", default="face_recognition")

# Threshold
DETECTION_THRESH = float(os.getenv("DETECTION_THRESH", default=0.7))
SPOOFING_THRESH = float(os.getenv("SPOOFING_THRESH", default=0.4))
RECOGNITION_THRESH = float(os.getenv("RECOGNITION_THRESH", default=0.4))
# Face db storage.
DB_NAME = os.getenv("DB_NAME", default="faces_collection")
FACES = QdrantFaceDatabase(
    collection_name=DB_NAME,
)
FACES_IMG_DIR = pathlib.Path(os.getenv("IMG_DIR", default="face_images"))
FACES_IMG_DIR.mkdir(exist_ok=True)

"""
Initialize Services
"""
service = FaceServiceV2(
    triton_server_url=TRITON_URL,
    is_grpc=IS_GRPC,
    headface_name=DETECTION_NAME,
    ghostfacenet_name=RECOGNITION_NAME,
    anti_spoofing_name=SPOOFING_NAME,
    facedb=FACES,
    detection_thresh=DETECTION_THRESH,
    spoofing_thresh=SPOOFING_THRESH,
    recognition_thresh=RECOGNITION_THRESH,
)

"""
Router
"""
router = APIRouter(prefix="/v1")

@router.post("/register")
async def register(id: str, request: FaceRequest, group_id: str = "default"):
    images = [base64.b64decode(x) for x in request.base64images]
    images = [Image.open(BytesIO(x)) for x in images]
    hashes_path = service.register_face(
        images=images, person_id=id, group_id=group_id, face_folder=FACES_IMG_DIR
    )
    return JSONResponse(
        content=hashes_path, status_code=status.HTTP_200_OK
    )

@router.post("/register/files")
async def register_upload(files: list[UploadFile], id: str, group_id: str = "default"):
    images = [Image.open(BytesIO(await x.read())).convert("RGB") for x in files]
    hashes_path = service.register_face(
        images=images, person_id=id, group_id=group_id, face_folder=FACES_IMG_DIR
    )
    return JSONResponse(
        content=hashes_path, status_code=status.HTTP_200_OK
    )

@router.get("/faces")
async def get_face_image(id: str|None = None, group_id: str|None = None):
    if not FACES.list_faces(person_id=id, group_id=group_id)[0]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="No image found, database empty"
        )
    res = [x for x in FACES.list_faces(person_id=id, group_id=group_id)[0] if x is not None]
    output = []
    for x in res:
        res_group = x.payload["group_id"]
        res_person = x.payload["person_id"]
        res_hash = x.id
        output.append(f"/imgs/{res_group}/{res_person}/{res_hash}.jpg")
    return output

@router.delete("/delete")
async def delete_face(face_id: str|None = None, id: str|None = None, group_id: str|None = None):
    message: dict = FACES.delete_face(
        face_id=face_id, 
        person_id=id, 
        group_id=group_id
    )
    return JSONResponse(content=message, status_code=status.HTTP_200_OK)

@router.post("/check/face")
async def check_face_images(request: FaceRequest, id: str|None = None):
    images = [base64.b64decode(x) for x in request.base64images]
    images = [Image.open(BytesIO(x)) for x in images]
    return [service.check_face(
        image=img, 
        thresh=RECOGNITION_THRESH, 
        person_id=id, 
    ) for img in images]

@router.post("/check/face/files")
async def check_face_images(
    files: list[UploadFile], 
    id: str|None = '0', 
):
    images = [Image.open(BytesIO(await x.read())).convert("RGB") for x in files]
    return [service.check_face(
        image=img, 
        thresh=RECOGNITION_THRESH, 
        person_id=id, 
    ) for img in images]

@router.post("/check/attendance")
async def check_face_images(request: FaceRequest, group_id: str|None = None):
    images = [base64.b64decode(x) for x in request.base64images]
    images = [Image.open(BytesIO(x)) for x in images]
    return [service.check_attendance(
        image=img, 
        thresh=RECOGNITION_THRESH, 
        group_id=group_id,
        face_folder=FACES_IMG_DIR,
    ) for img in images]

@router.post("/check/attendance/files")
async def check_face_images(
    files: list[UploadFile], 
    group_id: str|None = "default"
):
    images = [Image.open(BytesIO(await x.read())).convert("RGB") for x in files]
    return [service.check_attendance(
        image=img, 
        thresh=RECOGNITION_THRESH, 
        group_id=group_id,
        face_folder=FACES_IMG_DIR,
    ) for img in images]