
import os
import base64
import pathlib
from PIL import Image
from io import BytesIO
import gradio as gr

from fastapi import APIRouter, HTTPException, status, Request, File, UploadFile
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
DETECTION_THRESH = os.getenv("DETECTION_THRESH", default=0.5)
RECOGNITION_THRESH = os.getenv("RECOGNITION_THRESH", default=0.37)
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


def register_upload(files: list[str], id: str, group_id: str):
    print("="*50)
    print("Register files: ", files)
    if not files: return []
    images = [Image.open(x) for x in files]
    hash_imgs = service.register_face(images=images, id=id, group_id=group_id, face_folder=FACES_IMG_DIR)
    return [Image.open(
        f"{FACES_IMG_DIR}/{group_id}/{id}/{x}.jpg"
    ) for x in hash_imgs]


def get_face_image(id: str, group_id: str):
    if id == "": # gradio get "" instead of NoneType
        id = None
    if group_id == "":
        group_id = None

    retrieve_faces = FACES.list_faces(person_id=id, group_id=group_id)[0]
    if not retrieve_faces:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="No image found, database empty"
        )
    res = [x for x in retrieve_faces if x is not None]
    output = []
    for x in res:
        res_group = x.payload["group_id"]
        res_person = x.payload["person_id"]
        res_hash = "".join(x.id.split("-"))
        output.append(
            f"{FACES_IMG_DIR}/{res_group}/{res_person}/{res_hash}.jpg"
        )
    return [Image.open(x) for x in output]

def delete_face(face_id: str|None = None, id: str|None = None, group_id: str|None = None):
    return FACES.delete_face(
        face_id=face_id, 
        person_id=id, 
        group_id=group_id
    )


def check_faces(files: list[str], id: str|None = None, group_id: str = 'default'):
    images = [Image.open(x) for x in files]
    results_dict = service.check_faces(
        images=images, 
        thresh=RECOGNITION_THRESH,
        group_id=group_id
    )
    print("="*50)
    print("status_check: ", results_dict)
    result = results_dict.get('check_per_person') or results_dict.get('check_group')
    output, file_crop_paths = [], []
    if not result: 
        print("="*50)
        print("No faces accepted!")
        return []
    for x in result:
        if not x: continue
        res_group = x["group_id"]
        res_person = x["person_id"]
        res_hash = "".join(x['image_id'].split("-"))
        file_crop_paths.append(x['file_crop'])
        output.append(
            f"{FACES_IMG_DIR}/{res_group}/{res_person}/{res_hash}.jpg"
        )
    print("="*50)
    print("Faces check: ", output)

    # image && crop
    image_out = [Image.open(x) for x in output]
    image_crop = [Image.open(x) for x in file_crop_paths]
    merge = []
    for i in range(len(image_out)):
        merge.append(image_out[i])
        merge.append(image_crop[i])
    return  merge

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            files = gr.File(label="Register face",file_types=['image'], file_count="multiple")
            with gr.Row():
                group_id = gr.Textbox(value="default")
                person_id = gr.Textbox(value="0")
            run = gr.Button()
        outputs = gr.Gallery(type="filepath")

    with gr.Row():
        with gr.Column():
            check_files = gr.File(label="Check faces", file_count="multiple", file_types=['image'])
            check = gr.Button()
        checked_faces = gr.Gallery(type='filepath')

    with gr.Row():
        with gr.Column():
            with gr.Row():
                group_id_2 = gr.Textbox(label="Group ID")
                person_id_2 = gr.Textbox(label="Person ID")
            list_button = gr.Button("List faces")
        list_faces = gr.Gallery(type="filepath")
    # register
    # gallery
    # faces registers success
    event = run.click(
        register_upload, 
        [files, person_id, group_id], 
        outputs,
    )

    # checks
    # gallary
    # faces check succes + id
    event_2 = check.click(
        check_faces,
        check_files,
        checked_faces
    )

    event_3 = list_button.click(
        get_face_image,
        [person_id_2, group_id_2],
        list_faces
    )

if __name__ == "__main__":
    demo.launch()