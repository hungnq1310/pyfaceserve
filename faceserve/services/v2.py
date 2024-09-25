import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Any
from pathlib import Path

from trism import TritonModel
from faceserve.services.interface import InterfaceService
from faceserve.db.interface import InterfaceDatabase
from faceserve.utils import crop_image, align_5_points
from faceserve.utils.save_crop import save_crop



from fastapi import HTTPException, status

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

class FaceServiceV2(InterfaceService):
    '''FaceServiceV2'''
    def __init__(self, 
        triton_server_url: str,
        is_grpc: bool,
        headface_name: str,             
        ghostfacenet_name: str,
        anti_spoofing_name: str,
        facedb: InterfaceDatabase,
        detection_thresh: float,
        recognition_thresh: float,
    ):
        self.headface = TritonModel(headface_name, 0, triton_server_url, is_grpc)
        self.ghostfacenet = TritonModel(ghostfacenet_name, 0, triton_server_url, is_grpc)
        self.anti_spoofing = TritonModel(anti_spoofing_name, 0, triton_server_url, is_grpc)
        self.facedb = facedb
        self.detection_thresh = detection_thresh
        self.recognition_thresh = recognition_thresh


    def get_face_emb(self, images, **kwargs):
        """
        Get face embedding from face image
        
        Args:
            images (List[Any]): face images

        Returns:
            Tuple[Any, Any]: face bounding boxes, face embeddings
        """
        if isinstance(images, list):
            images = np.array(images)
        emds = self.ghostfacenet.run([images]) # get face embedding
        print("emds: ", emds)
        return emds
    
    def validate_face(self, images: List[Image.Image]):
        # TODO:
        # preprocess image with  shape (256, 256)
        ...
        # 1. get temp emb of each face -> List of List
        embeddings, valid_imgs, temp = [], [], self.get_face_emb(images=images)
        
        # 2. Check if face is valid by using spoofing model
        result_spoofing = self.anti_spoofing.run(data=[temp])
        result_softmax = softmax(result_spoofing['output'])

        for i in range(len(temp)):
            if result_softmax[i] > self.recognition_thresh:
                embeddings.append(temp[i])
                valid_imgs.append(images[i])
            else:
                embeddings.append(None)
                valid_imgs.append(None)

        # 3. Get face embedding of validate face-> List of List
        # get temp result
        return embeddings, valid_imgs
    
    def detect_face(self, images: List[np.ndarray]):
        # preprocess
        preprocess_images = [self.preprocess(image) for image in images]
        input_images = np.array([image[0] for image in preprocess_images])
        ratios = np.array([image[1] for image in preprocess_images])
        dwdhs = np.array([image[2] for image in preprocess_images])
        # call API
        outputs = self.headface.run(data=[input_images])['2833']
        index_images, bboxes, _, _, kpts = self.postprocess(
            outputs, ratios, dwdhs, det_thres=self.detection_thresh
        )
        return index_images, bboxes, kpts

    def check_face(self, images: List[np.ndarray], thresh: float, group_id: str) -> dict:
        """Check face images
        """
        valid_embeddings, valid_faces, file_paths = [], [], []
        # TODO:
        # 1. detect faces in each image -> List of List
        batch_bboxes, batch_kpts = self.detect_face(images=images)
        # 2. crop and align face -> List of List
        batch_crops= []
        for i, image in enumerate(images):
            crops_per_image = self.crop_and_align_face(image, batch_bboxes[i], batch_kpts[i]) # Image, Image's bboxes, Image's kpts
            batch_crops.append(crops_per_image)

        # 3. get valid face -> List of List
        batch_embeds, batch_crops = [], []
        for i, crops in enumerate(batch_crops):
            embeddings, valid_crops = self.validate_face(crops)
            for j, crop in enumerate(crops):
                if embeddings[j] is None:
                    embeddings.remove(embeddings[j])
                    valid_crops.remove(valid_crops[j])

            batch_embeds.append(embeddings)
            batch_crops.append(valid_crops)

        assert len(batch_embeds) == len(batch_bboxes), 'Number of embeddings and bboxes are not the same'
        assert len(batch_crops) == len(batch_bboxes), 'Number of crops and bboxes are not the same'

        for i, bboxes in enumerate(batch_bboxes):
            file_crop_paths = save_crop(
                bboxes=bboxes, 
                path=f"{group_id}_image_{i}_", 
                img=images[i], #* this is the original image 
                save_dir=Path("temp"), 
                names=['face']
            )
            file_paths.append(file_crop_paths)                

        # check face
        check_batch = []
        for embeddings in batch_embeds:
            check_batch.append([self.facedb.check_face(x, thresh) for x in embeddings])
        # print(checked)

        #TODO: turn check_batch into dict_checked
        dict_checked = []
        for i in range(len(check_batch)):
            if len(check_batch[i]) == 0:
                continue
            for j in range(len(check_batch[i])):
                if len(check_batch[i][j]) == 0:
                    continue
                dict_checked.append({
                    "image_id": check_batch[i][j][0].id,
                    "person_id": check_batch[i][j][0].payload['person_id'],
                    "group_id": check_batch[i][j][0].payload['group_id'],
                    'file_crop': file_paths[i][j]
                })
        # extract to csv
        self.dict_to_csv(dict_checked, group_id)

        # N images - M faces
        # if len(valid_crops) < len(images):
        return {"check_group": dict_checked}
        # N images - N faces
        # elif len(valid_crops) == len(images):
        #     return {"check_per_person": dict_checked, "num_detections": len(bboxes)}
        
        # raise HTTPException(
        #     status.HTTP_403_FORBIDDEN,
        #     f"Face checking fail (only {len(checked)}/{len(images)} passed), please try again.",
        # )
    
    def dict_to_csv(self, data: List[dict], group_id: str = 'default') -> None:
        import csv

        keys = ('image_id', 'person_id', 'group_id', 'file_crop')
        with open(f'{group_id}.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(data)

        
        

    def register_face(self, images: List[np.ndarray], person_id: str, group_id: str, face_folder: Path) -> dict:
        """
        Register face images

        Args:
            images (List[np.ndarray]): face images
            person_id (str): person id
            group_id (str): group id
            face_folder (Path): face folder

        Returns:
            dict: message
        """
        # 1. detect faces in each image -> List of List
        _, bboxes, kpts = self.detect_face(images=images)
        # 2. crop and align face -> List of List
        assert len(bboxes) == len(images), 'Number of batch bboxes and batch images are not the same'
        assert len(kpts) == len(images), 'Number of batch keypoints and batch images are not the same'
        batch_crops = []
        for i, image in enumerate(images):
            crops_per_image = self.crop_and_align_face(image, bboxes[i], kpts[i])
            batch_crops.extend(crops_per_image)
        # 3. get valid face -> List of List
        embeddings, valid_crops = self.validate_face(batch_crops)
        # 4. save crop to folder
        if len(valid_crops) < len(images) / 2:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Your face images is not valid, only {len(valid_crops)}/{len(images)} accepted images, please try again.",
            )
        # 5. save face embedding to local
        for i, crop in enumerate(valid_crops):
            cv2.imwrite(str(face_folder / f"{group_id}_{person_id}_{i}.jpg"), crop)
        # 6. save face embedding to database
        self.facedb.insert_faces(
            face_embs=embeddings,
            group_id=group_id,
            person_id=person_id
        )
        return {
            "message": "Register face successfully",
        } 

    
    def crop_and_align_face(self, image, xyxys, kpts):
        crops = []        
        # dets are of different sizes so batch preprocessing is not possible
        for box, kpt in zip(xyxys, kpts):
            x1, y1, _, _ = box
            crop = crop_image(image, box)
            # Align face
            # Scale the keypoints to the face size
            kpt[::3] = kpt[::3] - x1
            kpt[1::3] = kpt[1::3] - y1
            # Crop face
            crop = align_5_points(crop, kpt)
            crop = cv2.resize(crop, self.model_input_size)
            crop = (crop - 127.5) * 0.0078125
            crop = crop.transpose(2, 0, 1)
            crop = np.expand_dims(crop, axis=0)
            crops.append(crop)
        crops = np.concatenate(crops, axis=0)
        return crops

    def preprocess(self, 
        image: np.ndarray, 
        new_shape=(640, 640), 
        color=(114, 114, 114), 
        scaleup=True
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """ Preprocessing function with reshape and normalize input

        Args:
            im (np.array, optional): input image
            new_shape (tuple, optional): new shape to resize. Defaults to (640, 640).
            color (tuple, optional): _description_. Defaults to (114, 114, 114).
            scaleup (bool, optional): resize small to large input size. Defaults to True.

        Returns:
            im: image after normalize and resize
            r: scale ratio between original and new shape 
            dw, dh: padding follow by yolo processing
        """
        # Resize and pad image while meeting stride-multiple constraints
        if isinstance(image, Image.Image):
            image = np.array(image)
        shape = image.shape[:2]  # current shape [height, width]
        
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        
        image = image.transpose((2, 0, 1))
        image = np.ascontiguousarray(image, dtype=np.float32)
        image /= 255
        
        return image, r, (dw, dh)

    def postprocess(self, face_detect_batch, ratios, dwdhs, det_thres=0.5):
        """Processing output model match output format

        Args:
            face_detect_batch (np.array): output of face detection model
            ratios (float): ratio between original and new shape
            dwdhs (tuple): padding follow by yolo processing
            det_thres (float, optional): detection threshold. Defaults to 0.5.
            
        Returns:
            Tuple: index_images, det_bboxes, det_scores, det_labels, kpts
        """
        # wrap numpy
        if isinstance(face_detect_batch, list):
            face_detect_batch = np.array(face_detect_batch)
        # scale: x,y -> x,y,x,y
        padding = np.concatenate([dwdhs, dwdhs], axis=1)
        # get sample higher than threshold
        pred = face_detect_batch[face_detect_batch[:, 6] > det_thres] 
        # get index, bbox, score, label, ketpoint
        index_images, det_bboxes, det_scores, det_labels  = pred[:, 0], pred[:,1:5], pred[:,6], pred[:, 5]
        kpts = pred[:, 7:] if pred.shape[1] > 6 else None
        # Filter, Normalize
        det_bboxes -= np.array(padding)
        det_bboxes /= np.array(ratios)        
        if kpts is not None:
            kpts[:,0::3] = (kpts[:,0::3] - np.array(padding[:, 0])) / ratios
            kpts[:,1::3] = (kpts[:,1::3]- np.array(padding[:, 1])) / ratios
        # return
        return index_images, det_bboxes, det_scores, det_labels, kpts
        

