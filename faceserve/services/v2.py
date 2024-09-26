import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Any
from pathlib import Path

from trism import TritonModel
from faceserve.services.interface import InterfaceService
from faceserve.db.interface import InterfaceDatabase
from faceserve.utils import crop_image, align_5_points, preprocess

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


    
    ### Core Modules
    ###

    def get_face_emb(self, images: List[Image.Image]) -> Any:
        """
        Get face embedding from face image
        
        Args:
            images (List[Any]): face images

        Returns:
            Tuple[Any, Any]: face bounding boxes, face embeddings
        """
        images = [
            preprocess(image, new_shape=(112, 112), channel_first=False, normalize=True)[0] # get image only
            for image in images
        ]
        emds = self.ghostfacenet.run([np.array(images)]) # get face embedding
        print("emds: ", emds)
        return emds
    
    def validate_face(self, images: List[Image.Image]):
        # TODO:
        # 1. get temp emb of each face -> List of List
        embeddings, valid_imgs, temp = [], [], self.get_face_emb(images=images)
        temp = temp['embedding']
        # 2. Check if face is valid by using spoofing model
        images = np.array([
            preprocess(image, new_shape=(256, 256), channel_first=True, normalize=True)[0] # get image only
            for image in images
        ])

        result_spoofing = self.anti_spoofing.run(data=[images])
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
    
    def detect_face(self, images: List[Image.Image]):
        # preprocess
        preprocess_images = [
            preprocess(image, new_shape=(640, 640), channel_first=True, normalize=True)
            for image in images
        ]
        input_images = np.array([image[0] for image in preprocess_images])
        ratios = np.array([image[1] for image in preprocess_images])
        dwdhs = np.array([image[2] for image in preprocess_images])
        # call API
        outputs = self.headface.run(data=[input_images])['2833']
        index_images, bboxes, _, _, kpts = self.postprocess(
            outputs, ratios, dwdhs, det_thres=self.detection_thresh
        )
        return index_images, bboxes, kpts
    

    ### Base Modules
    ### 

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
        for i in range(len(det_bboxes)):
            det_bboxes[i] -= padding[i]
            det_bboxes[i] /= ratios[i]    
        if kpts is not None:
            for i in range(len(kpts)):
                kpts[i,0::3] = (kpts[i,0::3] - padding[i, 0]) / ratios[i]
                kpts[i,1::3] = (kpts[i,1::3]- padding[i, 1]) / ratios[i]
        # return
        return index_images, det_bboxes, det_scores, det_labels, kpts

    def crop_and_align_face(self, image, xyxys, kpts):
        """Crop and align face from image"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        crops = []        
        # dets are of different sizes so batch preprocessing is not possible
        for box, kpt in zip(xyxys, kpts):
            crop = crop_image(image, box)
            # Align face
            # Crop face
            crop = align_5_points(crop, kpt)
            crops.append(crop)
        return crops

    def dict_to_csv(self, data: List[dict], group_id: str = 'default') -> None:
        import csv

        keys = ('image_id', 'person_id', 'group_id', 'file_crop')
        with open(f'{group_id}.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(data)



    ### Main Modules
    ###
    def check_face(self, image: Image.Image, thresh: float, group_id: str) -> dict:
        """Check face images
        """
        # 1. detect faces in each image -> List of List
        _, batch_bboxes, batch_kpts = self.detect_face(images=[image])

        # 2. crop and align face -> List of List
        crops = self.crop_and_align_face(image, batch_bboxes, batch_kpts)
        # 3. get valid face -> List of List
        embeddings, valid_crops = self.validate_face(crops)

        # 5. save crop to folder
        file_crop_paths = save_crop(
            bboxes=batch_bboxes, 
            path=f"{group_id}_image_{i}_", 
            img=image, #* this is the original image 
            save_dir=Path("temp"), 
            names=['face']
        )               

        # check face
        check_batch = [self.facedb.check_face(x, thresh) for x in embeddings]
        print(check_batch)

        #TODO: turn check_batch into dict_checked
        dict_checked = []
        for i in range(len(check_batch)):
            if len(check_batch[i]) == 0:
                continue
            for j in range(len(check_batch[i])):
                if len(check_batch[i][j]) == 0:
                    continue
                dict_checked.append({
                    "image_id": check_batch[i][0].id,
                    "person_id": check_batch[i][0].payload['person_id'],
                    "group_id": check_batch[i][0].payload['group_id'],
                    'file_crop': file_crop_paths[i]
                })
        # extract to csv
        self.dict_to_csv(dict_checked, group_id)

        # N images - M faces
        if len(valid_crops) > 1:
            return {"check_group": dict_checked}
        # N images - N faces
        elif len(valid_crops) == 1:
            return {"check_per_person": dict_checked, "num_detections": len(batch_bboxes)}
        
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"Face checking fail (No detection: {len(check_batch)}/1), please try again.",
        )
    


    def register_face(self, images: List[Image.Image], person_id: str, group_id: str, face_folder: Path) -> dict:
        """
        Register face images

        Args:
            images (List[Image.Image]): face images
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
            crops_per_image = self.crop_and_align_face(image, [bboxes[i]], [kpts[i]])
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
        #? valide crops is 16 for 4 images
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