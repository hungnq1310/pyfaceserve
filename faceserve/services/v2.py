import numpy as np
from PIL import Image
from typing import List, Tuple, Any
from pathlib import Path
import hashlib
from collections import Counter

from trism import TritonModel
from faceserve.services.interface import InterfaceService
from faceserve.db.interface import InterfaceDatabase
from faceserve.utils import crop_image, face_align_landmarks_sk, preprocess

from faceserve.utils.save_crop import save_crop

def softmax(x):
    return 1 / np.sum(np.exp(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

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
        spoofing_thresh: float,
        recognition_thresh: float,
    ):
        self.headface = TritonModel(headface_name, 0, triton_server_url, is_grpc)
        self.ghostfacenet = TritonModel(ghostfacenet_name, 0, triton_server_url, is_grpc)
        self.anti_spoofing = TritonModel(anti_spoofing_name, 0, triton_server_url, is_grpc)
        self.facedb = facedb
        self.detection_thresh = detection_thresh
        self.spoofing_thresh = spoofing_thresh
        self.recognition_thresh = recognition_thresh


    
    ### Core Modules
    ###

    def get_face_emb(
        self, images: List[Image.Image | np.ndarray]
    ) -> np.ndarray:
        """
        Get face embedding from face image
        
        Args:
            images (List[Any]): face images

        Returns:
            Tuple[Any, Any]: face bounding boxes, face embeddings
        """
        images = [
            preprocess(image, new_shape=(112, 112), is_channel_first=False, normalize=True)[0] # get image only
            for image in images #* images are crops
        ]
        emds = self.ghostfacenet.run([np.array(images)]) # get face embedding
        return emds['embedding']
    
    def validate_face(
        self, images: List[Image.Image | np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate face images
        
        Args:
            images (List[Image.Image | np.ndarray]): list of images
            
        Returns:
            Tuple: embeddings, valid_imgs
        """
        # 1. get temp emb of each face -> List of List
        embeddings, valid_imgs, temp = [], [], self.get_face_emb(images=images) #* images are crops
        # 2. Check if face is valid by using spoofing model
        images = np.array([
            preprocess(image, new_shape=(256, 256), is_channel_first=True, normalize=True)[0] # get image only
            for image in images
        ])
        # 3. Call APi
        result_spoofing = self.anti_spoofing.run(data=[images])
        result_softmax = sigmoid(result_spoofing['output'])
        # 4. Filter valid face
        for i in range(len(temp)):
            if result_softmax[i] > self.spoofing_thresh:
                embeddings.append(temp[i])
                valid_imgs.append(images[i])
            else:
                embeddings.append(None)
                valid_imgs.append(None)

        # return
        return embeddings, valid_imgs
    
    def detect_face(
        self, images: List[Image.Image]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect face in image
        
        Args:
            images (List[Image.Image]): list of images
        
        Returns:
            Tuple: index_images, bboxes, kpts
        """
        # preprocess
        preprocess_images = [
            preprocess(image, new_shape=(640, 640), is_channel_first=True, normalize=True)
            for image in images
        ]
        input_images = np.array([image[0] for image in preprocess_images])
        ratios = np.array([image[1] for image in preprocess_images])
        dwdhs = np.array([image[2] for image in preprocess_images])
        # call API
        output_names = [meta.name for meta in self.headface.outputs] # get name from metadata
        outputs = self.headface.run(data=[input_images])[output_names[1]] # get face detection
        index_images, bboxes, _, _, kpts = self.postprocess(
            outputs, ratios, dwdhs, det_thres=self.detection_thresh
        )
        return index_images, bboxes, kpts
    

    ### Base Modules
    ### 

    def postprocess(
        self, 
        face_detect_batch: np.ndarray | List[np.ndarray], 
        ratios: np.ndarray | List[np.ndarray], 
        dwdhs: np.ndarray | List[np.ndarray], 
        det_thres: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        for i, pad in enumerate(padding):
            # filter bbox
            filter_index = [index for index, value in enumerate(index_images) if value == i]
            for j in filter_index:
                det_bboxes[j] = (det_bboxes[j] - pad ) / ratios[i]
                if kpts is not None:
                    kpts[j,0::3] = (kpts[j,0::3] - pad[0]) / ratios[i]
                    kpts[j,1::3] = (kpts[j,1::3]- pad[1]) / ratios[i]
        # return
        return index_images, det_bboxes, det_scores, det_labels, kpts

    def crop_and_align_face(
        self, 
        image: Image.Image | np.ndarray, 
        xyxys: List[List[float] | np.ndarray], 
        kpts: List[List[float] | np.ndarray],
    ) -> List[np.ndarray]:
        """Crop and align face from image"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        crops = []        
        for kpt in kpts:
            # wrap kpt from [1, 15] -> [[1, 2], [3, 4], ...]
            kpt_wrap = [[kpt[i], kpt[i+1]] for i in range(0, len(kpt), 3)]
            # Align face
            image_align = face_align_landmarks_sk(image, kpt_wrap)
            crops.append(image_align)
        return crops

    def dict_to_csv(self, data: List[dict], group_id: str = 'default', save_dir: str = 'temp') -> None:
        """Convert dict_checked (final result) to csv file"""
        import csv

        file_name = Path(group_id)
        save_dir = Path(save_dir)
        keys = ('face_id', 'person_id', 'group_id', 'bbox')

        if not (save_dir / "attendance").exists():
            (save_dir / "attendance").mkdir(parents=True, exist_ok=True)
        with open(f'{save_dir / "attendance" / file_name.stem}.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(data)
        return None


    ### Main Modules
    ###
    def check_face(
        self,  
        image: Image.Image, 
        thresh: float = 0.5, 
        person_id: str = '0',
    ) -> dict:
        """Check face images
        """
        # 1. detect faces in each image -> List of List
        _, batch_bboxes, batch_kpts = self.detect_face(images=[image])
        if len(batch_bboxes) == 0:
            return {
                "message": "Face checking fail (No detection), please try again.",
                "check": "false"
            }
        elif len(batch_bboxes) > 1:
            return {
                "message": "Only one person in one image, please try again.",
                "check": "false"
            }
    
        # 2. crop and align face -> List of List
        crops = self.crop_and_align_face(image, batch_bboxes, batch_kpts)
        assert len(crops) == len(batch_bboxes), "Number of crops and bboxes are not the same"

        # 3. get valid face -> List of List
        embeddings, valid_crops = self.validate_face(crops)

        # 4. Verify face
        if embeddings[0] is None:
            return {
                "message": "Detect fake face, please try again.",
                "check": "false"
            }
        check_batch = self.facedb.check_face(embeddings[0], thresh)
        if len(check_batch) != 0:
            # check if exist any point equal to person_id
            for point in check_batch:
                if point.payload['person_id'] == person_id:
                    return {
                        "message": "Face recognition success.",
                        "check": "true"
                    }
        return {
            "message": "Face recognition failed, please try again.",
            "check": "false"
        }

    def check_attendance(
        self,
        image: Image.Image,
        thresh: float = 0.5,
        group_id: str = 'default',
        face_folder: None | str = 'temp',
    ) -> dict:
        """Check attendance of face images
        """
        # 1. detect faces in each image -> List of List
        _, batch_bboxes, batch_kpts = self.detect_face(images=[image])
        if len(batch_bboxes) == 0:
            return {
                "check_attendance": "Node to check attendance, please try again."
            }
        # 2. crop and align face -> List of List
        crops = self.crop_and_align_face(image, batch_bboxes, batch_kpts)
        # 3. get valid face -> List of List
        embeddings, _ = self.validate_face(crops)
        # ensure embeddings equal to bboxes
        if len(embeddings) != len(batch_bboxes): 
            return {
            "check_attendance": "Fail to check attendance, please try again."
        }

        # 4. Verify face
        dict_checked = []
        for index, emb in enumerate(embeddings):
            if emb is None:
                dict_checked.append({
                    "face_id": "Unknown",
                    "person_id": "Unknown",
                    "group_id": group_id,
                    "bbox": batch_bboxes[index].tolist()
                })
            else:
                check_batch = self.facedb.check_face(emb, thresh)
                if len(check_batch) == 0:
                    dict_checked.append({
                        "face_id": "Unknown",
                        "person_id": "Unknown",
                        "group_id": group_id,
                        "bbox": batch_bboxes[index].tolist()
                    })
                else:
                    for point in check_batch:
                        if point.payload['group_id'] != group_id:
                            continue
                        dict_checked.append({
                            "face_id": point.id,
                            "person_id": point.payload['person_id'],
                            "group_id": group_id,
                            "bbox": batch_bboxes[index].tolist()
                        })
        # extract to csv
        # self.dict_to_csv(dict_checked, group_id, face_folder)
        return {
            "check_attendance": dict_checked,
        }


    def register_face(
        self, 
        images: List[Image.Image], 
        person_id: None | str = "0", 
        group_id: None | str = "default", 
        face_folder: None | str = "temp",
    ) -> dict:
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
        # 1. detect faces in each image
        index_images, bboxes, kpts = self.detect_face(images=images)
        counter = Counter(index_images)
        if any([value > 1 for value in counter.values()]):
            #! number detections of all image compare with list bboxes -> ensure one person in one image
            return {
                "message": f"Some images are invalid, only having one person per image.",  
            }
        if len(bboxes) == 0 or len(kpts) == 0:
            return {
                "message": "No face detected, please try again.",
            }
        # 2. crop and align face -> List of List
        batch_crops = []
        for i, image in enumerate(images):
            crops_per_image = self.crop_and_align_face(image, [bboxes[i]], [kpts[i]])
            batch_crops.extend(crops_per_image)
        # 3. get valid face -> List of List
        embeddings, valid_crops = self.validate_face(batch_crops)
        embeddings = [x.tolist() for x in embeddings if x is not None]
        valid_crops = [x for x in valid_crops if x is not None]

        # 4. verify
        if len(valid_crops) < len(images) / 2:
            return {
                "message": f"Your face images is not valid, only {len(valid_crops)}/{len(images)} accepted images, please try again.",
            }
        # 5. save and hash face embedding to local
        # foler_registry = Path(face_folder) / "registry"
        # foler_registry.mkdir(parents=True, exist_ok=True)

        hashes, crop_save_paths = [], []
        for i, crop in enumerate(valid_crops):
            # crop_save_path = f"{foler_registry}/{group_id}_{person_id}_{i}.jpg"
            # some preprocess
            if crop.shape[0] == 3:
                crop = np.transpose(crop, (1, 2, 0)) 
            crop_pil = Image.fromarray((crop*255).astype(np.uint8))
            # crop_pil.save(crop_save_path)
            # stuff
            hashes.append(hashlib.md5(crop_pil.tobytes()).hexdigest())
            # crop_save_paths.append(crop_save_path)
        # 6. save face embedding to database
        hash_in_db = self.facedb.insert_faces(
            face_embs=zip(hashes, embeddings),
            group_id=group_id,
            person_id=person_id
        )
        # return {
        #     f"{key}": f"{crop_save_path}" for key, crop_save_path in zip(hashes, crop_save_paths)
        # } 
        return hash_in_db