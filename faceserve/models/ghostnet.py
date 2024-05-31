from typing import Any
from pydantic import Field
import cv2
import numpy as np
import onnxruntime as ort
import os

from faceserve.utils import face, crop_image
from .interface import InterfaceModel

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


class GhostFaceNet(InterfaceModel):
    '''GhostFaceNet'''
    input_name: str = Field(default="input")
    output_name: str = Field(default="output")
    model_input_size: tuple = Field(default=(256, 256))

    def __init__(self, model_path) -> None:
        super().__init__()
        self._model = self.load_model(model_path)
        self.input_name = self._model.get_inputs()[0].name
        self.output_name = self._model.get_outputs()[0].name
        _, h, w, _ = self._model.get_inputs()[0].shape
        self.model_input_size = (w, h)

    def load_model(self, path: str | bytes | os.PathLike) -> ort.InferenceSession:
        return ort.InferenceSession(
            path,
            sess_options=sess_options,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

    def preprocess(self, image, xyxys, kpts):
        """ Preprocessing function with crops and align keypoints

        Args:
            image (np.array, optional): input image
            xyxys (tuple, optional): bbox prediction from headface model
            kpts (tuple, optional): keypoint prediction from headface model

        Returns:
            crops (np.array): list of corps (num_crops, 3, 112, 112)
        """
        crops = []
        # dets are of different sizes so batch preprocessing is not possible
        for box, kpt in zip(xyxys, kpts):
            x1, y1, _, _ = box
            # Crop face
            crop = crop_image(image, box)
            # Align face
            kpt[::3] = kpt[::3] - x1 # Scale the keypoints to the face size
            kpt[1::3] = kpt[1::3] - y1
            crop = face.align_5_points(crop, kpt)

            crop = cv2.resize(crop, self.model_input_size)
            crop = (crop - 127.5) * 0.0078125
            # crop = crop.transpose(2, 0, 1)
            crop = np.expand_dims(crop, axis=0)
            crops.append(crop)
        crops = np.concatenate(crops, axis=0)
        return crops

    def inference(self, image, xyxys, kpts, norm:bool=False):
        """ Get the embedding of the face

        Args:
            image (np.array): input image
            xyxys (tuple, optional): bbox prediction from headface model
            kpts (tuple, optional): keypoint prediction from headface model

        Returns:
            result ([batch, vector]): output of model - size: (batch, emb_size)
        """
        if isinstance(image, list):
            image = np.array(image)
        # preprocess
        crops = self.preprocess(image, xyxys, kpts)
        # check shape
        assert crops.shape[1] == 112, f'img.shape(1) == 112. You have shape {crops.shape[1]}'
        assert crops.shape[2] == 112, f'img.shape(2) == 112. You have shape {crops.shape[2]}'
        # Inference
        result = self._model.run(
            [self.output_name], {self.input_name: crops.astype("float32")}
        )[0]
        # Normalize
        if norm:
            result = result / np.linalg.norm(result, axis=1, keepdims=True)
            
        return result
    
    def postprocess(self, image, **kwargs) -> Any:
        raise NotImplementedError

    def batch_inference(self, images, **kwargs) -> Any:
        raise NotImplementedError