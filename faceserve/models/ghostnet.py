import cv2
import numpy as np
import onnxruntime as ort

from gfn.utils import image, face

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


class GhostFaceNet:
    def __init__(self, model_path) -> None:
        self.load_model(model_path)

    def load_model(self, model_path: str):
        self.model = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        _, h, w, _ = self.model.get_inputs()[0].shape
        self.model_input_size = (w, h)

    def preprocess(self, img, xyxys, kpts):
        crops = []
        h, w = img.shape[:2]
        # dets are of different sizes so batch preprocessing is not possible
        for box, kpt in zip(xyxys, kpts):
            x1, y1, x2, y2 = box
            # Limit the face to the image
            x1 = int(max(0, x1))
            y1 = int(max(0, y1))
            x2 = int(min(w, x2))
            y2 = int(min(h, y2))

            box = [x1, y1, x2, y2]
            crop = img[y1:y2, x1:x2]
            # Align face
            # Scale the keypoints to the face size
            kpt[::3] = kpt[::3] - x1
            kpt[1::3] = kpt[1::3] - y1
            
            crop = face.align_5_points(crop, kpt)
            
            crop = cv2.resize(crop, self.model_input_size)
            crop = (crop - 127.5) * 0.0078125
            # crop = crop.transpose(2, 0, 1)
            crop = np.expand_dims(crop, axis=0)
            crops.append(crop)
        crops = np.concatenate(crops, axis=0)
        return crops

    def forward(self, crops):
        embeddings = self.model.run(
            [self.output_name], {self.input_name: crops.astype("float32")}
        )[0]
        return embeddings

    def get_features(self, img, xyxys, kpts):
        if len(xyxys) == 0:
            return np.array([])
        
        crops = self.preprocess(img, xyxys, kpts)
        embeddings = self.forward(crops)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings