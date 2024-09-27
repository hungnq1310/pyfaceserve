from PIL import Image
import base64
from io import BytesIO
from typing import Tuple
import cv2
import numpy as np


def crop_image(image, bbox):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(w, x2))
    y2 = int(min(h, y2))
    return image[y1:y2, x1:x2]


def to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return str(img_str)


def from_base64(base64str: str) -> Image.Image:
    im = Image.open(BytesIO(base64.b64decode(base64str)))
    return im

def preprocess( 
        image: Image.Image, 
        new_shape=(640, 640), 
        is_channel_first=True,
        normalize=True,
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
        # I. Wrap image
        image = np.array(image, dtype=np.float32)
        # Resize and pad image while meeting stride-multiple constraints
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
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_NEAREST)
        
        # add border
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  
        
        # II. Channel first
        if is_channel_first:
            if image.shape[0] == 3:
                print('Image has already been in channel first format')
                pass
            first_shape = image.shape
            image = image.transpose((2, 0, 1))
            image = np.ascontiguousarray(image)
            print(f'Transposing from shape {first_shape} to {image.shape}.')

        # III. Normalize image   
        if normalize:
            # check image value pixel is in range 0-1
            if image.max() < 1 and image.min() >= 0:
                print('Image has already been normalized')
                pass
            image /= 255.0
            print('Normalizing image')
        
        # IV. Return image, scale ratio, padding
        return image, r, (dw, dh)