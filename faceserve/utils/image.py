from PIL import Image
import base64
from io import BytesIO


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
