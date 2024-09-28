import numpy as np
from skimage import transform


src = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.729904, 92.2041],
    ],
    dtype=np.float32,
)
original_size = 112

def face_align_landmarks_sk(img, landmarks, image_size=(112, 112), method="affine"):
    # Convert inputs to CuPy arrays if they are not already
    img_cp = np.asarray(img)
    landmarks_cp = np.asarray(landmarks)
    
    # Choose transformation method
    tform = transform.AffineTransform() if method == "affine" else transform.SimilarityTransform()
    
    # Compute scale factor outside of the loop or transformation process
    scale_factor = image_size[0] / original_size
    _src = src * scale_factor
    
    # Estimate transformation and apply warp
    tform.estimate(landmarks_cp, _src)
    warped_image = transform.warp(img_cp, tform.inverse, output_shape=image_size)
    # Convert back to uint8 and return
    output = (warped_image * 255).astype(np.uint8)
    return output