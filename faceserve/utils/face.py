import cv2
import math


def align_5_points(face, points):
    # 5 points align
    # left eye, right eye, nose, left mouth, right mouth
    left_eye = points[0:2]
    right_eye = points[3:5]
    # Calculate the vertical difference between eye keypoints
    dy = left_eye[1] - right_eye[1]
    # Rotation degree
    if dy < 0:
        # need to rotate counter-clockwise
        angle = math.degrees(math.atan(abs(dy) / (right_eye[0] - left_eye[0])))
    else:
        # need to rotate clockwise
        angle = -math.degrees(math.atan(abs(dy) / (right_eye[0] - left_eye[0])))
    # get the scale for eyes
    scale = 1
    # get the rotation matrix
    M = cv2.getRotationMatrix2D(left_eye.astype(float), angle, scale)
    # apply the affine transformation
    aligned = cv2.warpAffine(
        face, M, (face.shape[1], face.shape[0]), flags=cv2.INTER_CUBIC
    )
    return aligned
