import dlib
import cv2
import numpy as np
from utils import *
from glob import glob
from skimage import io

PRED_PATH = r"resources\shape_predictor_68_face_landmarks.dat"
FILES = r"test_images\*.jpg"
UPSCALE = 0
ADJUST = -0.5

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PRED_PATH)

for file in glob(FILES):
    print("File: " + file)
    image = io.imread(file)
    img_cv = np.array(image[:, :, ::-1])
    copy_cv = np.copy(img_cv)[:, :, :]
    
    try:
        (detection, *_), (score, *_), idx = detector.run(image, UPSCALE, ADJUST)
    except ValueError:
        print("No faces found.")
        cv2.imshow("Image", img_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        continue
    print("Face detected! (score " + str(score) + ")")
    
    landmarks = predictor(image, detection)
    face = Face(img_cv, detection, landmarks)

    lmin, lmax = np.array(face.left_eye.bounding_box(expand=EYE_EXPAND))
    rmin, rmax = np.array(face.right_eye.bounding_box(expand=EYE_EXPAND))
    cv2.circle(copy_cv, tuple(face.left_eye.find_pupil() + lmin.astype(np.int32)), 2, (0, 255, 0))
    cv2.circle(copy_cv, tuple(face.right_eye.find_pupil() + rmin.astype(np.int32)), 2, (0, 255, 0))
    
    print("Left eye aspect ratio:", face.eye_aspect_ratio(left=True))
    print("Right eye aspect ratio:", face.eye_aspect_ratio(right=True))
    print("Mouth aspect ratio:", face.mouth_aspect_ratio())
    print("Left eyebrow distance:", face.eyebrow_distance(left=True))
    print("Right eyebrow distance:", face.eyebrow_distance(right=True))

    left_center = face.left_eyebrow.center
    left_top = face.left_eye.top
    cv2.line(copy_cv, (int(left_center[0]), int(left_center[1])), (int(left_top[0]), int(left_top[1])), (0, 255, 0), 1)

    right_center = face.right_eyebrow.center
    right_top = face.right_eye.top
    cv2.line(copy_cv, (int(right_center[0]), int(right_center[1])), (int(right_top[0]), int(right_top[1])), (0, 255, 0), 1)
    
    f = lambda p: (p.x, p.y)
    for component in FACE_DISPLAY:
        for i, p in enumerate(component[:-1]):
            cv2.line(copy_cv, f(landmarks.part(p)), f(landmarks.part(component[i + 1])), (255, 0, 0), 1)

    for point in face.pnp_points:
        cv2.circle(copy_cv, (int(point[0]), int(point[1])), 2, (0, 0, 255))
    
    rotation, translation = face.PnP()
    print("Rotation vector:", tuple(rotation.reshape((1, 3))[0]))
    axis = np.double([[0, 0, 1000]]).reshape(1, 3)
    ((end_point, *_), *_), jacobian = cv2.projectPoints(axis, rotation, translation, *camera_internals(img_cv))
    cv2.arrowedLine(copy_cv, tuple(np.round(face.nose.tip).astype(np.int)), tuple(np.round(end_point).astype(np.int)), (0, 0, 255), 1)

    cv2.imshow("Facial Features", copy_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print()
