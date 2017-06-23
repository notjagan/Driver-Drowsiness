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
    
    print("Left eye aspect ratio:", face.eye_aspect_ratio(left=True))
    print("Right eye aspect ratio:", face.eye_aspect_ratio(right=True))
    print("Mouth aspect ratio:", face.mouth_aspect_ratio())

    for feature in face.features:
        p = feature.points
        for i, point in enumerate(p[:-1]):
            cv2.line(copy_cv, tuple(np.round(point).astype(np.int)), tuple(np.round(p[i + 1]).astype(np.int)), (255, 0, 0), 1)

    cv2.rectangle(copy_cv, *face.right_eye.bounding_box(), (0, 255, 0))    
    cv2.rectangle(copy_cv, *face.left_eye.bounding_box(), (0, 255, 0))

    print(face.right_eye.find_pupil())

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
