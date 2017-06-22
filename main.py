import dlib
import cv2
import numpy as np
from utils import Face
from glob import glob
from skimage import io

PRED_PATH = r"resources\shape_predictor_68_face_landmarks.dat"
FILES = r"test_images\*.jpg"
UPSCALE = 1
ADJUST = -0.5

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PRED_PATH)

for file in glob(FILES):
    print("File: " + file)
    image = io.imread(file)
    window = dlib.image_window()
    window.clear_overlay()
    window.set_image(image)
    try:
        (detection, *_), (score, *_), idx = detector.run(image, UPSCALE, ADJUST)
    except ValueError:
        print("No faces found.")
        window.wait_until_closed()
        continue
    print("Face detected! (score " + str(score) + ")")
    landmarks = predictor(image, detection)
    window.add_overlay(landmarks)
    face = Face(landmarks)
    print("Left eye EAR:", face.left_eye.EAR())
    print("Right eye EAR:", face.right_eye.EAR())
    window.wait_until_closed()
