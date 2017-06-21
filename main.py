import dlib
import cv2
import numpy
from utils import Face
from glob import glob
from skimage import io

PRED_PATH = r"resources\shape_predictor_68_face_landmarks.dat"
FILES = r"test_images\*.jpg"
UPSCALE = 1
ADJUST = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PRED_PATH)
window = dlib.image_window()

for file in glob(FILES):
    print("File: " + file)
    image = io.imread(file)
    window.clear_overlay()
    window.set_image(image)
    try:
        (detection, *_), scores, idx = detector.run(image, UPSCALE, ADJUST)
    except ValueError:
        print("No faces found.")
        continue
    landmarks = predictor(image, detection)
    window.add_overlay(landmarks)
