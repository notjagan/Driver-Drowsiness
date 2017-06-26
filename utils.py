import cv2
import numpy as np
from math import pi
from matplotlib.path import Path
from scipy.spatial.distance import euclidean as distance

MODEL_COORDS = np.array((
    (0, 0, 0),              # Tip of nose
    (0, -330, -65),         # Chin
    (-225, 170, -135),      # Left eye, left corner
    (225, 170, -135),       # Right eye, right corner
    (-150, -150, -125),     # Mouth, left corner
    (150, -150, -125)       # Mouth, right corner
), dtype=np.double)
PNP_METHOD = cv2.SOLVEPNP_ITERATIVE
FACE_DISPLAY = [
    list(range(17)),
    list(range(17, 22)),
    list(range(22, 27)),
    list(range(27, 36)) + [30],
    list(range(36, 42)) + [36],
    list(range(42, 48)) + [42],
    list(range(48, 60)) + [48],
    list(range(60, 68)) + [60]
]

def camera_internals(image):
    size = image.shape
    focal_length = size[1]
    center = (size[1] // 2, size[0] // 2)
    camera_matrix = np.array((
        (focal_length, 0, center[0]),
        (0, focal_length, center[1]),
        (0, 0, 1)
    ), dtype=np.double)
    dist_coeffs = np.zeros((4, 1))
    return camera_matrix, dist_coeffs

def polygon_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

class Feature:
    def __init__(self, image, points, expand=0):
        self.points = points
        bbox = list(zip(*self.bounding_box(expand)))
        self.image = image[slice(*bbox[1]), slice(*bbox[0])]

    def bounding_box(self, expand=0):
        bmin = self.points.min(axis=0)
        bmax = self.points.max(axis=0)
        diff = bmax - bmin
        return (tuple(np.round(bmin - diff * expand).astype(np.int)), tuple(np.round(bmax + diff * expand).astype(np.int)))

class Eyebrow(Feature):
    pass

class Eye(Feature):
    def __init__(self, image, points):
        super().__init__(image, points, expand=0.5)
        self.left_corner = points[0]
        self.right_corner = points[3]
    
    def aspect_ratio(self):
        p1, p2, p3, p4, p5, p6 = self.points
        return (distance(p2, p6) + distance(p3, p5)) / (2 * distance(p1, p4))

    def find_pupil(self):
        grayscale = cv2.equalizeHist(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY))
        ret, threshold = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY_INV)
        ret, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        circularity = 4 * pi * area / cv2.arcLength(largest, True) ** 2
        if circularity > 0.25 and area < polygon_area(*zip(*self.points)):
            M = cv2.moments(largest)
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

class Nose(Feature):
    def __init__(self, image, points):
        super().__init__(image, points)
        self.tip = points[3]

class Mouth(Feature):
    def __init__(self, image, points):
        super().__init__(image, points)
        self.left_corner = points[0]
        self.right_corner = points[6]

    def aspect_ratio(self):
        p1, p2, p3, p4, p5, p6, p7, p8 = self.points[12:]
        return (distance(p2, p8) + distance(p3, p7) + distance(p4, p6)) / (3 * distance(p1, p5))

class Jawline(Feature):
    def __init__(self, image, points):
        super().__init__(image, points)
        self.chin = points[8]

class Face:
    def __init__(self, image, detection, landmarks):
        self.original = image
        self.image = image[detection.top():detection.bottom(), detection.left():detection.right()]
        
        coords = np.zeros((landmarks.num_parts, 2), dtype=np.double)
        for i, part in enumerate(landmarks.parts()):
            coords[i] = (part.x, part.y)
            
        self.features = [
            self.jawline,
            self.left_eyebrow,
            self.right_eyebrow,
            self.nose,
            self.left_eye,
            self.right_eye,
            self.mouth
        ] = [
            Jawline(image, coords[:17]),
            Eyebrow(image, coords[17:22]),
            Eyebrow(image, coords[22:27]),
            Nose(image, coords[27:36]),
            Eye(image, coords[36:42]),
            Eye(image, coords[42:48]),
            Mouth(image, coords[48:68])
        ]
        
        self.pnp_points = np.array((
            self.nose.tip,
            self.jawline.chin,
            self.left_eye.left_corner,
            self.right_eye.right_corner,
            self.mouth.left_corner,
            self.mouth.right_corner
        ), dtype=np.double)

    def eye_aspect_ratio(self, left=False, right=False):
        if left:
            if right:
                return (self.left_eye.aspect_ratio() + self.right_eye.aspect_ratio()) / 2
            return self.left_eye.aspect_ratio()
        elif right:
            return self.right_eye.aspect_ratio()

    def mouth_aspect_ratio(self):
        return self.mouth.aspect_ratio()

    def PnP(self):
        success, rotation, translation = cv2.solvePnP(MODEL_COORDS, self.pnp_points, *camera_internals(self.original), flags=PNP_METHOD)
        return rotation, translation
