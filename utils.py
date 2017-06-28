import cv2
import numpy as np
from matplotlib.path import Path
from scipy.spatial.distance import euclidean as distance

MODEL_COORDS = np.array((
    (0, 0, 0),                      # Tip of nose
    (0, -330, -65),                 # Chin
    (-225, 170, -135),              # Left eye, left corner
    (225, 170, -135),               # Right eye, right corner
    (-150, -150, -125),             # Mouth, left corner
    (150, -150, -125)               # Mouth, right corner
), dtype=np.double)

FACE_DISPLAY = [
    list(range(17)),                # Jawline
    list(range(17, 22)),            # Left eyebrow
    list(range(22, 27)),            # Right eyebrow
    list(range(27, 36)) + [30],     # Nose
    list(range(36, 42)) + [36],     # Left eye
    list(range(42, 48)) + [42],     # Right eye
    list(range(48, 60)) + [48],     # Lips
    list(range(60, 68)) + [60]      # Mouth
]

PNP_METHOD = cv2.SOLVEPNP_ITERATIVE
DILATION_KERNEL = np.ones((3, 3), np.uint8)
MAX_AREA = 5000
EYE_EXPAND= 0.75

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

def gradient(mat):
    return np.apply_along_axis(lambda r: [r[1] - r[0], *(r[2:] - r[:-2]) / 2, r[-1] - r[-2]], 1, mat.astype(np.int32))

def gradxy(mat):
    return (gradient(mat), gradient(mat.T).T)

class Feature:
    def __init__(self, image, points, expand=0):
        self.points = points
        bbox = list(zip(*self.bounding_box(expand)))
        self.image = image[slice(*bbox[1]), slice(*bbox[0])]

    def bounding_box(self, expand=0):
        bmin = self.points.min(axis=0)
        bmax = self.points.max(axis=0)
        diff = bmax - bmin
        return (tuple(np.round(bmin - diff * expand).astype(np.int)),
                tuple(np.round(bmax + diff * expand).astype(np.int)))

    def centroid(self):
        M = cv2.moments(self.points)
        M_00 = M["m00"]
        return int(M["m10"] / M_00), int(M["01"] / M_00)

class Eyebrow(Feature):
    def __init__(self, image, points):
        super().__init__(image, points)
        self.center = self.points[2]

class Eye(Feature):
    def __init__(self, image, points):
        super().__init__(image, points, expand=EYE_EXPAND)
        self.left_corner = points[0]
        self.right_corner = points[3]
        self.top = ((self.points[1] + self.points[2]) / 2).astype(np.int32)
    
    def aspect_ratio(self):
        p1, p2, p3, p4, p5, p6 = self.points
        return (distance(p2, p6) + distance(p3, p5)) / (2 * distance(p1, p4))
    
    def find_pupil(self, histogram=True):
        blurred = cv2.GaussianBlur(self.image, (5, 5), 0)
        gray = cv2.equalizeHist(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY))
        area = gray.shape[0] * gray.shape[1]
        scale_factor = (MAX_AREA / area) ** 0.5
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        radius = gray.shape[0] * 0.375
        dxn, dyn = gradxy(gray)
        magnitudes = (dxn ** 2 + dyn ** 2) ** 0.5
        grad_filter = magnitudes > magnitudes.mean() + magnitudes.std() * 0.3
        lengths = magnitudes[grad_filter]
        gradx = dxn[grad_filter] / lengths
        grady = dyn[grad_filter] / lengths
        x_coords = np.tile(np.arange(dxn.shape[1]), [dxn.shape[0], 1])[grad_filter]
        y_coords = np.tile(np.arange(dxn.shape[0]), [dxn.shape[1], 1]).T[grad_filter]
        
        thresh = cv2.dilate(cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)[1], DILATION_KERNEL)
        mx = my = m = 0
        for cx in range(dxn.shape[1]):
            for cy in range(dxn.shape[0]):
                if not thresh[cy, cx]:
                    continue
                dx = x_coords - cx
                dy = y_coords - cy
                squared_lengths = dx ** 2 + dy ** 2
                dots = dx * gradx + dy * grady
                valid = dots > 0
                if histogram:
                    bins = [(1 + index) ** 2 for index in range(int(radius + 0.5))]
                    hist, edges = np.histogram(squared_lengths, bins)
                    max_bin = hist.argmax()
                    valid &= (squared_lengths > max(1, bins[max_bin] - 1)) & (squared_lengths < bins[max_bin + 1] + 1)
                dots = (dots[valid] ** 2 / squared_lengths[valid]) ** 2
                total = np.sum(dots)
                if total > m:
                    mx, my, m = cx, cy, total
                    
        return (np.array((mx, my)) // scale_factor).astype(np.int32)

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

    def eyebrow_distance(self, left=False, right=False):
        if left:
            l = np.linalg.norm(self.left_eyebrow.center - self.left_eye.top)
            if right:
                return (l + np.linalg.norm(self.right_eyebrow.center - self.right_eye.top)) / 2
            return l
        elif right:
            return np.linalg.norm(self.right_eyebrow.center - self.right_eye.top)
