import numpy as np
from scipy.spatial.distance import euclidean as distance

class Feature:
    def __init__(self, points):
        self.points = points

class Eyebrow(Feature):
    pass

class Eye(Feature):
    def __init__(self, points):
        super().__init__(points)
        self.left_corner = points[0]
        self.right_corner = points[3]
    
    def EAR(self):
        p1, p2, p3, p4, p5, p6 = self.points
        return (distance(p2, p6) + distance(p3, p5)) / (2 * distance(p1, p4))

class Nose(Feature):
    pass

class Mouth(Feature):
    pass

class Jawline(Feature):
    pass

class Face:
    def __init__(self, landmarks):
        coords = np.zeros((landmarks.num_parts, 2))
        for i, part in enumerate(landmarks.parts()):
            coords[i] = (part.x, part.y)
        self.jawline = Jawline(coords[:17])
        self.left_eyebrow = Eyebrow(coords[17:22])
        self.right_eyebrow = Eyebrow(coords[22:27])
        self.nose = Nose(coords[27:36])
        self.left_eye = Eye(coords[36:42])
        self.right_eye = Eye(coords[42:48])
        self.mouth = Mouth(coords[48:68])

    def EAR(self, left):
        if left:
            return self.left_eye.EAR()
        else:
            return self.right_eye.EAR()
