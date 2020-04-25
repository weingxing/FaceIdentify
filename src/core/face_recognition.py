import numpy as np

from face_db import FaceDB
from core.face_encoder import Encoder


class Recognition:
    def __init__(self):
        self.db = FaceDB()
        self.encoder = Encoder()

    def add_face(self, name, image):
        features = self.encoder.generate_features(image)
        face_id = features.tostring()
        self.db.insert_face(name, face_id)

    def get_faces(self):
        faces = self.db.get_all_faces()
        face_id = []
        name = []
        for face in faces:
            name.append(face[0])
            face_id.append(np.frombuffer(face[1], dtype=np.float32))
        return name, face_id

    def get_distance(self, face, faces):
        result = []
        for i in faces:
            distances = self.encoder.distance(face, i)
            result.append(distances)
        return result

    def result(self, emb):
        name, face_ids = self.get_faces()
        distance = self.get_distance(emb, face_ids)
        index = np.argmin(distance)
        return name[index], distance[index]
