import numpy as np

from face_db import FaceDB
from core.face_encoder import Encoder


class Recognition:
    def __init__(self):
        self.db = FaceDB()
        self.encoder = Encoder()

    def add_to_db(self, name, image):
        emb = self.encoder.generate_embedding(image)
        face_id = emb.tostring()
        self.db.insert_face(name, face_id)

    def get_embs(self):
        embs = self.db.get_all_faces()
        face_id = []
        name = []
        for emb in embs:
            name.append(emb[0])
            face_id.append(np.frombuffer(emb[1], dtype=np.float32))
        return name, face_id

    def get_distance(self, emb, embs):
        result = []
        for i in embs:
            distances = self.encoder.distance(emb, i)
            result.append(distances)
        return result

    def result(self, img):
        emb = self.encoder.generate_embedding(img)
        name, face_ids = self.get_embs()
        distance = self.get_distance(emb, face_ids)
        index = np.argmin(distance)
        return name[index], distance[index]

    def generate_face_id(self, image):
        return self.encoder.generate_embedding(image)
