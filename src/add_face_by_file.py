import cv2
from face_db import FaceDB
from core.face_recognition import Recognition
from core.face_alignment import Alignment

DB = FaceDB()
reconition = Recognition()
alignment = Alignment()

if __name__ == '__main__':
    path = 'C:\\Users\\24518\\Desktop\\123.jpg'
    image = cv2.imread(path)
    image = image[:, :, ::-1]
    image = alignment.align_face(image)
    reconition.add_to_db('刘涛', image)
