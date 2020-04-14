from face_db import FaceDB
from core.face_recognition import Encoder
from core.face_alignment import Alignment

DB = FaceDB()
encoder = Encoder()
alignment = Alignment()


def add_face_id(name, face_id):
    DB.insert_face(name, face_id)


if __name__ == '__main__':
    pass
    # path = 'C:\\Users\\24518\\Desktop\\123.jpg'
    # image = cv2.imread(path)
    # image = image[:, :, ::-1]
    # image = alignment.align_face(image)
    # plt.imshow(image)
    # plt.show()
    # face_id = encoder.generate_embedding(image)
    # print(face_id.tostring())
    # add_face_id('刘涛', face_id.tostring())
