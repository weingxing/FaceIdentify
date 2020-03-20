# 读取数据库中的人脸数据
import mysql.connector


class FaceDB:
    def __init__(self):
        self.db = mysql.connector.connect(host="localhost", user="root",
                                               passwd="123456", database="face")
        self.cursor = self.db.cursor()

    def get_all_faces(self):
        self.cursor.execute("SELECT * FROM face")
        result = self.cursor.fetchall()
        return result

    def insert_face(self, name, face_id):
        sql = "INSERT INTO face (name, face_id) VALUES (%s, %s)"
        val = (name, face_id)
        self.cursor.execute(sql, val)
        self.db.commit()
